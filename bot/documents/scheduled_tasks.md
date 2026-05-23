# Scheduled Tasks

## Overview

The scheduled tasks feature lets users instruct the bot to run prompts on a schedule — either recurring (cron-style) or one-off (delayed). Tasks belong to a Discord channel and are created, listed, edited, and managed through natural-language commands directed at the bot, the same way memory commands and guild settings are managed today.

When a scheduled task fires, the bot executes its stored prompt through the general query generator and posts the result to the task's channel. Execution reuses the existing generator wholesale; the new work is the scheduling system around it.

## Schedule Representation

### Cron Expressions

Recurring schedules are stored as standard cron expressions (5-field minute/hour/day-of-month/month/day-of-week). The well-maintained `croniter` library handles all forward and backward time arithmetic — computing next firings, computing the most recent past firing for catch-up logic, and previewing several upcoming firings for user confirmation at creation time.

### One-Off Tasks

A one-off task ("remind me tomorrow at 3pm") is stored on the same table as recurring tasks but with a null cron expression. The presence or absence of a cron expression is the sole differentiator between "advance after firing" and "delete after firing."

### Natural-Language Time Expressions

LLMs are good at translating recurrence patterns to cron (`"every Monday at 9am"` → `0 9 * * 1`) but unreliable at date arithmetic (`"tomorrow at 3pm"` → an exact timestamp). The system splits the responsibility: the LLM never produces absolute timestamps. For non-pattern times it produces a natural-language phrase that a deterministic library (`dateparser`) resolves to a timestamp, anchored at the current time and the task's timezone.

The LLM extraction therefore produces two complementary fields per task:
- `cron_expression` — populated for recurring schedules.
- `first_run_phrase` — populated for one-off tasks, or for recurring tasks where the user explicitly specifies the first firing.

Three legal combinations:

| Pattern | `cron_expression` | `first_run_phrase` | First firing computed from |
|---|---|---|---|
| Pure recurring | set | null | `croniter(cron, now)` |
| Recurring with explicit first run | set | set | `dateparser(phrase)`, then cron advances from there |
| Pure one-off | null | set | `dateparser(phrase)` |

Both null is an extraction failure — the bot asks the user to clarify.

`first_run_phrase` is consumed at creation/edit time to compute `next_run_at` and is *not* persisted on the row. All subsequent firings of a recurring task come from `croniter(cron, prev_run_at).get_next()` — the explicit first run only affects the initial firing.

If `dateparser` cannot resolve the phrase, the handler falls back to asking the LLM directly for an ISO timestamp (with current time and timezone injected into the prompt). Whichever path resolves it, the handler echoes the parsed firing time back in the confirmation message so the user can spot errors immediately.

### Timezones

A channel's members may live in different timezones, so "every Monday at 9am" needs to fire at the user's local 9am, not at 9am in whatever timezone the server happens to be in. The task row carries the cron expression as the user wrote it plus an IANA timezone name; cron arithmetic is anchored to a timezone-aware "now" at firing-decision time.

DST is out of scope for v1 — the targeted deployment timezones do not observe DST. If we later expand to DST zones, the timezone-aware cron arithmetic is the right foundation; the verification work would just shift to `croniter`'s transition-day behavior.

What the task row carries:
- The cron expression produced by the LLM from the user's natural-language description.
- An IANA timezone name (always populated — never null).
- `next_run_at` as a UTC instant: the engine's hot-path comparison column.

`next_run_at` is a moment in time; the engine compares `now_utc >= next_run_at` and dispatches. Timezone affects how that instant is *computed* from the cron expression, not how it is stored or queried.

### Timezone Resolution at Creation

Three sources, in priority order:
1. **Explicit in the user's utterance.** "every Monday 9am Tokyo" yields `Asia/Tokyo`. The cron-parser LLM returns the timezone alongside the cron expression in its structured output, validated against the IANA database.
2. **Guild default.** A new guild setting (alongside existing settings like `setArchiveChannel`, `enableCountryJokes`) — e.g., `setDefaultTimezone Europe/Berlin`. Used when the user's utterance does not specify one.
3. **System fallback.** UTC, if neither is set.

The resolved timezone is frozen on the task row at creation. Even if the task uses the guild default at creation time, the resolved IANA name is written to the row — so subsequent changes to the guild default do not silently shift when existing tasks fire.

### Display

Firing times are displayed in the task's timezone — both during creation preview and via `SCHEDULE_NEXT`. That is the timezone the user wrote and reasoned in. For channels with members in mixed timezones, this still works: members convert mentally if they need to. Surfacing all readers' timezones would be noise.

### Freeform Schedule Parsing

Users describe their schedule in natural language. An LLM normalizes the description into structured fields: a cron expression for recurring patterns, a natural-language phrase for one-off tasks or explicit first-run anchors. The handler then validates the output deterministically — cron expressions through `croniter`, phrases through `dateparser` — before persisting anything.

There is no preview-then-confirm step. The bot maintains no conversational state across messages, so a two-turn "here is what I parsed, confirm?" flow is unreliable. Instead, the LLM's output schema includes a `reason` field. On a successful parse the handler persists immediately and returns `reason` to the user — populated with a friendly confirmation that includes the resolved schedule and next firing time, so the user can spot any mistranslation and follow up with `edit` or `delete`. On an unsuccessful parse (LLM can't make sense of the request), `reason` carries the explanation and the bot relays it without persisting anything.

The cron-parsing fallback chain follows the existing linguistic-task pattern documented in `llm_fallback.md`: gemma (60-second retry) → kimi (fail fast). To be confirmed at implementation time.

## Database Schema

The canonical SQL definitions live in `db/init.sql`. The data model:

**`scheduled_tasks`** — one row per task.
- Identification: surrogate `task_id` (BIGSERIAL), guild ID, channel ID, creator user ID. Tasks are addressed by `task_id` for all management operations (delete, edit, pause, etc.) — IDs are stable across edits and easier to log and reference than free-form names.
- Definition: prompt text, normalized cron expression (nullable for one-off tasks), IANA timezone.
- State: next intended run time (the engine's hot-path query column, stored as a UTC instant), last actual run time.
- Audit: created at, updated at.

Things deliberately *not* persisted on the row:
- The human-readable schedule string ("every Monday 9am Europe/Berlin") — re-rendered from `cron_expression` + `timezone` on demand for `list` output and creation confirmation messages.
- Language code / name — detected at firing time by the same pipeline used for reactive mentions, with in-process caching keyed by prompt content if cost ever matters.

The principle: anything derivable from `prompt` + `cron_expression` + `timezone` should be derived rather than stored, so edits never leave secondary fields stale.

The schema deliberately does *not* track per-task failure counts. Failed firings show up in logs and telemetry; the engine never retries within a tick (because `next_run_at` is advanced before dispatch), so a task whose channel was deleted simply fails on every scheduled firing and surfaces in observability rather than spinning into a retry storm or self-pausing.

Indexes:
- `next_run_at` — the engine's tick-loop query.
- `(guild_id, channel_id)` — the `list` operation's channel-scoped query.

Timestamp columns use `TIMESTAMPTZ` to make timezone semantics explicit on a feature where timezone correctness is the central concern. This deviates from the rest of the schema (which uses `TIMESTAMP` and assumes the application stores UTC), and is a deliberate one-table exception.

## Management Commands

Scheduling is managed through natural-language mentions, parsed by the AI router. The router gains a `SCHEDULE` route alongside the existing `FAMOUS`, `GENERAL`, `FACT`, `NONE`. The router identifies the route and a sub-operation:

| Operation | User utterance shape |
|---|---|
| `create` | "schedule a weekly summary every Monday 9am" |
| `list` | "what's scheduled in this channel?" |
| `edit` | "change task 5's prompt to summarize the past two weeks" |
| `delete` | "delete task 5" |
| `run_now` | "run task 5 now" |

The router produces a `ScheduleParams` object containing only the `operation` discriminator (plus the standard language fields). It does *not* try to identify which task the user is referring to. Task identification, schedule parsing, and edit interpretation all happen in tier 3 inside the schedule handler, where the channel's task list is available as context. The original user message is passed through to the handler verbatim; each sub-operation makes its own LLM call against the relevant context (task list, current datetime, the existing task being edited, etc.) and its own dedicated output schema.

A new `schedule_handler` (mirroring `fact_handler`) owns all sub-operations against the database. `run_now` is part of the surface — without it, debugging a weekly task means waiting a week to see if it works.

### Operation Details

- **Create.** Handler passes the user's message to an LLM with the `ScheduleTaskFields` schema. If the LLM populates the required fields validly (`prompt`, `timezone`, and at least one of `cron_expression` / `first_run_phrase`), the handler validates via `croniter` / `dateparser`, computes `next_run_at`, inserts the row, and returns the LLM's `reason` field as confirmation (including the resolved schedule and next firing time, and the new task ID appended). If the LLM returns null fields with an explanation in `reason`, the handler returns `reason` without persisting anything.
- **List.** Handler reads all tasks for the channel, formats them as context, and asks an LLM to answer the user's request in that context. The user's request is freeform: "what's scheduled?", "what runs on Mondays?", "show only the morning ones." The LLM returns the answer text directly.
- **Edit.** Handler reads the existing task, passes (existing task fields + user's change request) to an LLM with the *same* `ScheduleTaskFields` schema as create. The LLM returns the full updated task — preserving fields the user did not change, modifying only what was asked. Same shape as the "remember" merge in `fact_handler` (load existing → merge with new info → save full result). Updated row replaces the old. v1 use cases lean toward prompt tweaks; cron edits are supported by the same flow.
- **Delete.** Handler deletes the row by `task_id` (scoped to the originating channel) and returns confirmation. No LLM call.
- **Run Now.** Handler reads the task and asks the schedule engine to fire it once, outside the schedule. The engine produces output identical to a regular firing and posts it to the channel; the handler returns a brief acknowledgment.

## Engine

### Tick Loop

A single asyncio background task wakes once per minute and queries for due tasks: rows whose `next_run_at` is in the past or present, ordered by `next_run_at`. The engine starts during application bootstrap, alongside other DI-container services.

### Dispatch

For each due task, the engine atomically advances `next_run_at` to the next future occurrence (or marks the task as completed for one-offs) *before* dispatching execution. This is critical: it prevents the same task from being picked up again on the next tick if its execution is still running. Execution proceeds as a separate coroutine, bounded by a concurrency semaphore to respect AI provider quotas.

After execution completes, `last_run_at` is updated. Per-firing history is not persisted; outcomes are visible through logs and telemetry, and produced messages live in Discord itself.

### Catch-Up After Downtime

The contract: a recurring task that missed its window while the bot was offline fires at most once on next startup, with its schedule advanced to the next future occurrence. Intermediate missed firings are skipped silently. This is the right default because Discord users do not want seven daily-summary posts in a row when the bot returns from a week-long outage.

A staleness threshold (e.g., 12 hours) guards against truly stale make-up firings: if the missed firing is older than the threshold, even the single make-up is skipped. The threshold is configurable per task — most tasks will use the default, but a "weekly important reminder" task may want a longer window.

One-off tasks are an explicit exception: a missed one-off *always* fires on next startup, regardless of how stale it is. The whole point of "remind me tomorrow at 3pm" is to fire eventually, even if the bot was down through the original time.

### Concurrency Safety

A long-running AI call could outlast the one-minute tick interval. The "advance `next_run_at` before dispatch" rule above eliminates double-firing from re-selection. If a task definition is edited while a firing is in flight, the running firing uses the snapshot it dispatched with; the next firing uses the new definition.

## Task Execution

When a task fires, the engine invokes the general query generator with the task's stored prompt, creator user ID (for memory context), and channel ID (for conversation history). A flag marks the invocation as scheduler-originated so the generator can suppress reactive-only behavior (replying to a triggering message that doesn't exist) and so telemetry can distinguish scheduled from reactive load.

That is the whole execution path. No new generator, no new prompt format, no routing decision at firing time. Language detection, memory injection, conversation history fetching, and AI fallback chains all work exactly as they do for a reactive mention.

Specialized generators (`JokeGenerator`, `FamousPersonGenerator`, `WisdomGenerator`) are intentionally not schedulable as distinct execution paths in v1. The general query generator is expressive enough to handle scheduled prompts that would have been routed to those generators reactively, accepting the trade-off that their specialized prompt-engineering is absent. A nullable `generator_override` column can be added later if usage demands it, without disturbing the architecture.

## Failure Modes

- **Channel deleted or bot lost access.** The firing fails; the failure is logged and traced. Because `next_run_at` is advanced before dispatch, a failing firing does not retry within a tick — the task just fails again at its next scheduled time. The system makes no attempt to self-heal; an operator who notices repeated failures in logs is expected to delete the task manually.
- **AI provider failures.** Inherited from the existing fallback chains. A task firing only fails outright if every provider in the chain fails.
- **Creator left the guild.** The task continues running. Memory context for the creator may be reduced but the task itself is owned by the channel, not the user.
- **Quota abuse.** Per-guild and per-creator caps on active task count and per-day firing count. Cap policy is an open question.
- **Edits during firing.** The running firing uses the snapshot it dispatched with. The next firing uses the new definition.

## Permissions

There is no per-operation permissions model. Tasks belong to the channel, and anyone in the channel can create, list, edit, delete, or run any task in that channel. The `creator_user_id` is recorded on the row for audit and telemetry purposes only — it does not gate any operation.

Channel-level visibility is implicit: a task is tied to a channel, so users who cannot see the channel cannot see or act on its tasks.

## Telemetry

Each firing produces a span with the task ID, intended run time, actual run time, drift (intended − actual), and status. Counters track scheduled-vs-reactive load distribution and missed-firing-due-to-downtime events.

The schedule-management surface produces spans tagged by intent (`SCHEDULE_CREATE`, etc.) for observability of how users actually use the feature.

## Open Questions

These design calls are not yet decided and need to be pinned down before implementation:

1. **Catch-up staleness threshold.** Default value (12 hours suggested) and whether it is per-task configurable or guild-wide.
2. **Quotas.** Specific caps on active task count per guild/channel/creator and per-day firing counts. Worth deferring concrete numbers until early usage data is available, but the *enforcement points* should exist from day one.
3. **Cron parser fallback chain.** Confirm gemma → kimi (linguistic-task default) is the right fit, or whether the cron-translation task's deterministic nature warrants a different chain.
4. **Command UX.** Stick with the existing freeform mention pattern (matches `remember`/`forget`/settings) or adopt slash commands (more discoverable for a new feature, but breaks pattern).
