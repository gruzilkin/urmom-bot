import hashlib
import logging

from ai_client import AIClient
from conversation_formatter import (
    CONVERSATION_STRUCTURE_INSTRUCTIONS,
    ConversationFormatter,
    chat_message_to_conversation_message,
)
from open_telemetry import Telemetry
from redis_cache import RedisCache
from schemas import TaskHistorySummary
from store import ChatMessage, ScheduledTask, Store

logger = logging.getLogger(__name__)

MIN_PREVIOUS_RUNS = 3

PROMPT_TEMPLATE = """You summarize the previous runs of a recurring scheduled task run by a Discord bot.

<task_prompt>
{task_prompt}
</task_prompt>

The <conversation_history> holds this task's reply chain, oldest first: the request that
created the task (if still available) and the bot's previous outputs for this task, which are
the messages from member_id {bot_user_id}.

{conversation_structure}

The summary is given to the next run of this task as its context about previous runs.
Summarize the previous runs according to what is important for the <task_prompt>: the task
prompt alone decides which information about previous runs matters. Record only what the
messages actually contain, keep it compact, and do not include any instructions for the
next run.
"""


class TaskHistorySummarizer:
    def __init__(
        self,
        ai_client: AIClient,
        store: Store,
        conversation_formatter: ConversationFormatter,
        redis_cache: RedisCache,
        telemetry: Telemetry,
        char_budget: int,
    ) -> None:
        self.ai_client = ai_client
        self.store = store
        self.conversation_formatter = conversation_formatter
        self.redis_cache = redis_cache
        self.telemetry = telemetry
        self.char_budget = char_budget

    async def summarize(self, task: ScheduledTask, bot_user_id: int) -> str | None:
        """Summarize the task's reply chain in the context of its prompt.

        Returns None when there are too few previous runs to need a summary (the
        conversation history already shows them verbatim) or when summarization fails.
        Summaries are cached by the selected chain messages and the task prompt, so runs
        that post nothing (and leave the chain unchanged) reuse the previous summary.
        """
        async with self.telemetry.async_create_span("summarize_task_history") as span:
            span.set_attribute("task_id", task.task_id)
            span.set_attribute("chain_length", len(task.chain_message_ids))

            try:
                messages = await self.store.get_task_chain_messages(task.task_id)
                history = trim_to_budget(messages, self.char_budget)
                previous_runs = sum(1 for m in history if m.user_id == bot_user_id)

                span.set_attribute("history_messages", len(history))
                span.set_attribute("history_chars", sum(len(m.message_text) for m in history))
                span.set_attribute("history_truncated", len(history) < len(messages))
                span.set_attribute("previous_runs", previous_runs)

                if previous_runs < MIN_PREVIOUS_RUNS:
                    span.set_attribute("status", "skipped")
                    return None

                messages_hash = hashlib.md5(",".join(str(m.message_id) for m in history).encode()).hexdigest()
                prompt_hash = hashlib.md5(task.prompt.encode()).hexdigest()
                cached = await self.redis_cache.get_task_history_summary(task.task_id, messages_hash, prompt_hash)
                span.set_attribute("cache_hit", cached is not None)
                if cached is not None:
                    logger.info(f"Task {task.task_id} history summary cache hit")
                    span.set_attribute("summary_chars", len(cached))
                    span.set_attribute("status", "success")
                    return cached

                chain_xml = await self.conversation_formatter.format_to_xml(
                    [chat_message_to_conversation_message(m) for m in history]
                )
                result: TaskHistorySummary = await self.ai_client.generate_content(
                    message=chain_xml,
                    prompt=PROMPT_TEMPLATE.format(
                        task_prompt=task.prompt,
                        bot_user_id=bot_user_id,
                        conversation_structure=CONVERSATION_STRUCTURE_INSTRUCTIONS,
                    ),
                    temperature=0.0,
                    response_schema=TaskHistorySummary,
                )
                logger.info(f"Task {task.task_id} history summary reason: {result.reason}")
                await self.redis_cache.set_task_history_summary(
                    task.task_id, messages_hash, prompt_hash, result.summary
                )

                span.set_attribute("summary_chars", len(result.summary))
                span.set_attribute("status", "success")
                return result.summary
            except Exception as e:
                span.set_attribute("status", "error")
                logger.error(f"Failed to summarize history for scheduled task {task.task_id}: {e}", exc_info=True)
                return None


def trim_to_budget(messages: list[ChatMessage], char_budget: int) -> list[ChatMessage]:
    """Keep the newest messages whose combined text fits char_budget, in chronological order."""
    kept: list[ChatMessage] = []
    total = 0
    for message in reversed(messages):
        total += len(message.message_text)
        if total > char_budget:
            break
        kept.append(message)
    kept.reverse()
    return kept
