"""Tier-1 route selection strategies for the AI router.

`AiRouter.route_request` delegates the "which route?" decision to a `RouteSelector`. Two
implementations exist: `LlmRouteSelector` asks an `AIClient` with the classic prompt, and
`JevRouteSelector` asks Jev a single choice question over the definite routes, answering NOTSURE when
the top probability is too low. `CompositeRouteSelector` chains them, falling through whenever a
selector answers NOTSURE or fails, mirroring `CompositeAIClient`.
"""

from __future__ import annotations

import logging
from typing import Protocol, get_args

from pydantic import BaseModel, create_model

from ai_client import AIClient
from fact_handler import FactHandler
from famous_person_generator import FamousPersonGenerator
from general_query_generator import GeneralQueryGenerator
from jev_client import Choice, JevClient, choice
from open_telemetry import Telemetry
from schedule_handler import ScheduleHandler
from schemas import DefiniteRouteName, RouteName, RouteSelection

logger = logging.getLogger(__name__)


NONE_ROUTE_DESCRIPTION = """NONE: For everything else
- Simple reactions, acknowledgments, or invalid queries
- Conversations about the BOT without a direct request to it
- Examples:
  - "lol", "nice", "ok", "haha", random gibberish
  - "Did you see BOT's last response? It was hilarious."
  - "I think BOT is getting smarter at answering questions."
  - "You should ask BOT about that."
  - "I like that new feature where you can mention BOT anywhere in the sentence."
  - Any message containing references to child sexual abuse"""

NOTSURE_ROUTE_DESCRIPTION = """NOTSURE: When uncertain about routing decision
- Message is ambiguous or could fit multiple categories
- User intent is unclear or lacks sufficient context
- You're unsure about the semantic meaning
- May trigger fallback to more capable model for re-evaluation"""


def build_route_descriptions(
    famous: FamousPersonGenerator,
    general: GeneralQueryGenerator,
    fact: FactHandler,
    schedule: ScheduleHandler,
) -> dict[str, str]:
    """Collect one description per selectable route, keyed by route name, including NOTSURE for the
    LLM prompt."""
    return {
        "FAMOUS": famous.get_route_description().strip(),
        "GENERAL": general.get_route_description().strip(),
        "FACT": fact.get_route_description().strip(),
        "SCHEDULE": schedule.get_route_description().strip(),
        "NONE": NONE_ROUTE_DESCRIPTION,
        "NOTSURE": NOTSURE_ROUTE_DESCRIPTION,
    }


class RouteSelector(Protocol):
    async def select(self, message: str, conversation_context: str) -> RouteSelection: ...


class LlmRouteSelector:
    """Route selection by an LLM with the full instruction prompt; may answer NOTSURE."""

    def __init__(self, ai_client: AIClient, route_descriptions: dict[str, str], telemetry: Telemetry) -> None:
        self.ai_client = ai_client
        self.route_descriptions = route_descriptions
        self.telemetry = telemetry

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.ai_client!r})"

    def build_prompt(self, conversation_context: str = "") -> str:
        if conversation_context:
            conversation_context = f"""
<conversation_context>
Route the LAST message.
Use earlier messages to resolve references like "this", "that", "it" in the last message.

{conversation_context}
</conversation_context>
"""

        route_blocks = "\n\n".join(
            f'<route route="{route}">\n{self.route_descriptions[route]}\n</route>' for route in get_args(RouteName)
        )

        return f"""
<system_instructions>
Analyze the user message and decide how to route it. Choose exactly one route.

**CRITICAL: ACCURACY IS THE TOP PRIORITY. DO NOT GUESS.**

**IMPORTANT: The user message can be in ANY language (English, Russian, French, Japanese, etc.).
Route based on the SEMANTIC MEANING and INTENT of the message, not specific keywords or language.**

**CONFIDENCE REQUIREMENTS:**
- Only choose a specific route when you are ABSOLUTELY CERTAIN about the user's intent
- If there is ANY doubt, ambiguity, or uncertainty - choose NOTSURE immediately
- DO NOT make routing decisions based on keyword presence alone
- ACCURACY over speed - being uncertain is better than being wrong
- When in doubt, choose NOTSURE - this is strongly preferred over incorrect routing

Instructions:
1. Check if the message contains references to child sexual abuse. If yes, choose NONE immediately.
2. Read the user message carefully, understanding its semantic meaning regardless of language.
3. Assess your confidence: Are you ABSOLUTELY CERTAIN about the intent?
4. If not absolutely certain, choose NOTSURE immediately - this is the preferred choice.
5. Consider that all route types can be expressed in any language:
   - Famous person requests: "What would X say?" / "Что бы сказал X?" / "¿Qué diría X?"
   - Memory operations: "Remember that..." / "Запомни что..." / "Recuerda que..."
   - General queries: "Explain..." / "Объясни..." / "Explica..."
6. ALWAYS provide a brief (1-2 sentence) reason for your decision.
7. Focus ONLY on route selection - parameter extraction happens later.
</system_instructions>
{conversation_context}
<route_definitions>
{route_blocks}
</route_definitions>
"""

    async def select(self, message: str, conversation_context: str) -> RouteSelection:
        async with self.telemetry.async_create_span("select_route") as span:
            span.set_attribute("route_selector", "llm")
            selection = await self.ai_client.generate_content(
                message=message,
                prompt=self.build_prompt(conversation_context),
                temperature=0.0,
                response_schema=RouteSelection,
            )
            span.set_attribute("route", selection.route)
            return selection


JEV_ROUTE_INSTRUCTIONS = (
    "Decide how to route the user message in `message`. The message can be in any language; route by "
    "semantic meaning and intent, not keywords. If `conversation_context` is present, route only the "
    "message in `message` and use the earlier conversation to resolve references like 'this', 'that', "
    "'it'. Any message referencing child sexual abuse is NONE."
)


class JevRouteSelector:
    """Route selection by a single Jev choice question over the definite routes; the highest-probability
    route wins unless it falls below `min_probability`, in which case the answer is NOTSURE."""

    def __init__(
        self,
        jev_client: JevClient,
        route_descriptions: dict[str, str],
        telemetry: Telemetry,
        min_probability: float = 0.9,
    ) -> None:
        self.jev_client = jev_client
        self.telemetry = telemetry
        self.min_probability = min_probability
        self.questions: type[BaseModel] = create_model(
            "RouteDecision",
            route=(
                Choice[DefiniteRouteName],
                choice(
                    instructions=JEV_ROUTE_INSTRUCTIONS,
                    criteria={route: route_descriptions[route] for route in get_args(DefiniteRouteName)},
                ),
            ),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.jev_client!r})"

    def resolve(self, answer: Choice) -> RouteName:
        return answer.choice if answer.top_probability >= self.min_probability else "NOTSURE"

    async def select(self, message: str, conversation_context: str) -> RouteSelection:
        async with self.telemetry.async_create_span("select_route") as span:
            span.set_attribute("route_selector", "jev")
            state: dict[str, str] = {"message": message}
            if conversation_context:
                state["conversation_context"] = conversation_context

            decision = await self.jev_client.ask(state=state, questions=self.questions)
            answer: Choice = decision.route
            route = self.resolve(answer)

            ranked = sorted(answer.probabilities.items(), key=lambda item: item[1], reverse=True)
            reason = "jev " + ", ".join(f"{name}={probability:.2f}" for name, probability in ranked)

            span.set_attribute("route", route)
            span.set_attribute("confidence", answer.confidence)
            span.set_attribute("top_probability", answer.top_probability)
            logger.info(f"Jev route selection: {route} ({reason})")
            return RouteSelection(route=route, reason=reason)


class CompositeRouteSelector:
    """Try selectors in order; move on when one answers NOTSURE or raises."""

    def __init__(self, selectors: list[RouteSelector], telemetry: Telemetry) -> None:
        if not selectors:
            raise ValueError("CompositeRouteSelector requires at least one selector")
        self.selectors = tuple(selectors)
        self.telemetry = telemetry

    def __repr__(self) -> str:
        return f"{type(self).__name__}([{', '.join(repr(s) for s in self.selectors)}])"

    async def select(self, message: str, conversation_context: str) -> RouteSelection:
        last_error: Exception | None = None
        async with self.telemetry.async_create_span("composite_select_route") as span:
            for index, selector in enumerate(self.selectors):
                try:
                    selection = await selector.select(message, conversation_context)
                except Exception as e:
                    last_error = e
                    logger.error(f"Route selector {selector!r} failed: {e}", exc_info=True)
                    continue
                if selection.route != "NOTSURE":
                    span.set_attribute("selected_by", type(selector).__name__)
                    span.set_attribute("selector_index", index)
                    return selection
                logger.info(f"Route selector {selector!r} was not sure, trying next")
            raise RuntimeError("All route selectors failed or were unsure") from last_error


__all__ = [
    "CompositeRouteSelector",
    "JevRouteSelector",
    "LlmRouteSelector",
    "NONE_ROUTE_DESCRIPTION",
    "NOTSURE_ROUTE_DESCRIPTION",
    "RouteSelector",
    "build_route_descriptions",
]
