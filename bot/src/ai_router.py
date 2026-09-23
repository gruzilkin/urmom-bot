from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from ai_client import AIClient
from conversation_formatter import ConversationFormatter
from conversation_graph import ConversationMessage
from open_telemetry import Telemetry
from route_selector import RouteSelector
from schemas import FamousParams, FactParams, GeneralParams, ScheduleParams
from language_detector import LanguageDetector

logger = logging.getLogger(__name__)


# RouterDecision class removed - now using tuple return (route_name, params_object)


class AiRouter:
    def __init__(
        self,
        ai_client: AIClient,
        telemetry: Telemetry,
        language_detector: LanguageDetector,
        famous_generator,
        general_generator,
        fact_handler,
        conversation_formatter: ConversationFormatter,
        schedule_handler,
        memory_manager,
        route_selector: RouteSelector,
    ):
        self.ai_client = ai_client
        self.route_selector = route_selector
        self.telemetry = telemetry
        self.language_detector = language_detector
        self.famous_generator = famous_generator
        self.general_generator = general_generator
        self.fact_handler = fact_handler
        self.conversation_formatter = conversation_formatter
        self.schedule_handler = schedule_handler
        self.memory_manager = memory_manager

    async def _extract_parameters(
        self,
        route: str,
        message: str,
        conversation_context: str = "",
    ) -> FamousParams | GeneralParams | FactParams | ScheduleParams | None:
        """Extract parameters for the selected route (tier 2)."""
        if route == "NONE":
            return None

        # Get the appropriate route handler
        if route == "FAMOUS":
            handler = self.famous_generator
        elif route == "GENERAL":
            handler = self.general_generator
        elif route == "FACT":
            handler = self.fact_handler
        elif route == "SCHEDULE":
            handler = self.schedule_handler
        else:
            raise ValueError(f"Unknown route: {route}")

        # Get schema and prompt from the handler
        param_schema = handler.get_parameter_schema()
        extraction_prompt = handler.get_parameter_extraction_prompt(conversation_context)

        async with self.telemetry.async_create_span("extract_parameters") as span:
            span.set_attribute("route", route)
            span.set_attribute("message", message)
            params = await self.ai_client.generate_content(
                message=message,
                prompt=extraction_prompt,
                temperature=0.0,  # Deterministic parameter extraction
                response_schema=param_schema,
            )
            logger.info(f"Extracted parameters for {route}: {params}")
            return params

    async def extract_general_params(self, message: str, conversation_context: str = "") -> GeneralParams:
        """Extract GENERAL-route parameters without route selection.

        For flows that must never be routed elsewhere — e.g. scheduled task firings,
        which would otherwise be able to create new scheduled tasks."""
        params, language_code = await asyncio.gather(
            self._extract_parameters("GENERAL", message, conversation_context),
            self.language_detector.detect_language(message),
        )
        params.language_code = language_code
        params.language_name = await self.language_detector.get_language_name(language_code)
        return params

    async def route_request(
        self,
        message: str,
        conversation_fetcher: Callable[[], Awaitable[list[ConversationMessage]]],
        guild_id: int,
    ) -> tuple[str, FamousParams | GeneralParams | FactParams | ScheduleParams | None]:
        """Route a message using 2-tier approach: route selection (delegated to the RouteSelector),
        language detection, then parameter extraction."""
        async with self.telemetry.async_create_span("route_request") as span:
            span.set_attribute("message", message)

            conversation_context = ""
            conversation = await conversation_fetcher()
            if conversation:
                conversation_context = await self.conversation_formatter.format_to_xml(guild_id, conversation)
                member_ids: set[int] = {msg.author_id for msg in conversation}
                for msg in conversation:
                    member_ids.update(msg.mentioned_user_ids)
                memories_block = await self.memory_manager.build_memory_prompt(guild_id, member_ids)
                if memories_block:
                    conversation_context = f"{memories_block}\n{conversation_context}"

            # Start language detection in background while routing proceeds
            lang_task = asyncio.create_task(self.language_detector.detect_language(message))

            # Tier 1: Route selection (Jev and/or LLM, see route_selector.py)
            route_selection = await self.route_selector.select(message, conversation_context)

            logger.info(f"Final route selection: {route_selection.route}, Reason: {route_selection.reason}")
            span.set_attribute("route", route_selection.route)
            span.set_attribute("route_reason", route_selection.reason)

            # Tier 2: Parameter extraction (starts immediately, doesn't wait for language detection)
            try:
                params = await self._extract_parameters(route_selection.route, message, conversation_context)
            except Exception:
                language_code = await lang_task
                self.telemetry.metrics.route_selections_counter.add(
                    1, {"route": route_selection.route, "outcome": "error", "language_code": language_code}
                )
                raise

            language_code = await lang_task
            language_name = await self.language_detector.get_language_name(language_code)
            span.set_attribute("language_code", language_code)
            span.set_attribute("language_name", language_name)
            logger.info(f"Detected language: {language_code} ({language_name})")

            # Count route selection success
            self.telemetry.metrics.route_selections_counter.add(
                1, {"route": route_selection.route, "outcome": "success", "language_code": language_code}
            )

            # Add language information to the extracted parameters
            if params:
                if hasattr(params, "language_code"):
                    params.language_code = language_code
                if hasattr(params, "language_name"):
                    params.language_name = language_name

            # Add parameter details to telemetry
            if params:
                if route_selection.route == "FAMOUS":
                    span.set_attribute("famous_person", params.famous_person)
                elif route_selection.route == "GENERAL":
                    span.set_attribute("ai_backend", params.ai_backend)
                    span.set_attribute("ai_backend_reason", params.reason)
                    span.set_attribute("temperature", params.temperature)
                elif route_selection.route == "FACT":
                    span.set_attribute("fact_operation", params.operation)
                    if params.member_id is not None:
                        span.set_attribute("fact_member_id", params.member_id)
                    span.set_attribute("fact_content", params.fact_content)
                elif route_selection.route == "SCHEDULE":
                    span.set_attribute("schedule_operation", params.operation)

            return (route_selection.route, params)
