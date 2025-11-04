from __future__ import annotations

import asyncio
import logging

from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import FamousParams, GeneralParams, FactParams, RouteSelection
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
    ):
        self.ai_client = ai_client
        self.telemetry = telemetry
        self.language_detector = language_detector
        self.famous_generator = famous_generator
        self.general_generator = general_generator
        self.fact_handler = fact_handler

    def _build_route_selection_prompt(self) -> str:
        """Build focused prompt for route selection only (tier 1)."""
        famous_desc = self.famous_generator.get_route_description()
        general_desc = self.general_generator.get_route_description()
        fact_desc = self.fact_handler.get_route_description()

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

<route_definitions>
<route route="FAMOUS">
{famous_desc}
</route>

<route route="GENERAL">
{general_desc}
</route>

<route route="FACT">
{fact_desc}
</route>

<route route="NONE">
NONE: For everything else
- Simple reactions, acknowledgments, or invalid queries
- Conversations about the BOT without a direct request to it
- Examples:
  - "lol", "nice", "ok", "haha", random gibberish
  - "Did you see BOT's last response? It was hilarious."
  - "I think BOT is getting smarter at answering questions."
  - "You should ask BOT about that."
  - "I like that new feature where you can mention BOT anywhere in the sentence."
  - Any message containing references to child sexual abuse
</route>

<route route="NOTSURE">
NOTSURE: When uncertain about routing decision
- Message is ambiguous or could fit multiple categories
- User intent is unclear or lacks sufficient context
- You're unsure about the semantic meaning
- May trigger fallback to more capable model for re-evaluation
</route>
</route_definitions>
"""

    async def _extract_parameters(
        self,
        route: str,
        message: str,
    ) -> FamousParams | GeneralParams | FactParams | None:
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
        else:
            raise ValueError(f"Unknown route: {route}")

        # Get schema and prompt from the handler
        param_schema = handler.get_parameter_schema()
        extraction_prompt = handler.get_parameter_extraction_prompt()

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

    async def route_request(
        self,
        message: str,
    ) -> tuple[str, FamousParams | GeneralParams | FactParams | None]:
        """Route a message using 2-tier approach: route selection, language detection, then parameter extraction."""
        async with self.telemetry.async_create_span("route_request") as span:
            span.set_attribute("message", message)

            # Tier 1: Route selection and language detection (run in parallel)
            route_prompt = self._build_route_selection_prompt()

            route_selection, language_code = await asyncio.gather(
                self.ai_client.generate_content(
                    message=message,
                    prompt=route_prompt,
                    temperature=0.0,  # Deterministic route selection
                    response_schema=RouteSelection,
                ),
                self.language_detector.detect_language(message),
            )

            logger.info(f"Final route selection: {route_selection.route}, Reason: {route_selection.reason}")

            span.set_attribute("route", route_selection.route)
            span.set_attribute("reason", route_selection.reason)

            language_name = await self.language_detector.get_language_name(language_code)
            span.set_attribute("language_code", language_code)
            span.set_attribute("language_name", language_name)
            logger.info(f"Detected language: {language_code} ({language_name})")

            # Tier 2: Parameter extraction (if needed)
            try:
                params = await self._extract_parameters(route_selection.route, message)
            except Exception:
                # Record routing attempt with error outcome
                self.telemetry.metrics.route_selections_counter.add(
                    1, {"route": route_selection.route, "outcome": "error", "language_code": language_code}
                )
                raise

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
                    span.set_attribute("temperature", params.temperature)
                elif route_selection.route == "FACT":
                    span.set_attribute("fact_operation", params.operation)
                    span.set_attribute("fact_user_mention", params.user_mention)
                    span.set_attribute("fact_content", params.fact_content)

            return (route_selection.route, params)
