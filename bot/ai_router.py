import logging
from typing import Union, Optional, Tuple
from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import FamousParams, GeneralParams, FactParams, RouteSelection
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


# RouterDecision class removed - now using tuple return (route_name, params_object)


class AiRouter:
    def __init__(self, ai_client: AIClient, telemetry: Telemetry, 
                 famous_generator, general_generator, fact_handler):
        self.ai_client = ai_client
        self.telemetry = telemetry
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

**IMPORTANT: The user message can be in ANY language (English, Russian, French, Japanese, etc.). 
Route based on the SEMANTIC MEANING and INTENT of the message, not specific keywords or language.**

Instructions:
1. Read the user message carefully, understanding its meaning regardless of language.
2. Determine which route best matches the intent semantically.
3. Consider that all route types can be expressed in any language:
   - Famous person requests: "What would X say?" / "Что бы сказал X?" / "¿Qué diría X?"
   - Memory operations: "Remember that..." / "Запомни что..." / "Recuerda que..."
   - General queries: "Explain..." / "Объясни..." / "Explica..."
4. ALWAYS provide a brief (1-2 sentence) reason for your decision.
5. Focus ONLY on route selection - parameter extraction happens later.
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
</route>
</route_definitions>
"""

    async def _extract_parameters(self, route: str, message: str) -> Union[FamousParams, GeneralParams, FactParams, None]:
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
                response_schema=param_schema
            )
            
            logger.info(f"Extracted parameters for {route}: {params}")
            return params
    
    async def route_request(self, message: str) -> Tuple[str, Union[FamousParams, GeneralParams, FactParams, None]]:
        """Route a message using 2-tier approach: route selection then parameter extraction."""
        async with self.telemetry.async_create_span("route_request") as span:
            span.set_attribute("message", message)
            
            # Tier 1: Route selection
            route_prompt = self._build_route_selection_prompt()
            route_message = f"{route_prompt}\n<user_input>{message}</user_input>"
            
            route_selection = await self.ai_client.generate_content(
                message=route_message,
                prompt="",
                temperature=0.0,  # Deterministic route selection
                response_schema=RouteSelection
            )
            
            span.set_attribute("route", route_selection.route)
            span.set_attribute("reason", route_selection.reason)
            logger.info(f"Route selection: {route_selection.route}, Reason: {route_selection.reason}")
            
            # Tier 2: Parameter extraction (if needed)
            params = await self._extract_parameters(route_selection.route, message)
            
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
    
    