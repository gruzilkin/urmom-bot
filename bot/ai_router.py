import logging
from typing import Union, Optional
from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import FamousParams, GeneralParams, FactParams
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    """Schema for routing decisions."""
    route: Literal["FAMOUS", "GENERAL", "FACT", "NONE"] = Field(description="Route decision")
    famous_params: Optional[FamousParams] = Field(
        default=None,
        description="Parameters for the FAMOUS route. Only present if route is FAMOUS."
    )
    general_params: Optional[GeneralParams] = Field(
        default=None,
        description="Parameters for the GENERAL route. Only present if route is GENERAL."
    )
    fact_params: Optional[FactParams] = Field(
        default=None,
        description="Parameters for the FACT route. Only present if route is FACT."
    )
    reason: str = Field(description="Reason for choosing the route.")


class AiRouter:
    def __init__(self, ai_client: AIClient, telemetry: Telemetry, 
                 famous_generator, general_generator, fact_handler):
        self.ai_client = ai_client
        self.telemetry = telemetry
        self.famous_generator = famous_generator
        self.general_generator = general_generator
        self.fact_handler = fact_handler
    
    def _build_combined_prompt(self) -> str:
        famous_desc = self.famous_generator.get_route_description()
        general_desc = self.general_generator.get_route_description()
        fact_desc = self.fact_handler.get_route_description()
        
        famous_guidelines = self.famous_generator.get_parameter_extraction_guidelines()
        general_guidelines = self.general_generator.get_parameter_extraction_guidelines()
        fact_guidelines = self.fact_handler.get_parameter_extraction_guidelines()
        
        return f"""
<system_instructions>
Analyze the user message and decide how to route it. Choose exactly one route.

Instructions:
1. Read the user message carefully.
2. Determine which route best matches the intent.
3. Follow the parameter extraction guidelines for your chosen route.
4. ALWAYS provide a brief (1-2 sentence) reason for your decision, regardless of the chosen route (including NONE).
5. Return your decision with the appropriate parameters and the mandatory reason field filled in.
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
- No parameters needed
</route>
</route_definitions>

<parameter_extraction_guidelines>
<route route="FAMOUS">
{famous_guidelines}
</route>

<route route="GENERAL">
{general_guidelines}
</route>

<route route="FACT">
{fact_guidelines}
</route>

<route route="NONE">
NONE route parameter extraction:
- No parameters needed for NONE route
</route>
</parameter_extraction_guidelines>
"""

    async def route_request(self, message: str) -> RouterDecision:
        system_instructions = self._build_combined_prompt()
        full_message_for_ai = f"{system_instructions}\n<user_input>{message}</user_input>"
        
        async with self.telemetry.async_create_span("route_request") as span:
            span.set_attribute("message", message)
            
            response = await self.ai_client.generate_content(
                message=full_message_for_ai,
                prompt="",
                response_schema=RouterDecision
            )
            
            span.set_attribute("route", response.route)
            span.set_attribute("reason", response.reason)
            if response.route == "FAMOUS" and response.famous_params:
                span.set_attribute("famous_person", response.famous_params.famous_person)
            elif response.route == "GENERAL" and response.general_params:
                span.set_attribute("ai_backend", response.general_params.ai_backend)
                span.set_attribute("temperature", response.general_params.temperature)
            elif response.route == "FACT" and response.fact_params:
                span.set_attribute("fact_operation", response.fact_params.operation)
                span.set_attribute("fact_user_mention", response.fact_params.user_mention)
            
            logger.info(f"Routing decision: {response.route}, Reason: {response.reason}")
            return response
    
    