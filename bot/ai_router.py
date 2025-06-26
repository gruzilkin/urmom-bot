import logging
from typing import Union, Optional
from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import FamousParams, GeneralParams
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    """Schema for routing decisions."""
    route: Literal["FAMOUS", "GENERAL", "NONE"] = Field(description="Route decision")
    famous_params: Optional[FamousParams] = Field(
        default=None,
        description="Parameters for the FAMOUS route. Only present if route is FAMOUS."
    )
    general_params: Optional[GeneralParams] = Field(
        default=None,
        description="Parameters for the GENERAL route. Only present if route is GENERAL."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for choosing the route."
    )


class AiRouter:
    def __init__(self, ai_client: AIClient, telemetry: Telemetry, 
                 famous_generator, general_generator):
        self.ai_client = ai_client
        self.telemetry = telemetry
        self.famous_generator = famous_generator
        self.general_generator = general_generator
    
    def _build_combined_prompt(self) -> str:
        famous_desc = self.famous_generator.get_route_description()
        general_desc = self.general_generator.get_route_description()
        
        return f"""
        Analyze the user message and decide how to route it. Choose exactly one route:

        {famous_desc}

        {general_desc}

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

        Instructions:
        1. Read the user message carefully
        2. Determine which route best matches the intent
        3. Follow the parameter extraction guidelines for your chosen route
        4. Provide a brief (1-2 sentence) reason for your decision.
        5. Return your decision with the appropriate parameters and reason filled in.
        """
    
    async def route_request(self, message: str) -> RouterDecision:
        prompt = self._build_combined_prompt()
        
        async with self.telemetry.async_create_span("route_request") as span:
            span.set_attribute("message", message)
            
            response = await self.ai_client.generate_content(
                message=message,
                prompt=prompt,
                response_schema=RouterDecision
            )
            
            span.set_attribute("route", response.route)
            if response.reason:
                span.set_attribute("reason", response.reason)
            if response.route == "FAMOUS" and response.famous_params:
                span.set_attribute("famous_person", response.famous_params.famous_person)
            elif response.route == "GENERAL" and response.general_params:
                span.set_attribute("ai_backend", response.general_params.ai_backend)
                span.set_attribute("temperature", response.general_params.temperature)
            
            logger.info(f"Routing decision: {response.route}, Reason: {response.reason}")
            return response