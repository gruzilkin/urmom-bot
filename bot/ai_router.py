import logging
from typing import Union
from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import FamousParams, GeneralParams
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    """Schema for routing decisions."""
    route: Literal["FAMOUS", "GENERAL", "NONE"] = Field(description="Route decision")
    parameters: Union[FamousParams, GeneralParams, None] = Field(
        description="Parameters for the chosen route", 
        default=None
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
        - Examples: "lol", "nice", "ok", "haha", random gibberish
        - No parameters needed

        Instructions:
        1. Read the user message carefully
        2. Determine which route best matches the intent
        3. Follow the parameter extraction guidelines for your chosen route
        4. Return your decision with the appropriate parameters filled in
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
            if response.route == "FAMOUS" and response.parameters:
                span.set_attribute("famous_person", response.parameters.famous_person)
            elif response.route == "GENERAL" and response.parameters:
                span.set_attribute("ai_backend", response.parameters.ai_backend)
                span.set_attribute("temperature", response.parameters.temperature)
            
            logger.info(f"Routing decision: {response.route}")
            return response