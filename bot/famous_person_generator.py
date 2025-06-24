import logging

from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import FamousParams

logger = logging.getLogger(__name__)


class FamousPersonGenerator:
    def __init__(self, ai_client: AIClient, telemetry: Telemetry):
        self.ai_client = ai_client
        self.telemetry = telemetry

    def get_route_description(self) -> str:
        return """
        FAMOUS: For celebrity/character impersonation requests
        - Detects when users ask what a famous person would say or how they would respond
        - Examples: "What would Trump say?", "How would Darth Vader respond?", "What if Einstein explained this?"
        - Handles variations like "What would X say about Y?" or "How would X feel about this?"
        - Extract the person's name (can be real celebrities, fictional characters, historical figures)
        - Parameter extraction: Set famous_person to the name of the person to impersonate
        """

    async def is_famous_person_request(self, message: str) -> str | None:
        """
        Check if a message is asking what a famous person would say.
        
        Args:
            message (str): The message to check
            
        Returns:
            str | None: The name of the famous person if it's a request, None otherwise
        """
        # Use AI client's generate_content method with specialized prompt
        prompt = """You need to check if the user message is asking to impersonate a famous person and reply with the person's name.

        Example 1:
        Input: What would Trump say?
        Output: Trump

        Example 2:
        Input: What's the weather today?
        Output: None

        Example 3:
        Input: What would Jesus say if he spoke like Trump?
        Output: Jesus

        Example 4:
        Input: How would Darth Vader feel about this?
        Output: Darth Vader

        Example 5:
        Input: What if Eminen did tldr?
        Output: Eminen

        Example 6:
        Input: How would Sigmund Freud respond to this?
        Output: Sigmund Freud

        Only extract the person's name if the message is clearly asking to impersonate them.
        If it's not a request to impersonate someone then respond with 'None'."""

        async with self.telemetry.async_create_span("is_famous_person_request") as span:
            span.set_attribute("message", message)
            
            response_text = await self.ai_client.generate_content(
                message=message,
                prompt=prompt
            )
            
            response_text = response_text.strip()
            # Convert "None" string to actual None
            person_name = None if response_text == "None" else response_text
            
            span.set_attribute("person_detected", person_name is not None)
            if person_name:
                span.set_attribute("person", person_name)
            
            logger.info(f"Detection result: '{person_name}'")
            return person_name

    async def generate_famous_person_response(self, extracted_message: str, person: str, conversation_fetcher) -> str:
        """
        Generate a famous person response. Returns the response string.
        
        Args:
            extracted_message (str): The user's message with bot mentions removed
            person (str): The name of the famous person to impersonate
            conversation_fetcher: Parameterless async function that returns conversation history
            
        Returns:
            str: The response string ready to be sent by the caller
        """
        async with self.telemetry.async_create_span("generate_famous_person_response") as span:
            span.set_attribute("person", person)
            span.set_attribute("extracted_message", extracted_message)
            
            conversation = await conversation_fetcher()
            
            # Build conversation context as a single message with timestamps
            conversation_text = "\n".join([
                f"{msg.timestamp} {msg.author_name}: {msg.content}" 
                for msg in conversation
            ])
            
            prompt = f"""You are {person}. Generate a response as if you were {person}, 
            using their communication style, beliefs, values, and knowledge.
            Make the response thoughtful, authentic to {person}'s character, and relevant to the conversation.
            Stay in character completely and respond directly as {person} would.
            Keep your response length similar to the average message length in the conversation.
            Feel free to tease and poke fun at the message authors, especially Florent.
            The user specifically asked: '{extracted_message}'
            Your response should be in the form of direct speech - exactly as if {person} is speaking directly, without quotation marks or attributions.
            
            Keep responses under 2000 characters due to Discord's message limit.
            
            Here is the conversation context:
            {conversation_text}"""
            
            logger.info(f"Generating response as {person}")
            logger.info(f"Conversation: {conversation}")
            
            response = await self.ai_client.generate_content(
                message="",  # The conversation is included in the prompt
                prompt=prompt
            )
            
            logger.info(f"Generated response: {response}")
            return f"**{person.title()} would say:**\n\n{response}"
    
    async def handle_request(self, params: FamousParams, extracted_message: str, conversation_fetcher) -> str:
        """
        Handle a famous person request using the provided parameters.
        
        Args:
            params (FamousParams): Parameters containing the famous person's name
            extracted_message (str): The user's message with bot mentions removed
            conversation_fetcher: Parameterless async function that returns conversation history
            
        Returns:
            str: The response string ready to be sent by the caller
        """
        return await self.generate_famous_person_response(extracted_message, params.famous_person, conversation_fetcher)


