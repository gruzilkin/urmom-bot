import logging
from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import YesNo, GeneralParams


logger = logging.getLogger(__name__)


class GeneralQueryGenerator:
    def __init__(self, ai_client: AIClient, telemetry: Telemetry):
        self.ai_client = ai_client
        self.telemetry = telemetry

    def get_route_description(self) -> str:
        return """
        GENERAL: For valid questions/requests needing AI assistance
        - Handles legitimate questions, requests for information, explanations, or help
        - Valid queries: "What's the weather?", "Explain quantum physics", "How do I cook pasta?"
        - Context-dependent questions: "What about this?", "How does that work?"
        - Invalid: Simple reactions like "lol", "nice", "haha that's funny"
        
        Parameter extraction:
        - ai_backend selection:
          * gemini_pro: Complex reasoning, research, detailed explanations, technical questions
          * gemini_flash: Simple questions, quick facts, basic explanations (faster, cheaper)  
          * grok: Creative tasks, current events, social media context, unconventional perspectives
          * Handle explicit requests: "ask grok about...", "use gemini pro for..."
        
        - temperature selection:
          * 0.0-0.2: Factual data, calculations, precise information, technical explanations
          * 0.3-0.6: Balanced responses, general questions, moderate creativity
          * 0.7-1.0: Creative writing, brainstorming, "go crazy" requests, artistic content
        
        - cleaned_query extraction:
          * Remove explicit backend requests: "ask grok to..." → just the actual question
          * Remove temperature instructions: "use high temperature and..." → just the core request
          * Remove routing hints: "be creative with..." → keep the creative context but remove routing language
          * Examples:
            - "ask grok to write a poem about cats" → "write a poem about cats"
            - "use gemini pro to explain quantum physics" → "explain quantum physics"
            - "with high creativity, write a story" → "write a story"
            - "be factual and explain the weather" → "explain the weather"
        """

    async def is_general_query(self, message: str) -> bool:
        """
        Check if a message is a reasonable general query that should be answered by an LLM.
        
        Args:
            message (str): The message to check
            
        Returns:
            bool: True if it's a valid general query, False otherwise
        """
        prompt = """You need to check if the user message is a reasonable general query or request that an AI assistant should answer.
        Determine if the message contains a clear question or request for information/assistance.

        A valid general query should contain:
        - A question (who, what, where, when, why, how)
        - A request for information, explanation, or help
        - A request to do something (explain, analyze, summarize, etc.)
        - Context-dependent questions that could make sense within a conversation

        Valid general queries:
        - What's the weather today?
        - Can you explain quantum physics?
        - How do I cook pasta?
        - Tell me about the history of Rome
        - Why is the sky blue?
        - When was it? (context-dependent but valid)
        - What about this? (context-dependent but valid)
        - How does that work? (context-dependent but valid)

        Invalid queries (reactions, acknowledgments, no clear request):
        - lol
        - nice
        - haha that's funny"""

        async with self.telemetry.async_create_span("is_general_query") as span:
            span.set_attribute("message", message)
            
            response = await self.ai_client.generate_content(
                message=message,
                prompt=prompt,
                response_schema=YesNo
            )
            
            is_query = response.answer == "YES"
            
            span.set_attribute("is_query", is_query)
            
            logger.info(f"Detection result: {is_query}")
            return is_query

    async def generate_general_response(self, extracted_message: str, conversation_fetcher) -> str:
        """
        Generate a general response to a user query using conversation context.
        
        Args:
            extracted_message (str): The user's message with bot mentions removed
            conversation_fetcher: Parameterless async function that returns conversation history
            
        Returns:
            str: The response string ready to be sent by the caller
        """
        async with self.telemetry.async_create_span("generate_general_response") as span:
            span.set_attribute("extracted_message", extracted_message)
            
            conversation = await conversation_fetcher()
            
            # Build conversation context as a single message
            conversation_text = "\n".join([f"{username}: {content}" for username, content in conversation])
            
            prompt = f"""You are a helpful AI assistant in a Discord chat. Please respond to the user's question or request.

            Use the conversation context to better understand what the user is asking about.
            Keep your response conversational and appropriate for a Discord chat.
            If the question relates to something mentioned in the conversation, reference it appropriately.
            Keep your response length below 1000 symbols.
            
            User's question/request: '{extracted_message}'
            
            Conversation context:
            {conversation_text}"""
            
            logger.info(f"Generating response")
            logger.info(f"User message: {extracted_message}")
            logger.info(f"Conversation context: {conversation}")
            
            response = await self.ai_client.generate_content(
                message="",  # The conversation is included in the prompt
                prompt=prompt,
                enable_grounding=True  # Enable grounding for general queries to get current information
            )
            
            logger.info(f"Generated response: {response}")
            return response
    
    async def handle_request(self, params: GeneralParams, conversation_fetcher) -> str:
        """
        Handle a general query request using the provided parameters.
        
        Args:
            params (GeneralParams): Parameters containing ai_backend, temperature, and cleaned_query
            conversation_fetcher: Parameterless async function that returns conversation history
            
        Returns:
            str: The response string ready to be sent by the caller
        """
        # Use the cleaned query from the router
        # TODO: In the future, this should use params.ai_backend and params.temperature
        # to select the appropriate AI client and configure temperature
        logger.info(f"Processing general request with params: {params}")
        return await self.generate_general_response(params.cleaned_query, conversation_fetcher)
