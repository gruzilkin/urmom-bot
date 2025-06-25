import logging

from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import GeneralParams
from response_summarizer import ResponseSummarizer

logger = logging.getLogger(__name__)


class GeneralQueryGenerator:
    def __init__(self, gemini_flash: AIClient, grok: AIClient, claude: AIClient, gemma: AIClient, response_summarizer: ResponseSummarizer, telemetry: Telemetry):
        self.gemini_flash = gemini_flash
        self.grok = grok
        self.claude = claude
        self.gemma = gemma
        self.response_summarizer = response_summarizer
        self.telemetry = telemetry
    
    def _get_ai_client(self, ai_backend: str) -> AIClient:
        """Select the appropriate AI client based on backend name."""
        if ai_backend == "gemini_flash":
            return self.gemini_flash
        elif ai_backend == "grok":
            return self.grok
        elif ai_backend == "claude":
            return self.claude
        elif ai_backend == "gemma":
            return self.gemma
        else:
            raise ValueError(f"Unknown ai_backend: {ai_backend}")

    def get_route_description(self) -> str:
        return """
        GENERAL: For valid questions/requests needing AI assistance
        - Handles legitimate questions, requests for information, explanations, or help
        - Valid queries: "What's the weather?", "Explain quantum physics", "How do I cook pasta?"
        - Context-dependent questions: "What about this?", "How does that work?"
        - Invalid: Simple reactions like "lol", "nice", "haha that's funny"
        
        Parameter extraction:
        - ai_backend selection:
          * gemini_flash: General questions, explanations, factual information
          * grok: Creative tasks, uncensored content, real-time news/current events, wild requests
          * claude: Coding help, technical explanations, detailed analysis, complex reasoning
          * gemma: Do not select unless explicitly requested
          * Handle explicit requests: "ask grok about...", "use gemini flash for...", "ask claude to..."
        
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
            - "use gemini flash to explain quantum physics" → "explain quantum physics"
            - "with high creativity, write a story" → "write a story"
            - "be factual and explain the weather" → "explain the weather"
        """


    
    async def handle_request(self, params: GeneralParams, conversation_fetcher) -> str:
        """
        Handle a general query request using the provided parameters.
        
        Args:
            params (GeneralParams): Parameters containing ai_backend, temperature, and cleaned_query
            conversation_fetcher: Parameterless async function that returns conversation history
            
        Returns:
            str: The response string ready to be sent by the caller
        """
        logger.info(f"Processing general request with params: {params}")
        
        # Select the appropriate AI client based on parameters
        ai_client = self._get_ai_client(params.ai_backend)
        
        async with self.telemetry.async_create_span("generate_general_response") as span:
            span.set_attribute("ai_backend", params.ai_backend)
            span.set_attribute("temperature", params.temperature)
            span.set_attribute("cleaned_query", params.cleaned_query)
            
            conversation = await conversation_fetcher()
            
            # Build conversation context as a single message with timestamps
            conversation_text = "\n".join([
                f"{msg.timestamp} {msg.author_name}: {msg.content}" 
                for msg in conversation
            ])
            
            prompt = f"""You are a helpful AI assistant in a Discord chat. Please respond to the user's question or request.
            Keep responses under 2000 characters due to Discord's message limit but no need to report on the length of the response.

            Use the conversation context to better understand what the user is asking about.
            If the question relates to something mentioned in the conversation, reference it appropriately.
            For complex topics, provide a brief summary with key points rather than detailed explanations.
            
            Conversation context:
            {conversation_text}"""
            
            logger.info(f"Generating response using {params.ai_backend}")
            logger.info(f"User message: {params.cleaned_query}")
            logger.info(f"Conversation context: {conversation}")
            
            response = await ai_client.generate_content(
                message=params.cleaned_query,
                prompt=prompt,
                temperature=params.temperature,
                enable_grounding=True  # Enable grounding for general queries to get current information
            )
            
            logger.info(f"Generated response: {response}")
            
            # Process response (summarize if too long, or truncate as fallback)
            processed_response = await self.response_summarizer.process_response(response)
            
            return processed_response
