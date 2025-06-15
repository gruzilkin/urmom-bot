from ai_client import AIClient
from open_telemetry import Telemetry


class GeneralQueryGenerator:
    def __init__(self, ai_client: AIClient, telemetry: Telemetry):
        self.ai_client = ai_client
        self.telemetry = telemetry

    async def is_general_query(self, message: str) -> bool:
        """
        Check if a message is a reasonable general query that should be answered by an LLM.
        
        Args:
            message (str): The message to check
            
        Returns:
            bool: True if it's a valid general query, False otherwise
        """
        # Use AI client's generate_content method with specialized prompt
        prompt = """You need to check if the user message is a reasonable general query or request that an AI assistant should answer.
        
        Do not try to answer the query itself, a follow up query will handle that.
        Only respond with 'Yes' if the message contains a clear question or request for information/assistance.
        Respond with 'No' if it's just a reaction, acknowledgment, or doesn't contain a clear query.

        A valid general query should contain:
        - A question (who, what, where, when, why, how)
        - A request for information, explanation, or help
        - A request to do something (explain, analyze, summarize, etc.)

        Example 1:
        Input: What's the weather today?
        Output: Yes

        Example 2:
        Input: Can you explain quantum physics?
        Output: Yes

        Example 3:
        Input: lol
        Output: No

        Example 4:
        Input: How do I cook pasta?
        Output: Yes

        Example 5:
        Input: nice
        Output: No

        Example 6:
        Input: Tell me about the history of Rome
        Output: Yes

        Example 7:
        Input: Why is the sky blue?
        Output: Yes

        Example 8:
        Input: haha that's funny
        Output: No"""

        async with self.telemetry.async_create_span("is_general_query") as span:
            span.set_attribute("message", message)
            
            response_text = await self.ai_client.generate_content(
                message=message,
                prompt=prompt
            )
            
            response_text = response_text.strip().lower()
            is_query = response_text == "yes"
            
            span.set_attribute("is_query", is_query)
            
            print(f"[GENERAL_QUERY] Detection result: {is_query}")
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
            
            print(f"[GENERAL_QUERY] Generating response")
            print(f"[GENERAL_QUERY] User message: {extracted_message}")
            print(f"[GENERAL_QUERY] Conversation context: {conversation}")
            
            response = await self.ai_client.generate_content(
                message="",  # The conversation is included in the prompt
                prompt=prompt,
                enable_grounding=True  # Enable grounding for general queries to get current information
            )
            
            print(f"[GENERAL_QUERY] Generated response: {response}")
            return response
