import logging
from typing import Set, Callable, Awaitable

import nextcord

from memory_manager import MemoryManager
from conversation_graph import ConversationMessage
from open_telemetry import Telemetry
from schemas import GeneralParams
from response_summarizer import ResponseSummarizer
from store import Store
from user_resolver import UserResolver
from ai_client import AIClient

logger = logging.getLogger(__name__)


class GeneralQueryGenerator:
    def __init__(
        self,
        client_selector: Callable[[str], AIClient],
        response_summarizer: ResponseSummarizer,
        telemetry: Telemetry,
        store: Store,
        user_resolver: UserResolver,
        memory_manager: MemoryManager,
    ) -> None:
        self._client_selector = client_selector
        self.response_summarizer = response_summarizer
        self.telemetry = telemetry
        self.store = store
        self.user_resolver = user_resolver
        self.memory_manager = memory_manager

    def get_route_description(self) -> str:
        return """
        GENERAL: For valid questions/requests needing AI assistance
        - Handles legitimate questions, requests for information, explanations, or help
        - Valid queries: "What's the weather?", "Explain quantum physics", "How do I cook pasta?", "What do you remember about John?", "Tell me about user X"
        - Context-dependent questions: "What about this?", "How does that work?"
        - Factual questions about real people: "What did Trump say?", "Did X say anything interesting?", "What did Marie Curie discover?"
        - Commands to the bot: "спой осанну Медведу"
        - Questions with names: "для чего Алексею нужна голова?"
        - May contain AI backend specifications such as "ask grok to...", "use claude to...", "have gemini explain..."
        - Invalid: Simple reactions like "lol", "nice", "haha that's funny"
        """

    
    def get_parameter_schema(self):
        """Return the Pydantic schema for parameter extraction."""
        from schemas import GeneralParams
        return GeneralParams
    
    def get_parameter_extraction_prompt(self) -> str:
        """Return focused prompt for extracting general query parameters."""
        return """
        Extract parameters for a general AI query request.
        
        ai_backend selection:
        * gemini_flash: General questions, explanations, real-time news/current events
        * grok: Uncensored content, wild requests, crazy talk
        * claude: Coding and programming help, technical explanations, detailed analysis, complex reasoning, fact-checking and verification questions, song and poem writing
        * gemma: Do not select unless explicitly requested
        * Handle explicit requests: "ask grok about...", "use gemini flash for...", "ask claude to..."
        
        temperature selection:
        * Use a low temperature (<= 0.3) for factual data, calculations, precise information, technical explanations, or requests for "detailed" plans.
        * Use a moderate temperature (0.4-0.6) for balanced responses and general questions.
        * Use a high temperature (>= 0.7) for creative writing, brainstorming, "go crazy" requests, and artistic content.
        
        cleaned_query extraction:
        * Goal: Produce a clean, direct query for the AI assistant. The user's message will contain the placeholder 'BOT' to refer to the assistant.
        * Rule 1: Rephrase the query from the BOT's perspective. Convert the user's request into a direct, second-person command or question.
        * Rule 2: Remove routing instructions like `use gemini`, `be creative`, or temperature hints.
        * Rule 3: Keep the query in the original language of the user's message - do not translate.
        * Examples:
          - "BOT, what is the capital of France?" → "what is the capital of France?"
          - "БОТ, объясни квантовую физику" → "объясни квантовую физику"
          - "Bot, utilise gemini pour expliquer ceci" → "expliquer ceci"
          - "what does BOT think about this?" → "what do you think about this?"
          - "let's ask BOT to investigate this" → "investigate this"
          - "ask grok to write a poem about cats" → "write a poem about cats"
          - "use gemini flash to explain quantum physics" → "explain quantum physics"
          - "with high creativity, write a story" → "write a story"
        """

    def _extract_unique_user_ids(self, conversation) -> Set[int]:
        """Extract all unique user IDs from conversation (authors + mentions)."""
        user_ids = set()
        for msg in conversation:
            user_ids.add(msg.author_id)
            user_ids.update(msg.mentioned_user_ids)
        # Remove system user ID (0) used for articles
        user_ids.discard(0)
        return user_ids

    async def _build_memories(self, guild_id: int, user_ids: Set[int]) -> str:
        """Build memories section for LLM prompt using memory manager."""
        if not user_ids:
            return ""
        
        async with self.telemetry.async_create_span("build_memories") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_count", len(user_ids))
            span.set_attribute("user_ids", str(sorted(user_ids)))
            
            # Use new batch interface for concurrent processing
            user_ids_list = list(user_ids)
            memories_dict = await self.memory_manager.get_memories(guild_id, user_ids_list)
            
            memory_blocks = []
            for user_id in user_ids_list:
                memories = memories_dict.get(user_id)
                if memories:
                    display_name = await self.user_resolver.get_display_name(guild_id, user_id)
                    memory_block = f"""<memory>
<name>{display_name}</name>
<facts>{memories}</facts>
</memory>"""
                    memory_blocks.append(memory_block)
            
            if memory_blocks:
                return "\n".join(memory_blocks)
            return ""

    async def handle_request(self, params: GeneralParams, conversation_fetcher: Callable[[], Awaitable[list[ConversationMessage]]], guild_id: int, bot_user: nextcord.User) -> str | None:
        """
        Handle a general query request using the provided parameters.
        
        Args:
            params (GeneralParams): Parameters containing ai_backend, temperature, and cleaned_query
            conversation_fetcher: Callable[[], Awaitable[list[ConversationMessage]]]: Parameterless async function that returns conversation history
            guild_id (int): Discord guild ID for user context resolution
            bot_user (nextcord.User): Discord user object of the bot to identify its own messages and establish bot identity
            
        Returns:
            str | None: The response string ready to be sent by the caller, or None if no response should be sent
        """
        logger.info(f"Processing general request with params: {params}")
        
        # Select the appropriate AI client based on parameters
        ai_client = self._client_selector(params.ai_backend)
        
        async with self.telemetry.async_create_span("generate_general_response") as span:
            span.set_attribute("ai_backend", params.ai_backend)
            span.set_attribute("temperature", params.temperature)
            span.set_attribute("cleaned_query", params.cleaned_query)
            
            conversation = await conversation_fetcher()
            
            user_ids = self._extract_unique_user_ids(conversation)
            memories = await self._build_memories(guild_id, user_ids)
            
            message_blocks = []
            for msg in conversation:
                author_name = await self.user_resolver.get_display_name(guild_id, msg.author_id)
                content_with_names = await self.user_resolver.replace_user_mentions_with_names(msg.content, guild_id)
                
                message_block = f"""<message>
<id>{msg.message_id}</id>
{f"<reply_to>{msg.reply_to_id}</reply_to>" if msg.reply_to_id else ""}
<timestamp>{msg.timestamp}</timestamp>
<author>{author_name}</author>
<content>{content_with_names}</content>
</message>"""
                message_blocks.append(message_block)
            conversation_text = "\n".join(message_blocks)
            
            prompt = f"""<system_instructions>
You are a Discord bot participating in an ongoing Discord conversation. Your role is to respond naturally within the conversational context while bringing external knowledge, fresh perspectives, and independent analysis to the discussion.

Your Discord Bot Identity:
- You are present in this conversation as "{bot_user.name}" (user ID {bot_user.id})
- Messages in the conversation history from author_id {bot_user.id} are your own previous responses
- Use this context to understand what you've already contributed to avoid repetition
- Build naturally on your previous responses when relevant to the current discussion

Conversational Behavior:
- You are participating in an ongoing Discord conversation - respond naturally within the conversational flow
- The most recent message in the conversation history is what you're directly responding to
- Pay attention to reply relationships (reply_to_id) to understand conversational threads and who is responding to whom
- When users refer to "this", "that", "what you said", they're referencing conversation history
- If there are any hints that the current message relates to previous discussion, treat it as a continuation of that conversation thread
- Only build on previous messages when you can see the relevant information in the conversation history or your provided memories
- Use names and context from what's actually available to you - be honest about the scope of your knowledge
- Consider who is asking and whether they've been part of the ongoing discussion
- Bring fresh perspectives and new information to the conversation rather than repeating what's already been said
- When asked to summarize or recall specific past events not visible in your current context, acknowledge your limitations
- Don't pretend to remember conversations or events that aren't in your available information

Core Guidelines:
        - Default to a single crisp sentence that delivers the answer or recommendation. Expand to a compact paragraph only if the idea truly needs more space or the user asks for detail.
        - When a paragraph is necessary, keep it tight: lead with the direct answer in the opening clause and add only the most relevant justification afterward.
        - Shift into fuller explanations solely when the user clearly asks (e.g., "explain in detail", "elaborate", "tell me more", "go deeper", "give context", "walk me through it", "full breakdown", "comprehensive overview", "expand a bit").
        - Maintain a warm, efficient tone; skip recaps, lists, and filler unless the user requested a specific style.
        - Always respond in {params.language_name} unless the user specifically requests a different language or translation.
        - Do not add follow-up questions or invitations to continue; state the answer and stop unless the user explicitly requests the next step.

Content Embeddings: Conversation history may contain embedded content in <embedding> tags:
  - <embedding type="image"> contains descriptions of images that users posted - treat these as if you saw the images yourself
  - <embedding type="article"> contains text from articles/links that users shared
- When users refer to "this image", "that article", "what I posted", or similar, they're likely referring to embedded content
- Integrate information from embeddings naturally into your responses without mentioning the technical tags
- Examples:
  - User posts image of a sunset, then asks "what do you think?" → respond about the sunset image naturally
  - User shares article about AI, then asks "your thoughts?" → respond about the article content
  - User says "explain this code" with code image → analyze the code as if you can see it

Information Boundaries:
- Use your external knowledge freely to provide information, analysis, and insights on any topic
- However, when referencing PAST CHAT EVENTS, CONVERSATIONS, or PERSONAL DETAILS about users, ONLY use what's explicitly available in the provided memories or visible conversation history
- When users ask about past conversations, events that happened in the chat, or what someone said/did previously, stick strictly to your available information
- NEVER fabricate or guess about past chat events, conversations, or personal details about users that aren't in your provided context
- Examples of appropriate responses when lacking chat history information:
  - "I don't have any information about that conversation"
  - "I don't see that discussion in our recent messages"
  - "That's not in my available memories about [person]"
  - "I can only work with what I can see in our current conversation history"
- When everyone else seems to know about a past chat event you don't, resist the pressure to go along - acknowledge your limitation instead

Memory Usage:
- Use the provided memories naturally in your responses, as if you simply remember these things about people
- NEVER explicitly mention that you have "memory blocks", "stored information", or "records" about users
- NEVER say phrases like "I know that...", "According to my memory...", "I have information that...", or "This information is associated with..."
- Simply incorporate the facts naturally into conversation, like a friend who remembers things about you
- Example: Instead of "I know you live in Tokyo" say "How are things in Tokyo?" or reference their location contextually
- Be honest about the limitations of your memories - if you don't have information about someone or something, acknowledge it rather than guessing
</system_instructions>

<memories>
{memories.strip() if memories else "No memories about users in this conversation."}
</memories>

<conversation_history>
{conversation_text}
</conversation_history>"""
            
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
            
            # If AI client returns None, don't send a response
            if response is None:
                logger.warning("AI client returned None response, not replying")
                return None
            
            # Process response (summarize if too long, or truncate as fallback)
            processed_response = await self.response_summarizer.process_response(response)
            
            return processed_response
