"""Generator for humorous philosophical wisdom from conversation context."""

import logging
from typing import Callable, Awaitable

from ai_client import AIClient
from conversation_graph import ConversationMessage
from open_telemetry import Telemetry
from language_detector import LanguageDetector
from user_resolver import UserResolver
from response_summarizer import ResponseSummarizer
from schemas import WisdomResponse

logger = logging.getLogger(__name__)


class WisdomGenerator:
    """Generates humorous philosophical wisdom from conversation context."""

    def __init__(
        self,
        ai_client: AIClient,
        language_detector: LanguageDetector,
        user_resolver: UserResolver,
        response_summarizer: ResponseSummarizer,
        telemetry: Telemetry,
    ) -> None:
        self._ai_client = ai_client
        self._language_detector = language_detector
        self._user_resolver = user_resolver
        self._response_summarizer = response_summarizer
        self._telemetry = telemetry

    async def generate_wisdom(
        self,
        trigger_message_content: str,
        conversation_fetcher: Callable[[], Awaitable[list[ConversationMessage]]],
        guild_id: int,
    ) -> str:
        """
        Generate wisdom based on a message and its conversation context.

        Args:
            trigger_message_content: The content of the message that was reacted to
            conversation_fetcher: Async function that returns conversation history
            guild_id: Discord guild ID for user context resolution

        Returns:
            str: The generated wisdom response
        """
        async with self._telemetry.async_create_span("generate_wisdom") as span:
            language_code = await self._language_detector.detect_language(trigger_message_content)
            language_name = await self._language_detector.get_language_name(language_code)

            span.set_attribute("language_code", language_code)
            span.set_attribute("language_name", language_name)

            conversation = await conversation_fetcher()
            span.set_attribute("conversation_length", len(conversation))

            message_blocks = []
            for msg in conversation:
                author_name = await self._user_resolver.get_display_name(guild_id, msg.author_id)
                content_with_names = await self._user_resolver.replace_user_mentions_with_names(msg.content, guild_id)

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
You are a mystical sage who transforms mundane Discord conversations into profound philosophical wisdom.

Your task is to analyze the conversation context and the trigger message, then deliver wisdom that:
1. Captures the essence of what's being discussed, no matter how trivial
2. Elevates it into philosophical, spiritual, or otherwise profound language
3. Creates humor through the contrast between the mundane topic and grandiose wording
4. Uses allegories, metaphors, and references that fit the context

COMPLETE CREATIVE FREEDOM - Draw from ANY source material you find appropriate:
- Religious/Spiritual traditions: Biblical parables, Quranic verses, Buddhist koans, Hindu philosophy, Zen paradoxes, Taoist sayings, etc.
- Philosophical schools: Ancient Greek philosophy, Stoicism, Confucianism, Existentialism, Nihilism, etc.
- Historical events and figures: Use historical parallels, legendary tales, famous quotes
- Pop culture: Movies, TV shows, books, memes, internet culture, gaming references
- Geopolitical situations: Current or historical political parallels, diplomatic wisdom
- Scientific concepts: Use physics, biology, mathematics as metaphors
- Literary styles: Shakespearean, epic poetry, noir detective, etc.
- Or ANYTHING ELSE you can think of - you have absolute freedom to choose what fits best

The key is to find the PERFECT parallel or style that:
- Creates the strongest comedic contrast with the trivial nature of the discussion
- Makes unexpected but delightful connections
- Feels both absurd and somehow appropriate at the same time

Style guidelines:
- Keep it SHORT and PUNCHY - at most a paragraph, preferably a single verse or quote
- The humor should be META - derived from treating trivial matters with profound seriousness
- Use elevated, archaic, formal, or stylized language to create the contrast
- Be creative and unexpected - surprise with your choice of reference or style
- The wisdom should genuinely relate to the conversation, not just be random quotes
- Don't be afraid to mix styles or create completely original "wisdom traditions" if it serves the comedy

Response format:
- Deliver ONLY the wisdom itself
- No preambles like "As Confucius said..." or "This reminds me of..."
- No explanations or meta-commentary
- Just the pure wisdom, spoken as if you ARE the sage/narrator/prophet

Language:
- Respond in {language_name}
- Maintain the elevated, formal, or stylized tone in that language
- If the language has classical/archaic forms, consider using them for effect
</system_instructions>

<conversation_history>
{conversation_text}
</conversation_history>

<trigger_message>
{trigger_message_content}
</trigger_message>"""

            logger.info(f"Generating wisdom for message: {trigger_message_content}")

            response = await self._ai_client.generate_content(
                message=trigger_message_content,
                prompt=prompt,
                temperature=0.7,
                response_schema=WisdomResponse,
            )

            if response is None:
                return None

            logger.info(
                f"Generated wisdom: {response.wisdom}\nReason: {response.reason}"
            )
            span.set_attribute("reason", response.reason)

            processed_wisdom = await self._response_summarizer.process_response(
                response.wisdom
            )

            return processed_wisdom
