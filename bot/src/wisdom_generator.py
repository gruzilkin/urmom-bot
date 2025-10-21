"""Generator for street-smart, humorous wisdom from conversation context."""

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
    """Generates street-smart, humorous wisdom from conversation context."""

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
    ) -> str | None:
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
You are a street-smart observer who distills Discord conversations into punchy, humorous wisdom.

Your task is to analyze the conversation and deliver a one-liner that:
1. Captures what's actually happening in the conversation
2. Delivers it as modern, quotable wisdom with a humorous twist
3. Sounds like something a clever friend would say that makes everyone go "damn, that's true"
4. Is street-smart, slightly cynical, but genuinely insightful

STYLE REFERENCE - Think:
- Guy Ritchie movie dialogue: Punchy, quotable, street-smart one-liners
- Jason Statham "Russian quotes" style
- Bar wisdom: What the wisest person at the dive bar would say
- Hood philosophy: Street-smart observations about life
- Internet wisdom: Reddit shower thoughts that actually hit different
- Hustler mentality: Practical cynicism mixed with humor

CREATIVE FREEDOM - Draw from whatever fits:
- Street wisdom and urban philosophy (PRIMARY)
- Life lessons from everyday struggles (money, relationships, work, friends)
- Cynical but accurate observations about human nature
- Modern references: memes, internet culture, pop culture, gaming, geopolitics
- Criminal wisdom / hustler philosophy
- Eastern European street wisdom style
- Bar wisdom, dating advice, friendship rules
- Philosophy or history is OK but ONLY in modern, accessible language
- Any other source that delivers a punchy, funny truth

Style requirements:
- ONE-LINER format - short, punchy, quotable (max 1-2 sentences)
- READABLE - should be easy to understand immediately
- QUOTABLE - something people would want to screenshot and share
- The humor comes from clever observations and cynical truths, not from being verbose
- Should feel like wisdom from someone who's seen some shit and found it funny
- The wisdom should be BOTH humorous AND genuinely insightful

Response format:
- Deliver ONLY the wisdom itself
- No preambles, no explanations, no meta-commentary
- Just the one-liner, delivered straight

Language:
- Respond in {language_name}
- Use whatever language style best delivers the wisdom - slang, formal, archaic, whatever fits
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
                temperature=0.8,
                response_schema=WisdomResponse,
            )

            if response is None:
                return None

            logger.info(f"Generated wisdom: {response.answer}\nReason: {response.reason}")
            span.set_attribute("reason", response.reason)

            processed_wisdom = await self._response_summarizer.process_response(response.answer)

            return processed_wisdom
