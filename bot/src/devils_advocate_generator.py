"""Generator for analytical counter-arguments from conversation context."""

import logging
from collections.abc import Callable, Awaitable

from ai_client import AIClient
from conversation_graph import ConversationMessage
from open_telemetry import Telemetry
from language_detector import LanguageDetector
from response_summarizer import ResponseSummarizer
from memory_manager import MemoryManager
from conversation_formatter import ConversationFormatter
from schemas import DevilsAdvocateResponse

logger = logging.getLogger(__name__)


class DevilsAdvocateGenerator:
    """Generates analytical counter-arguments from conversation context."""

    def __init__(
        self,
        ai_client: AIClient,
        language_detector: LanguageDetector,
        conversation_formatter: ConversationFormatter,
        response_summarizer: ResponseSummarizer,
        memory_manager: MemoryManager,
        telemetry: Telemetry,
    ) -> None:
        self._ai_client = ai_client
        self._language_detector = language_detector
        self._conversation_formatter = conversation_formatter
        self._response_summarizer = response_summarizer
        self._memory_manager = memory_manager
        self._telemetry = telemetry

    def _extract_unique_user_ids(self, conversation: list[ConversationMessage]) -> set[int]:
        """Extract all unique user IDs from conversation (authors + mentions)."""
        user_ids = set()
        for msg in conversation:
            user_ids.add(msg.author_id)
            user_ids.update(msg.mentioned_user_ids)
        return user_ids

    async def generate_counter_argument(
        self,
        trigger_message_content: str,
        conversation_fetcher: Callable[[], Awaitable[list[ConversationMessage]]],
        guild_id: int,
    ) -> str | None:
        """
        Generate counter-argument based on a message and its conversation context.

        Args:
            trigger_message_content: The content of the message that was reacted to
            conversation_fetcher: Async function that returns conversation history
            guild_id: Discord guild ID for user context resolution

        Returns:
            str: The generated counter-argument response, or None if generation fails
        """
        async with self._telemetry.async_create_span("generate_counter_argument") as span:
            language_code = await self._language_detector.detect_language(trigger_message_content)
            language_name = await self._language_detector.get_language_name(language_code)

            span.set_attribute("language_code", language_code)
            span.set_attribute("language_name", language_name)

            conversation = await conversation_fetcher()

            user_ids = self._extract_unique_user_ids(conversation)
            memories_block = await self._memory_manager.build_memory_prompt(guild_id, user_ids)

            conversation_block = await self._conversation_formatter.format_to_xml(guild_id, conversation)

            prompt = f"""<system_instructions>
You are a devil's advocate.

Your task is to analyze the conversation and deliver a counter-argument that:
1. Traces the author's arguments across the entire conversation - they may have made multiple points in different messages
2. Identifies the main claim or overall position the author is taking
3. Challenges assumptions, finds logical flaws, or presents alternative viewpoints
4. Argues persuasively for an opposing or alternative perspective
5. Maintains intellectual honesty (acknowledge valid points while arguing the opposite)
6. Synthesizes your counter-argument from the full conversational context, not just the single trigger message
7. Don't be afraid to go after the author or other chat participants personally

Response format and style:
- Decide whether the reply should use the TL;DR style or the Expanded style before you start writing.
- TL;DR style (default): deliver the counter-argument in a single crisp sentence or short paragraph; use inline Markdown emphasis (bold, italic, inline code) when it sharpens the message, but avoid headings, lists, or blockquotes.
- Expanded style: use this only when the argument truly demands structured breakdown or when countering multiple complex points.
- When using Expanded style, lead with the direct counter-argument in the opening clause, then add tightly edited support using Markdown structure (headings, lists) to improve readability.
- Maintain a punchy, efficient tone; skip recaps and filler.
- Do not add follow-up questions or invitations to continue; state the counter-argument and stop.
- Deliver the counter-argument directly, without any meta commentary.

Language:
- Respond in {language_name}

Personalization:
- You have memories about some users in this conversation - use them to make the counter-argument more relevant
- Reference their known positions or characteristics when it strengthens your argument
</system_instructions>

{memories_block}

{conversation_block}

<trigger_message>
{trigger_message_content}
</trigger_message>"""

            logger.info(f"Generating counter-argument for message: {trigger_message_content}")

            response = await self._ai_client.generate_content(
                message=trigger_message_content,
                prompt=prompt,
                temperature=0.7,
                response_schema=DevilsAdvocateResponse,
            )

            if response is None:
                return None

            logger.info(f"Generated counter-argument: {response.answer}\nReason: {response.reason}")
            span.set_attribute("reason", response.reason)

            processed_counter_argument = await self._response_summarizer.process_response(response.answer)

            return processed_counter_argument
