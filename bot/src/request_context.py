"""Shared prompt context for generators that answer inside a Discord conversation.

Builds the two context blocks every conversational generator sends alongside its own
instructions: ``<memories>`` for the members involved (with display names and aliases) and
``<conversation_history>``, optionally with research handoffs embedded in the bot's earlier messages.
"""

from dataclasses import dataclass

from conversation_formatter import ConversationFormatter
from conversation_graph import ConversationMessage
from memory_manager import MemoryManager
from open_telemetry import Telemetry
from research_handoff import ResearchHandoffService


@dataclass
class RequestContext:
    memories_block: str
    conversation_block: str


def collect_member_ids(conversation: list[ConversationMessage], request: ConversationMessage) -> set[int]:
    """Members whose memories are relevant: message authors, mentioned users, and the request author."""
    member_ids = {request.author_id}
    for message in conversation:
        member_ids.add(message.author_id)
        member_ids.update(message.mentioned_user_ids)
    return member_ids


class RequestContextBuilder:
    def __init__(
        self,
        conversation_formatter: ConversationFormatter,
        memory_manager: MemoryManager,
        handoff_service: ResearchHandoffService,
        telemetry: Telemetry,
    ) -> None:
        self._conversation_formatter = conversation_formatter
        self._memory_manager = memory_manager
        self._handoff_service = handoff_service
        self._telemetry = telemetry

    async def build(
        self,
        request: ConversationMessage,
        conversation: list[ConversationMessage],
        extra_user_ids: set[int] | None = None,
        include_handoffs: bool = False,
    ) -> RequestContext:
        """Build the memories and conversation blocks for a request.

        Args:
            request: The message being handled; its author's memories are always loaded
            conversation: Conversation history for the request
            extra_user_ids: Additional members whose memories should be loaded
            include_handoffs: Embed research handoffs attached to the bot's earlier messages
        """
        async with self._telemetry.async_create_span("build_request_context") as span:
            member_ids = collect_member_ids(conversation, request)
            if extra_user_ids:
                member_ids.update(extra_user_ids)
            span.set_attribute("memory_user_count", len(member_ids))
            memories_block = await self._memory_manager.build_memory_prompt(request.guild_id, member_ids)

            span.set_attribute("include_handoffs", include_handoffs)
            handoffs = await self._handoff_service.load_for_conversation(conversation) if include_handoffs else None
            conversation_block = await self._conversation_formatter.format_to_xml(conversation, handoffs)

            return RequestContext(memories_block=memories_block, conversation_block=conversation_block)
