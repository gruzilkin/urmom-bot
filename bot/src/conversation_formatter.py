"""Formats conversation messages into XML blocks for LLM prompts."""

from conversation_graph import ConversationMessage
from user_resolver import UserResolver

CONVERSATION_STRUCTURE_INSTRUCTIONS = """Conversation Structure:
- <conversation_history> contains earlier messages in chronological
  order, each in its own <message> block
- <request> is the message currently being handled. It has the same
  fields as <message> and may also appear as the last message in
  <conversation_history>
- Message fields:
  - <id>: Discord message ID
  - <reply_to>: present only when the message is a Discord reply;
    holds the <id> of the message being replied to
  - <timestamp>: when the message was sent
  - <member_id>: Discord user ID of the author
  - <content>: message text"""


def format_request_xml(
    author_id: int,
    content: str,
    message_id: int | None = None,
    reply_to_id: int | None = None,
    timestamp: str | None = None,
) -> str:
    """Format the message being handled as a <request> block mirroring the <message> structure.

    Fields unknown for the request (e.g. scheduled task firings have no Discord message) are omitted.
    """
    fields = [
        f"<id>{message_id}</id>" if message_id is not None else "",
        f"<reply_to>{reply_to_id}</reply_to>" if reply_to_id is not None else "",
        f"<timestamp>{timestamp}</timestamp>" if timestamp is not None else "",
        f"<member_id>{author_id}</member_id>",
        f"<content>{content}</content>",
    ]
    body = "\n".join(field for field in fields if field)
    return f"<request>\n{body}\n</request>"


def format_request_message_xml(request: ConversationMessage, content: str | None = None) -> str:
    """Format a Discord message as the <request> block; `content` replaces its text (e.g. with a cleaned query)."""
    return format_request_xml(
        author_id=request.author_id,
        content=request.content if content is None else content,
        message_id=request.message_id,
        reply_to_id=request.reply_to_id,
        timestamp=request.timestamp,
    )


class ConversationFormatter:
    """Formats conversation messages into standardized XML blocks for LLM prompts."""

    def __init__(self, user_resolver: UserResolver) -> None:
        self._user_resolver = user_resolver

    async def format_to_xml(
        self,
        guild_id: int,
        conversation: list[ConversationMessage],
        handoffs: dict[int, str] | None = None,
    ) -> str:
        """Format conversation messages into XML blocks with outer tags.

        Args:
            guild_id: Discord guild ID for user context resolution
            conversation: List of conversation messages to format
            handoffs: Optional research handoff text keyed by message ID; embedded as a
                <research_handoff> element inside the matching message

        Returns:
            Complete XML conversation history block (<conversation_history>...</conversation_history>),
            or empty string if no messages
        """
        if not conversation:
            return ""

        handoffs = handoffs or {}
        message_blocks = []
        for msg in conversation:
            content_with_names = await self._user_resolver.replace_user_mentions_with_names(msg.content, guild_id)
            handoff = handoffs.get(msg.message_id)

            message_block = f"""<message>
<id>{msg.message_id}</id>
{f"<reply_to>{msg.reply_to_id}</reply_to>" if msg.reply_to_id else ""}
<timestamp>{msg.timestamp}</timestamp>
<member_id>{msg.author_id}</member_id>
<content>{content_with_names}</content>
{f"<research_handoff>{handoff}</research_handoff>" if handoff else ""}
</message>"""
            message_blocks.append(message_block)

        messages_xml = "\n".join(message_blocks)
        return f"<conversation_history>\n{messages_xml}\n</conversation_history>"
