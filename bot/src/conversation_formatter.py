"""Formats conversation messages into XML blocks for LLM prompts."""

from datetime import datetime
from zoneinfo import ZoneInfo

from conversation_graph import ConversationMessage
from store import Store
from user_resolver import UserResolver

CONVERSATION_STRUCTURE_INSTRUCTIONS = """Conversation Structure:
- <conversation_history> contains earlier messages in chronological
  order, each in its own <message> block
- <request>, if present, is the message currently being handled. It
  has the same fields as <message> and may also appear as the last
  message in <conversation_history>
- Message fields:
  - <id>: Discord message ID
  - <reply_to>: present only when the message is a Discord reply;
    holds the <id> of the message being replied to
  - <timestamp>: when the message was sent, in the server's local time
    with its UTC offset
  - <member_id>: Discord user ID of the author
  - <content>: message text"""


def format_timestamp(timestamp: datetime, tz: ZoneInfo) -> str:
    """Render a timestamp in local time with weekday and UTC offset, e.g. '2026-10-15 Thu 21:42 +09:00'."""
    local = timestamp.astimezone(tz)
    offset = local.strftime("%z")
    return f"{local.strftime('%Y-%m-%d %a %H:%M')} {offset[:3]}:{offset[3:]}"


class ConversationFormatter:
    """Formats conversation messages into standardized XML blocks for LLM prompts."""

    def __init__(self, user_resolver: UserResolver, store: Store) -> None:
        self._user_resolver = user_resolver
        self._store = store

    async def format_request_message_xml(self, request: ConversationMessage, content: str | None = None) -> str:
        """Format the message being handled as a <request> block mirroring the <message> structure.

        `content` replaces its text (e.g. with a cleaned query). Fields a request lacks (scheduled task
        firings have no Discord message id) are omitted.
        """
        tz = await self._store.get_guild_timezone(request.guild_id)
        fields = [
            f"<id>{request.message_id}</id>" if request.message_id is not None else "",
            f"<reply_to>{request.reply_to_id}</reply_to>" if request.reply_to_id is not None else "",
            f"<timestamp>{format_timestamp(request.timestamp, tz)}</timestamp>",
            f"<member_id>{request.author_id}</member_id>",
            f"<content>{request.content if content is None else content}</content>",
        ]
        body = "\n".join(field for field in fields if field)
        return f"<request>\n{body}\n</request>"

    async def format_to_xml(
        self,
        conversation: list[ConversationMessage],
        handoffs: dict[int, str] | None = None,
    ) -> str:
        """Format conversation messages into XML blocks with outer tags.

        Args:
            conversation: List of conversation messages to format, all from one guild
            handoffs: Optional research handoff text keyed by message ID; embedded as a
                <research_handoff> element inside the matching message

        Returns:
            Complete XML conversation history block (<conversation_history>...</conversation_history>),
            or empty string if no messages
        """
        if not conversation:
            return ""

        handoffs = handoffs or {}
        tz = await self._store.get_guild_timezone(conversation[0].guild_id)
        message_blocks = []
        for msg in conversation:
            content_with_names = await self._user_resolver.replace_user_mentions_with_names(msg.content, msg.guild_id)
            handoff = handoffs.get(msg.message_id)

            message_block = f"""<message>
<id>{msg.message_id}</id>
{f"<reply_to>{msg.reply_to_id}</reply_to>" if msg.reply_to_id else ""}
<timestamp>{format_timestamp(msg.timestamp, tz)}</timestamp>
<member_id>{msg.author_id}</member_id>
<content>{content_with_names}</content>
{f"<research_handoff>{handoff}</research_handoff>" if handoff else ""}
</message>"""
            message_blocks.append(message_block)

        messages_xml = "\n".join(message_blocks)
        return f"<conversation_history>\n{messages_xml}\n</conversation_history>"
