"""Formats conversation messages into XML blocks for LLM prompts."""

from conversation_graph import ConversationMessage
from user_resolver import UserResolver


class ConversationFormatter:
    """Formats conversation messages into standardized XML blocks for LLM prompts."""

    def __init__(self, user_resolver: UserResolver) -> None:
        self._user_resolver = user_resolver

    async def format_to_xml(self, guild_id: int, conversation: list[ConversationMessage]) -> str:
        """Format conversation messages into XML blocks with outer tags.

        Args:
            guild_id: Discord guild ID for user context resolution
            conversation: List of conversation messages to format

        Returns:
            Complete XML conversation history block (<conversation_history>...</conversation_history>),
            or empty string if no messages
        """
        if not conversation:
            return ""

        message_blocks = []
        for msg in conversation:
            author_name = await self._user_resolver.get_display_name(guild_id, msg.author_id)
            content_with_names = await self._user_resolver.replace_user_mentions_with_names(msg.content, guild_id)

            message_block = f"""<message>
<id>{msg.message_id}</id>
{f"<reply_to>{msg.reply_to_id}</reply_to>" if msg.reply_to_id else ""}
<timestamp>{msg.timestamp}</timestamp>
<author_id>{msg.author_id}</author_id>
<author>{author_name}</author>
<content>{content_with_names}</content>
</message>"""
            message_blocks.append(message_block)

        messages_xml = "\n".join(message_blocks)
        return f"<conversation_history>\n{messages_xml}\n</conversation_history>"
