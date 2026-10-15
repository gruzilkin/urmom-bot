from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from zoneinfo import ZoneInfo

from conversation_graph import ConversationMessage
from store import Store


def make_request(content: str, author_id: int = 1, guild_id: int = 12345) -> ConversationMessage:
    """Build the message being routed, as app.py does for a live Discord message."""
    return ConversationMessage(
        message_id=1000,
        guild_id=guild_id,
        author_id=author_id,
        content=content,
        timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        mentioned_user_ids=[],
    )


def utc_store() -> Mock:
    """Store mock for components that only need the guild timezone; every guild is on UTC."""
    store = Mock(spec=Store)
    store.get_guild_timezone = AsyncMock(return_value=ZoneInfo("UTC"))
    return store
