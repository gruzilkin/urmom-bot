from conversation_graph import ConversationMessage


def make_request(content: str, author_id: int = 1) -> ConversationMessage:
    """Build the message being routed, as app.py does for a live Discord message."""
    return ConversationMessage(
        message_id=1000,
        author_id=author_id,
        content=content,
        timestamp="2024-01-01 12:00:00",
        mentioned_user_ids=[],
    )
