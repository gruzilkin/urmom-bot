import datetime
from dataclasses import dataclass


@dataclass
class MessageNode:
    """Materialized message representation with processed embeds."""
    id: int
    content: str
    author_id: int
    mentioned_user_ids: list[int]
    created_at: datetime.datetime
    reference_id: int | None = None
    
    def __post_init__(self) -> None:
        if self.mentioned_user_ids is None:
            self.mentioned_user_ids = []