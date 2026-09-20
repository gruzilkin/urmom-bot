import unittest
from unittest.mock import AsyncMock, Mock

from conversation_formatter import ConversationFormatter
from conversation_graph import ConversationMessage


def _msg(message_id: int, author_id: int, content: str) -> ConversationMessage:
    return ConversationMessage(
        message_id=message_id,
        author_id=author_id,
        content=content,
        timestamp="2026-09-20 10:00:00",
        mentioned_user_ids=[],
    )


class TestConversationFormatter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.user_resolver = Mock()
        self.user_resolver.replace_user_mentions_with_names = AsyncMock(side_effect=lambda content, guild_id: content)
        self.formatter = ConversationFormatter(self.user_resolver)

    async def test_empty_conversation_returns_empty_string(self):
        self.assertEqual(await self.formatter.format_to_xml(1, []), "")

    async def test_messages_without_handoffs_have_no_handoff_element(self):
        xml = await self.formatter.format_to_xml(1, [_msg(10, 5, "hello"), _msg(11, 999, "answer")])

        self.assertIn("<content>hello</content>", xml)
        self.assertIn("<content>answer</content>", xml)
        self.assertNotIn("<research_handoff>", xml)

    async def test_handoff_is_embedded_inside_its_own_message(self):
        conversation = [_msg(10, 5, "question"), _msg(11, 999, "short answer"), _msg(12, 5, "follow-up")]
        handoffs = {11: "Verified: version B added it (https://example.com/notes)."}

        xml = await self.formatter.format_to_xml(1, conversation, handoffs)

        messages = xml.split("<message>")[1:]
        self.assertEqual(len(messages), 3)
        self.assertNotIn("<research_handoff>", messages[0])
        self.assertIn("<id>11</id>", messages[1])
        self.assertIn(
            "<content>short answer</content>\n<research_handoff>Verified: version B added it"
            " (https://example.com/notes).</research_handoff>",
            messages[1],
        )
        self.assertNotIn("<research_handoff>", messages[2])

    async def test_handoffs_for_unknown_messages_are_ignored(self):
        xml = await self.formatter.format_to_xml(1, [_msg(10, 5, "hi")], {77: "stale"})
        self.assertNotIn("<research_handoff>", xml)


if __name__ == "__main__":
    unittest.main()
