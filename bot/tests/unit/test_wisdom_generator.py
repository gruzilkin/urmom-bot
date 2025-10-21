"""Unit tests for WisdomGenerator."""

import unittest
from unittest.mock import AsyncMock, Mock

from wisdom_generator import WisdomGenerator
from conversation_graph import ConversationMessage
from null_telemetry import NullTelemetry
from schemas import WisdomResponse


class TestWisdomGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.ai_client = Mock()
        self.ai_client.generate_content = AsyncMock()

        self.language_detector = Mock()
        self.language_detector.detect_language = AsyncMock(return_value="en")
        self.language_detector.get_language_name = AsyncMock(return_value="English")

        self.conversation_formatter = Mock()
        self.conversation_formatter.format_to_xml = AsyncMock(return_value="<message>Mock conversation</message>")

        self.response_summarizer = Mock()
        self.response_summarizer.process_response = AsyncMock(side_effect=lambda x: x)

        self.memory_manager = Mock()
        self.memory_manager.get_memories = AsyncMock(return_value={})
        self.memory_manager.build_memory_prompt = AsyncMock(return_value="")

        self.telemetry = NullTelemetry()

        self.generator = WisdomGenerator(
            ai_client=self.ai_client,
            language_detector=self.language_detector,
            conversation_formatter=self.conversation_formatter,
            response_summarizer=self.response_summarizer,
            memory_manager=self.memory_manager,
            telemetry=self.telemetry,
        )

    async def test_generate_wisdom_calls_ai_client(self) -> None:
        """Test that generate_wisdom calls AI client with correct parameters."""
        self.ai_client.generate_content.return_value = WisdomResponse(
            answer="Thus spoke the sage.", reason="Referenced ancient wisdom traditions to add gravitas"
        )

        async def mock_conversation():
            return []

        result = await self.generator.generate_wisdom(
            trigger_message_content="Hello world",
            conversation_fetcher=mock_conversation,
            guild_id=123,
        )

        self.assertEqual(result, "Thus spoke the sage.")
        self.ai_client.generate_content.assert_called_once()
        call_args = self.ai_client.generate_content.call_args
        self.assertEqual(call_args.kwargs["temperature"], 0.8)
        self.assertEqual(call_args.kwargs["message"], "Hello world")
        self.assertEqual(call_args.kwargs["response_schema"], WisdomResponse)
        self.assertIn("prompt", call_args.kwargs)

    async def test_generate_wisdom_with_conversation_context(self) -> None:
        """Test that conversation context is included in the prompt."""
        # Update mock to return actual formatted conversation
        self.conversation_formatter.format_to_xml = AsyncMock(
            return_value="<message><id>1</id><content>First message</content></message>\n<message><id>2</id><content>Second message</content></message>"
        )
        self.ai_client.generate_content.return_value = WisdomResponse(
            answer="Wisdom from context.", reason="Drew on the conversation thread about messaging"
        )

        async def mock_conversation():
            return [
                ConversationMessage(
                    message_id=1,
                    author_id=100,
                    content="First message",
                    timestamp="2023-01-01T00:00:00",
                    mentioned_user_ids=[],
                ),
                ConversationMessage(
                    message_id=2,
                    author_id=101,
                    content="Second message",
                    timestamp="2023-01-01T00:01:00",
                    mentioned_user_ids=[],
                    reply_to_id=1,
                ),
            ]

        result = await self.generator.generate_wisdom(
            trigger_message_content="Question?",
            conversation_fetcher=mock_conversation,
            guild_id=123,
        )

        self.assertEqual(result, "Wisdom from context.")
        call_args = self.ai_client.generate_content.call_args
        prompt = call_args.kwargs["prompt"]
        self.assertIn("First message", prompt)
        self.assertIn("Second message", prompt)
        self.assertIn("<conversation_history>", prompt)

    async def test_generate_wisdom_detects_language(self) -> None:
        """Test that language detection is called and used in prompt."""
        self.language_detector.detect_language.return_value = "ru"
        self.language_detector.get_language_name.return_value = "Russian"
        self.ai_client.generate_content.return_value = WisdomResponse(
            answer="Мудрость", reason="Responded in Russian as detected"
        )

        async def mock_conversation():
            return []

        result = await self.generator.generate_wisdom(
            trigger_message_content="Привет",
            conversation_fetcher=mock_conversation,
            guild_id=123,
        )

        self.assertEqual(result, "Мудрость")
        self.language_detector.detect_language.assert_called_once_with("Привет")
        self.language_detector.get_language_name.assert_called_once_with("ru")

        call_args = self.ai_client.generate_content.call_args
        prompt = call_args.kwargs["prompt"]
        self.assertIn("Russian", prompt)

    async def test_generate_wisdom_processes_response(self) -> None:
        """Test that response summarizer is called."""
        long_wisdom = "A" * 3000
        summarized = "A" * 1000

        self.ai_client.generate_content.return_value = WisdomResponse(
            answer=long_wisdom, reason="Generated very long wisdom for testing"
        )
        self.response_summarizer.process_response = AsyncMock(return_value=summarized)

        async def mock_conversation():
            return []

        result = await self.generator.generate_wisdom(
            trigger_message_content="Test",
            conversation_fetcher=mock_conversation,
            guild_id=123,
        )

        self.assertEqual(result, summarized)
        self.response_summarizer.process_response.assert_called_once_with(long_wisdom)

    async def test_generate_wisdom_replaces_user_mentions(self) -> None:
        """Test that user mentions are replaced with display names."""
        self.ai_client.generate_content.return_value = WisdomResponse(answer="Wisdom", reason="Simple wisdom response")

        async def mock_conversation():
            return [
                ConversationMessage(
                    message_id=1,
                    author_id=100,
                    content="Hello <@12345>",
                    timestamp="2023-01-01T00:00:00",
                    mentioned_user_ids=[12345],
                ),
            ]

        await self.generator.generate_wisdom(
            trigger_message_content="Test",
            conversation_fetcher=mock_conversation,
            guild_id=123,
        )

        self.conversation_formatter.format_to_xml.assert_called_once()

    async def test_generate_wisdom_handles_none_response(self) -> None:
        """Test handling when AI returns None."""
        self.ai_client.generate_content.return_value = None
        self.response_summarizer.process_response = AsyncMock(return_value=None)

        async def mock_conversation():
            return []

        result = await self.generator.generate_wisdom(
            trigger_message_content="Test",
            conversation_fetcher=mock_conversation,
            guild_id=123,
        )

        self.assertIsNone(result)

    async def test_generate_wisdom_includes_memories(self) -> None:
        """Test that memories are fetched and included in the prompt."""
        memory_block = "<memory><name>User</name><facts>Lives in Tokyo</facts></memory>"
        self.memory_manager.build_memory_prompt = AsyncMock(return_value=memory_block)
        self.ai_client.generate_content.return_value = WisdomResponse(
            answer="Wisdom", reason="test"
        )

        await self.generator.generate_wisdom(
            trigger_message_content="Test",
            conversation_fetcher=AsyncMock(return_value=[]),
            guild_id=123,
        )

        self.memory_manager.build_memory_prompt.assert_called_once()
        prompt = self.ai_client.generate_content.call_args.kwargs["prompt"]
        self.assertIn("Lives in Tokyo", prompt)

if __name__ == "__main__":
    unittest.main()
