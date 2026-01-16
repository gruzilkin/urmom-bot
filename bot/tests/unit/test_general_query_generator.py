import unittest
from unittest.mock import AsyncMock, Mock
from general_query_generator import GeneralQueryGenerator
from schemas import GeneralParams
from conversation_graph import ConversationMessage
from null_telemetry import NullTelemetry


class TestGeneralQueryGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock AI clients and map them by backend label
        self.mock_gemini_flash = Mock()
        self.mock_gemini_flash.generate_content = AsyncMock()

        self.mock_grok = Mock()
        self.mock_grok.generate_content = AsyncMock()

        self.mock_claude = Mock()
        self.mock_claude.generate_content = AsyncMock()

        self.mock_gemma = Mock()
        self.mock_gemma.generate_content = AsyncMock()

        self.client_map = {
            "gemini_flash": self.mock_gemini_flash,
            "grok": self.mock_grok,
            "claude": self.mock_claude,
            "gemma": self.mock_gemma,
        }

        # Mock response summarizer that returns what it receives (passthrough)
        self.mock_response_summarizer = Mock()
        self.mock_response_summarizer.process_response = AsyncMock(side_effect=lambda x, **kwargs: x)

        # Mock store and conversation_formatter
        self.mock_store = Mock()
        self.mock_store.get_user_facts = AsyncMock(return_value=None)

        self.mock_conversation_formatter = Mock()
        self.mock_conversation_formatter.format_to_xml = AsyncMock(
            return_value="<conversation_history>\n<message>Mock conversation</message>\n</conversation_history>"
        )

        self.mock_memory_manager = Mock()
        self.mock_memory_manager.get_memories = AsyncMock(return_value={})
        self.mock_memory_manager.build_memory_prompt = AsyncMock(return_value="")

        # Mock bot user for tests
        self.mock_bot_user = Mock()
        self.mock_bot_user.name = "urmom-bot"
        self.mock_bot_user.id = 99999

        # Mock requesting user
        self.mock_requesting_user = Mock()
        self.mock_requesting_user.id = 123
        self.mock_requesting_user.display_name = "TestUser"

        self.telemetry = NullTelemetry()

        def selector(backend: str):
            try:
                return self.client_map[backend]
            except KeyError as exc:
                raise ValueError(f"Unknown ai_backend: {backend}") from exc

        self.generator = GeneralQueryGenerator(
            client_selector=selector,
            response_summarizer=self.mock_response_summarizer,
            telemetry=self.telemetry,
            store=self.mock_store,
            conversation_formatter=self.mock_conversation_formatter,
            memory_manager=self.mock_memory_manager,
        )

    async def test_handle_request_with_gemini_flash(self):
        """Test handling request with gemini_flash backend"""
        self.mock_gemini_flash.generate_content.return_value = "Quick answer!"

        params = GeneralParams(ai_backend="gemini_flash", temperature=0.3, cleaned_query="Simple question?")

        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            msg = ConversationMessage(
                message_id=12345,
                author_id=123,
                content="Test message",
                timestamp="2024-01-01 12:00:00",
                mentioned_user_ids=[],
            )
            return [msg]

        result = await self.generator.handle_request(
            params, mock_conversation_fetcher, guild_id=12345, bot_user=self.mock_bot_user,
            requesting_user=self.mock_requesting_user
        )

        self.assertEqual(result, "Quick answer!")
        self.mock_gemini_flash.generate_content.assert_called_once()
        self.mock_response_summarizer.process_response.assert_called_once()

    async def test_handle_request_with_grok(self):
        """Test handling request with grok backend"""
        self.mock_grok.generate_content.return_value = "Creative response!"

        params = GeneralParams(ai_backend="grok", temperature=0.8, cleaned_query="Be creative!")

        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            return [
                ConversationMessage(
                    message_id=10001,
                    author_id=1000,
                    content="Let's get creative",
                    timestamp="2024-01-01 12:00:00",
                    mentioned_user_ids=[],
                )
            ]

        result = await self.generator.handle_request(
            params, mock_conversation_fetcher, guild_id=12345, bot_user=self.mock_bot_user,
            requesting_user=self.mock_requesting_user
        )

        self.assertEqual(result, "Creative response!")
        self.mock_grok.generate_content.assert_called_once()
        self.mock_response_summarizer.process_response.assert_called_once()

    async def test_handle_request_none_response(self):
        """Test handling when AI client returns None response"""
        self.mock_gemini_flash.generate_content.return_value = None

        params = GeneralParams(ai_backend="gemini_flash", temperature=0.5, cleaned_query="Test query")

        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            msg = ConversationMessage(
                message_id=12346,
                author_id=123,
                content="Test message",
                timestamp="2024-01-01 12:00:00",
                mentioned_user_ids=[],
            )
            return [msg]

        result = await self.generator.handle_request(
            params, mock_conversation_fetcher, guild_id=12345, bot_user=self.mock_bot_user,
            requesting_user=self.mock_requesting_user
        )

        # Should return None when AI client returns None
        self.assertIsNone(result)
        self.mock_gemini_flash.generate_content.assert_called_once()
        # Response summarizer should not be called when response is None
        self.mock_response_summarizer.process_response.assert_not_called()

    async def test_handle_request_error_handling(self):
        """Test error handling in handle_request - exceptions should propagate"""
        self.mock_gemini_flash.generate_content.side_effect = Exception("API error")

        params = GeneralParams(ai_backend="gemini_flash", temperature=0.5, cleaned_query="What's the weather today?")

        # Mock conversation fetcher
        async def mock_conversation_fetcher():
            msg = ConversationMessage(
                message_id=12347,
                author_id=123,
                content="Test message",
                timestamp="2024-01-01 12:00:00",
                mentioned_user_ids=[],
            )
            return [msg]

        # Exception should propagate instead of being caught
        with self.assertRaises(Exception) as context:
            await self.generator.handle_request(
                params, mock_conversation_fetcher, guild_id=12345, bot_user=self.mock_bot_user,
                requesting_user=self.mock_requesting_user
            )

        self.assertEqual(str(context.exception), "API error")
        # Response summarizer should not be called due to exception
        self.mock_response_summarizer.process_response.assert_not_called()

    async def test_client_selector_called_with_backend(self):
        """Ensure the client selector is invoked with the requested backend."""
        params = GeneralParams(ai_backend="gemini_flash", temperature=0.3, cleaned_query="Test selector")

        async def mock_conversation_fetcher():
            return [
                ConversationMessage(
                    message_id=1, author_id=2, content="Hello", timestamp="2024-01-01", mentioned_user_ids=[]
                )
            ]

        self.mock_gemini_flash.generate_content.return_value = "Hi"

        await self.generator.handle_request(
            params, mock_conversation_fetcher, guild_id=1, bot_user=self.mock_bot_user,
            requesting_user=self.mock_requesting_user
        )

        self.mock_gemini_flash.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
