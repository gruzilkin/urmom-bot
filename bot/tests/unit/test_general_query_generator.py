import unittest
from unittest.mock import AsyncMock, Mock
from discord_formatting import DISCORD_FORMATTING_INSTRUCTIONS
from general_query_generator import GeneralQueryGenerator
from research_handoff import (
    HANDOFF_CONTEXT_INSTRUCTIONS,
    HANDOFF_INSTRUCTIONS,
    ProcessorResult,
    ResearchHandoffService,
)
from schemas import GeneralParams, GeneralQueryResponse
from conversation_graph import ConversationMessage
from null_telemetry import NullTelemetry


class TestGeneralQueryGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock AI clients and map them by backend label
        self.mock_gemini_flash = Mock()
        self.mock_gemini_flash.generate_content = AsyncMock()

        self.mock_grok = Mock()
        self.mock_grok.generate_content = AsyncMock()

        self.mock_gemma = Mock()
        self.mock_gemma.generate_content = AsyncMock()

        self.client_map = {
            "gemini_flash": self.mock_gemini_flash,
            "grok": self.mock_grok,
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

        # Requesting user is now passed by id; display name is resolved internally.
        self.requesting_user_id = 123

        self.mock_user_resolver = Mock()
        self.mock_user_resolver.get_display_name = AsyncMock(return_value="TestUser")

        self.telemetry = NullTelemetry()

        # No stored handoffs by default
        self.mock_handoff_service = Mock(spec=ResearchHandoffService)
        self.mock_handoff_service.load_for_conversation = AsyncMock(return_value={})

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
            user_resolver=self.mock_user_resolver,
            handoff_service=self.mock_handoff_service,
        )

    async def test_handle_request_with_gemini_flash(self):
        """Test handling request with gemini_flash backend"""
        self.mock_gemini_flash.generate_content.return_value = GeneralQueryResponse(reply="Quick answer!")

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
            params,
            mock_conversation_fetcher,
            guild_id=12345,
            bot_user=self.mock_bot_user,
            requesting_user_id=self.requesting_user_id,
        )

        self.assertIsInstance(result, ProcessorResult)
        self.assertEqual(result.text, "Quick answer!")
        self.assertIsNone(result.handoff)
        self.mock_gemini_flash.generate_content.assert_called_once()
        self.mock_response_summarizer.process_response.assert_called_once()
        call_kwargs = self.mock_gemini_flash.generate_content.await_args.kwargs
        self.assertIn(DISCORD_FORMATTING_INSTRUCTIONS, call_kwargs["prompt"])
        self.assertIn(HANDOFF_INSTRUCTIONS, call_kwargs["prompt"])
        self.assertIn(HANDOFF_CONTEXT_INSTRUCTIONS, call_kwargs["prompt"])
        self.assertIs(call_kwargs["response_schema"], GeneralQueryResponse)

    async def test_handle_request_with_grok(self):
        """Test handling request with grok backend"""
        self.mock_grok.generate_content.return_value = GeneralQueryResponse(reply="Creative response!")

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
            params,
            mock_conversation_fetcher,
            guild_id=12345,
            bot_user=self.mock_bot_user,
            requesting_user_id=self.requesting_user_id,
        )

        self.assertEqual(result.text, "Creative response!")
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
            params,
            mock_conversation_fetcher,
            guild_id=12345,
            bot_user=self.mock_bot_user,
            requesting_user_id=self.requesting_user_id,
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
                params,
                mock_conversation_fetcher,
                guild_id=12345,
                bot_user=self.mock_bot_user,
                requesting_user_id=self.requesting_user_id,
            )

        self.assertEqual(str(context.exception), "API error")
        # Response summarizer should not be called due to exception
        self.mock_response_summarizer.process_response.assert_not_called()

    async def test_requesting_user_included_in_memory_lookup(self):
        # The requesting user's facts must be loaded into the memory block even
        # when they have no recent message in the conversation — otherwise scheduled
        # tasks (where the creator is rarely present in recent history) lose all
        # personalization.
        params = GeneralParams(ai_backend="gemini_flash", temperature=0.3, cleaned_query="hi")
        self.mock_gemini_flash.generate_content.return_value = GeneralQueryResponse(reply="Hello")

        async def mock_conversation_fetcher():
            return [
                ConversationMessage(
                    message_id=1, author_id=42, content="Hi", timestamp="2024-01-01", mentioned_user_ids=[]
                )
            ]

        await self.generator.handle_request(
            params,
            mock_conversation_fetcher,
            guild_id=1,
            bot_user=self.mock_bot_user,
            requesting_user_id=999,  # not present in conversation
        )

        self.mock_memory_manager.build_memory_prompt.assert_awaited_once()
        passed_user_ids = self.mock_memory_manager.build_memory_prompt.await_args.args[1]
        self.assertIn(999, passed_user_ids)

    async def test_client_selector_called_with_backend(self):
        """Ensure the client selector is invoked with the requested backend."""
        params = GeneralParams(ai_backend="gemini_flash", temperature=0.3, cleaned_query="Test selector")

        async def mock_conversation_fetcher():
            return [
                ConversationMessage(
                    message_id=1, author_id=2, content="Hello", timestamp="2024-01-01", mentioned_user_ids=[]
                )
            ]

        self.mock_gemini_flash.generate_content.return_value = GeneralQueryResponse(reply="Hi")

        await self.generator.handle_request(
            params,
            mock_conversation_fetcher,
            guild_id=1,
            bot_user=self.mock_bot_user,
            requesting_user_id=self.requesting_user_id,
        )

        self.mock_gemini_flash.generate_content.assert_called_once()

    # ---- research handoff ----

    async def _run(self, params: GeneralParams) -> ProcessorResult | None:
        async def mock_conversation_fetcher():
            return [
                ConversationMessage(
                    message_id=1, author_id=123, content="Question", timestamp="2024-01-01", mentioned_user_ids=[]
                )
            ]

        return await self.generator.handle_request(
            params,
            mock_conversation_fetcher,
            guild_id=12345,
            bot_user=self.mock_bot_user,
            requesting_user_id=self.requesting_user_id,
        )

    async def test_handoff_is_returned_alongside_reply(self):
        handoff = (
            "Verified: version B added it (release notes, https://example.com/notes)."
            " Unverified: behavior on the mobile client."
        )
        self.mock_gemini_flash.generate_content.return_value = GeneralQueryResponse(
            reply="Short answer", handoff=handoff
        )
        params = GeneralParams(ai_backend="gemini_flash", temperature=0.3, cleaned_query="Which version added it?")

        result = await self._run(params)

        self.assertEqual(result.text, "Short answer")
        self.assertEqual(result.handoff, handoff)

    async def test_summarization_affects_reply_only(self):
        handoff = "Kept intact"
        self.mock_gemini_flash.generate_content.return_value = GeneralQueryResponse(
            reply="A very long reply", handoff=handoff
        )
        self.mock_response_summarizer.process_response = AsyncMock(return_value="Condensed reply")
        params = GeneralParams(ai_backend="gemini_flash", temperature=0.3, cleaned_query="Explain")

        result = await self._run(params)

        self.mock_response_summarizer.process_response.assert_awaited_once_with("A very long reply")
        self.assertEqual(result.text, "Condensed reply")
        self.assertEqual(result.handoff, handoff)

    async def test_stored_handoffs_are_embedded_in_conversation(self):
        handoffs = {42: "Verified: version B added it (https://example.com/notes)."}
        self.mock_handoff_service.load_for_conversation.return_value = handoffs
        self.mock_gemini_flash.generate_content.return_value = GeneralQueryResponse(reply="Follow-up answer")
        params = GeneralParams(ai_backend="gemini_flash", temperature=0.3, cleaned_query="And on mobile?")

        await self._run(params)

        passed_conversation, passed_bot_id = self.mock_handoff_service.load_for_conversation.await_args.args
        self.assertEqual(passed_bot_id, self.mock_bot_user.id)
        self.assertEqual(len(passed_conversation), 1)
        # Handoffs travel into the conversation XML so each one sits inside its own message
        self.mock_conversation_formatter.format_to_xml.assert_awaited_once_with(12345, passed_conversation, handoffs)

    async def test_empty_reply_is_treated_as_no_response(self):
        self.mock_gemini_flash.generate_content.return_value = GeneralQueryResponse(reply="   ")
        params = GeneralParams(ai_backend="gemini_flash", temperature=0.3, cleaned_query="hi")

        result = await self._run(params)

        self.assertIsNone(result)
        self.mock_response_summarizer.process_response.assert_not_called()


if __name__ == "__main__":
    unittest.main()
