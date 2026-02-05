"""
Integration tests for AiRouter that verify routing behaviour across multiple
client configurations (Gemma/Gemini and Ollama Kimi).
"""

import os
import unittest
from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv

from ai_client import AIClient
from ai_client_wrappers import CompositeAIClient
from ai_router import AiRouter
from conversation_formatter import ConversationFormatter
from fact_handler import FactHandler
from famous_person_generator import FamousPersonGenerator
from gemini_client import GeminiClient
from gemma_client import GemmaClient
from general_query_generator import GeneralQueryGenerator
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry
from ollama_client import OllamaClient

load_dotenv()


@dataclass(frozen=True)
class RouterProfile:
    """Container for router instance and identifying metadata."""

    name: str
    router: AiRouter


class TestAiRouterIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for AiRouter."""

    async def asyncSetUp(self):
        """Create router profiles that mirror production client mixes."""
        self.telemetry = NullTelemetry()
        self.profiles: list[RouterProfile] = []

        # Default test values for route_request calls
        self.default_guild_id = 12345
        self.default_conversation_fetcher = AsyncMock(return_value=[])

        gemini_profile = self._build_gemini_profile()
        if gemini_profile:
            self.profiles.append(gemini_profile)

        gpt_oss_profile = self._build_gpt_oss_profile()
        if gpt_oss_profile:
            self.profiles.append(gpt_oss_profile)

        kimi_profile = self._build_kimi_profile()
        if kimi_profile:
            self.profiles.append(kimi_profile)

        if not self.profiles:
            self.skipTest("No AI router profiles configured; ensure Gemini and/or Ollama credentials are set.")

    def _build_gemini_profile(self) -> RouterProfile | None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemma_api_key = os.getenv("GEMMA_API_KEY")
        gemma_model = os.getenv("GEMMA_MODEL")
        flash_model = os.getenv("GEMINI_FLASH_MODEL")

        if not all([gemini_api_key, gemma_api_key, gemma_model, flash_model]):
            return None

        gemma_client = GemmaClient(
            api_key=gemma_api_key,
            model_name=gemma_model,
            telemetry=self.telemetry,
        )
        flash_client = GeminiClient(
            api_key=gemini_api_key,
            model_name=flash_model,
            telemetry=self.telemetry,
        )

        router_client = CompositeAIClient(
            [gemma_client, flash_client],
            telemetry=self.telemetry,
            is_bad_response=lambda r: getattr(r, "route", None) == "NOTSURE",
        )

        router = self._create_router(router_client, language_detector_client=gemma_client)
        return RouterProfile(name="gemma_gemini", router=router)

    def _build_gpt_oss_profile(self) -> RouterProfile | None:
        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        model_name = os.getenv("OLLAMA_GPT_OSS_MODEL", "gpt-oss:120b-cloud")
        if not ollama_api_key:
            return None

        gpt_oss_client = OllamaClient(
            api_key=ollama_api_key,
            model_name=model_name,
            telemetry=self.telemetry,
            temperature=0.1,
        )

        router_client = CompositeAIClient(
            [gpt_oss_client],
            telemetry=self.telemetry,
            is_bad_response=lambda r: getattr(r, "route", None) == "NOTSURE",
        )

        router = self._create_router(router_client, language_detector_client=gpt_oss_client)
        return RouterProfile(name="ollama_gpt_oss", router=router)

    def _build_kimi_profile(self) -> RouterProfile | None:
        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        model_name = os.getenv("OLLAMA_KIMI_MODEL", "kimi-k2:1t-cloud")
        if not ollama_api_key:
            return None

        kimi_client = OllamaClient(
            api_key=ollama_api_key,
            model_name=model_name,
            telemetry=self.telemetry,
            temperature=0.1,
        )

        router_client = CompositeAIClient(
            [kimi_client],
            telemetry=self.telemetry,
            is_bad_response=lambda r: getattr(r, "route", None) == "NOTSURE",
        )

        router = self._create_router(router_client, language_detector_client=kimi_client)
        return RouterProfile(name="ollama_kimi", router=router)

    def _create_router(self, router_client: AIClient, *, language_detector_client: AIClient) -> AiRouter:
        """Instantiate AiRouter with shared mocks and specified AI client."""
        mock_user_resolver = Mock()
        mock_store = Mock()
        conversation_formatter = ConversationFormatter(mock_user_resolver)

        famous_generator = FamousPersonGenerator(
            ai_client=None,
            response_summarizer=None,
            telemetry=self.telemetry,
            conversation_formatter=conversation_formatter,
        )
        general_generator = GeneralQueryGenerator(
            client_selector=lambda _: AsyncMock(spec=AIClient),
            response_summarizer=None,
            telemetry=self.telemetry,
            store=mock_store,
            conversation_formatter=conversation_formatter,
            memory_manager=AsyncMock(),
        )
        fact_handler = FactHandler(
            ai_client=None,
            store=None,
            telemetry=self.telemetry,
            user_resolver=mock_user_resolver,
        )

        language_detector = LanguageDetector(
            ai_client=language_detector_client,
            telemetry=self.telemetry,
        )

        return AiRouter(
            ai_client=router_client,
            telemetry=self.telemetry,
            language_detector=language_detector,
            famous_generator=famous_generator,
            general_generator=general_generator,
            fact_handler=fact_handler,
            conversation_formatter=conversation_formatter,
        )

    async def test_route_request_with_perspective_shift(self):
        """
        Test that the router correctly rephrases a third-person request
        into a direct, second-person query.
        """
        user_message = "let's ask BOT to use grok and be creative to tell me a joke"
        expected_cleaned_query = "tell me a joke"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(route, "GENERAL")
                self.assertIsNotNone(params)
                self.assertEqual(params.ai_backend, "grok")
                self.assertEqual(params.cleaned_query.lower(), expected_cleaned_query.lower())
                self.assertGreaterEqual(
                    params.temperature,
                    0.7,
                    "Temperature should be high for a 'creative' request",
                )

    async def test_route_request_with_direct_command(self):
        """
        Test that the router correctly handles a direct command to the bot,
        stripping out only the bot's name and routing hints.
        """
        user_message = "BOT, ask claude to write a technical blog post, be very detailed"
        expected_cleaned_query = "write a technical blog post"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(route, "GENERAL")
                self.assertIsNotNone(params)
                self.assertEqual(params.ai_backend, "claude")
                self.assertEqual(params.cleaned_query.lower(), expected_cleaned_query.lower())
                self.assertLessEqual(
                    params.temperature,
                    0.3,
                    "Temperature should be low for a 'detailed' request",
                )

    async def test_route_request_memory_remember(self):
        """
        Test that the router correctly identifies and routes memory remember commands.
        """
        user_message = "Bot remember that <@1333878858138652682> works at TechCorp"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(route, "FACT")
                self.assertIsNotNone(params)
                self.assertEqual(params.operation, "remember")
                self.assertEqual(params.user_mention, "1333878858138652682")
                self.assertIn("TechCorp", params.fact_content)

    async def test_route_request_memory_forget(self):
        """
        Test that the router correctly identifies and routes memory forget commands.
        """
        user_message = "Bot forget that gruzilkin likes pizza"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(route, "FACT")
                self.assertIsNotNone(params)
                self.assertEqual(params.operation, "forget")
                self.assertEqual(params.user_mention, "gruzilkin")
                self.assertIn("pizza", params.fact_content.lower())
                self.assertIn("like", params.fact_content.lower())

    async def test_route_request_famous_person_news_not_impersonation(self):
        user_message = "What did Trump say yesterday?"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(
                    route,
                    "GENERAL",
                    "Questions about actual statements should route to GENERAL for news search",
                )
                self.assertIsNotNone(params)
                self.assertIn("Trump", params.cleaned_query)

    async def test_route_request_memory_verb_not_bot_memory(self):
        user_message = "I can't remember where I put my keys"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertNotEqual(
                    route,
                    "FACT",
                    "Personal memory statements should not trigger bot memory operations",
                )
                self.assertIn(
                    route,
                    ["GENERAL", "NONE"],
                    "Personal statements should route to GENERAL or NONE",
                )

                if route == "NONE":
                    self.assertIsNone(params)
                else:
                    self.assertIsNotNone(params)

    async def test_route_request_quote_lookup_not_impersonation(self):
        user_message = "Einstein said something about imagination"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(
                    route,
                    "GENERAL",
                    "Quote lookup requests should route to GENERAL, not FAMOUS",
                )
                self.assertIsNotNone(params)
                self.assertIn("Einstein", params.cleaned_query)

    async def test_route_request_riddle_with_famous_names_not_impersonation(self):
        user_message = "кто до Путина, если после Путина Агутин?"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertNotEqual(
                    route,
                    "FAMOUS",
                    "Riddles mentioning famous people should not trigger impersonation",
                )
                self.assertIn(
                    route,
                    ["GENERAL", "NONE"],
                    "Riddles should route to GENERAL (question) or NONE (wordplay)",
                )

                if route == "NONE":
                    self.assertIsNone(params)
                else:
                    self.assertIsNotNone(params)

    async def test_route_request_wordplay_question_not_impersonation(self):
        user_message = "у Дональда Трампа козырная фамилия?"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(
                    route,
                    "GENERAL",
                    "Wordplay questions about names should route to GENERAL, not FAMOUS",
                )
                self.assertIsNotNone(params, "GENERAL route should have parameters")
                self.assertIn("Трамп", params.cleaned_query, "Should preserve the name in the query")

    async def test_route_request_statement_about_person_not_impersonation(self):
        user_message = "Медвед бы так не сказал"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertNotEqual(
                    route,
                    "FAMOUS",
                    "Statements about what someone would say should not trigger impersonation",
                )
                self.assertIn(
                    route,
                    ["NONE", "GENERAL"],
                    "Should route to NONE (statement) or GENERAL (opinion)",
                )

                if route == "NONE":
                    self.assertIsNone(params)
                else:
                    self.assertIsNotNone(params)

    async def test_route_request_praise_command_not_impersonation(self):
        user_message = "спой осанну Медведу"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(
                    route,
                    "GENERAL",
                    "Commands to praise someone should route to GENERAL, not FAMOUS",
                )
                self.assertIsNotNone(params, "GENERAL route should have parameters")
                self.assertIn("осанну", params.cleaned_query, "Should preserve the praise request")

    async def test_route_request_question_with_name_not_memory_operation(self):
        user_message = "для чего Алексею нужна голова?"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(
                    route,
                    "GENERAL",
                    "Questions with names should route to GENERAL, not FACT",
                )
                self.assertIsNotNone(params, "GENERAL route should have parameters")
                self.assertIn("Алексею", params.cleaned_query, "Should preserve the name in the query")

    async def test_route_request_chatgpt_alias_selects_codex(self):
        """Test that chatgpt/openai requests are routed to codex backend."""
        user_message = "BOT use chatgpt to research climate change"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(route, "GENERAL")
                self.assertIsNotNone(params)
                self.assertEqual(
                    params.ai_backend,
                    "codex",
                    "ChatGPT requests should select codex backend",
                )

    async def test_route_request_song_writing_selects_claude(self):
        """Test that song writing requests are routed to Claude backend."""
        user_message = "BOT write a song about summer"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                route, params = await profile.router.route_request(
                    user_message, self.default_conversation_fetcher, self.default_guild_id
                )

                self.assertEqual(route, "GENERAL", "Song requests should route to GENERAL")
                self.assertIsNotNone(params, "GENERAL route should have parameters")
                self.assertEqual(
                    params.ai_backend,
                    "claude",
                    "Song requests should select Claude backend",
                )
                self.assertIn(
                    "song",
                    params.cleaned_query.lower(),
                    "Should preserve song request in query",
                )


if __name__ == "__main__":
    unittest.main()
