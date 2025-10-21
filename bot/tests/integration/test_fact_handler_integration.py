"""
Integration tests for FactHandler across multiple AI clients.

Each test exercises the same memory scenarios (remember, merge, forget, routing)
against Gemma and Ollama cloud models to ensure parity in behaviour and language
preservation.
"""

import os
import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

from dotenv import load_dotenv

from ai_client import AIClient
from ai_router import AiRouter
from conversation_formatter import ConversationFormatter
from fact_handler import FactHandler
from famous_person_generator import FamousPersonGenerator
from gemma_client import GemmaClient
from general_query_generator import GeneralQueryGenerator
from language_detector import LanguageDetector
from null_telemetry import NullTelemetry
from ollama_client import OllamaClient
from test_store import TestStore

load_dotenv()


@dataclass(frozen=True)
class FactClientProfile:
    """Declarative container for FactHandler client configuration."""

    name: str
    client: AIClient


@dataclass
class FactTestContext:
    """Fresh state for a single test run."""

    fact_handler: FactHandler
    ai_router: AiRouter
    store: TestStore
    guild_id: int
    target_user_id: int


class TestFactHandlerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for FactHandler with real AI clients."""

    async def asyncSetUp(self):
        """Configure the available FactHandler client profiles."""
        self.telemetry = NullTelemetry()
        self.target_user = "Einstein"
        self.profiles: list[FactClientProfile] = []

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemma_model = os.getenv("GEMINI_GEMMA_MODEL")

        if gemini_api_key and gemma_model:
            gemma_client = GemmaClient(
                api_key=gemini_api_key,
                model_name=gemma_model,
                telemetry=self.telemetry,
                temperature=0.0,
            )
            self.profiles.append(FactClientProfile(name="gemma", client=gemma_client))

        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        if ollama_api_key:
            gpt_oss_model = os.getenv("OLLAMA_GPT_OSS_MODEL", "gpt-oss:120b-cloud")
            kimi_model = os.getenv("OLLAMA_KIMI_MODEL", "kimi-k2:1t-cloud")

            gpt_oss_client = OllamaClient(
                api_key=ollama_api_key,
                model_name=gpt_oss_model,
                telemetry=self.telemetry,
                temperature=0.0,
            )
            self.profiles.append(FactClientProfile(name="ollama_gpt_oss", client=gpt_oss_client))

            kimi_client = OllamaClient(
                api_key=ollama_api_key,
                model_name=kimi_model,
                telemetry=self.telemetry,
                temperature=0.0,
            )
            self.profiles.append(FactClientProfile(name="ollama_kimi", client=kimi_client))

        if not self.profiles:
            self.skipTest("No FactHandler AI clients configured; ensure Gemma or Ollama credentials are set.")

    def _build_context(self, client: AIClient) -> FactTestContext:
        """Create a fresh FactHandler + router stack for a single test run."""
        store = TestStore()
        user_resolver = store.user_resolver
        conversation_formatter = ConversationFormatter(user_resolver)

        fact_handler = FactHandler(
            ai_client=client,
            store=store,
            telemetry=self.telemetry,
            user_resolver=user_resolver,
        )

        famous_generator = FamousPersonGenerator(
            ai_client=Mock(),
            response_summarizer=Mock(),
            telemetry=self.telemetry,
            conversation_formatter=conversation_formatter,
        )
        famous_generator.get_parameter_schema = Mock(
            side_effect=Exception("Should not extract FAMOUS params in FACT tests")
        )
        famous_generator.get_parameter_extraction_prompt = Mock(
            side_effect=Exception("Should not extract FAMOUS params in FACT tests")
        )

        general_generator = GeneralQueryGenerator(
            client_selector=lambda _: AsyncMock(spec=AIClient),
            response_summarizer=Mock(),
            telemetry=self.telemetry,
            store=Mock(),
            conversation_formatter=conversation_formatter,
            memory_manager=Mock(),
        )
        general_generator.get_parameter_schema = Mock(
            side_effect=Exception("Should not extract GENERAL params in FACT tests")
        )
        general_generator.get_parameter_extraction_prompt = Mock(
            side_effect=Exception("Should not extract GENERAL params in FACT tests")
        )

        language_detector = LanguageDetector(ai_client=client, telemetry=self.telemetry)

        ai_router = AiRouter(
            ai_client=client,
            telemetry=self.telemetry,
            language_detector=language_detector,
            famous_generator=famous_generator,
            general_generator=general_generator,
            fact_handler=fact_handler,
        )

        guild_id = store.physics_guild_id
        physicist_ids = store.physicist_ids

        return FactTestContext(
            fact_handler=fact_handler,
            ai_router=ai_router,
            store=store,
            guild_id=guild_id,
            target_user_id=physicist_ids[self.target_user],
        )

    async def test_remember_fact_for_new_user(self):
        """Remembering a new fact should persist memory and return specific confirmation."""
        fact_content = "they like chocolate cake"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)

                response = await ctx.fact_handler._remember_fact(
                    ctx.guild_id, ctx.target_user_id, fact_content, "English"
                )

                saved_memory = await ctx.store.get_user_facts(ctx.guild_id, ctx.target_user_id)
                self.assertIsNotNone(saved_memory, "Memory should be saved to store")

                saved_memory_lower = saved_memory.lower()
                self.assertIn("chocolate", saved_memory_lower, "Memory should contain 'chocolate'")
                self.assertIn("cake", saved_memory_lower, "Memory should contain 'cake'")

                self.assertIsNotNone(response, "Should return confirmation message")
                self.assertGreater(len(response), 10, "Confirmation should be substantial")
                response_lower = response.lower()
                fact_terms = ["chocolate", "cake", "remember"]
                self.assertGreaterEqual(
                    sum(1 for term in fact_terms if term in response_lower),
                    2,
                    f"Confirmation should mention specific fact content. Got: {response}",
                )

    async def test_remember_additional_fact_merges_memory(self):
        """Adding a second fact should merge with existing memory."""
        existing_memory = "they like chocolate cake"
        new_fact = "they work at Google"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)
                ctx.store.set_user_facts(ctx.guild_id, ctx.target_user_id, existing_memory)

                response = await ctx.fact_handler._remember_fact(ctx.guild_id, ctx.target_user_id, new_fact, "English")

                updated_memory = await ctx.store.get_user_facts(ctx.guild_id, ctx.target_user_id)
                updated_memory_lower = updated_memory.lower()

                self.assertIn(
                    "chocolate",
                    updated_memory_lower,
                    "Should retain original chocolate fact",
                )
                self.assertIn(
                    "cake",
                    updated_memory_lower,
                    "Should retain original cake fact",
                )
                self.assertIn(
                    "google",
                    updated_memory_lower,
                    "Should contain new Google fact",
                )
                self.assertIn(
                    "work",
                    updated_memory_lower,
                    "Should contain work information",
                )

                self.assertIn(
                    "google",
                    response.lower(),
                    f"Confirmation should mention Google. Got: {response}",
                )

    async def test_forget_existing_fact(self):
        """Forgetting an existing fact should remove it and confirm the removal."""
        existing_memory = "they like chocolate cake and they work at Google"
        fact_to_forget = "they work at Google"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)
                ctx.store.set_user_facts(ctx.guild_id, ctx.target_user_id, existing_memory)

                response = await ctx.fact_handler._forget_fact(
                    ctx.guild_id, ctx.target_user_id, fact_to_forget, "English"
                )

                updated_memory = await ctx.store.get_user_facts(ctx.guild_id, ctx.target_user_id)
                updated_memory_lower = updated_memory.lower()

                self.assertNotIn(
                    "google",
                    updated_memory_lower,
                    "Should have removed Google fact",
                )
                self.assertNotIn(
                    "work",
                    updated_memory_lower,
                    "Should have removed work information",
                )
                self.assertIn(
                    "chocolate",
                    updated_memory_lower,
                    "Should retain chocolate fact",
                )
                self.assertIn(
                    "cake",
                    updated_memory_lower,
                    "Should retain cake fact",
                )

                response_lower = response.lower()
                forgotten_terms = ["forgotten", "google", "work"]
                self.assertGreaterEqual(
                    sum(1 for term in forgotten_terms if term in response_lower),
                    2,
                    f"Confirmation should mention what was forgotten. Got: {response}",
                )

    async def test_forget_nonexistent_fact(self):
        """Forgetting a non-existent fact should leave memory untouched and explain why."""
        existing_memory = "they like chocolate cake"
        nonexistent_fact = "they dislike pizza"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)
                ctx.store.set_user_facts(ctx.guild_id, ctx.target_user_id, existing_memory)

                response = await ctx.fact_handler._forget_fact(
                    ctx.guild_id, ctx.target_user_id, nonexistent_fact, "English"
                )

                unchanged_memory = await ctx.store.get_user_facts(ctx.guild_id, ctx.target_user_id)
                self.assertEqual(unchanged_memory, existing_memory, "Memory should remain unchanged")

                response_lower = response.lower()
                not_found_terms = ["not found", "couldn't find", "don't have"]
                self.assertTrue(
                    any(term in response_lower for term in not_found_terms),
                    f"Confirmation should indicate fact not found. Got: {response}",
                )

    async def test_forget_fact_no_existing_memory(self):
        """Forgetting from empty memory should clarify that nothing is stored."""
        fact_to_forget = "they like pizza"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)

                response = await ctx.fact_handler._forget_fact(
                    ctx.guild_id, ctx.target_user_id, fact_to_forget, "English"
                )

                memory = await ctx.store.get_user_facts(ctx.guild_id, ctx.target_user_id)
                self.assertIsNone(memory, "No memory should exist for user")

                response_lower = response.lower()
                no_memory_terms = ["no memory", "don't have", "no information"]
                self.assertTrue(
                    any(term in response_lower for term in no_memory_terms),
                    f"Confirmation should indicate no memory exists. Got: {response}",
                )

    async def test_language_preservation_english_fact(self):
        """English facts should stay in English in both memory and confirmation."""
        english_fact = "they are a software engineer"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)

                response = await ctx.fact_handler._remember_fact(
                    ctx.guild_id, ctx.target_user_id, english_fact, "English"
                )

                saved_memory = await ctx.store.get_user_facts(ctx.guild_id, ctx.target_user_id)
                self.assertIn(
                    "engineer",
                    saved_memory.lower(),
                    "Memory should contain engineering fact",
                )

                self.assertIn(
                    "engineer",
                    response.lower(),
                    "Confirmation should mention engineer",
                )
                non_english_patterns = ["я ", "что ", "они "]
                self.assertFalse(
                    any(pattern in response.lower() for pattern in non_english_patterns),
                    f"English fact should generate English confirmation. Got: {response}",
                )

    async def test_confirmation_message_quality(self):
        """Confirmation messages should be specific, not generic boilerplate."""
        specific_fact = "they graduated from MIT in 2020"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)

                response = await ctx.fact_handler._remember_fact(
                    ctx.guild_id, ctx.target_user_id, specific_fact, "English"
                )

                response_lower = response.lower()
                specific_terms = ["mit", "graduated", "2020"]
                self.assertGreaterEqual(
                    sum(1 for term in specific_terms if term in response_lower),
                    2,
                    f"Confirmation should mention specific details. Got: {response}",
                )

                generic_phrases = [
                    "i'll remember that about",
                    "information stored",
                    "data saved",
                ]
                self.assertFalse(
                    any(phrase in response_lower for phrase in generic_phrases),
                    f"Confirmation should be specific, not generic. Got: {response}",
                )

                action_words = ["remember", "recall", "note"]
                self.assertTrue(
                    any(word in response_lower for word in action_words),
                    f"Confirmation should indicate the action taken. Got: {response}",
                )

    async def test_end_to_end_russian_language_preservation(self):
        """Russian input should stay in Russian through routing and storage."""
        russian_message = "БОТ запомни что Rutherford так же известен как Крокодил"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)

                route, params = await ctx.ai_router.route_request(russian_message)

                self.assertEqual(route, "FACT", "Should route Russian command to FACT")
                self.assertIsNotNone(params, "Should extract parameters for FACT route")
                self.assertEqual(params.operation, "remember")
                self.assertEqual(params.user_mention, "Rutherford")

                fact_content_lower = params.fact_content.lower()
                russian_terms = ["известен", "крокодил"]
                self.assertGreaterEqual(
                    sum(1 for term in russian_terms if term in fact_content_lower),
                    1,
                    f"Fact content should preserve Russian words. Got: {params.fact_content}",
                )
                english_translations = ["known as", "also known"]
                self.assertFalse(
                    any(term in fact_content_lower for term in english_translations),
                    f"Fact content should NOT be translated. Got: {params.fact_content}",
                )
                third_person_indicators = ["он", "они"]
                self.assertTrue(
                    any(pronoun in fact_content_lower for pronoun in third_person_indicators),
                    f"Should convert to third person in Russian. Got: {params.fact_content}",
                )

                response = await ctx.fact_handler.handle_request(params, ctx.guild_id)

                rutherford_id = ctx.store.physicist_ids["Rutherford"]
                saved_memory = await ctx.store.get_user_facts(ctx.guild_id, rutherford_id)
                self.assertIsNotNone(saved_memory, "Memory should be saved")
                self.assertIn(
                    "крокодил",
                    saved_memory.lower(),
                    "Saved memory should contain Russian nickname",
                )

                response_lower = response.lower()
                russian_confirmation_terms = ["крокодил", "запомн"]
                self.assertTrue(
                    any(term in response_lower for term in russian_confirmation_terms),
                    f"Confirmation should contain Russian elements. Got: {response}",
                )

    async def test_end_to_end_english_still_works(self):
        """English end-to-end flow should continue to behave correctly."""
        english_message = "BOT remember that Einstein likes chocolate"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                ctx = self._build_context(profile.client)

                route, params = await ctx.ai_router.route_request(english_message)

                self.assertEqual(route, "FACT")
                self.assertIsNotNone(params)
                self.assertEqual(params.operation, "remember")
                self.assertEqual(params.user_mention, "Einstein")
                self.assertIn("chocolate", params.fact_content.lower())
                self.assertIn("like", params.fact_content.lower())

                await ctx.fact_handler.handle_request(params, ctx.guild_id)

                saved_memory = await ctx.store.get_user_facts(ctx.guild_id, ctx.target_user_id)
                self.assertIsNotNone(saved_memory)
                self.assertIn("chocolate", saved_memory.lower())


if __name__ == "__main__":
    unittest.main()
