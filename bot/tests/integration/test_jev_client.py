"""Integration tests for JevClient against the real TypeSafe AI API.

Covers the whole question surface in two requests: a Choice[Literal] with described criteria, a
Noul with outcome descriptions, and a Score, all answered in one call; then a Choice[str] whose
labels come from a plain list. Requires JEV_API_KEY in the environment or .env.
"""

import os
import unittest
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from jev_client import Choice, JevClient, Noul, Score, choice, noul, score
from null_telemetry import NullTelemetry

load_dotenv()


class SupportTriage(BaseModel):
    category: Choice[Literal["billing", "technical", "other"]] = choice(
        instructions="What is this support message about?",
        criteria={
            "billing": "Charges, refunds, invoices, payment methods",
            "technical": "Errors, outages, bugs, things not working",
            "other": "Anything else",
        },
    )
    is_urgent: Noul = noul(
        instructions="Does the message convey urgency?",
        true="The user needs help right now or is blocked",
        false="The user can wait or is just asking casually",
    )
    frustration: Score = score(
        instructions="How frustrated is the user?",
        levels=["calm", "mildly annoyed", "angry"],
    )


class LanguageGuess(BaseModel):
    language: Choice[str] = choice(
        instructions="Which language is the message written in? Answer with its ISO 639-1 code.",
        criteria=["en", "ru", "de", "fr", "ja"],
    )


@unittest.skipUnless(os.getenv("JEV_API_KEY"), "JEV_API_KEY not set")
class TestJevClientIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = JevClient(api_key=os.environ["JEV_API_KEY"], telemetry=NullTelemetry(), timeout_seconds=15.0)

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_choice_noul_and_score_in_one_request(self) -> None:
        result = await self.client.ask(
            state={"message": "I was charged twice this month and I need it fixed TODAY, this is ridiculous."},
            questions=SupportTriage,
        )

        self.assertIsInstance(result, SupportTriage)

        self.assertEqual(result.category.choice, "billing")
        self.assertEqual(set(result.category.probabilities), {"billing", "technical", "other"})
        self.assertAlmostEqual(sum(result.category.probabilities.values()), 1.0, places=1)
        self.assertGreater(result.category.top_probability, 0.5)
        self.assertTrue(0.0 <= result.category.confidence <= 1.0)

        self.assertGreater(result.is_urgent.probability, 0.5)

        self.assertEqual(result.frustration.legend, {0: "calm", 1: "mildly annoyed", 2: "angry"})
        self.assertEqual(set(result.frustration.probabilities), {0, 1, 2})
        self.assertGreater(result.frustration.score, 1.0)
        self.assertTrue(0.0 <= result.frustration.confidence <= 1.0)

    async def test_choice_with_plain_label_list(self) -> None:
        result = await self.client.ask(
            state={"message": "Подскажи, пожалуйста, как доехать до вокзала?"},
            questions=LanguageGuess,
        )

        self.assertEqual(result.language.choice, "ru")
        self.assertEqual(set(result.language.probabilities), {"en", "ru", "de", "fr", "ja"})


if __name__ == "__main__":
    unittest.main()
