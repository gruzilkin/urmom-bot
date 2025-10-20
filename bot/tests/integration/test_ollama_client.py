"""
Integration tests for OllamaClient that exercise the AIClient contract against a
shared manifest of Ollama cloud models.

All tests honour the capabilities declared on each model profile to avoid
copy/pasting suites per model while still verifying shared behaviour.
"""

import base64
import os
from dataclasses import dataclass
from unittest import IsolatedAsyncioTestCase

from dotenv import load_dotenv
from ollama_client import OllamaClient
from null_telemetry import NullTelemetry
from pydantic import BaseModel

from schemas import YesNo

load_dotenv()


@dataclass(frozen=True)
class ModelProfile:
    """Declarative manifest describing a model's supported capabilities."""

    slug: str
    supports_text: bool = True
    supports_structured: bool = True
    supports_tools: bool = False
    supports_vision: bool = False


DEFAULT_MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile("kimi-k2:1t-cloud"),
    ModelProfile("deepseek-v3.1:671b-cloud", supports_tools=True),
    ModelProfile("qwen3-coder:480b-cloud"),
    ModelProfile(
        "qwen3-vl:235b-cloud",
        supports_text=False,
        supports_structured=False,
        supports_vision=True,
    ),
    ModelProfile("gpt-oss:120b-cloud", supports_tools=True),
)

class CurrentEvent(BaseModel):
    """Structured response schema for web search tests."""

    topic: str
    date: str
    summary: str


class TestOllamaClientIntegration(IsolatedAsyncioTestCase):
    """Integration tests for Ollama cloud client behaviour."""

    async def asyncSetUp(self) -> None:
        self.api_key = os.getenv("OLLAMA_API_KEY")
        if not self.api_key:
            self.skipTest("OLLAMA_API_KEY not set, skipping integration tests")

        self.telemetry = NullTelemetry()
        self.models = list(DEFAULT_MODEL_PROFILES)
        if not self.models:
            self.skipTest("No Ollama models configured for integration tests")

    def _client_for(self, profile: ModelProfile, **overrides) -> OllamaClient:
        """Instantiate OllamaClient using shared telemetry and credentials."""
        params = {
            "api_key": self.api_key,
            "model_name": profile.slug,
            "telemetry": self.telemetry,
        }
        params.update(overrides)
        return OllamaClient(**params)

    async def test_basic_generation(self):
        """Plain text responses should work across all configured models."""

        text_models = [profile for profile in self.models if profile.supports_text]
        if not text_models:
            self.skipTest("No text-capable models configured")

        for profile in text_models:
            with self.subTest(model=profile.slug):
                client = self._client_for(profile)
                response = await client.generate_content(
                    message="What is 2 + 2?",
                    prompt="Answer concisely.",
                )
                self.assertIsInstance(response, str)
                self.assertIn("4", response)

    async def test_structured_output(self):
        """Models marked for structured output must obey the schema contract."""

        structured_models = [
            profile
            for profile in self.models
            if profile.supports_text and profile.supports_structured
        ]
        if not structured_models:
            self.skipTest("No structured-output models configured")

        for profile in structured_models:
            with self.subTest(model=profile.slug):
                client = self._client_for(profile, temperature=0.0)
                response = await client.generate_content(
                    message="Is Paris the capital of France?",
                    prompt="You are a helpful assistant.",
                    response_schema=YesNo,
                )
                self.assertIsInstance(response, YesNo)
                self.assertEqual(response.answer, "YES")

    async def test_web_search_grounding(self):
        """Models that support tools should handle grounding via web search."""

        tool_models = [
            profile
            for profile in self.models
            if profile.supports_text and profile.supports_tools
        ]
        if not tool_models:
            self.skipTest("No models configured with tool support")

        for profile in tool_models:
            with self.subTest(model=profile.slug):
                client = self._client_for(profile)
                response = await client.generate_content(
                    message="What are the latest developments in AI in 2025?",
                    prompt=(
                        "You are a helpful assistant. Use web search to provide current information."
                    ),
                    enable_grounding=True,
                )
                self.assertIsInstance(response, str)
                self.assertTrue(
                    "2025" in response or "recent" in response.lower(),
                    "Response should reference a current time frame.",
                )

    async def test_web_search_with_structured_output(self):
        """Tool-capable models should combine structured output with grounding."""

        tool_structured_models = [
            profile
            for profile in self.models
            if profile.supports_text
            and profile.supports_tools
            and profile.supports_structured
        ]
        if not tool_structured_models:
            self.skipTest(
                "No models configured with both tool and structured output support"
            )

        for profile in tool_structured_models:
            with self.subTest(model=profile.slug):
                client = self._client_for(profile)
                response = await client.generate_content(
                    message="Find the latest major tech event or announcement in 2025.",
                    prompt=(
                        "Use web search to find current information and extract it as structured data."
                    ),
                    enable_grounding=True,
                    response_schema=CurrentEvent,
                )
                self.assertIsInstance(response, CurrentEvent)
                self.assertGreater(len(response.topic), 0)
                self.assertIn("2025", response.date)

    async def test_image_to_text(self):
        """Vision-capable models should return a text description for images."""

        vision_models = [profile for profile in self.models if profile.supports_vision]
        if not vision_models:
            self.skipTest("No models configured with vision support")

        red_pixel_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

        for profile in vision_models:
            with self.subTest(model=profile.slug):
                client = self._client_for(profile)
                response = await client.generate_content(
                    message="What color is this image?",
                    prompt="You are a helpful assistant. Analyze the image and describe what you see.",
                    image_data=red_pixel_png,
                    image_mime_type="image/png",
                )
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
                self.assertIn(
                    "red",
                    response.lower(),
                    "Vision model should identify the red pixel.",
                )

    async def test_few_shot_learning(self):
        """Few-shot examples should steer responses across text models."""

        text_models = [profile for profile in self.models if profile.supports_text]
        if not text_models:
            self.skipTest("No text-capable models configured")

        samples = [
            ("Secret challenge: alpha?", "Access code: XR-17."),
            ("Secret challenge: beta?", "Access code: QZ-58."),
        ]

        for profile in text_models:
            with self.subTest(model=profile.slug):
                client = self._client_for(profile)
                response = await client.generate_content(
                    message="Secret challenge: alpha?",
                    prompt=(
                        "You are the keeper of access codes. Use the examples above to respond. "
                        "If a challenge is unknown, reply with 'ACCESS DENIED'."
                    ),
                    samples=samples,
                )
                self.assertIsInstance(response, str)
                self.assertEqual("Access code: XR-17.", response.strip())
