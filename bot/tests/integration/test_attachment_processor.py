"""
Integration tests for the AttachmentProcessor.

These tests verify that the AttachmentProcessor can correctly download and process
real-world image and article content from URLs across multiple AI client profiles.
"""

import os
import unittest
from dataclasses import dataclass
from unittest.mock import Mock

from dotenv import load_dotenv

from attachment_processor import AttachmentProcessor
from gemma_client import GemmaClient
from null_telemetry import NullTelemetry
from ollama_client import OllamaClient

load_dotenv()


@dataclass(frozen=True)
class ProcessorProfile:
    """Container for attachment processor instance and metadata."""

    name: str
    processor: AttachmentProcessor


class TestAttachmentProcessorIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for AttachmentProcessor using live URLs."""

    async def asyncSetUp(self):
        """Configure processor profiles mirroring production model choices."""
        self.telemetry = NullTelemetry()
        self.profiles: list[ProcessorProfile] = []

        gemma_api_key = os.getenv("GEMMA_API_KEY")
        gemma_model = os.getenv("GEMMA_MODEL")

        if gemma_api_key and gemma_model:
            gemma_client = GemmaClient(
                api_key=gemma_api_key,
                model_name=gemma_model,
                telemetry=self.telemetry,
            )
            self.profiles.append(
                ProcessorProfile(
                    name="gemma",
                    processor=AttachmentProcessor(
                        ai_client=gemma_client,
                        telemetry=self.telemetry,
                        max_file_size_mb=20,
                    ),
                )
            )

        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        qwen_model = os.getenv("OLLAMA_QWEN_VL_MODEL", "qwen3-vl:235b-cloud")

        if ollama_api_key:
            qwen_client = OllamaClient(
                api_key=ollama_api_key,
                model_name=qwen_model,
                telemetry=self.telemetry,
                temperature=0.0,
            )
            self.profiles.append(
                ProcessorProfile(
                    name="ollama_qwen3_vl",
                    processor=AttachmentProcessor(
                        ai_client=qwen_client,
                        telemetry=self.telemetry,
                        max_file_size_mb=20,
                    ),
                )
            )

        if not self.profiles:
            self.skipTest(
                "No attachment processor profiles configured; ensure Gemma or Ollama credentials are set."
            )

    async def test_process_image_attachment_from_url(self):
        """Test processing a real image from a URL."""
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2018_Shibuya_Crossing.jpg/1024px-2018_Shibuya_Crossing.jpg"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                mock_attachment = Mock()
                mock_attachment.filename = "2018_Shibuya_Crossing.jpg"
                mock_attachment.content_type = "image/jpeg"
                mock_attachment.size = 5 * 1024 * 1024
                mock_attachment.url = image_url
                mock_attachment.id = f"{profile.name}_image"

                embeddings = await profile.processor._process_attachments(
                    [mock_attachment]
                )

                self.assertEqual(len(embeddings), 1)
                embedding_text = embeddings[0]
                self.assertTrue(embedding_text.startswith('<embedding type="image"'))
                self.assertIn('filename="2018_Shibuya_Crossing.jpg"', embedding_text)
                self.assertTrue(embedding_text.endswith("</embedding>"))

                description = embedding_text.split(">", 1)[1].rsplit("<", 1)[0].lower()
                self.assertGreater(
                    len(description), 100, "Description should be substantial."
                )

    def test_process_article_embed_from_url(self):
        """Test processing a real article from a URL."""
        article_url = "https://en.wikipedia.org/wiki/Tokyo"

        for profile in self.profiles:
            with self.subTest(profile=profile.name):
                mock_embed = Mock()
                mock_embed.url = article_url

                embeddings = profile.processor._process_embeds([mock_embed])

                self.assertEqual(len(embeddings), 1)
                embedding_text = embeddings[0]
                self.assertTrue(embedding_text.startswith('<embedding type="article"'))
                self.assertIn(f'url="{article_url}"', embedding_text)
                self.assertTrue(embedding_text.endswith("</embedding>"))

                article_content = embedding_text.split(">", 1)[1].rsplit("<", 1)[0].lower()
                self.assertGreater(
                    len(article_content), 1000, "Article content should be substantial."
                )
                keywords = [
                    "tokyo",
                    "capital",
                    "japan",
                    "emperor",
                    "government",
                    "population",
                ]
                found_keywords = [kw for kw in keywords if kw in article_content]
                self.assertGreaterEqual(
                    len(found_keywords),
                    4,
                    f"Expected to find at least 4 keywords, but found: {found_keywords}",
                )


if __name__ == "__main__":
    unittest.main()
