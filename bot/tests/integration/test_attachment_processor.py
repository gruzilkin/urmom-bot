"""
Integration tests for the AttachmentProcessor.

These tests verify that the AttachmentProcessor can correctly download and process
real-world image and article content from URLs.
"""

import os
import unittest
from unittest.mock import Mock
from dotenv import load_dotenv

from attachment_processor import AttachmentProcessor
from gemma_client import GemmaClient
from tests.null_telemetry import NullTelemetry

load_dotenv()


class TestAttachmentProcessorIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for AttachmentProcessor using live URLs."""

    def setUp(self):
        """Set up test dependencies, including a real AI client."""
        self.telemetry = NullTelemetry()

        # API Keys and Models
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        gemma_model = os.getenv('GEMINI_GEMMA_MODEL')

        if not all([gemini_api_key, gemma_model]):
            self.skipTest("Missing GEMINI_API_KEY or GEMINI_GEMMA_MODEL environment variables.")

        # The processor requires a real AI client for image analysis
        self.gemma_client = GemmaClient(api_key=gemini_api_key, model_name=gemma_model, telemetry=self.telemetry)

        # The component under test
        self.attachment_processor = AttachmentProcessor(
            ai_client=self.gemma_client,
            telemetry=self.telemetry,
            max_file_size_mb=20  # Increase size for real-world images
        )

    async def test_process_image_attachment_from_url(self):
        """Test processing a real image from a URL."""
        # Arrange
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2018_Shibuya_Crossing.jpg/1024px-2018_Shibuya_Crossing.jpg"
        # Create a mock attachment that points to the real URL
        mock_attachment = Mock()
        mock_attachment.filename = "2018_Shibuya_Crossing.jpg"
        mock_attachment.content_type = "image/jpeg"
        mock_attachment.size = 5 * 1024 * 1024  # Assume a plausible size within limits
        mock_attachment.url = image_url

        # Act
        embeddings = await self.attachment_processor.process_attachments([mock_attachment])

        # Assert
        self.assertEqual(len(embeddings), 1)
        embedding_text = embeddings[0]
        self.assertTrue(embedding_text.startswith('<embedding type="image"'))
        self.assertIn('filename="2018_Shibuya_Crossing.jpg"', embedding_text)
        self.assertTrue(embedding_text.endswith('</embedding>'))

        # Check for plausible content in the AI-generated description
        description = embedding_text.split('>', 1)[1].rsplit('<', 1)[0].lower()
        self.assertGreater(len(description), 100, "Description should be substantial.")

    def test_process_article_embed_from_url(self):
        """Test processing a real article from a URL."""
        # Arrange
        article_url = "https://en.wikipedia.org/wiki/Tokyo"
        # Create a mock embed that points to the real URL
        mock_embed = Mock()
        mock_embed.url = article_url

        # Act
        embeddings = self.attachment_processor.process_embeds([mock_embed])

        # Assert
        self.assertEqual(len(embeddings), 1)
        embedding_text = embeddings[0]
        self.assertTrue(embedding_text.startswith('<embedding type="article"'))
        self.assertIn(f'url="{article_url}"', embedding_text)
        self.assertTrue(embedding_text.endswith('</embedding>'))

        # Check for plausible content in the extracted article
        article_content = embedding_text.split('>', 1)[1].rsplit('<', 1)[0].lower()
        self.assertGreater(len(article_content), 1000, "Article content should be substantial.")
        keywords = ["tokyo", "capital", "japan", "emperor", "government", "population"]
        found_keywords = [kw for kw in keywords if kw in article_content]
        self.assertGreaterEqual(len(found_keywords), 4, f"Expected to find at least 4 keywords, but found: {found_keywords}")


if __name__ == '__main__':
    unittest.main()
