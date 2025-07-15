"""
Unit tests for the AttachmentProcessor.
"""

import unittest
from unittest.mock import Mock, AsyncMock, patch

from attachment_processor import AttachmentProcessor, AttachmentData
from ai_client import AIClient
from null_telemetry import NullTelemetry


class TestAttachmentProcessorUnit(unittest.IsolatedAsyncioTestCase):
    """Unit tests for AttachmentProcessor."""

    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()
        self.mock_ai_client = AsyncMock(spec=AIClient)
        self.mock_goose = Mock()
        self.attachment_processor = AttachmentProcessor(
            ai_client=self.mock_ai_client,
            telemetry=self.telemetry,
            max_file_size_mb=1,  # 1MB for easier testing
            goose=self.mock_goose
        )

    def create_mock_attachment(self, filename, content_type, size, url):
        """Helper to create a mock discord.Attachment."""
        attachment = Mock()
        attachment.filename = filename
        attachment.content_type = content_type
        attachment.size = size
        attachment.url = url
        return attachment

    def test_initialization(self):
        """Test that max_file_size_bytes is correctly calculated."""
        self.assertEqual(self.attachment_processor.max_file_size_bytes, 1 * 1024 * 1024)
        # Test with a different value
        processor_5mb = AttachmentProcessor(
            self.mock_ai_client, self.telemetry, max_file_size_mb=5
        )
        self.assertEqual(processor_5mb.max_file_size_bytes, 5 * 1024 * 1024)

    def test_is_supported_image(self):
        """Test the is_supported_image method with various scenarios using subtests."""
        scenarios = [
            ("Supported type, size within limit", "image/png", 500 * 1024, True),
            (
                "Supported type, size at exact limit",
                "image/jpeg",
                1 * 1024 * 1024,
                True,
            ),
            (
                "Supported type, size over limit",
                "image/gif",
                1 * 1024 * 1024 + 1,
                False,
            ),
            (
                "Unsupported type, size within limit",
                "application/pdf",
                500 * 1024,
                False,
            ),
        ]

        for name, content_type, size, expected in scenarios:
            with self.subTest(msg=name):
                attachment = self.create_mock_attachment(
                    "testfile", content_type, size, "url"
                )
                self.assertEqual(
                    self.attachment_processor._is_supported_image(attachment), expected
                )

    @patch("aiohttp.ClientSession.get")
    async def test_download_attachment_success(self, mock_get):
        """Test successful download of a supported attachment."""
        # Arrange
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = b"test_image_data"
        mock_get.return_value.__aenter__.return_value = mock_response

        attachment = self.create_mock_attachment(
            "test.png", "image/png", 500, "http://example.com/test.png"
        )

        # Act
        result = await self.attachment_processor._download_attachment(attachment)

        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(result.binary_data, b"test_image_data")
        self.assertEqual(result.mime_type, "image/png")
        self.assertEqual(result.filename, "test.png")

    @patch("aiohttp.ClientSession.get")
    async def test_download_attachment_http_error(self, mock_get):
        """Test download failure on HTTP error."""
        # Arrange
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_get.return_value.__aenter__.return_value = mock_response

        attachment = self.create_mock_attachment(
            "test.png", "image/png", 500, "http://example.com/notfound.png"
        )

        # Act
        result = await self.attachment_processor._download_attachment(attachment)

        # Assert
        self.assertIsNone(result)

    async def test_download_attachment_unsupported(self):
        """Test that unsupported attachments are not downloaded."""
        attachment = self.create_mock_attachment(
            "test.pdf", "application/pdf", 500, "http://example.com/test.pdf"
        )

        with patch.object(
            self.attachment_processor, "_download_from_url", new_callable=AsyncMock
        ) as mock_download:
            result = await self.attachment_processor._download_attachment(attachment)
            self.assertIsNone(result)
            mock_download.assert_not_called()

    async def test_analyze_image_success(self):
        """Test successful image analysis."""
        # Arrange
        attachment_data = AttachmentData(b"data", "image/png", "test.png", 4)
        self.mock_ai_client.generate_content.return_value = (
            "A detailed description of the image."
        )

        # Act
        description = await self.attachment_processor._analyze_image(attachment_data)

        # Assert
        self.assertEqual(description, "A detailed description of the image.")
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_analyze_image_failure(self):
        """Test that exceptions from the AI client are propagated."""
        # Arrange
        attachment_data = AttachmentData(b"data", "image/png", "test.png", 4)
        self.mock_ai_client.generate_content.side_effect = Exception("AI client failed")

        # Act & Assert
        with self.assertRaises(Exception) as context:
            await self.attachment_processor._analyze_image(attachment_data)
        self.assertTrue("AI client failed" in str(context.exception))

    def test_extract_article_success(self):
        """Test successful article extraction."""
        # Arrange
        mock_article = Mock()
        mock_article.cleaned_text = "This is the article content."
        self.mock_goose.extract.return_value = mock_article

        url = "http://example.com/article"

        # Act
        content = self.attachment_processor._extract_article(url)

        # Assert
        self.assertEqual(content, "This is the article content.")
        self.mock_goose.extract.assert_called_once_with(url=url)

    def test_extract_article_failure(self):
        """Test that an empty string is returned on extraction failure."""
        # Arrange
        self.mock_goose.extract.side_effect = Exception("Goose failed")
        url = "http://example.com/failing-article"

        # Act
        content = self.attachment_processor._extract_article(url)

        # Assert
        self.assertEqual(content, "")

    def test_extract_article_caching(self):
        """Test that article extraction is cached."""
        # Arrange
        mock_article = Mock()
        mock_article.cleaned_text = "Article content"
        self.mock_goose.extract.return_value = mock_article
        url = "http://example.com/cached-article"

        # Act
        result1 = self.attachment_processor._extract_article(url)
        result2 = self.attachment_processor._extract_article(url)

        # Assert
        self.assertEqual(result1, "Article content")
        self.assertEqual(result2, "Article content")
        # goose.extract should only be called once due to caching
        self.mock_goose.extract.assert_called_once_with(url=url)

    async def test_process_all_content_mixed(self):
        """Test processing a mix of attachments and embeds."""
        # Arrange
        # Mock attachment processing
        mock_attachment = self.create_mock_attachment(
            "test.png", "image/png", 500, "http://example.com/test.png"
        )
        with patch.object(
            self.attachment_processor, "_download_attachment", new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = AttachmentData(
                b"data", "image/png", "test.png", 4
            )

            with patch.object(
                self.attachment_processor, "_analyze_image", new_callable=AsyncMock
            ) as mock_analyze:
                mock_analyze.return_value = "Image description"

                # Mock embed processing
                mock_embed = Mock()
                mock_embed.url = "http://example.com/article"
                with patch.object(
                    self.attachment_processor, "_extract_article"
                ) as mock_extract:
                    mock_extract.return_value = "Article content"

                    # Act
                    result = await self.attachment_processor.process_all_content(
                        attachments=[mock_attachment], embeds=[mock_embed]
                    )

        # Assert
        expected_result = '<embeddings><embedding type="article" url="http://example.com/article">Article content</embedding><embedding type="image" filename="test.png">Image description</embedding></embeddings>'
        self.assertEqual(result, expected_result)

    async def test_process_all_content_empty(self):
        """Test processing with no attachments or embeds."""
        result = await self.attachment_processor.process_all_content([], [])
        self.assertEqual(result, "")

    async def test_processed_attachment_cache_hit_subsequent_calls(self):
        """Test that subsequent calls to process_attachments use cache."""
        # Arrange
        attachment = self.create_mock_attachment(
            "test.png", "image/png", 1024 * 600, "http://example.com/test.png"
        )

        with patch.object(
            self.attachment_processor, "_download_attachment", new_callable=AsyncMock
        ) as mock_download:
            mock_download.return_value = AttachmentData(
                b"test_image_data" * 150, "image/png", "test.png", 1350
            )

            with patch.object(
                self.attachment_processor, "_analyze_image", new_callable=AsyncMock
            ) as mock_analyze:
                mock_analyze.return_value = "Detailed image analysis"

                # Act - First call
                result1 = await self.attachment_processor._process_attachments(
                    [attachment]
                )

                # Act - Second call (should use cache)
                result2 = await self.attachment_processor._process_attachments(
                    [attachment]
                )

                # Assert
                self.assertEqual(result1, result2)
                self.assertEqual(len(result1), 1)
                self.assertEqual(len(result2), 1)

                # Verify methods were only called once (first call)
                mock_download.assert_called_once_with(attachment)
                mock_analyze.assert_called_once()

    async def test_processed_attachment_cache_failed_operations_not_cached(self):
        """Test that failed operations do not populate cache."""
        # Arrange
        attachment = self.create_mock_attachment(
            "test.png", "image/png", 1024 * 700, "http://example.com/test.png"
        )

        with patch.object(
            self.attachment_processor, "_download_attachment", new_callable=AsyncMock
        ) as mock_download:
            # First call fails
            mock_download.side_effect = Exception("Download failed")

            # Act - First call should fail
            result1 = await self.attachment_processor._process_attachments([attachment])

            # Assert - No results due to failure
            self.assertEqual(len(result1), 0)

            # Arrange - Second call succeeds
            mock_download.side_effect = None
            mock_download.return_value = AttachmentData(
                b"recovered_image" * 180, "image/png", "test.png", 1620
            )

            with patch.object(
                self.attachment_processor, "_analyze_image", new_callable=AsyncMock
            ) as mock_analyze:
                mock_analyze.return_value = "Recovery analysis"

                # Act - Second call should succeed and not use cache
                result2 = await self.attachment_processor._process_attachments(
                    [attachment]
                )

                # Assert
                self.assertEqual(len(result2), 1)
                self.assertIn("Recovery analysis", result2[0])

                # Verify download was called again (not cached)
                self.assertEqual(mock_download.call_count, 2)
                mock_analyze.assert_called_once()


if __name__ == "__main__":
    unittest.main()
