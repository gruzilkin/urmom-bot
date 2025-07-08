"""
Attachment Processor for handling Discord message attachments.

This module provides image analysis and embedding generation for Discord attachments,
extracting attachment processing logic from the main app flow into a focused component.
"""

import logging
from dataclasses import dataclass

import aiohttp
import nextcord
from goose3 import Goose
from opentelemetry.trace import SpanKind, Status, StatusCode
from cachetools import LRUCache

from ai_client import AIClient
from open_telemetry import Telemetry

logger = logging.getLogger(__name__)


@dataclass
class AttachmentData:
    """Represents processed attachment data."""
    binary_data: bytes
    mime_type: str
    filename: str
    size: int


class AttachmentProcessor:
    """Handles processing of Discord message attachments and embeds."""
    
    # Supported image MIME types
    SUPPORTED_IMAGE_TYPES = {
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'
    }
    
    def __init__(self, ai_client: AIClient, telemetry: Telemetry, max_file_size_mb: int = 10):
        """
        Initialize the attachment processor.
        
        Args:
            ai_client: AI client for image analysis (Gemma)
            telemetry: Telemetry instance for tracking metrics
            max_file_size_mb: Maximum file size in MB for processing
        """
        self.ai_client = ai_client
        self.telemetry = telemetry
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._download_cache = LRUCache(maxsize=128)
        self._article_cache = LRUCache(maxsize=128)
        self._processed_attachment_cache = LRUCache(maxsize=128)
        
    def is_supported_image(self, attachment: nextcord.Attachment) -> bool:
        """
        Check if attachment is a supported image type.
        
        Args:
            attachment: Discord attachment object
            
        Returns:
            True if attachment is a supported image type
        """
        return (attachment.content_type in self.SUPPORTED_IMAGE_TYPES and 
                attachment.size <= self.max_file_size_bytes)
    
    async def _download_from_url(self, url: str) -> bytes | None:
        """Download binary data from URL with caching."""
        if url in self._download_cache:
            return self._download_cache[url]

        async with self.telemetry.async_create_span("download_from_url", kind=SpanKind.CLIENT) as span:
            span.set_attribute("url", url)
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status != 200:
                            span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status}"))
                            return None
                        
                        binary_data = await response.read()
                        span.set_attribute("downloaded_size", len(binary_data))
                        self._download_cache[url] = binary_data
                        return binary_data
                        
            except Exception as e:
                logger.error(f"Error downloading from URL {url}: {e}", exc_info=True)
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                return None
    
    async def download_attachment(self, attachment: nextcord.Attachment) -> AttachmentData | None:
        """
        Download and validate a single attachment.
        
        Args:
            attachment: Discord attachment object
            
        Returns:
            AttachmentData object or None if download failed
        """
        if not self.is_supported_image(attachment):
            return None
            
        binary_data = await self._download_from_url(attachment.url)
        if not binary_data:
            return None
            
        return AttachmentData(
            binary_data=binary_data,
            mime_type=attachment.content_type,
            filename=attachment.filename,
            size=len(binary_data)
        )
    
    async def analyze_image(self, attachment_data: AttachmentData) -> str:
        """
        Generate AI description for image data.
        
        Args:
            attachment_data: AttachmentData object with image information
            
        Returns:
            AI-generated description of the image
        """
        async with self.telemetry.async_create_span("analyze_image", kind=SpanKind.CLIENT) as span:
            span.set_attribute("filename", attachment_data.filename)
            span.set_attribute("mime_type", attachment_data.mime_type)
            span.set_attribute("size", attachment_data.size)
            
            try:
                description = await self.ai_client.generate_content(
                    message="Provide an extremely detailed description of this image. Extract and transcribe ALL visible text exactly as it appears, including any writing, signs, labels, captions, usernames, timestamps, UI elements, or any other textual content. Then provide a comprehensive visual description of every element in the image - objects, people, colors, positioning, background, lighting, emotions, actions, and spatial relationships. Be so thorough that someone who cannot see the image would have a complete understanding of what is depicted.",
                    prompt="You are an expert image analyst who extracts every detail from images. Your primary task is to capture ALL textual content verbatim, then provide exhaustive visual descriptions. Never summarize or be concise - provide maximum detail for each element you observe.",
                    image_data=attachment_data.binary_data,
                    image_mime_type=attachment_data.mime_type
                )
                
                span.set_attribute("description_length", len(description))
                logger.info(f"Generated description for {attachment_data.filename}: {description}")
                
                return description
                
            except Exception as e:
                logger.error(f"Error analyzing image {attachment_data.filename}: {e}", exc_info=True)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    async def process_attachments(self, attachments: list[nextcord.Attachment]) -> list[str]:
        """
        Process Discord attachments and return embedding-formatted descriptions.
        
        Args:
            attachments: List of Discord attachments
            
        Returns:
            List of image description strings in embedding format
        """
        async with self.telemetry.async_create_span("process_attachments") as span:
            descriptions = []
            
            for attachment in attachments:
                cache_key = attachment.url
                if cache_key in self._processed_attachment_cache:
                    descriptions.append(self._processed_attachment_cache[cache_key])
                    continue

                try:
                    # Download attachment
                    attachment_data = await self.download_attachment(attachment)
                    if not attachment_data:
                        continue
                    
                    # Analyze image
                    description = await self.analyze_image(attachment_data)
                    
                    # Format as embedding similar to article embeddings
                    embedding_text = f'<embedding type="image" filename="{attachment.filename}">{description}</embedding>'
                    descriptions.append(embedding_text)
                    self._processed_attachment_cache[cache_key] = embedding_text
                    
                except Exception as e:
                    logger.error(f"Error processing attachment {attachment.filename}: {e}", exc_info=True)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    continue
            
            span.set_attribute("success_count", len(descriptions))
            
            return descriptions
    
    def extract_article(self, url: str) -> str:
        """Extract article content with LRU caching."""
        if url in self._article_cache:
            return self._article_cache[url]

        with self.telemetry.create_span("extract_article") as span:
            span.set_attribute("url", url)
            try:
                article = Goose().extract(url=url)
                content = article.cleaned_text if article.cleaned_text else ""
                span.set_attribute("content_length", len(content))
                if content:
                    logger.info(f"Extracted article content from {url}: {content[:500]}{'...' if len(content) > 500 else ''}")
                else:
                    logger.info(f"No content extracted from {url}")
                self._article_cache[url] = content
                return content
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.error(f"Error extracting article from {url}: {e}", exc_info=True)
                return ""
    
    def process_embeds(self, embeds: list[nextcord.Embed]) -> list[str]:
        """Process Discord embeds and return article embedding strings.
        
        Args:
            embeds: List of Discord embeds
            
        Returns:
            List of article embedding strings in XML format
        """
        with self.telemetry.create_span("process_embeds") as span:
            article_contents = []
            
            for embed in embeds:
                if hasattr(embed, 'url') and embed.url:
                    article_text = self.extract_article(embed.url)
                    if article_text:
                        embedding_text = f'<embedding type="article" url="{embed.url}">{article_text}</embedding>'
                        article_contents.append(embedding_text)
            
            span.set_attribute("success_count", len(article_contents))
            
            return article_contents
    
    async def process_all_content(self, attachments: list[nextcord.Attachment], embeds: list[nextcord.Embed]) -> str:
        """Process both attachments and embeds, returning wrapped XML embeddings.
        
        Args:
            attachments: List of Discord attachments
            embeds: List of Discord embeds
            
        Returns:
            XML-wrapped embeddings string, or empty string if no content processed
        """
        async with self.telemetry.async_create_span("process_all_content") as span:
            all_embeddings = []
            
            # Process article embeds
            if embeds:
                article_embeddings = self.process_embeds(embeds)
                all_embeddings.extend(article_embeddings)
            
            # Process image attachments
            if attachments:
                image_embeddings = await self.process_attachments(attachments)
                all_embeddings.extend(image_embeddings)
            
            # Return wrapped XML or empty string
            if all_embeddings:
                result = f"<embeddings>{''.join(all_embeddings)}</embeddings>"
                span.set_attribute("result_length", len(result))
                return result
            else:
                return ""
