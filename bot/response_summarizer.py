"""
Response Summarizer for handling long AI responses.

This module provides a reusable utility for summarizing AI responses that exceed
Discord's 2000 character limit, with fallback to truncation if summarization fails.
"""

import logging
from ai_client import AIClient
from open_telemetry import Telemetry
from opentelemetry.trace import SpanKind

logger = logging.getLogger(__name__)


class ResponseSummarizer:
    """Handles summarization of long responses using gemma as a fallback."""
    
    def __init__(self, gemma_client: AIClient, telemetry: Telemetry):
        """
        Initialize the response summarizer.
        
        Args:
            gemma_client: The gemma AI client for summarization
            telemetry: Telemetry instance for tracking metrics
        """
        self.gemma_client = gemma_client
        self.telemetry = telemetry
        self.summarization_temperature = 0.1  # Low temperature for focused summarization
    
    async def process_response(self, original_response: str, max_length: int = 2000) -> str:
        """
        Process a response, summarizing if too long, or truncating as fallback.
        
        Args:
            original_response: The original AI response
            max_length: Maximum allowed length (default: 2000 for Discord)
            
        Returns:
            str: The processed response (original, summarized, or truncated)
        """
        if len(original_response) <= max_length:
            return original_response
        
        # Calculate target length as 90% of max length, rounded down to nearest 100 for clarity
        target_length = int(max_length * 0.9 // 100) * 100
        
        logger.info(f"Response length {len(original_response)} exceeds limit of {max_length}, attempting summarization")
        
        async with self.telemetry.async_create_span("summarize_long_response", kind=SpanKind.CLIENT) as span:
            span.set_attribute("original_length", len(original_response))
            span.set_attribute("max_length", max_length)
            
            try:
                # Attempt summarization using gemma
                summarized_response = await self._summarize_with_gemma(original_response, target_length)
                
                if len(summarized_response) <= max_length:
                    logger.info(f"Summarization successful: {len(original_response)} → {len(summarized_response)} characters")
                    span.set_attribute("final_length", len(summarized_response))
                    return summarized_response
                else:
                    logger.warning(f"Summarization still too long: {len(summarized_response)} characters, falling back to truncation")
                    
            except Exception as e:
                logger.error(f"Summarization failed with error: {e}", exc_info=True)
            
            # Fallback to truncation
            truncated_response = self._truncate_response(original_response, max_length)
            logger.info(f"Falling back to truncation: {len(original_response)} → {len(truncated_response)} characters")
            span.set_attribute("final_length", len(truncated_response))
            return truncated_response
    
    async def _summarize_with_gemma(self, original_response: str, target_length: int) -> str:
        """
        Summarize the response using gemma client.
        
        Args:
            original_response: The response to summarize
            target_length: Target length for the summary
            
        Returns:
            str: The summarized response
            
        Raises:
            Exception: If summarization fails
        """
        prompt = f"""Summarize the following response to approximately {target_length} characters while preserving all key information, main points, and the original tone. 
        
The summary should be comprehensive and maintain the same style as the original response. Aim for close to {target_length} characters - use the full space available to provide a detailed summary. Do not add any meta-commentary about the summarization process.

Original response to summarize:
{original_response}"""

        logger.info("Attempting summarization with gemma")
        
        summarized = await self.gemma_client.generate_content(
            message="Please summarize the response provided in the system prompt.",
            prompt=prompt,
            temperature=self.summarization_temperature
        )
        
        result = summarized.strip()
        
        # Check for empty or whitespace-only result to ensure fallback truncation
        if not result:
            raise ValueError("Gemma returned empty summary")
            
        return result
    
    def _truncate_response(self, response: str, max_length: int) -> str:
        """
        Truncate response to fit within the maximum length.
        
        Args:
            response: The response to truncate
            max_length: Maximum allowed length
            
        Returns:
            str: The truncated response with "..." suffix
        """
        if len(response) <= max_length:
            return response
        
        return response[:max_length - 3] + "..."