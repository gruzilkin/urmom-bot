"""
Integration tests for ResponseSummarizer with real Gemma client.

Tests the actual summarization behavior with the Gemma API to verify
the integration works correctly in practice.
"""

import os
import unittest
from dotenv import load_dotenv
from gemma_client import GemmaClient
from response_summarizer import ResponseSummarizer
from null_telemetry import NullTelemetry

load_dotenv()


class TestResponseSummarizerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for ResponseSummarizer with real Gemma client."""
    
    def setUp(self):
        """Set up test dependencies."""
        self.telemetry = NullTelemetry()
        
        # Check for API key and model name
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_GEMMA_MODEL')
        
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        if not self.model_name:
            self.skipTest("GEMINI_GEMMA_MODEL environment variable not set")
            
        self.gemma_client = GemmaClient(
            api_key=self.api_key,
            model_name=self.model_name,
            telemetry=self.telemetry,
            temperature=0.1  # Fixed temperature for test stability
        )
        
        self.summarizer = ResponseSummarizer(self.gemma_client, self.telemetry)
    
    async def test_short_response_no_summarization(self):
        """Test that short responses pass through without API call."""
        short_response = "This is a short response that doesn't need summarization."
        
        result = await self.summarizer.process_response(short_response, max_length=2000)
        
        # Should return the original response unchanged
        self.assertEqual(result, short_response)
    
    async def test_long_response_gets_summarized(self):
        """Test that long responses are actually summarized by Gemma."""
        # Create a realistic long response about software development practices (~5000 chars)
        long_response = """
        Software development has evolved dramatically over the past few decades, transforming from a largely individual pursuit to a highly collaborative, methodical discipline that powers virtually every aspect of modern life. The journey from early programming practices to today's sophisticated development methodologies represents one of the most significant technological advances of our time.

        In the early days of computing, software development was often an ad-hoc process where individual programmers would write code with minimal documentation, limited testing, and virtually no standardized practices. Programs were typically small, single-purpose applications that could be understood and maintained by their original authors. However, as software systems grew in complexity and organizations began to rely more heavily on technology, it became clear that more structured approaches were needed.

        The introduction of structured programming in the 1960s and 1970s marked the first major shift toward disciplined software development. This approach emphasized the use of clear control structures, modular design, and systematic code organization. Pioneers like Edsger Dijkstra advocated for programming practices that would make code more readable, maintainable, and less prone to errors. The concept of "goto considered harmful" became a rallying cry for better programming practices.

        Object-oriented programming emerged in the 1980s as another revolutionary approach, introducing concepts like encapsulation, inheritance, and polymorphism. Languages like C++ and later Java popularized these concepts, making it possible to create more complex, reusable software components. This paradigm shift enabled developers to model real-world problems more naturally and create more maintainable codebases.

        The 1990s brought the rise of the internet and web development, which introduced new challenges and opportunities. Suddenly, software needed to be accessible from anywhere in the world, handle concurrent users, and integrate with diverse systems. This era saw the emergence of new programming languages, frameworks, and architectural patterns specifically designed for distributed computing.

        Agile methodologies revolutionized software development in the early 2000s, emphasizing iterative development, customer collaboration, and adaptability over rigid planning and documentation. The Agile Manifesto, published in 2001, articulated principles that prioritized working software, individual interactions, and responding to change. This approach proved particularly effective in rapidly changing technological landscapes where requirements often evolved during development.

        DevOps culture emerged as a natural evolution of agile practices, breaking down traditional silos between development and operations teams. This movement emphasized automation, continuous integration, continuous deployment, and shared responsibility for software quality and reliability. Tools like Jenkins, Docker, and Kubernetes became essential components of modern development workflows.

        Today's software development landscape is characterized by cloud computing, microservices architectures, artificial intelligence integration, and an unprecedented focus on security and privacy. Modern developers must be proficient not only in programming languages and frameworks but also in understanding distributed systems, cybersecurity principles, data protection regulations, and user experience design.

        The rise of open-source software has democratized access to high-quality tools and libraries, enabling developers worldwide to collaborate on projects that would have been impossible for individual organizations to create. Platforms like GitHub have transformed how developers share code, collaborate on projects, and learn from each other.

        Looking forward, emerging technologies like quantum computing, edge computing, and advanced AI systems promise to create new paradigms for software development. The industry continues to evolve rapidly, with new languages, frameworks, and methodologies constantly emerging to address the changing needs of users and organizations.

        Despite all these changes, the fundamental principles of good software development remain constant: write clear, maintainable code; test thoroughly; collaborate effectively; and always keep the end user's needs at the center of the development process. These timeless principles provide stability in an ever-changing technological landscape.
        """
        
        result = await self.summarizer.process_response(long_response, max_length=2000)
        
        # Verify the result is shorter than the original
        self.assertLess(len(result), len(long_response))
        # Verify it fits within the limit
        self.assertLessEqual(len(result), 2000)
        # Verify it contains key concepts from different parts of the text
        self.assertTrue(any(term in result for term in ["software development", "programming", "agile", "development"]))
        # Verify it's a meaningful summary, not just the start of the original
        self.assertNotEqual(result, long_response[:len(result)])
    


if __name__ == '__main__':
    unittest.main()