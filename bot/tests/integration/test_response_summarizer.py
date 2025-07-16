"""
Integration tests for ResponseSummarizer with real Gemma client.

Tests the actual summarization behavior with the Gemma API to verify
the integration works correctly in practice.
"""

import os
import re
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
    
    async def test_russian_language_preservation(self):
        """Test that Russian scientific terms are preserved when summarizing Russian text."""
        # Create a long Russian text about Einstein's work with key scientific terms
        russian_einstein_text = """
        Альберт Эйнштейн был одним из величайших физиков в истории человечества. Его работы революционизировали наше понимание пространства, времени и гравитации. Родившись в Ульме в 1879 году, Эйнштейн провел свои самые продуктивные годы в создании теорий, которые изменили науку навсегда.

        Специальная теория относительности, опубликованная в 1905 году, стала одним из самых важных достижений в физике. Эта теория установила, что скорость света в вакууме является постоянной для всех наблюдателей, независимо от их движения. Знаменитая формула E=mc² показывает эквивалентность массы и энергии, что имело огромные последствия для понимания ядерной физики.

        Фотоэлектрический эффект, за который Эйнштейн получил Нобелевскую премию в 1921 году, продемонстрировал квантовую природу света. Его работа показала, что свет состоит из дискретных пакетов энергии, называемых фотонами. Это открытие стало основой для развития квантовой механики, которая полностью изменила наше понимание атомного и субатомного мира.

        Общая теория относительности, представленная в 1915 году, описывает гравитацию не как силу, а как искривление пространства-времени массивными объектами. Эта теория предсказала существование черных дыр, гравитационных волн и расширение Вселенной. Многие из этих предсказаний были подтверждены только десятилетия спустя с помощью современных технологий.

        Работа Эйнштейна в области статистической механики и броуновского движения также внесла значительный вклад в понимание молекулярной структуры материи. Его теоретические исследования подтвердили существование атомов и молекул, что было спорным вопросом в начале XX века.

        Космологические исследования Эйнштейна привели к пониманию того, что Вселенная может расширяться или сжиматься. Хотя сначала он сопротивлялся этой идее, введя космологическую постоянную, позже он признал, что динамическая Вселенная является естественным следствием его теории.

        Влияние Эйнштейна на современную физику невозможно переоценить. Его работы заложили основы для квантовой теории поля, теории струн и современной космологии. Принцип относительности и квантовая механика стали двумя столпами современной физики, хотя попытки их объединения в единую теорию квантовой гравитации продолжаются и по сей день.
        """
        
        result = await self.summarizer.process_response(russian_einstein_text)
        
        # Verify the result is shorter than the original
        self.assertLess(len(result), len(russian_einstein_text))
        # Verify it fits within the limit
        self.assertLessEqual(len(result), 2000)
        
        # Verify key Russian scientific terms are preserved (using regex for inflected forms)
        
        key_russian_patterns = [
            (r"теор\w* относит", "теория относительности"),
            (r"квантов\w* механик", "квантовая механика"), 
            (r"фотоэлектрическ\w* эффект", "фотоэлектрический эффект"),
            (r"Нобелевск\w* преми", "Нобелевская премия")
        ]
        
        for pattern, term_name in key_russian_patterns:
            found = re.search(pattern, result)
            self.assertTrue(found, f"Russian term '{term_name}' (pattern: {pattern}) should be preserved in summary")
        
        # Verify it's a meaningful summary, not just the start of the original
        self.assertNotEqual(result, russian_einstein_text[:len(result)])
    
    async def test_multilingual_quotes_preservation(self):
        """Test that foreign language quotes and technical terms are preserved in multilingual text."""
        # Create a long multilingual text about Einstein with foreign quotes and technical terms
        multilingual_einstein_text = """
        Albert Einstein's revolutionary contributions to physics fundamentally changed our understanding of the universe. His work bridged classical and modern physics, establishing principles that continue to guide scientific research today. Born in Germany in 1879, Einstein's intellectual journey took him across Europe and eventually to the United States, where he spent his final years at Princeton.

        The development of quantum mechanics involved collaboration with many brilliant scientists. Werner Heisenberg, one of the founding fathers of quantum mechanics, once remarked: "Die Unschärferelation ist fundamental für unser Verstehen der Quantenwelt" (The uncertainty principle is fundamental to our understanding of the quantum world). This principle, along with Einstein's work on the photoelectric effect, laid the groundwork for quantum theory.

        Einstein's special theory of relativity, known in Japanese as 相対性理論 (sōtaisei riron), revolutionized our understanding of space and time. The theory established that the speed of light is constant for all observers, regardless of their motion. This concept, along with 量子力学 (ryōshi rikigaku, quantum mechanics), forms the foundation of modern physics.

        The Russian physicist Lev Landau, who made significant contributions to theoretical physics, once observed: "Эйнштейн изменил наше понимание пространства и времени навсегда" (Einstein changed our understanding of space and time forever). Landau's work on phase transitions and condensed matter physics complemented Einstein's broader theoretical framework.

        French mathematician Henri Poincaré, who made important contributions to the mathematical foundations of relativity, stated: "La relativité n'est qu'une convention, mais c'est une convention très commode" (Relativity is just a convention, but it is a very convenient convention). Poincaré's insights into the mathematical structure of spacetime helped establish the formal framework for Einstein's theories.

        General relativity, Einstein's theory of gravitation, describes gravity not as a force but as the curvature of spacetime caused by mass and energy. This theory predicted the existence of black holes, gravitational waves, and the expansion of the universe. In Japanese physics terminology, 重力波 (jūryokuha, gravitational waves) were finally detected in 2015, confirming another of Einstein's predictions.

        Einstein's work on Brownian motion and statistical mechanics provided crucial evidence for the atomic theory of matter. His theoretical analysis of the random motion of particles suspended in a fluid helped establish the reality of atoms and molecules, which was still debated at the beginning of the 20th century.

        The famous mass-energy equivalence formula E=mc² demonstrated the fundamental relationship between mass and energy. This principle underlies nuclear physics and has profound implications for our understanding of stellar processes and nuclear reactions. The German physicist Max Planck noted: "Einsteins Entdeckung der Masse-Energie-Äquivalenz war revolutionär" (Einstein's discovery of mass-energy equivalence was revolutionary).

        Einstein's cosmological investigations led to the understanding that the universe could be expanding or contracting. Although he initially resisted this idea by introducing the cosmological constant, he later acknowledged that a dynamic universe was a natural consequence of his theory. Modern cosmology, with concepts like 暗黒物質 (ankoku busshitsu, dark matter) and 暗黒エネルギー (ankoku enerugī, dark energy), continues to build upon Einstein's foundational work.
        """
        
        # Test: Should preserve all multilingual content
        result = await self.summarizer.process_response(multilingual_einstein_text)
        
        # Verify the result is shorter than the original
        self.assertLess(len(result), len(multilingual_einstein_text))
        # Verify it fits within the limit
        self.assertLessEqual(len(result), 2000)
        
        # Verify foreign language quotes and technical terms are preserved
        foreign_terms = [
            "Die Unschärferelation ist fundamental",  # German quote
            "Эйнштейн изменил наше понимание",  # Russian quote
            "La relativité n'est qu'une convention",  # French quote
            "相対性理論",  # Japanese: relativity theory
            "量子力学",   # Japanese: quantum mechanics
            "重力波"      # Japanese: gravitational waves
        ]
        
        preserved_terms = 0
        for term in foreign_terms:
            if term in result:
                preserved_terms += 1
        
        # At least half of the foreign terms should be preserved
        self.assertGreaterEqual(preserved_terms, len(foreign_terms) // 2, 
                               f"At least {len(foreign_terms) // 2} foreign terms should be preserved, but only {preserved_terms} were found")
        
        # Result should be meaningful summary, not just truncation
        self.assertNotEqual(result, multilingual_einstein_text[:len(result)])


if __name__ == '__main__':
    unittest.main()