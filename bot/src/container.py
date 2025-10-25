from store import Store
from joke_generator import JokeGenerator
from famous_person_generator import FamousPersonGenerator
from general_query_generator import GeneralQueryGenerator
from gemini_client import GeminiClient
from gemma_client import GemmaClient
from grok_client import GrokClient
from claude_client import ClaudeClient
from country_resolver import CountryResolver
from open_telemetry import Telemetry
from ai_router import AiRouter
from response_summarizer import ResponseSummarizer
from attachment_processor import AttachmentProcessor
from fact_handler import FactHandler
from user_resolver import UserResolver
from memory_manager import MemoryManager
from language_detector import LanguageDetector
from conversation_formatter import ConversationFormatter
from config import AppConfig
from ai_client_wrappers import CompositeAIClient, RetryAIClient
from ai_client import AIClient
from ollama_client import OllamaClient
from wisdom_generator import WisdomGenerator


class Container:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()

        self.telemetry = Telemetry(
            service_name=self.config.otel_service_name,
            endpoint=self.config.otel_exporter_otlp_endpoint,
        )

        self.store = Store(
            telemetry=self.telemetry,
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
            database=self.config.postgres_db,
            weight_coef=self.config.sample_jokes_coef,
        )

        # Create specific AI clients for GeneralQueryGenerator (both required)
        self.gemini_flash = GeminiClient(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_flash_model,
            telemetry=self.telemetry,
            temperature=self.config.gemini_temperature,
        )

        self.gemini_pro = GeminiClient(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_pro_model,
            telemetry=self.telemetry,
            temperature=self.config.gemini_temperature,
        )

        self.gemma = GemmaClient(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_gemma_model,
            telemetry=self.telemetry,
            temperature=self.config.gemini_temperature,
        )

        self.grok = GrokClient(
            api_key=self.config.grok_api_key,
            model_name=self.config.grok_model,
            telemetry=self.telemetry,
            temperature=self.config.grok_temperature,
        )

        self.claude = ClaudeClient(telemetry=self.telemetry)

        self.ollama_kimi = OllamaClient(
            api_key=self.config.ollama_api_key,
            model_name=self.config.ollama_kimi_model,
            telemetry=self.telemetry,
            base_url=self.config.ollama_base_url,
            temperature=self.config.ollama_temperature,
        )
        self.ollama_kimi_long_timeout = OllamaClient(
            api_key=self.config.ollama_api_key,
            model_name=self.config.ollama_kimi_model,
            telemetry=self.telemetry,
            base_url=self.config.ollama_base_url,
            temperature=self.config.ollama_temperature,
            timeout=90.0,
        )
        self.ollama_gpt_oss = OllamaClient(
            api_key=self.config.ollama_api_key,
            model_name=self.config.ollama_gpt_oss_model,
            telemetry=self.telemetry,
            base_url=self.config.ollama_base_url,
            temperature=self.config.ollama_temperature,
        )
        self.ollama_qwen_vl = OllamaClient(
            api_key=self.config.ollama_api_key,
            model_name=self.config.ollama_qwen_vl_model,
            telemetry=self.telemetry,
            base_url=self.config.ollama_base_url,
            temperature=0.0,
            timeout=60.0,
        )

        # Apply retry policy for rate-limited services (Gemma/Grok only)
        self.retrying_gemma = RetryAIClient(self.gemma, telemetry=self.telemetry, max_time=60, jitter=True)
        self.retrying_grok = RetryAIClient(self.grok, telemetry=self.telemetry, max_tries=3)

        # Composite for components needing gemma → grok fallback
        self.gemma_with_grok_fallback = CompositeAIClient(
            [self.retrying_gemma, self.retrying_grok],
            telemetry=self.telemetry,
        )

        # Shuffled composite for wisdom - gives both clients equal chance
        self.shuffled_grok_kimi = CompositeAIClient(
            [self.retrying_grok, self.ollama_kimi],
            telemetry=self.telemetry,
            shuffle=True,
        )

        self.kimi_with_gemma_fallback = CompositeAIClient(
            [self.ollama_kimi, self.retrying_gemma],
            telemetry=self.telemetry,
        )

        self.gemma_with_kimi_fallback = CompositeAIClient(
            [self.retrying_gemma, self.ollama_kimi],
            telemetry=self.telemetry,
        )

        self.flash_with_kimi_fallback = CompositeAIClient(
            [self.gemini_flash, self.ollama_kimi],
            telemetry=self.telemetry,
        )

        self.pro_flash_with_kimi_long_timeout = CompositeAIClient(
            [self.gemini_pro, self.gemini_flash, self.ollama_kimi_long_timeout],
            telemetry=self.telemetry,
        )

        self.qwen_with_gemma_fallback = CompositeAIClient(
            [self.ollama_qwen_vl, self.retrying_gemma],
            telemetry=self.telemetry,
        )

        self.response_summarizer = ResponseSummarizer(
            self.kimi_with_gemma_fallback,
            self.telemetry,
        )

        # Initialize language detector early since it's needed by multiple components
        self.language_detector = LanguageDetector(ai_client=self.gemma_with_kimi_fallback, telemetry=self.telemetry)

        self.attachment_processor = AttachmentProcessor(
            ai_client=self.qwen_with_gemma_fallback,
            telemetry=self.telemetry,
            max_file_size_mb=10,
        )

        self.joke_generator = JokeGenerator(
            joke_writer_client=self.retrying_grok,
            joke_classifier_client=CompositeAIClient(
                [self.ollama_kimi, self.retrying_grok],
                telemetry=self.telemetry,
            ),
            store=self.store,
            telemetry=self.telemetry,
            language_detector=self.language_detector,
            sample_count=self.config.sample_jokes_count,
        )

        # UserResolver is initialized here but needs bot client to be set later
        self.user_resolver = UserResolver(self.telemetry)

        self.conversation_formatter = ConversationFormatter(self.user_resolver)

        self.famous_person_generator = FamousPersonGenerator(
            self.retrying_grok,
            self.response_summarizer,
            self.telemetry,
            self.conversation_formatter,
        )

        fact_handler_client = CompositeAIClient(
            [self.ollama_kimi, self.ollama_gpt_oss, self.retrying_gemma],
            telemetry=self.telemetry,
        )
        self.fact_handler = FactHandler(
            ai_client=fact_handler_client,
            store=self.store,
            telemetry=self.telemetry,
            user_resolver=self.user_resolver,
        )

        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.store,
            gemini_client=self.pro_flash_with_kimi_long_timeout,
            gemma_client=self.gemma_with_kimi_fallback,
            user_resolver=self.user_resolver,
        )

        self.general_query_generator = GeneralQueryGenerator(
            client_selector=self._build_general_ai_client,
            response_summarizer=self.response_summarizer,
            telemetry=self.telemetry,
            store=self.store,
            conversation_formatter=self.conversation_formatter,
            memory_manager=self.memory_manager,
        )

        # The router client will be a composite client that handles the NOTSURE fallback.
        router_client = CompositeAIClient(
            [self.ollama_kimi, self.retrying_gemma, self.retrying_grok],
            telemetry=self.telemetry,
            is_bad_response=lambda r: getattr(r, "route", None) == "NOTSURE",
        )

        self.ai_router = AiRouter(
            router_client,
            self.telemetry,
            self.language_detector,
            self.famous_person_generator,
            self.general_query_generator,
            self.fact_handler,
        )

        self.country_resolver = CountryResolver(self.gemma_with_grok_fallback, self.telemetry)

        self.wisdom_generator = WisdomGenerator(
            ai_client=self.shuffled_grok_kimi,
            language_detector=self.language_detector,
            conversation_formatter=self.conversation_formatter,
            response_summarizer=self.response_summarizer,
            memory_manager=self.memory_manager,
            telemetry=self.telemetry,
        )

    def _build_general_ai_client(self, preferred_backend: str) -> AIClient:
        """Create a composite client matching the fallback rules for general queries."""
        client_map: dict[str, AIClient] = {
            "gemini_flash": self.gemini_flash,
            "claude": self.claude,
            "grok": self.retrying_grok,
            "gemma": self.retrying_gemma,
        }

        if preferred_backend not in client_map:
            raise ValueError(f"Unknown ai_backend: {preferred_backend}")

        fallback_order = ["gemini_flash", "claude", "grok"]
        ordered_labels = [preferred_backend] + [label for label in fallback_order if label != preferred_backend]

        chain = [client_map[label] for label in ordered_labels]

        return CompositeAIClient(chain, telemetry=self.telemetry)


# Create a single instance to be imported by other modules
container = Container()
