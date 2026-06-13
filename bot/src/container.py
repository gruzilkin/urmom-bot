from store import Store
from joke_generator import JokeGenerator
from famous_person_generator import FamousPersonGenerator
from general_query_generator import GeneralQueryGenerator
from gemini_client import GeminiClient
from gemma_client import GemmaClient
from grok_client import GrokClient
from claude_client import ClaudeClient
from codex_client import CodexClient
from deepseek_client import DeepSeekClient
from country_resolver import CountryResolver
from open_telemetry import Telemetry
from ai_router import AiRouter
from response_summarizer import ResponseSummarizer
from attachment_processor import AttachmentProcessor
from fact_handler import FactHandler
from schedule_handler import ScheduleHandler
from schedule_engine import ScheduleEngine
from user_resolver import UserResolver
from memory_manager import MemoryManager
from language_detector import LanguageDetector
from conversation_formatter import ConversationFormatter
from config import AppConfig
from ai_client_wrappers import CompositeAIClient, RetryAIClient
from ai_client import AIClient
from wisdom_generator import WisdomGenerator
from devils_advocate_generator import DevilsAdvocateGenerator
from cobalt_client import CobaltClient
from tinyurl_client import TinyURLClient
from video_compressor import VideoCompressor
from redis_cache import RedisCache
from video_embedder import MAX_FILE_SIZE_BYTES, VideoEmbedder


class Container:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()

        self.telemetry = Telemetry(
            service_name=self.config.otel_service_name,
            endpoint=self.config.otel_exporter_otlp_endpoint,
        )

        self.redis_cache = RedisCache(
            host=self.config.redis_host,
            port=self.config.redis_port,
            telemetry=self.telemetry,
        )

        self.cobalt_client = CobaltClient(
            base_url=self.config.cobalt_url,
            telemetry=self.telemetry,
        )

        self.tinyurl_client = TinyURLClient(
            api_token=self.config.tinyurl_api_token,
            telemetry=self.telemetry,
        )

        self.video_compressor = VideoCompressor(
            telemetry=self.telemetry,
            target_size_bytes=MAX_FILE_SIZE_BYTES,
        )

        self.video_embedder = VideoEmbedder(
            cobalt_client=self.cobalt_client,
            video_compressor=self.video_compressor,
            tinyurl_client=self.tinyurl_client,
            telemetry=self.telemetry,
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

        # Gemini Flash client for general queries and daily summaries
        self.gemini_flash = GeminiClient(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_flash_model,
            telemetry=self.telemetry,
            temperature=self.config.gemini_temperature,
        )

        self.gemma = self._build_gemma_client()

        self.grok = GrokClient(
            api_key=self.config.grok_api_key,
            model_name=self.config.grok_model,
            telemetry=self.telemetry,
            temperature=self.config.grok_temperature,
        )

        self.claude = ClaudeClient(telemetry=self.telemetry, model_name="opus")
        self.claude_haiku = ClaudeClient(telemetry=self.telemetry, model_name="haiku")

        self.codex = CodexClient(telemetry=self.telemetry, model_name="gpt-5.4")
        self.codex_mini = CodexClient(telemetry=self.telemetry, model_name="gpt-5.4-mini")

        self.deepseek = DeepSeekClient(
            api_key=self.config.deepseek_api_key,
            model_name=self.config.deepseek_model,
            telemetry=self.telemetry,
            base_url=self.config.deepseek_base_url,
            temperature=self.config.deepseek_temperature,
        )

        # Apply retry policy for rate-limited services (Gemma/Grok only)
        self.retrying_gemma = RetryAIClient(self.gemma, telemetry=self.telemetry, max_time=60, jitter=True)
        self.retrying_grok = RetryAIClient(self.grok, telemetry=self.telemetry, max_tries=3)

        self.lightweight_fallback = CompositeAIClient(
            [self.gemma, self.codex_mini, self.retrying_gemma, self.claude_haiku, self.deepseek, self.retrying_grok],
            telemetry=self.telemetry,
        )

        # Shuffled composite for jokes and wisdom - gives both clients equal chance
        self.shuffled_grok_gemini = CompositeAIClient(
            [self.retrying_grok, self.gemini_flash],
            telemetry=self.telemetry,
            shuffle=True,
        )

        self.codex_claude_deepseek_flash_fallback = CompositeAIClient(
            [self.codex, self.claude_haiku, self.deepseek, self.gemini_flash],
            telemetry=self.telemetry,
        )

        # Composite for the devil's advocate generator
        self.codex_gemini_grok = CompositeAIClient(
            [self.codex, self.gemini_flash, self.retrying_grok],
            telemetry=self.telemetry,
        )

        self.response_summarizer = ResponseSummarizer(
            self.lightweight_fallback,
            self.telemetry,
        )

        # Initialize language detector early since it's needed by multiple components
        self.language_detector = LanguageDetector(
            ai_client=self.lightweight_fallback,
            telemetry=self.telemetry,
        )

        self.attachment_processor = AttachmentProcessor(
            ai_client=CompositeAIClient(
                [self.codex, self.retrying_gemma],
                telemetry=self.telemetry,
            ),
            telemetry=self.telemetry,
            redis_cache=self.redis_cache,
            max_file_size_mb=10,
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
            [self.codex, self.retrying_gemma, self.claude_haiku, self.deepseek, self.retrying_grok],
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
            summary_client=self.codex_claude_deepseek_flash_fallback,
            alias_client=self.lightweight_fallback,
            merge_client=self.lightweight_fallback,
            user_resolver=self.user_resolver,
            redis_cache=self.redis_cache,
        )

        self.joke_generator = JokeGenerator(
            joke_writer_client=self.shuffled_grok_gemini,
            joke_classifier_client=self.lightweight_fallback,
            store=self.store,
            telemetry=self.telemetry,
            language_detector=self.language_detector,
            conversation_formatter=self.conversation_formatter,
            memory_manager=self.memory_manager,
            sample_count=self.config.sample_jokes_count,
        )

        self.general_query_generator = GeneralQueryGenerator(
            client_selector=self._build_general_ai_client,
            response_summarizer=self.response_summarizer,
            telemetry=self.telemetry,
            store=self.store,
            conversation_formatter=self.conversation_formatter,
            memory_manager=self.memory_manager,
            user_resolver=self.user_resolver,
        )

        self.schedule_engine = ScheduleEngine(
            store=self.store,
            telemetry=self.telemetry,
            general_query_generator=self.general_query_generator,
        )

        self.schedule_handler = ScheduleHandler(
            ai_client=self.lightweight_fallback,
            store=self.store,
            telemetry=self.telemetry,
            schedule_engine=self.schedule_engine,
            conversation_formatter=self.conversation_formatter,
        )

        # The router client will be a composite client that handles the NOTSURE fallback.
        router_client = CompositeAIClient(
            [self.gemma, self.codex_mini, self.retrying_gemma, self.claude_haiku, self.deepseek, self.retrying_grok],
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
            self.conversation_formatter,
            self.schedule_handler,
        )

        # Late-bound to break the engine → router → schedule_handler → engine cycle
        self.schedule_engine.ai_router = self.ai_router

        self.country_resolver = CountryResolver(self.lightweight_fallback, self.telemetry)

        self.wisdom_generator = WisdomGenerator(
            ai_client=self.shuffled_grok_gemini,
            language_detector=self.language_detector,
            conversation_formatter=self.conversation_formatter,
            response_summarizer=self.response_summarizer,
            memory_manager=self.memory_manager,
            telemetry=self.telemetry,
        )

        self.devils_advocate_generator = DevilsAdvocateGenerator(
            ai_client=self.codex_gemini_grok,
            language_detector=self.language_detector,
            conversation_formatter=self.conversation_formatter,
            response_summarizer=self.response_summarizer,
            memory_manager=self.memory_manager,
            telemetry=self.telemetry,
        )

    def _build_gemma_client(self) -> AIClient:
        """Build the Gemma client, shuffling across two models when GEMMA_MODEL_2 is set.

        The two Gemma models have independent rate-limit quotas, so a shuffled composite
        doubles the effective free tier.
        """
        model_names = [self.config.gemma_model]
        if self.config.gemma_model_2:
            model_names.append(self.config.gemma_model_2)

        clients = [
            GemmaClient(
                api_key=self.config.gemma_api_key,
                model_name=model_name,
                telemetry=self.telemetry,
                temperature=self.config.gemini_temperature,
            )
            for model_name in model_names
        ]

        if len(clients) == 1:
            return clients[0]
        return CompositeAIClient(clients, telemetry=self.telemetry, shuffle=True)

    def _build_general_ai_client(self, preferred_backend: str) -> AIClient:
        """Create a composite client matching the fallback rules for general queries."""
        client_map: dict[str, AIClient] = {
            "gemini_flash": self.gemini_flash,
            "claude": self.claude,
            "grok": self.retrying_grok,
            "gemma": self.retrying_gemma,
            "codex": self.codex,
            "deepseek": self.deepseek,
        }

        if preferred_backend not in client_map:
            raise ValueError(f"Unknown ai_backend: {preferred_backend}")

        fallback_order = ["codex", "claude", "deepseek", "gemini_flash", "grok"]
        ordered_labels = [preferred_backend] + [label for label in fallback_order if label != preferred_backend]

        chain = [client_map[label] for label in ordered_labels]

        return CompositeAIClient(chain, telemetry=self.telemetry)


# Create a single instance to be imported by other modules
container = Container()
