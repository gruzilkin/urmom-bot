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
from fact_handler import FactHandler
from user_resolver import UserResolver
from memory_manager import MemoryManager
from language_detector import LanguageDetector
from config import AppConfig

class Container:
    def __init__(self, config: AppConfig | None = None):
        # Load configuration from environment or use provided config
        self.config = config or AppConfig()
        
        # Initialize Telemetry with configuration
        self.telemetry = Telemetry(
            service_name=self.config.otel_service_name, 
            endpoint=self.config.otel_exporter_otlp_endpoint
        )

        # Initialize Store
        self.store = Store(
            telemetry=self.telemetry,
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
            database=self.config.postgres_db,
            weight_coef=self.config.sample_jokes_coef
        )

        # Create specific AI clients for GeneralQueryGenerator (both required)
        self.gemini_flash = GeminiClient(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_flash_model,
            telemetry=self.telemetry,
            temperature=self.config.gemini_temperature
        )
        
        self.gemma = GemmaClient(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_gemma_model,
            telemetry=self.telemetry,
            temperature=self.config.gemini_temperature
        )
        
        self.grok = GrokClient(
            api_key=self.config.grok_api_key,
            model_name=self.config.grok_model,
            telemetry=self.telemetry,
            temperature=self.config.grok_temperature
        )
        
        self.claude = ClaudeClient(
            telemetry=self.telemetry
        )

        self.ai_client = self._get_ai_client()

        # Create response summarizer for handling long responses
        self.response_summarizer = ResponseSummarizer(self.gemma, self.telemetry)

        # Initialize language detector early since it's needed by multiple components
        self.language_detector = LanguageDetector(
            ai_client=self.gemma,
            telemetry=self.telemetry
        )

        self.joke_generator = JokeGenerator(
            self.grok, 
            self.store, 
            self.telemetry,
            self.language_detector,
            sample_count=self.config.sample_jokes_count
        )

        # UserResolver is initialized here but needs bot client to be set later
        self.user_resolver = UserResolver(self.telemetry)

        self.famous_person_generator = FamousPersonGenerator(
            self.grok, self.response_summarizer, self.telemetry, self.user_resolver
        )
        
        self.fact_handler = FactHandler(
            ai_client=self.gemma,  # Use Gemma for memory operations
            store=self.store,
            telemetry=self.telemetry,
            user_resolver=self.user_resolver
        )
        
        self.memory_manager = MemoryManager(
            telemetry=self.telemetry,
            store=self.store,
            gemini_client=self.gemini_flash,
            gemma_client=self.gemma,
            user_resolver=self.user_resolver
        )
        
        self.general_query_generator = GeneralQueryGenerator(
            gemini_flash=self.gemini_flash, 
            grok=self.grok,
            claude=self.claude,
            gemma=self.gemma,
            response_summarizer=self.response_summarizer,
            telemetry=self.telemetry,
            store=self.store,
            user_resolver=self.user_resolver,
            memory_manager=self.memory_manager
        )
        
        self.ai_router = AiRouter(
            self.ai_client,
            self.telemetry,
            self.language_detector,
            self.famous_person_generator,
            self.general_query_generator,
            self.fact_handler
        )

        self.country_resolver = CountryResolver(self.ai_client, self.telemetry)

    def _get_ai_client(self):
        if self.config.ai_provider == "FLASH":
            return self.gemini_flash
        elif self.config.ai_provider == "GEMMA":
            return self.gemma
        elif self.config.ai_provider == "GROK":
            return self.grok
        elif self.config.ai_provider == "CLAUDE":
            return self.claude
        else:
            # This should not happen due to Pydantic validation, but keeping for safety
            raise ValueError(f"Invalid AI_PROVIDER: {self.config.ai_provider}")

# Create a single instance to be imported by other modules
container = Container()