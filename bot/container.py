from dotenv import load_dotenv
import os
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

class Container:
    def __init__(self):
        load_dotenv()
        
        # Initialize Store
        self.store = Store(
            host=self._get_env('POSTGRES_HOST'),
            port=int(self._get_env('POSTGRES_PORT')),
            user=self._get_env('POSTGRES_USER'),
            password=self._get_env('POSTGRES_PASSWORD'),
            database=self._get_env('POSTGRES_DB'),
            weight_coef=float(self._get_env('SAMPLE_JOKES_COEF'))
        )

        # Initialize Telemetry with environment variables
        service_name = os.getenv("OTEL_SERVICE_NAME", "urmom-bot")
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "192.168.0.2:4317")
        self.telemetry = Telemetry(service_name=service_name, endpoint=endpoint)

        # Create specific AI clients for GeneralQueryGenerator (both required)
        self.gemini_flash = GeminiClient(
            api_key=self._get_env("GEMINI_API_KEY"),
            model_name=self._get_env("GEMINI_FLASH_MODEL"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            telemetry=self.telemetry
        )
        
        self.gemma = GemmaClient(
            api_key=self._get_env("GEMINI_API_KEY"),
            model_name=self._get_env("GEMINI_GEMMA_MODEL"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            telemetry=self.telemetry
        )
        
        self.grok = GrokClient(
            api_key=self._get_env("GROK_API_KEY"),
            model_name=self._get_env("GROK_MODEL"),
            temperature=float(os.getenv("GROK_TEMPERATURE", "0.7")),
            telemetry=self.telemetry
        )
        
        self.claude = ClaudeClient(
            telemetry=self.telemetry
        )

        self.ai_client = self._get_ai_client()

        self.joke_generator = JokeGenerator(
            self.grok, 
            self.store, 
            self.telemetry, 
            sample_count=int(self._get_env('SAMPLE_JOKES_COUNT'))
        )

        self.famous_person_generator = FamousPersonGenerator(self.grok, self.telemetry)

        self.general_query_generator = GeneralQueryGenerator(
            gemini_flash=self.gemini_flash, 
            grok=self.grok,
            claude=self.claude,
            telemetry=self.telemetry
        )

        self.ai_router = AiRouter(
            self.ai_client, 
            self.telemetry, 
            self.famous_person_generator, 
            self.general_query_generator
        )

        self.country_resolver = CountryResolver(self.ai_client, self.telemetry)
    
    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Environment variable {key} is not set")
        return value

    def _get_ai_client(self):
        FLASH = "FLASH"
        GEMMA = "GEMMA"
        GROK = "GROK"
        CLAUDE = "CLAUDE"
        ai_provider = self._get_env("AI_PROVIDER").upper()
        
        if ai_provider == FLASH:
            return self.gemini_flash
        elif ai_provider == GEMMA:
            return self.gemma
        elif ai_provider == GROK:
            return self.grok
        elif ai_provider == CLAUDE:
            return self.claude
        else:
            raise ValueError(f"Invalid AI_PROVIDER: {ai_provider}. Must be either {FLASH}, {GEMMA}, {GROK}, or {CLAUDE}")

# Create a single instance to be imported by other modules
container = Container()