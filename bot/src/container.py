from dotenv import load_dotenv
import os
from store import Store
from joke_generator import JokeGenerator
from gemini_client import GeminiClient
from grok_client import GrokClient
from country_resolver import CountryResolver
from open_telemetry import Telemetry

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

        # Initialize AI Client
        self.ai_client = self._create_ai_client()

        # Initialize JokeGenerator with AI Client
        self.joke_generator = JokeGenerator(self.ai_client, self.store, sample_count=int(self._get_env('SAMPLE_JOKES_COUNT')))

        # Initialize CountryResolver with AI Client
        self.country_resolver = CountryResolver(self.ai_client)
    
    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Environment variable {key} is not set")
        return value

    def _create_ai_client(self):
        GEMINI = "GEMINI"
        GROK = "GROK"
        ai_provider = self._get_env("AI_PROVIDER").upper()
        
        if ai_provider == GEMINI:
            return GeminiClient(
                api_key=self._get_env("GEMINI_API_KEY"),
                model_name=self._get_env("GEMINI_MODEL"),
                temperature=float(self._get_env("GEMINI_TEMPERATURE"))
            )
            
        elif ai_provider == GROK:
            return GrokClient(
                api_key=self._get_env("GROK_API_KEY"),
                model_name=self._get_env("GROK_MODEL"),
                temperature=float(self._get_env("GROK_TEMPERATURE"))
            )
        else:
            raise ValueError(f"Invalid AI_PROVIDER: {ai_provider}. Must be either {GEMINI} or {GROK}")

# Create a single instance to be imported by other modules
container = Container()