from dotenv import load_dotenv
import os
from store import Store
from joke_generator import JokeGenerator
from famous_person_generator import FamousPersonGenerator
from general_query_generator import GeneralQueryGenerator
from gemini_client import GeminiClient
from grok_client import GrokClient
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

        self.ai_client = self._create_ai_client()
        
        # Create specific AI clients for GeneralQueryGenerator
        self.gemini_pro = GeminiClient(
            api_key=self._get_env("GEMINI_API_KEY"),
            model_name=self._get_env("GEMINI_PRO_MODEL"),
            temperature=float(self._get_env("GEMINI_PRO_TEMPERATURE")),
            telemetry=self.telemetry
        )
        
        self.gemini_flash = GeminiClient(
            api_key=self._get_env("GEMINI_API_KEY"),
            model_name=self._get_env("GEMINI_FLASH_MODEL"),
            temperature=float(self._get_env("GEMINI_FLASH_TEMPERATURE")),
            telemetry=self.telemetry
        )
        
        self.grok = GrokClient(
            api_key=self._get_env("GROK_API_KEY"),
            model_name=self._get_env("GROK_MODEL"),
            temperature=float(self._get_env("GROK_TEMPERATURE")),
            telemetry=self.telemetry
        )

        self.joke_generator = JokeGenerator(
            self.ai_client, 
            self.store, 
            self.telemetry, 
            sample_count=int(self._get_env('SAMPLE_JOKES_COUNT'))
        )

        self.famous_person_generator = FamousPersonGenerator(self.ai_client, self.telemetry)

        self.general_query_generator = GeneralQueryGenerator(
            gemini_pro=self.gemini_pro,
            gemini_flash=self.gemini_flash, 
            grok=self.grok,
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

    def _create_ai_client(self):
        PRO = "PRO"
        FLASH = "FLASH"
        GROK = "GROK"
        ai_provider = self._get_env("AI_PROVIDER").upper()
        
        if ai_provider == PRO:
            return GeminiClient(
                api_key=self._get_env("GEMINI_API_KEY"),
                model_name=self._get_env("GEMINI_PRO_MODEL"),
                temperature=float(self._get_env("GEMINI_PRO_TEMPERATURE")),
                telemetry=self.telemetry
            )
        elif ai_provider == FLASH:
            return GeminiClient(
                api_key=self._get_env("GEMINI_API_KEY"),
                model_name=self._get_env("GEMINI_FLASH_MODEL"),
                temperature=float(self._get_env("GEMINI_FLASH_TEMPERATURE")),
                telemetry=self.telemetry
            )
        elif ai_provider == GROK:
            return GrokClient(
                api_key=self._get_env("GROK_API_KEY"),
                model_name=self._get_env("GROK_MODEL"),
                temperature=float(self._get_env("GROK_TEMPERATURE")),
                telemetry=self.telemetry
            )
        else:
            raise ValueError(f"Invalid AI_PROVIDER: {ai_provider}. Must be either {PRO}, {FLASH}, or {GROK}")

# Create a single instance to be imported by other modules
container = Container()