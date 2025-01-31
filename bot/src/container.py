from dotenv import load_dotenv
import os
from store import Store
from joke_generator import JokeGenerator
from gemini_client import GeminiClient
from country_resolver import CountryResolver

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

        # Initialize GeminiClient
        self.gemini_client = GeminiClient(
            api_key=self._get_env('GEMINI_API_KEY'),
            model_name=self._get_env('GEMINI_MODEL'),
            temperature=float(self._get_env('GEMINI_TEMPERATURE'))
        )

        # Initialize JokeGenerator with GeminiClient
        self.joke_generator = JokeGenerator(self.gemini_client, self.store, sample_count=int(self._get_env('SAMPLE_JOKES_COUNT')))

        # Initialize CountryResolver with GeminiClient
        self.country_resolver = CountryResolver(self.gemini_client)
    
    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Environment variable {key} is not set")
        return value

# Create a single instance to be imported by other modules
container = Container()