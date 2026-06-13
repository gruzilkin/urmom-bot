from open_telemetry import Telemetry
from openai_client import OpenAIClient


class GrokClient(OpenAIClient):
    def __init__(self, api_key: str, model_name: str, telemetry: Telemetry, temperature: float = 0.1):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            telemetry=telemetry,
            base_url="https://api.x.ai/v1",
            service="GROK",
            temperature=temperature,
        )
