from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional


class AppConfig(BaseSettings):
    """Centralized configuration model for all environment variables."""
    
    # Database configuration
    postgres_host: str = Field(env="POSTGRES_HOST")
    postgres_port: int = Field(env="POSTGRES_PORT")
    postgres_user: str = Field(env="POSTGRES_USER")
    postgres_password: str = Field(env="POSTGRES_PASSWORD")
    postgres_db: str = Field(env="POSTGRES_DB")
    
    # AI Provider configuration
    ai_provider: str = Field(env="AI_PROVIDER")
    
    # Gemini/Gemma configuration
    gemini_api_key: str = Field(env="GEMINI_API_KEY")
    gemini_flash_model: str = Field(env="GEMINI_FLASH_MODEL")
    gemini_gemma_model: str = Field(env="GEMINI_GEMMA_MODEL")
    gemini_temperature: float = Field(default=0.7, env="GEMINI_TEMPERATURE")
    
    # Grok configuration
    grok_api_key: str = Field(env="GROK_API_KEY")
    grok_model: str = Field(env="GROK_MODEL")
    grok_temperature: float = Field(default=0.7, env="GROK_TEMPERATURE")
    
    # Discord configuration
    discord_token: str = Field(env="DISCORD_TOKEN")
    
    # Joke generation configuration
    sample_jokes_count: int = Field(env="SAMPLE_JOKES_COUNT")
    sample_jokes_coef: float = Field(env="SAMPLE_JOKES_COEF")
    
    # OpenTelemetry configuration
    otel_service_name: str = Field(default="urmom-bot", env="OTEL_SERVICE_NAME")
    otel_exporter_otlp_endpoint: str = Field(default="localhost:4317", env="OTEL_EXPORTER_OTLP_ENDPOINT")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }
    
    @field_validator("ai_provider")
    @classmethod
    def validate_ai_provider(cls, v: str) -> str:
        """Validate AI provider is one of the supported options."""
        valid_providers = {"FLASH", "GEMMA", "GROK", "CLAUDE"}
        if v.upper() not in valid_providers:
            raise ValueError(f"AI_PROVIDER must be one of {valid_providers}")
        return v.upper()
    
    @field_validator("postgres_port")
    @classmethod
    def validate_postgres_port(cls, v: int) -> int:
        """Validate PostgreSQL port is within valid range."""
        if not (1 <= v <= 65535):
            raise ValueError("POSTGRES_PORT must be between 1 and 65535")
        return v
    
    @field_validator("sample_jokes_count")
    @classmethod
    def validate_sample_jokes_count(cls, v: int) -> int:
        """Validate sample jokes count is positive."""
        if v <= 0:
            raise ValueError("SAMPLE_JOKES_COUNT must be positive")
        return v
    
    @field_validator("sample_jokes_coef")
    @classmethod
    def validate_sample_jokes_coef(cls, v: float) -> float:
        """Validate sample jokes coefficient is positive."""
        if v <= 0:
            raise ValueError("SAMPLE_JOKES_COEF must be positive")
        return v
    
    @field_validator("gemini_temperature", "grok_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within valid range."""
        if not (0.0 <= v <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v