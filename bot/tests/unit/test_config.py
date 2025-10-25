import unittest
import os
from unittest.mock import patch
from config import AppConfig


class TestAppConfig(unittest.TestCase):
    """Test the centralized AppConfig model."""

    def test_config_validation_success(self):
        """Test that valid configuration is accepted."""
        config = AppConfig(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_user="test",
            postgres_password="test",
            postgres_db="test",
            gemini_api_key="test-key",
            gemini_flash_model="gemini-1.5-flash",
            gemini_pro_model="gemini-2.5-pro",
            gemini_gemma_model="gemini-2.0-flash-exp",
            grok_api_key="test-grok-key",
            grok_model="grok-beta",
            discord_token="test-token",
            sample_jokes_count=5,
            sample_jokes_coef=1.0
        )
        
        self.assertEqual(config.postgres_host, "localhost")
        self.assertEqual(config.postgres_port, 5432)
        
    
    def test_port_validation(self):
        """Test that invalid port raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            AppConfig(
                postgres_host="localhost",
                postgres_port=99999,  # Invalid port
                postgres_user="test",
                postgres_password="test",
                postgres_db="test",
                gemini_api_key="test-key",
                gemini_flash_model="gemini-1.5-flash",
                gemini_pro_model="gemini-2.5-pro",
                gemini_gemma_model="gemini-2.0-flash-exp",
                grok_api_key="test-grok-key",
                grok_model="grok-beta",
                discord_token="test-token",
                sample_jokes_count=5,
                sample_jokes_coef=1.0
            )
        self.assertIn("POSTGRES_PORT must be between", str(cm.exception))
    
    def test_temperature_validation(self):
        """Test that invalid temperature raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            AppConfig(
                postgres_host="localhost",
                postgres_port=5432,
                postgres_user="test",
                postgres_password="test",
                postgres_db="test",
                gemini_api_key="test-key",
                gemini_flash_model="gemini-1.5-flash",
                gemini_pro_model="gemini-2.5-pro",
                gemini_gemma_model="gemini-2.0-flash-exp",
                gemini_temperature=3.0,  # Invalid temperature
                grok_api_key="test-grok-key",
                grok_model="grok-beta",
                discord_token="test-token",
                sample_jokes_count=5,
                sample_jokes_coef=1.0
            )
        self.assertIn("Temperature must be between", str(cm.exception))
    
    @patch.dict(os.environ, {
        'POSTGRES_HOST': 'test-host',
        'POSTGRES_PORT': '5433',
        'POSTGRES_USER': 'test-user',
        'POSTGRES_PASSWORD': 'test-pass',
        'POSTGRES_DB': 'test-db',
        'GEMINI_API_KEY': 'test-gemini-key',
        'GEMINI_FLASH_MODEL': 'test-flash-model',
        'GEMINI_PRO_MODEL': 'test-pro-model',
        'GEMINI_GEMMA_MODEL': 'test-gemma-model',
        'GROK_API_KEY': 'test-grok-key',
        'GROK_MODEL': 'test-grok-model',
        'DISCORD_TOKEN': 'test-discord-token',
        'SAMPLE_JOKES_COUNT': '10',
        'SAMPLE_JOKES_COEF': '2.5',
        'GEMINI_TEMPERATURE': '0.8',
        'GROK_TEMPERATURE': '0.9'
    })
    def test_from_env_method(self):
        """Test that BaseSettings automatically loads configuration from environment variables."""
        config = AppConfig()
        
        self.assertEqual(config.postgres_host, 'test-host')
        self.assertEqual(config.postgres_port, 5433)
        self.assertEqual(config.postgres_user, 'test-user')
        self.assertEqual(config.postgres_password, 'test-pass')
        self.assertEqual(config.postgres_db, 'test-db')
        self.assertEqual(config.gemini_api_key, 'test-gemini-key')
        self.assertEqual(config.gemini_flash_model, 'test-flash-model')
        self.assertEqual(config.gemini_pro_model, 'test-pro-model')
        self.assertEqual(config.gemini_gemma_model, 'test-gemma-model')
        self.assertEqual(config.grok_api_key, 'test-grok-key')
        self.assertEqual(config.grok_model, 'test-grok-model')
        self.assertEqual(config.discord_token, 'test-discord-token')
        self.assertEqual(config.sample_jokes_count, 10)
        self.assertEqual(config.sample_jokes_coef, 2.5)
        self.assertEqual(config.gemini_temperature, 0.8)
        self.assertEqual(config.grok_temperature, 0.9)


if __name__ == '__main__':
    unittest.main()