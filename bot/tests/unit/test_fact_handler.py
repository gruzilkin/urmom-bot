import unittest
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv

from fact_handler import FactHandler
from gemma_client import GemmaClient
from schemas import FactParams, MemoryUpdate, MemoryForget
from store import Store
from tests.null_telemetry import NullTelemetry
from user_resolver import UserResolver

load_dotenv()

class TestFactHandlerIntegration(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.telemetry = NullTelemetry()
        
        # Mock the AI client to return predictable responses
        self.mock_ai_client = Mock(spec=GemmaClient)
        
        # Mock the store
        self.mock_store = Mock(spec=Store)
        self.mock_store.get_user_facts = AsyncMock(return_value=None)
        self.mock_store.save_user_facts = AsyncMock()
        
        # Mock the user resolver
        self.mock_user_resolver = Mock(spec=UserResolver)
        
        # The component under test
        self.fact_handler = FactHandler(
            ai_client=self.mock_ai_client,
            store=self.mock_store,
            telemetry=self.telemetry,
            user_resolver=self.mock_user_resolver
        )

    async def test_remember_fact_for_new_user(self):
        """Test remembering a fact for a user with no prior memory."""
        # Arrange
        params = FactParams(operation="remember", user_mention="testuser", fact_content="they like chocolate", language_code="en", language_name="English")
        self.mock_user_resolver.resolve_user_id.return_value = 12345
        self.mock_ai_client.generate_content = AsyncMock(
            return_value=MemoryUpdate(updated_memory="they like chocolate", confirmation_message="I'll remember that they like chocolate.")
        )
        
        # Act
        response = await self.fact_handler.handle_request(params, guild_id=67890)
        
        # Assert
        self.assertEqual(response, "I'll remember that they like chocolate.")
        self.mock_store.get_user_facts.assert_awaited_once_with(67890, 12345)
        self.mock_store.save_user_facts.assert_awaited_once_with(67890, 12345, "they like chocolate")
        self.mock_ai_client.generate_content.assert_called_once() # AI call now needed for confirmation message generation

    async def test_remember_fact_with_existing_memory(self):
        """Test remembering a fact for a user who already has a memory blob."""
        # Arrange
        params = FactParams(operation="remember", user_mention="testuser", fact_content="they are a developer", language_code="en", language_name="English")
        self.mock_user_resolver.resolve_user_id.return_value = 12345
        self.mock_store.get_user_facts.return_value = "they like chocolate"
        # Set up mock to return merging response
        self.mock_ai_client.generate_content = AsyncMock(
            return_value=MemoryUpdate(updated_memory="they like chocolate and they are a developer", confirmation_message="I'll remember that they are a developer.")
        )
        
        # Act
        response = await self.fact_handler.handle_request(params, guild_id=67890)
        
        # Assert
        self.assertEqual(response, "I'll remember that they are a developer.")
        self.mock_store.save_user_facts.assert_awaited_once_with(67890, 12345, "they like chocolate and they are a developer")
        # Expect one AI call for merging
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_forget_fact_not_found(self):
        """Test forgetting a fact that does not exist in the user's memory."""
        # Arrange
        params = FactParams(operation="forget", user_mention="testuser", fact_content="they dislike broccoli", language_code="en", language_name="English")
        self.mock_user_resolver.resolve_user_id.return_value = 12345
        self.mock_store.get_user_facts.return_value = "they like chocolate"
        # Set up mock to return forget operation response  
        self.mock_ai_client.generate_content = AsyncMock(
            return_value=MemoryForget(updated_memory="they like chocolate", fact_found=False, confirmation_message="I couldn't find that information about broccoli in my memory.")
        )
        
        # Act
        response = await self.fact_handler.handle_request(params, guild_id=67890)
        
        # Assert
        self.assertEqual(response, "I couldn't find that information about broccoli in my memory.")
        self.mock_store.save_user_facts.assert_not_awaited()
        # Expect one AI call for forget operation
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_forget_fact_found(self):
        """Test forgetting a fact that exists in the user's memory."""
        # Arrange
        params = FactParams(operation="forget", user_mention="testuser", fact_content="they like chocolate", language_code="en", language_name="English")
        self.mock_user_resolver.resolve_user_id.return_value = 12345
        self.mock_store.get_user_facts.return_value = "they like chocolate and they are a developer"
        # Set up mock to return forget operation response
        self.mock_ai_client.generate_content = AsyncMock(
            return_value=MemoryForget(updated_memory="they are a developer", fact_found=True, confirmation_message="I've forgotten that they like chocolate.")
        )
        
        # Act
        response = await self.fact_handler.handle_request(params, guild_id=67890)
        
        # Assert
        self.assertEqual(response, "I've forgotten that they like chocolate.")
        self.mock_store.save_user_facts.assert_awaited_once_with(67890, 12345, "they are a developer")
        # Expect one AI call for forget operation
        self.mock_ai_client.generate_content.assert_called_once()

    async def test_handle_request_with_unresolvable_user(self):
        """Test that the handler returns a helpful message for unresolvable users."""
        # Arrange
        params = FactParams(operation="remember", user_mention="nonexistentuser", fact_content="they like testing", language_code="en", language_name="English")
        self.mock_user_resolver.resolve_user_id.return_value = None
        
        # Act
        response = await self.fact_handler.handle_request(params, guild_id=67890)
        
        # Assert
        self.assertIn("couldn't identify the user", response)
        self.mock_store.get_user_facts.assert_not_awaited()
        self.mock_store.save_user_facts.assert_not_awaited()

if __name__ == '__main__':
    unittest.main()