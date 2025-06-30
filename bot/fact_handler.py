import logging
from typing import Dict, Any
from ai_client import AIClient
from open_telemetry import Telemetry
from schemas import FactParams, MemoryUpdate, MemoryForget
from store import Store
import re

logger = logging.getLogger(__name__)


from user_resolver import UserResolver


class FactHandler:
    def __init__(self, ai_client: AIClient, store: Store, telemetry: Telemetry, user_resolver: UserResolver):
        self.ai_client = ai_client
        self.store = store
        self.telemetry = telemetry
        self.user_resolver = user_resolver
    
    def get_route_description(self) -> str:
        return """
        FACT: For imperative memory operations (remember/forget facts about users)
        - Strictly for commands to store or remove permanent facts about users.
        - Examples:
          * "Bot remember that gruzilkin is Sergey"
          * "Bot, remember this about Florent: he likes pizza"
          * "Bot forget that gruzilkin likes pizza"
          * "remember <@987654321098765432> works at TechCorp"
          * "forget <@123456789012345678>'s birthday"
        
        Non-examples (NOT a FACT request - these are GENERAL queries):
        - "What do you remember about X?"
        - "Does John like apples?"
        - "What is X's name?"
        - "What food does <@123456789012345678> like?"
        """

    
    def get_parameter_schema(self):
        """Return the Pydantic schema for parameter extraction."""
        from schemas import FactParams
        return FactParams
    
    def get_parameter_extraction_prompt(self) -> str:
        """Return focused prompt for extracting fact operation parameters."""
        return """
        Extract parameters for a memory fact operation (remember/forget).
        
        operation: "remember" or "forget" based on an EXPLICIT and IMPERATIVE command.
        user_mention: Extract user reference (Discord ID for <@1333878858138652682> or nickname)
        fact_content: The specific fact to remember or forget, converted to third-person perspective using appropriate pronouns. This can be extracted both from the user message and inferred from the conversation history.
        
        For fact_content conversion to third-person perspective:
        - Use "they/them" as default pronouns when gender is unknown
        
        Examples:
        - "Bot remember that gruzilkin is Sergey" → operation: "remember", user_mention: "gruzilkin", fact_content: "He is Sergey"
        - "Bot, remember this about Florent: he likes pizza" → operation: "remember", user_mention: "Florent", fact_content: "he likes pizza"
        - "Bot forget that <@1333878858138652682> likes pizza" → operation: "forget", user_mention: "1333878858138652682", fact_content: "they like pizza"
        - "remember Florent works at Google" → operation: "remember", user_mention: "Florent", fact_content: "they work at Google"
        - "Bot remember I live in Tokyo" (about speaker) → operation: "remember", user_mention: "[infer from context]", fact_content: "they live in Tokyo"
        """
    
    
    
    async def _remember_fact(self, guild_id: int, user_id: int, fact_content: str) -> str:
        """Add or update a fact about a user. Fact content is already in third-person perspective from parameter extraction."""
        async with self.telemetry.async_create_span("remember_fact") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("fact_content", fact_content)
            
            # Get existing memory blob
            current_memory = await self.store.get_user_facts(guild_id, user_id)
            
            if not current_memory:
                # No existing memory, use the fact content directly
                updated_memory = fact_content
            else:
                # Merge with existing memory using AI
                prompt = f"""
                You need to update a user's memory by incorporating new information.
                
                Current memory: {current_memory}
                New information: {fact_content}
                
                Merge the new information with the existing memory, resolving any conflicts intelligently 
                and maintaining a natural narrative flow. Maintain third-person perspective.
                """
                
                memory_response = await self.ai_client.generate_content(
                    message=fact_content,
                    prompt=prompt,
                    temperature=0.0,  # Deterministic for memory operations
                    response_schema=MemoryUpdate
                )
                updated_memory = memory_response.updated_memory
            
            # Save updated memory to database
            await self.store.save_user_facts(guild_id, user_id, updated_memory)
            
            logger.info(f"Updated memory for user {user_id} in guild {guild_id}")
            return f"I'll remember that about the user."
    
    async def _forget_fact(self, guild_id: int, user_id: int, fact_content: str) -> str:
        """Remove a specific fact about a user. Fact content is already in third-person perspective from parameter extraction."""
        async with self.telemetry.async_create_span("forget_fact") as span:
            span.set_attribute("guild_id", guild_id)
            span.set_attribute("user_id", user_id)
            span.set_attribute("fact_content", fact_content)
            
            current_memory = await self.store.get_user_facts(guild_id, user_id)
            
            if not current_memory:
                return "I don't have any memory about that user to forget."
            
            # Use AI to remove specific information while maintaining coherence
            prompt = f"""
            You need to determine if specific information exists in a user's memory and remove it if found.
            
            Current memory: {current_memory}
            Information to remove: {fact_content}
            
            If the information exists in the memory, remove it and return the updated memory with fact_found=true.
            If the information is not found, set fact_found=false (the updated_memory field will be ignored).
            Maintain third-person perspective.
            """
            
            forget_response = await self.ai_client.generate_content(
                message=fact_content,
                prompt=prompt,
                temperature=0.0,  # Deterministic for memory operations
                response_schema=MemoryForget
            )
            
            # Check if the fact was found and removed
            if not forget_response.fact_found:
                return "I couldn't find that specific information in my memory about the user."
            
            updated_memory = forget_response.updated_memory
            
            # Save updated memory
            await self.store.save_user_facts(guild_id, user_id, updated_memory)
            
            logger.info(f"Removed fact from memory for user {user_id} in guild {guild_id}")
            return f"I've forgotten that about the user."
    

    async def handle_request(self, params: FactParams, guild_id: int) -> str:
        """
        Handle a fact operation request using the provided parameters.
        
        Args:
            params (FactParams): Parameters containing operation, user_mention, and fact_content
            guild_id (int): Discord guild ID for context
            
        Returns:
            str: The response string ready to be sent by the caller
        """
        logger.info(f"Processing fact request with params: {params}")
        
        # Resolve user mention to user ID
        user_id = await self.user_resolver.resolve_user_id(guild_id, params.user_mention)
        if user_id is None:
            return f"I couldn't identify the user '{params.user_mention}'. Please use a standard Discord mention, user ID, or a recognizable nickname."
        
        if params.operation == "remember":
            return await self._remember_fact(guild_id, user_id, params.fact_content)
        elif params.operation == "forget":
            return await self._forget_fact(guild_id, user_id, params.fact_content)
        else:
            return f"Unknown operation: {params.operation}"