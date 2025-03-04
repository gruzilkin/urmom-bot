from google import genai
from google.genai.types import Content, Part, GenerateContentConfig, GenerateContentResponse
from ai_client import AIClient
from typing import List, Tuple
from open_telemetry import Telemetry

class GeminiClient(AIClient):
    def __init__(self, api_key: str, model_name: str, temperature: float = 1.2, telemetry: Telemetry = None):
        if not api_key:
            raise ValueError("Gemini API key not provided!")
        if not model_name:
            raise ValueError("Gemini model name not provided!")

        self.client = genai.Client(api_key=api_key)
        self.temperature = temperature
        self.model_name = model_name
        self.telemetry = telemetry
        
    def _track_completion_metrics(self, response: GenerateContentResponse, method_name: str, **additional_attributes):
        """Track metrics from Gemini response with detailed attributes"""
        if self.telemetry:
            try:
                usage_metadata = response.usage_metadata
                
                prompt_tokens = usage_metadata.prompt_token_count
                completion_tokens = usage_metadata.candidates_token_count
                total_tokens = usage_metadata.total_token_count
                    
                attributes = {
                    "service": "GEMINI",
                    "model": self.model_name,
                    "method": method_name
                }
                
                # Add any additional attributes passed in
                attributes.update(additional_attributes)
                
                self.telemetry.track_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    attributes=attributes
                )
                print(f"[GEMINI] Token usage tracked - Method: {method_name}, Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            except Exception as e:
                print(f"[GEMINI] Error tracking token usage: {e}")

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None) -> str:
        samples = samples or []
        contents = []

        for msg, joke in samples:
            contents.append(Content(parts=[Part(text=msg)], role="user"))
            contents.append(Content(parts=[Part(text=joke)], role="model"))

        contents.append(Content(parts=[Part(text=message)], role="user"))

        print(f"[GEMINI] system_instruction: {prompt}")
        print(f"[GEMINI] Request contents: {contents}")

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfig(
                temperature=self.temperature,
                system_instruction = prompt,
            )
        )

        print(response)
        print(f"[GEMINI] Response: {response}")
        self._track_completion_metrics(response, method_name="generate_content")
        return response.text

    async def is_joke(self, original_message: str, response_message: str) -> bool:
        prompt = f"""Tell me if the response is a joke, a wordplay or a sarcastic remark to the original message, reply in English with only yes or no:
original message: {original_message}
response: {response_message}
No? Think again carefully. The response might be a joke, wordplay, or sarcastic remark.
Is it actually a joke? Reply only yes or no."""
        
        print(f"[GEMINI] Checking if message is a joke:")
        print(f"[GEMINI] Original: {original_message}")
        print(f"[GEMINI] Response: {response_message}")

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1,
                stop_sequences=["\n", "."]
            )
        )
        
        print(f"[GEMINI] Raw response object: {response}")
        result = response.text.strip().lower() == "yes"
        self._track_completion_metrics(response, method_name="is_joke", is_joke=result)
        print(f"[GEMINI] AI response: {response.text}")
        print(f"[GEMINI] Is joke: {result}")
        return result

    async def generate_famous_person_response(self, conversation: List[Tuple[str, str]], person: str, original_message: str = "") -> str:
        """
        Generate a response in the style of a famous person based on the conversation context.
        
        Args:
            conversation (List[Tuple[str, str]]): List of (username, message) tuples
            person (str): The name of the famous person
            original_message (str): The original user request with bot mention removed
            
        Returns:
            str: A response in the style of the famous person
        """

        system_instruction = f"""You are {person}. Generate a response as if you were {person}, 
            using their communication style, beliefs, values, and knowledge.
            Make the response thoughtful, authentic to {person}'s character, and relevant to the conversation.
            Stay in character completely and respond directly as {person} would.
            Keep your response length similar to the average message length in the conversation.
            The user specifically asked: '{original_message}'
            Your response should be in the form of direct speech - exactly as if {person} is speaking directly, without quotation marks or attributions."""
        
        contents = []
        for username, content in conversation:
            contents.append(Content(parts=[Part(text=f"{username}: {content}")], role="user"))
        
        print(f"[GEMINI] Generating response as {person}")
        print(f"[GEMINI] Conversation: {conversation}")
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=system_instruction,
            )
        )
        
        self._track_completion_metrics(
            response, 
            method_name="generate_famous_person_response", 
            person=person
        )
        print(f"[GEMINI] Famous person response: {response.text}")
        return response.text

    async def is_famous_person_request(self, message: str) -> str | None:
        """Check if a message is asking what a famous person would say"""
        
        system_instruction = """If the user message is a question similar to "What would X say?" 
        and then reply with X - the person's name.

        Only extract the person's name if the message is clearly asking what they would say.
        If it's not a request about what someone would say, respond with 'None'.
        """
        
        print(f"[GEMINI] Checking if message is a famous person request: {message}")
        
        config = GenerateContentConfig(
            temperature=0.1,
            system_instruction=system_instruction,
        )
        print(f"[GEMINI] Request config: {config}")

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=[message],
            config=config
        )
        
        print(f"[GEMINI] Full response object: {response}")
        response_text = response.text.strip()
        
        # Convert "None" string to actual None
        person_name = None if response_text == "None" else response_text
        self._track_completion_metrics(
            response, 
            method_name="is_famous_person_request", 
            person_detected=(person_name is not None),
            person=person_name
        )
        print(f"[GEMINI] Famous person detection result: '{person_name}'")
        
        return person_name
