from openai import OpenAI
from openai.types.chat import ChatCompletion
from ai_client import AIClient
from typing import List, Tuple
from opentelemetry.trace import SpanKind
from open_telemetry import Telemetry

class GrokClient(AIClient):
    def __init__(self, api_key: str, model_name: str = "grok-2-latest", temperature: float = 0.7, telemetry: Telemetry = None):
        if not api_key:
            raise ValueError("Grok API key not provided!")
        if not model_name:
            raise ValueError("Grok model name not provided!")

        self.model = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        self.model_name = model_name
        self.temperature = temperature
        self.telemetry = telemetry
        
    def _track_completion_metrics(self, completion: ChatCompletion, method_name: str, **additional_attributes):
        """Track metrics from completion response with detailed attributes"""
        if self.telemetry:
            usage = completion.usage
            attributes = {
                "service": "GROK",
                "model": self.model_name,
                "method": method_name
            }
            
            # Add any additional attributes passed in
            attributes.update(additional_attributes)
            
            self.telemetry.track_token_usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                attributes=attributes
            )

    async def generate_content(self, message: str, prompt: str = None, samples: List[Tuple[str, str]] = None) -> str:
        async with self.telemetry.async_create_span("generate_content", kind=SpanKind.CLIENT) as span:
            messages = []
            if prompt:
                messages.append({"role": "system", "content": prompt})
            
            if samples:
                for user_msg, assistant_msg in samples:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
                
            messages.append({"role": "user", "content": message})

            print(messages)

            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )

            print(completion)
            self._track_completion_metrics(completion, method_name="generate_content")

            return completion.choices[0].message.content

    async def is_joke(self, original_message: str, response_message: str) -> bool:
        async with self.telemetry.async_create_span("is_joke", kind=SpanKind.CLIENT) as span:
            prompt = f"""Tell me if the response is a joke, a wordplay or a sarcastic remark to the original message, reply in English with only yes or no:
    original message: {original_message}
    response: {response_message}
    No? Think again carefully. The response might be a joke, wordplay, or sarcastic remark.
    Is it actually a joke? Reply only yes or no."""

            print(f"[GROK] Checking if message is a joke:")
            print(f"[GROK] Original: {original_message}")
            print(f"[GROK] Response: {response_message}")

            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            print(f"[GROK] Raw completion object: {completion}")
            response_text = completion.choices[0].message.content.strip().lower()
            # Remove any punctuation and check if the response is 'yes'
            result = response_text.rstrip('.,!?') == "yes"
            self._track_completion_metrics(completion, method_name="is_joke", is_joke=result)
            print(f"[GROK] AI response: {response_text}")
            print(f"[GROK] Is joke: {result}")
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
        async with self.telemetry.async_create_span("generate_famous_person_response", kind=SpanKind.CLIENT) as span:
            system_content = f"""You are {person}. Generate a response as if you were {person}, 
                using their communication style, beliefs, values, and knowledge.
                Make the response thoughtful, authentic to {person}'s character, and relevant to the conversation.
                Stay in character completely and respond directly as {person} would.
                Keep your response length similar to the average message length in the conversation.
                The user specifically asked: '{original_message}'
                Your response should be in the form of direct speech - exactly as if {person} is speaking directly, without quotation marks or attributions."""
                
            system_message = {
                "role": "system", 
                "content": system_content
            }
            
            messages = [system_message]
            
            for username, content in conversation:
                messages.append({
                    "role": "user",
                    "content": f"{username}: {content}"
                })
            
            print(f"[GROK] Generating response as {person}")
            print(f"[GROK] Messages: {messages}")
            
            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            
            print(f"[GROK] Raw completion object: {completion}")
            self._track_completion_metrics(completion, 
                                        method_name="generate_famous_person_response", 
                                        person=person)
            
            return completion.choices[0].message.content

    async def is_famous_person_request(self, message: str) -> str | None:
        """Check if a message is asking what a famous person would say"""
        async with self.telemetry.async_create_span("is_famous_person_request", kind=SpanKind.CLIENT) as span:
            prompt = """You need to check if the user message is asking to impersonate a famous person and reply with the person's name.

            Example 1:
            Input: What would Trump say?
            Output: Trump

            Example 2:
            Input: What's the weather today?
            Output: None

            Example 3:
            Input: What would Jesus say if he spoke like Trump?
            Output: Jesus

            Example 4:
            Input: How would Darth Vader feel about this?
            Output: Darth Vader

            Example 5:
            Input: What if Eminen did tldr?
            Output: Eminen

            Example 5:
            Input: How would Sigmund Freud respond to this?
            Output: Sigmund Freud

            Only extract the person's name if the message is clearly asking to impersonate them.
            If it's not a request to impersonate someone then respond with 'None'."""
            

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}
            ]
            print(f"[GROK] Request messages: {messages}")
            
            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1
            )
            
            print(f"[GROK] Response object: {completion}")
            response_text = completion.choices[0].message.content.strip()
            
            # Convert "None" string to actual None
            person = None if response_text == "None" else response_text
            self._track_completion_metrics(completion, 
                                        method_name="is_famous_person_request", 
                                        person_detected=(person is not None),
                                        person=person)
            
            print(f"[GROK] Famous person detection result: '{person}'")
            
            return person
