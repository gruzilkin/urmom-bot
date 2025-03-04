import os
import logging
from types import SimpleNamespace
import nextcord

# OpenTelemetry imports
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

class Telemetry:
    def __init__(self, service_name="urmom-bot", endpoint="192.168.0.2:4317"):
        self.service_name = service_name
        self.endpoint = endpoint
        self.metrics = self.setup_metrics()
            
    def setup_metrics(self):
        """Set up OpenTelemetry metrics with debug logging"""
        print("Setting up OpenTelemetry metrics...")
        
        print(f"Service name: {self.service_name}")
        print(f"OTLP endpoint: {self.endpoint}")
        
        resource = Resource.create({
            "service.name": self.service_name
        })
        
        # Create the OTLP exporter for metrics
        otlp_exporter = OTLPMetricExporter(
            endpoint=self.endpoint,
            insecure=True
        )
        print(f"Created OTLP exporter targeting {self.endpoint}")
        
        # Create metric readers
        otlp_reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=15000  # Export every 15 seconds
        )
        
        # Set the meter provider with the readers
        provider = MeterProvider(metric_readers=[otlp_reader], resource=resource)
        metrics.set_meter_provider(provider)
        print("Meter provider configured with OTLP exporter")
        
        # Create a meter
        meter = metrics.get_meter("urmom_bot_metrics")
        print("Created meter: urmom_bot_metrics")
        
        # Define a single counter for message events
        message_counter = meter.create_counter(
            name="discord_messages",
            description="Number of Discord messages received",
            unit="1"
        )
        print("Created counter: discord_messages")
        
        # Define a counter for reaction events
        reaction_counter = meter.create_counter(
            name="discord_reactions",
            description="Number of Discord reactions received",
            unit="1"
        )
        print("Created counter: discord_reactions")
        
        # Define counters for token usage metrics
        prompt_tokens_counter = meter.create_counter(
            name="llm_prompt_tokens",
            description="Number of tokens used in prompts",
            unit="tokens"
        )
        print("Created counter: llm_prompt_tokens")
        
        completion_tokens_counter = meter.create_counter(
            name="llm_completion_tokens",
            description="Number of tokens used in completions",
            unit="tokens"
        )
        print("Created counter: llm_completion_tokens")
        
        total_tokens_counter = meter.create_counter(
            name="llm_total_tokens",
            description="Total number of tokens used",
            unit="tokens"
        )
        print("Created counter: llm_total_tokens")
        
        return SimpleNamespace(
            message_counter=message_counter,
            reaction_counter=reaction_counter,
            prompt_tokens_counter=prompt_tokens_counter,
            completion_tokens_counter=completion_tokens_counter,
            total_tokens_counter=total_tokens_counter
        )
    
    def increment_message_counter(self, message: nextcord.Message):
        """Increment the message counter and log the action"""
        try:
            channel_id = message.channel.id
            guild_id = message.guild.id if message.guild else None
            
            attributes = {
                "channel_id": str(channel_id),
                "guild_id": str(guild_id) if guild_id else "dm"
            }
            self.metrics.message_counter.add(1, attributes)
        except Exception as e:
            print(f"Error incrementing counter: {e}")
            
    def increment_reaction_counter(self, payload: nextcord.RawReactionActionEvent):
        """Increment the reaction counter with emoji, channel and guild information"""
        try:
            emoji_str = str(payload.emoji)
            channel_id = payload.channel_id
            guild_id = payload.guild_id
            
            attributes = {
                "channel_id": str(channel_id),
                "guild_id": str(guild_id) if guild_id else "dm",
                "emoji": emoji_str
            }
            self.metrics.reaction_counter.add(1, attributes)
        except Exception as e:
            print(f"Error incrementing reaction counter: {e}")
            
    def track_token_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, attributes: dict = None):
        """Track token usage from LLM API calls with custom attributes"""
        try:
            if attributes is None:
                attributes = {}
            
            self.metrics.prompt_tokens_counter.add(prompt_tokens, attributes)
            self.metrics.completion_tokens_counter.add(completion_tokens, attributes)
            self.metrics.total_tokens_counter.add(total_tokens, attributes)
            
            print(f"Token usage tracked - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Attributes: {attributes}")
        except Exception as e:
            print(f"Error tracking token usage: {e}")