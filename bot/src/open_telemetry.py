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
        
        # Add console exporter for debugging
        console_exporter = ConsoleMetricExporter()
        
        # Create metric readers
        otlp_reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=15000  # Export every 15 seconds
        )
        
        console_reader = PeriodicExportingMetricReader(
            exporter=console_exporter,
            export_interval_millis=15000
        )
        
        # Set the meter provider with the readers
        provider = MeterProvider(metric_readers=[otlp_reader], resource=resource)
        metrics.set_meter_provider(provider)
        print("Meter provider configured with OTLP and Console exporters")
        
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
        
        return SimpleNamespace(
            message_counter=message_counter
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
            return True
        except Exception as e:
            print(f"Error incrementing counter: {e}")