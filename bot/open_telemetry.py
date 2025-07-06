import uuid
import contextlib
import logging
import sys
import os
from types import SimpleNamespace

import nextcord
# OpenTelemetry imports
from opentelemetry import metrics, trace, baggage
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind, Status, StatusCode
from contextlib import asynccontextmanager, contextmanager

# OpenTelemetry logging imports
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

# JSON logging for OpenTelemetry
from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)

class Telemetry:
    def __init__(self, service_name="urmom-bot", endpoint="192.168.0.2:4317"):
        self.service_name = service_name
        self.endpoint = endpoint
        
        # Create shared resource for both metrics and tracing
        self.resource = Resource.create({
            "service.name": self.service_name,
            "service.instance.id": self.get_container_id(),
        })
        
        self.setup_logging()
        
        # Setup both metrics and tracing
        self.metrics = self.setup_metrics()
        self.tracer = self.setup_tracing()

    def get_container_id(self):
        """Get the Docker container ID or generate a unique ID if not in Docker"""
        # Try to get container ID from cgroup (most reliable in Docker)
        try:
            with open('/proc/self/cgroup', 'r') as f:
                for line in f:
                    if '/docker/' in line:
                        return line.split('/')[-1][:12]  # Get the 12-char container ID
        except:
            pass
        
        # Try hostname file as backup
        try:
            with open('/etc/hostname', 'r') as f:
                hostname = f.read().strip()
                # Check if this looks like a container ID (hexadecimal)
                if len(hostname) == 12 and all(c in '0123456789abcdef' for c in hostname):
                    return hostname
        except:
            pass
            
        # Fall back to a generated UUID if we can't get container ID
        return uuid.uuid4().hex[:12]

    def setup_logging(self):
        """Configure structured logging with standard stdout and OpenTelemetry integration."""
        
        stdout_handler = logging.StreamHandler(sys.stdout)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(stdout_handler)
        
        # Set a custom formatter for console logs to include the module name as a prefix
        console_formatter = logging.Formatter('[%(name)s] %(message)s')
        stdout_handler.setFormatter(console_formatter)
        
        otel_logger_provider = LoggerProvider(resource=self.resource)
        set_logger_provider(otel_logger_provider)

        otlp_log_exporter = OTLPLogExporter(
            endpoint=self.endpoint,
            insecure=True
        )
        otel_logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(otlp_log_exporter)
        )

        otel_handler = LoggingHandler(
            level=logging.NOTSET, 
            logger_provider=otel_logger_provider
        )

        json_formatter = jsonlogger.JsonFormatter()
        otel_handler.setFormatter(json_formatter)

        root_logger.addHandler(otel_handler)

        logger.info("OpenTelemetry logging configured")

    def setup_metrics(self):
        """Set up OpenTelemetry metrics with debug logging"""
        logger.info("Setting up OpenTelemetry metrics...")
        
        logger.info(f"Service name: {self.service_name}")
        logger.info(f"OTLP endpoint: {self.endpoint}")
        
        # Create the OTLP exporter for metrics
        otlp_exporter = OTLPMetricExporter(
            endpoint=self.endpoint,
            insecure=True
        )
        logger.info(f"Created OTLP exporter targeting {self.endpoint}")
        
        # Create metric readers
        otlp_reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=15000  # Export every 15 seconds
        )
        
        provider = MeterProvider(metric_readers=[otlp_reader], resource=self.resource)
        metrics.set_meter_provider(provider)
        logger.info("Meter provider configured with OTLP exporter")
        
        # Create a meter
        meter = metrics.get_meter("urmom_bot_metrics")
        logger.info("Created meter: urmom_bot_metrics")
        
        # Define a single counter for message events
        message_counter = meter.create_counter(
            name="discord_messages",
            description="Number of Discord messages received",
            unit="1"
        )
        logger.info("Created counter: discord_messages")
        
        # Define a counter for reaction events
        reaction_counter = meter.create_counter(
            name="discord_reactions",
            description="Number of Discord reactions received",
            unit="1"
        )
        logger.info("Created counter: discord_reactions")
        
        # Define counters for token usage metrics
        prompt_tokens_counter = meter.create_counter(
            name="llm_prompt_tokens",
            description="Number of tokens used in prompts",
            unit="tokens"
        )
        logger.info("Created counter: llm_prompt_tokens")
        
        completion_tokens_counter = meter.create_counter(
            name="llm_completion_tokens",
            description="Number of tokens used in completions",
            unit="tokens"
        )
        logger.info("Created counter: llm_completion_tokens")
        
        total_tokens_counter = meter.create_counter(
            name="llm_total_tokens",
            description="Total number of tokens used",
            unit="tokens"
        )
        logger.info("Created counter: llm_total_tokens")
        
        return SimpleNamespace(
            message_counter=message_counter,
            reaction_counter=reaction_counter,
            prompt_tokens_counter=prompt_tokens_counter,
            completion_tokens_counter=completion_tokens_counter,
            total_tokens_counter=total_tokens_counter
        )
    
    def setup_tracing(self):
        """Set up OpenTelemetry tracing"""
        logger.info("Setting up OpenTelemetry tracing...")
        
        # Create the OTLP exporter for tracing
        otlp_span_exporter = OTLPSpanExporter(
            endpoint=self.endpoint,
            insecure=True
        )
        logger.info(f"Created OTLP trace exporter targeting {self.endpoint}")
        
        # Create a span processor for the exporter
        span_processor = BatchSpanProcessor(otlp_span_exporter)
        
        # Set up the tracer provider with the processor and shared resource
        trace_provider = TracerProvider(resource=self.resource)
        trace_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(trace_provider)
        logger.info("Tracer provider configured with OTLP exporter")
        
        # Create a tracer
        tracer = trace.get_tracer("urmom_bot_tracer")
        logger.info("Created tracer: urmom_bot_tracer")
        
        return tracer
    
    @contextmanager
    def create_span(self, name, kind=SpanKind.INTERNAL, attributes=None):
        """Create a span as a context manager for tracing operations"""
        if attributes is None:
            attributes = {}
            
        span = self.tracer.start_span(name, kind=kind, attributes=attributes)
        try:
            with trace.use_span(span, end_on_exit=False):
                yield span
                span.set_status(Status(StatusCode.OK))
                span.end()
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            span.end()
            raise
    
    @asynccontextmanager
    async def async_create_span(self, name, kind=SpanKind.INTERNAL, attributes=None):
        """Create a span as an async context manager for async operations"""
        span = self.tracer.start_span(name, kind=kind, attributes=attributes or {})
        try:
            with trace.use_span(span, end_on_exit=False):
                yield span
                span.set_status(Status(StatusCode.OK))
                span.end()
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            span.end()
            raise

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
            logger.error(f"Error incrementing counter: {e}", exc_info=True)
            
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
            logger.error(f"Error incrementing reaction counter: {e}", exc_info=True)
            
    def track_token_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, attributes: dict = None):
        """Track token usage from LLM API calls with custom attributes"""
        try:
            if attributes is None:
                attributes = {}

            # Add guild_id from baggage if available
            guild_id = baggage.get_baggage("guild_id")
            if guild_id:
                attributes["guild_id"] = guild_id

            attributes = {k: v for k, v in attributes.items() if v is not None}

            self.metrics.prompt_tokens_counter.add(prompt_tokens, attributes)
            self.metrics.completion_tokens_counter.add(completion_tokens, attributes)
            self.metrics.total_tokens_counter.add(total_tokens, attributes)
            
            logger.info(f"Token usage tracked - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Attributes: {attributes}")
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}", exc_info=True)