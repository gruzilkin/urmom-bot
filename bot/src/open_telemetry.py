import uuid
import contextlib
from types import SimpleNamespace

import nextcord
# OpenTelemetry imports
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind, Status, StatusCode

class Telemetry:
    def __init__(self, service_name="urmom-bot", endpoint="192.168.0.2:4317"):
        self.service_name = service_name
        self.endpoint = endpoint
        
        # Create shared resource for both metrics and tracing
        self.resource = Resource.create({
            "service.name": self.service_name,
            "service.instance.id": self.get_container_id(),
        })
        
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
            
    def setup_metrics(self):
        """Set up OpenTelemetry metrics with debug logging"""
        print("Setting up OpenTelemetry metrics...")
        
        print(f"Service name: {self.service_name}")
        print(f"OTLP endpoint: {self.endpoint}")
        
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
        
        provider = MeterProvider(metric_readers=[otlp_reader], resource=self.resource)
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
    
    def setup_tracing(self):
        """Set up OpenTelemetry tracing"""
        print("Setting up OpenTelemetry tracing...")
        
        # Create the OTLP exporter for tracing
        otlp_span_exporter = OTLPSpanExporter(
            endpoint=self.endpoint,
            insecure=True
        )
        print(f"Created OTLP trace exporter targeting {self.endpoint}")
        
        # Create a span processor for the exporter
        span_processor = BatchSpanProcessor(otlp_span_exporter)
        
        # Set up the tracer provider with the processor and shared resource
        trace_provider = TracerProvider(resource=self.resource)
        trace_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(trace_provider)
        print("Tracer provider configured with OTLP exporter")
        
        # Create a tracer
        tracer = trace.get_tracer("urmom_bot_tracer")
        print("Created tracer: urmom_bot_tracer")
        
        return tracer
    
    @contextlib.contextmanager
    def create_span(self, name, kind=SpanKind.INTERNAL, attributes=None):
        """Create a span as a context manager for tracing operations"""
        if attributes is None:
            attributes = {}
            
        span = self.tracer.start_span(name, kind=kind, attributes=attributes)
        try:
            with trace.use_span(span, end_on_exit=True):
                yield span
                span.set_status(Status(StatusCode.OK))  # Set status to OK if no exception occurs
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
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