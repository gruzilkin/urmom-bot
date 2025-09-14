import uuid
import logging
import sys
from types import SimpleNamespace
import time

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
    def __init__(self, service_name, endpoint):
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
        except (FileNotFoundError, OSError, IOError):
            pass
        
        # Try hostname file as backup
        try:
            with open('/etc/hostname', 'r') as f:
                hostname = f.read().strip()
                # Check if this looks like a container ID (hexadecimal)
                if len(hostname) == 12 and all(c in '0123456789abcdef' for c in hostname):
                    return hostname
        except (FileNotFoundError, OSError, IOError):
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

        # Routing metrics
        route_selections_counter = meter.create_counter(
            name="route_selections_total",
            description="Total number of route selections",
            unit="1",
        )


        # LLM request metrics
        llm_requests = meter.create_counter(
            name="llm_requests_total",
            description="Total LLM requests",
            unit="1",
        )

        llm_latency = meter.create_histogram(
            name="llm_latency",
            description="LLM request latency in milliseconds",
            unit="ms",
            explicit_bucket_boundaries=[
                100.0, 250.0, 500.0, 750.0,  # Sub-second (quick errors/cached responses)
                1000.0, 2000.0, 5000.0,      # Quick replies (1-5 seconds)
                10000.0, 15000.0, 30000.0,   # Medium responses (10-30 seconds)  
                60000.0, 120000.0, 180000.0, 300000.0  # Long responses (1-5 minutes)
            ],
        )

        structured_output_failures = meter.create_counter(
            name="llm_structured_output_failures_total",
            description="LLM structured output (schema) parse failures",
            unit="1",
        )

        # Bot/joke flow metrics
        message_latency = meter.create_histogram(
            name="bot_message_latency",
            description="Latency from message handling start to reply",
            unit="ms",
            explicit_bucket_boundaries=[
                1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 750.0,  # Fast non-LLM responses
                1000.0, 2000.0, 5000.0,                                    # Quick LLM replies (1-5 seconds)
                10000.0, 15000.0, 30000.0,                                 # Medium LLM responses (10-30 seconds)  
                60000.0, 120000.0, 180000.0                                # Long LLM responses (1-3 minutes)
            ],
        )

        jokes_generated = meter.create_counter(
            name="jokes_generated_total",
            description="Total jokes generated",
            unit="1",
        )

        message_deletions = meter.create_counter(
            name="message_deletions_total",
            description="Total bot message deletions by reason",
            unit="1",
        )



        # DB metrics
        db_latency = meter.create_histogram(
            name="db_query_latency",
            description="Database operation latency",
            unit="ms",
        )



        # Memory/daily summary metrics
        daily_summary_jobs = meter.create_counter(
            name="daily_summary_jobs_total",
            description="Daily summary jobs by outcome",
            unit="1",
        )

        daily_summary_messages = meter.create_histogram(
            name="daily_summary_messages_per_job",
            description="Number of messages processed per daily summary job",
            unit="1",
        )

        memory_merges = meter.create_counter(
            name="memory_merges_total",
            description="Memory merge outcomes",
            unit="1",
        )

        # Attachment processing metrics
        attachment_process = meter.create_counter(
            name="attachment_process_total",
            description="Attachment processing results",
            unit="1",
        )


        attachment_analysis_latency = meter.create_histogram(
            name="attachment_analysis_latency",
            description="Attachment analysis latency",
            unit="ms",
            explicit_bucket_boundaries=[
                100.0, 250.0, 500.0, 750.0,  # Sub-second (quick errors/cached responses)
                1000.0, 2000.0, 5000.0,      # Quick replies (1-5 seconds)
                10000.0, 15000.0, 30000.0,   # Medium responses (10-30 seconds)  
                60000.0, 120000.0, 180000.0, 300000.0  # Long responses (1-5 minutes)
            ],
        )

        # User resolution metric (success/error)
        user_resolution = meter.create_counter(
            name="user_resolution_total",
            description="User resolution outcomes by method",
            unit="1",
        )
        
        # Readable elapsed-time helper
        def timer():
            t0 = time.monotonic()
            def elapsed_ms():
                return (time.monotonic() - t0) * 1000.0
            return elapsed_ms

        return SimpleNamespace(
            message_counter=message_counter,
            reaction_counter=reaction_counter,
            prompt_tokens_counter=prompt_tokens_counter,
            completion_tokens_counter=completion_tokens_counter,
            total_tokens_counter=total_tokens_counter,
            route_selections_counter=route_selections_counter,
            llm_requests=llm_requests,
            llm_latency=llm_latency,
            structured_output_failures=structured_output_failures,
            message_latency=message_latency,
            jokes_generated=jokes_generated,
            message_deletions=message_deletions,
            db_latency=db_latency,
            attachment_process=attachment_process,
            attachment_analysis_latency=attachment_analysis_latency,
            daily_summary_jobs=daily_summary_jobs,
            daily_summary_messages=daily_summary_messages,
            memory_merges=memory_merges,
            user_resolution=user_resolution,
            timer=timer
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
            guild_id = message.guild.id
            
            attributes = {
                "channel_id": str(channel_id),
                "guild_id": str(guild_id)
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
                "guild_id": str(guild_id),
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


            attributes = {k: v for k, v in attributes.items() if v is not None}

            self.metrics.prompt_tokens_counter.add(prompt_tokens, attributes)
            self.metrics.completion_tokens_counter.add(completion_tokens, attributes)
            self.metrics.total_tokens_counter.add(total_tokens, attributes)
            
            logger.info(f"Token usage tracked - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}, Attributes: {attributes}")
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}", exc_info=True)
