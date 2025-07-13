import uuid
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind, Status, StatusCode

# OpenTelemetry logging imports
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

# JSON logging for OpenTelemetry
from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)

class SimpleTelemetry:
    """OpenTelemetry telemetry for web application"""
    
    def __init__(self, service_name: str, endpoint: str):
        self.service_name = service_name
        self.endpoint = endpoint
        
        # Create shared resource
        self.resource = Resource.create({
            "service.name": self.service_name,
            "service.instance.id": self.get_container_id(),
        })
        
        self.setup_logging()
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
        tracer = trace.get_tracer("joke_admin_web_tracer")
        logger.info("Created tracer: joke_admin_web_tracer")
        
        return tracer
    
    @asynccontextmanager
    async def async_create_span(self, name: str, kind=SpanKind.INTERNAL, attributes=None):
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

