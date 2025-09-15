from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

class NullTelemetry:
    """A no-op implementation of Telemetry for testing"""
    
    def __init__(self):
        def _counter():
            return SimpleNamespace(add=lambda *args, **kwargs: None)

        def _histogram():
            return SimpleNamespace(record=lambda *args, **kwargs: None)

        def _timer():
            # Return callable that returns 0.0 seconds when invoked
            return lambda: 0.0

        self.metrics = SimpleNamespace(
            # existing
            message_counter=_counter(),
            reaction_counter=_counter(),
            prompt_tokens_counter=_counter(),
            completion_tokens_counter=_counter(),
            total_tokens_counter=_counter(),
            # newly used by code under test
            route_selections_counter=_counter(),
            attachment_process=_counter(),
            attachment_analysis_latency=_histogram(),
            daily_summary_jobs=_counter(),
            daily_summary_messages=_histogram(),
            memory_merges=_counter(),
            message_latency=_histogram(),
            message_deletions=_counter(),
            db_latency=_histogram(),
            user_resolution=_counter(),
            llm_requests=_counter(),
            llm_latency=_histogram(),
            structured_output_failures=_counter(),
            timer=_timer
        )
    
    @contextmanager
    def create_span(self, name, kind=None, attributes=None):
        """No-op span for synchronous code"""
        yield SimpleNamespace(
            set_attribute=lambda *args: None, 
            set_status=lambda *args: None,
            record_exception=lambda *args: None
        )
    
    @asynccontextmanager
    async def async_create_span(self, name, kind=None, attributes=None):
        """No-op span for async code"""
        yield SimpleNamespace(
            set_attribute=lambda *args: None, 
            set_status=lambda *args: None,
            record_exception=lambda *args: None
        )
    
    def increment_message_counter(self, *args, **kwargs):
        pass
        
    def increment_reaction_counter(self, *args, **kwargs):
        pass
        
    def track_token_usage(self, *args, **kwargs):
        pass
