from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

class NullTelemetry:
    """A no-op implementation of Telemetry for testing"""
    
    def __init__(self):
        self.metrics = SimpleNamespace(
            message_counter=SimpleNamespace(add=lambda *args, **kwargs: None),
            reaction_counter=SimpleNamespace(add=lambda *args, **kwargs: None),
            prompt_tokens_counter=SimpleNamespace(add=lambda *args, **kwargs: None),
            completion_tokens_counter=SimpleNamespace(add=lambda *args, **kwargs: None),
            total_tokens_counter=SimpleNamespace(add=lambda *args, **kwargs: None)
        )
    
    @contextmanager
    def create_span(self, name, kind=None, attributes=None):
        """No-op span for synchronous code"""
        yield SimpleNamespace(set_attribute=lambda *args: None, set_status=lambda *args: None)
    
    @asynccontextmanager
    async def async_create_span(self, name, kind=None, attributes=None):
        """No-op span for async code"""
        yield SimpleNamespace(set_attribute=lambda *args: None, set_status=lambda *args: None)
    
    def increment_message_counter(self, *args, **kwargs):
        pass
        
    def increment_reaction_counter(self, *args, **kwargs):
        pass
        
    def track_token_usage(self, *args, **kwargs):
        pass