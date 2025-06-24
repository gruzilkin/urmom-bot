# Dependency Injection Refactoring Investigation

## Current Implementation Analysis

### Current Container Pattern
- **File**: `container.py`
- **Pattern**: Java Spring-style dependency injection container
- **Implementation**: Single `Container` class with all dependencies as instance attributes
- **Usage**: Global singleton instance (`container = Container()`)

**Strengths:**
- Clear dependency relationships
- Centralized configuration management
- Familiar pattern for developers from Java/Spring background
- Works well for the current Discord bot use case

**Limitations:**
- Hard to test with different configurations
- Cannot override environment variables for testing
- Tightly coupled to environment variables

## Pythonic Alternatives Investigated

### 1. Factory Functions Pattern (Recommended)

**Key Components:**
- `AppConfig` dataclass for configuration management
- `AppServices` dataclass as service container
- Factory functions for creating dependencies

**Benefits:**
- **Testability**: Easy to inject custom configs for testing
- **Immutability**: Services stored in dataclass, harder to accidentally modify
- **Functional Style**: Pure functions for dependency creation
- **Explicit Dependencies**: Each factory shows exactly what it needs

**Structure:**
```python
@dataclass
class AppConfig:
    # Configuration with from_env() factory method

@dataclass 
class AppServices:
    # All services as typed attributes

def create_app_services(config: AppConfig | None = None) -> AppServices:
    # Composition root - wires everything together
```

### 2. Other Python DI Approaches Considered

**Third-party Libraries:**
- `dependency-injector`: Most Spring-like, but adds complexity
- `punq`: Lightweight container, good for larger applications
- `returns`: Functional approach, steep learning curve

**Built-in Python Patterns:**
- Context managers: Good for resource management, overkill for services
- `functools.lru_cache`: Useful for expensive singletons
- Module-level factories: Simple but less flexible

## Migration Strategy

### Phase 1: Add Configuration Layer
1. Create `AppConfig` dataclass with `from_env()` method
2. Modify existing `Container` to accept optional config parameter
3. Update tests to inject test configurations

### Phase 2: Extract Factory Functions
1. Create factory functions for major service groups
2. Gradually refactor `Container` to use factories internally
3. Maintain backward compatibility

### Phase 3: Switch to Factory Pattern
1. Create `AppServices` dataclass
2. Implement `create_app_services()` composition root
3. Update `app.py` to use new pattern
4. Remove old `Container` class

## Implementation Notes

### Composition Root Pattern
- **Location**: `app.py` becomes the composition root
- **Responsibility**: Single place where all dependencies are wired together
- **Usage**: Call `create_app_services()` once at application startup

### Testing Benefits
```python
# Before: Hard to test with different configs
container = Container()  # Always uses environment variables

# After: Easy to inject test configuration
test_config = AppConfig(postgres_host="localhost", ...)
services = create_app_services(test_config)
```


## Recommendation

**For urmom-bot project**: The factory function approach provides the best balance of:
- Pythonic code style
- Improved testability (main pain point identified)
- Maintainable complexity
- Clear migration path from current implementation

The current container pattern is actually quite good for Python - the main improvement needed is better testability through configuration injection.

## Files to Create/Modify

1. **New**: `container_pythonic.py` - Factory-based implementation
2. **Modify**: `container.py` - Add config injection capability
3. **Modify**: Test files - Use injected configurations
4. **Update**: `app.py` - Switch to composition root pattern

## Decision: Proceed with Factory Function Refactoring

The investigation shows clear benefits for testability and maintainability while preserving the simplicity that works well for the Discord bot use case.