# urmom-bot Project Information

## Testing

### Unit Tests
Run unit tests only (excludes integration tests):
```bash
source .venv/bin/activate && PYTHONPATH=bot python -m unittest discover -s bot/tests/unit -p "*test*.py" -v
```

### Test Structure
- Unit tests: `bot/tests/unit/`
- Integration tests: `bot/tests/integration/`
- Testing framework: `unittest.IsolatedAsyncioTestCase` for async code
- Telemetry mocking: Use `NullTelemetry()` from `tests.null_telemetry`

### Project Structure
- Main code: `bot/` directory
- Virtual environment: `.venv/`
- Use `PYTHONPATH=bot` to set Python path instead of `cd bot`