# LLM Fallback Strategy

Free-tier AI services are inherently fragile, so every feature needs a clear fallback path.

**Available clients**
- gemma
- flash
- claude
- grok

**Client notes**
- Gemma and Gemini share rate limits, so falling back between them rarely helps.
- Claude tends to respond slowly; plan for longer latency.
- Grok is paid but reliable and relatively fast, so it is a good last-resort fallback.

**Retry policies**
- **Gemma (rate-limited)**: Time-based retry up to 60 seconds with exponential backoff and jitter to respect tokens-per-minute limits
- **Grok (paid API)**: Attempt-based retry up to 3 attempts with exponential backoff
- Both Gemma and Grok are invoked through `RetryAIClient` wrappers so composites consistently honor these defaults
- Underlying clients log errors; retry wrappers reissue calls when exceptions are raised
- Circuit breakers are not used; continue retrying because most errors are transient
- Implement retries with the `backoff` Python library using parametrized `RetryAIClient` wrapper

**Scenario guidelines**
1. **General query generator**
   Prefer the configured smart LLM. Call order: preferred smart model ➝ flash ➝ claude ➝ grok.
2. **AI router**
   Use gemma (60s retry with jitter) ➝ grok (3-attempt retry) ➝ flash (on NOTSURE).
3. **Memory**
   Use gemma (60s retry with jitter) ➝ grok (3-attempt retry).
4. **Language detector**
   Use gemma (60s retry with jitter) ➝ grok (3-attempt retry).
5. **Fact handler**
   Use gemma (60s retry with jitter) ➝ grok (3-attempt retry).
6. **Country resolver**
   Use gemma (60s retry with jitter) ➝ grok (3-attempt retry).
7. **Chat history summary**
   Use flash only. Surface the failure if flash cannot produce a summary and ensure the caller logs the exception; it will retry later.
8. **Response summarizer**
   Attempt gemma, claude, then grok (no retries - fast failover).
9. **Attachment processor**
   Use gemma only with 60s retry and jitter (no fallback needed).
10. **Joke generator & famous person generator**
    Grok-exclusive with 3-attempt retry policy.

**Metrics**
- Each underlying client already reports success and error counters, so the composite client does not emit additional metrics—surface the original exceptions to preserve that visibility.

**Hard failures**
- When fallbacks are exhausted the composite client should throw a new exception.
- Log the underlying exceptions as warnings to preserve context without overwhelming error monitoring.
