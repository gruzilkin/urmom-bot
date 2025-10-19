# LLM Fallback Strategy

Free-tier AI services are inherently fragile, so every feature needs a clear fallback path.

**Available clients**
- gemini flash
- gemma
- ollama kimi-k2
- ollama gpt-oss
- ollama qwen3-vl (vision only)
- claude
- grok

**Client notes**
- Gemma and Gemini share rate limits, so falling back between them rarely helps unless a second provider is in the chain.
- Ollama cloud is bursty and quota-based. We do **not** wrap Ollama clients in retries—fail fast and let the composite move on.
- Claude tends to respond slowly; plan for longer latency.
- Grok is paid but reliable and relatively fast, so it is a good last-resort fallback.

**Retry policies**
- **Gemma (rate-limited)**: Time-based retry up to 60 seconds with exponential backoff and jitter to respect tokens-per-minute limits
- **Grok (paid API)**: Attempt-based retry up to 3 attempts with exponential backoff
- Both Gemma and Grok are invoked through `RetryAIClient` wrappers so composites consistently honor these defaults
- Ollama clients (Kimi, GPT-OSS, Qwen3-VL) are **not** wrapped in retries; they typically fail for the remainder of the quota window, so we immediately fall back.
- Underlying clients log errors; retry wrappers reissue calls when exceptions are raised
- Circuit breakers are not used; continue retrying because most errors are transient
- Implement retries with the `backoff` Python library using parametrized `RetryAIClient` wrapper

**Scenario guidelines**
1. **General query generator**
   Prefer the configured smart LLM. Call order: preferred smart model ➝ flash ➝ claude ➝ grok.
2. **AI router**
   Use kimi (fail fast) ➝ gemma (60s retry with jitter) ➝ grok (3-attempt retry). Fallback is triggered when the router returns `NOTSURE`.
3. **Memory**
   Daily summaries run on flash only. Historical merge path uses kimi (fail fast) ➝ gemma (60s retry with jitter).
4. **Language detector**
   Use kimi (fail fast) ➝ gemma (60s retry with jitter).
5. **Fact handler**
   Use kimi (fail fast) ➝ gpt-oss (fail fast) ➝ gemma (60s retry with jitter).
6. **Country resolver**
   Use gemma (60s retry with jitter) ➝ grok (3-attempt retry).
7. **Chat history summary**
   Use flash only. Surface the failure if flash cannot produce a summary and ensure the caller logs the exception; it will retry later.
8. **Response summarizer**
   Attempt kimi (fail fast) ➝ gemma (60s retry with jitter).
9. **Attachment processor**
   Use qwen3-vl (fail fast, required for vision) ➝ gemma (60s retry with jitter).
10. **Joke generator & famous person generator**
    Joke writing: grok with 3-attempt retry policy. Joke classification: kimi (fail fast) ➝ grok (3-attempt retry). Famous person generator continues to use grok with retry.

**Metrics**
- Each underlying client already reports success and error counters, so the composite client does not emit additional metrics—surface the original exceptions to preserve that visibility.

**Hard failures**
- When fallbacks are exhausted the composite client should throw a new exception.
- Log the underlying exceptions as warnings to preserve context without overwhelming error monitoring.
