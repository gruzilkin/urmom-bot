# Ollama Cloud Notes

The codebase now treats the Ollama models as production peers to Gemini/Gemma/Grok.  
This document captures the practical quirks we uncovered while wiring them in.

## Active Usage
- **kimi-k2:** primary for linguistic-heavy tasks (AI router, fact handler, language detector) and joke classification fallback.
- **gpt-oss:** secondary text fallback for the fact handler when Kimi fails.
- **qwen3-vl:** vision path for the attachment processor (image descriptions).
- **Ollama clients are never retried.** They run behind hourly quota windows; when a call fails we fall through to Gemma/Grok immediately.

## Behavioural Quirks

### Structured Output
- Models frequently drift on `Literal`/`enum` fields (e.g., `"Yes"` instead of `"YES"`).
- `OllamaClient` wraps every structured call in a validation loop that feeds Pydantic errors back to the model; do not remove this loop or duplicate schema instructions in prompts.
- Keep responses pure JSON. Kimi occasionally emits `<|constrain|>` helpers on the first attempt; the retry loop resolves it automatically.

### Tools & Grounding
- Tool execution (`web_search`, `web_fetch`) is disabled. We saw unreliable behaviour and formatting regressions when tools were present, so the client simply logs a warning when `enable_grounding=True` and ignores the request.
- Because tools are off, prompts should not mention tool usage with Ollama models.

### Timeouts & Latency
- `OllamaClient` enforces a 20 s timeout. If the service stalls beyond that we raise and let the composite pick the next provider.
- Rate limits reset hourly. A quota hit usually persists for the full window; leaving retries off prevents wasting time.
- Vision (qwen3-vl) can take several seconds per request. The attachment processor is the only path that tolerates that latency.

### Prompting Tips
- Avoid markdown fences around structured output; the client strips them, but some models are more likely to drift if you ask for fences.
- When extending prompts, keep schema/format instructions inside the client logic, not the feature layer.

## Configuration Summary
Environment variables live in `AppConfig`:

```
OLLAMA_API_KEY
OLLAMA_BASE_URL                # defaults to https://ollama.com
OLLAMA_KIMI_MODEL              # default kimi-k2:1t-cloud
OLLAMA_GPT_OSS_MODEL           # default gpt-oss:120b-cloud
OLLAMA_QWEN_VL_MODEL           # default qwen3-vl:235b-cloud
OLLAMA_TEMPERATURE             # default 0.1 (overridden per feature when needed)
```

## Troubleshooting
- Look for `ollama_chat` spans in telemetry to time individual calls; `validation_retry` and `tool_iteration` attributes indicate how many correction loops ran.
- Quota failures surface as 4xx/5xx responses from the API; once you see one, expect the service to reject calls until the top of the next hour.
- If structured output keeps failing after the allowed retries, the client raises the final validation error—most feature composites already fall back to Gemma/Grok automatically.
