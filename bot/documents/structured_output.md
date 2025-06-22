# Structured LLM Output Guide

> **Purpose**
> This guide shows how to define your Pydantic schemas **once** and reuse them unchanged with
>
> * **Google Gemini** via `google-generativeai`
> * **OpenAI-compatible clients** via the OpenAI client
>
> It includes:
>
> 1. Shared schema definitions
> 2. End‑to‑end examples (Yes/No gate, Action decision)
> 3. Tips & troubleshooting

---

## 1  Shared Schemas

```python
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field

# --- Scenario A: YES / NO -----------------------------------------
class YesNo(BaseModel):
    answer: Literal["YES", "NO"] = Field(description="Return exactly YES or NO, uppercase")

# --- Scenario B: Choice + Free‑form reason -------------------------
class Action(str, Enum):
    EMAIL = "EMAIL"
    CALL = "CALL"
    ESCALATE = "ESCALATE"

class ActionDecision(BaseModel):
    action: Action = Field(description="Selected action")
    reason: str    = Field(description="Explanation of the choice")
```

---

## 2  Client Usage

### Gemini Client

```python
from typing import Type, TypeVar
from pydantic import BaseModel
from google import genai

T = TypeVar("T", bound=BaseModel)

client = genai.Client(api_key=api_key)

def gemini_call(model: str, prompt, schema: Type[T]) -> T:
    rsp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    if rsp.parsed is None:
        raise ValueError(f"Gemini failed schema → {rsp.text}")
    return rsp.parsed
```

### OpenAI Client

```python
from typing import Type, TypeVar, List, Dict
from pydantic import BaseModel
from openai import OpenAI

T = TypeVar("T", bound=BaseModel)

client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

def openai_call(model: str, messages: List[Dict[str,str]], schema: Type[T]) -> T:
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=schema,
    )
    return completion.choices[0].message.parsed
```

---

## 3  Usage Examples

### 3.1  YES / NO Gate

```python
# Gemini
is_safe = gemini_call("gemini-2.5-flash", "Is 2025 a leap year?", YesNo)
print(is_safe.answer)  # → "NO"

# OpenAI-compatible
msg = [{"role": "user", "content": "Should I bring an umbrella tomorrow?"}]
umbrella = openai_call("grok-3-mini", msg, YesNo)
print(umbrella.answer)
```

### 3.2  Action Decision

```python
customer_text = "Customer contacted twice, furious, demands refund."

# Gemini
decision = gemini_call("gemini-2.5-pro", customer_text, ActionDecision)
print(decision.action, decision.reason)

# OpenAI-compatible, with system role
decision = openai_call(
    "grok-3-mini",
    [
        {"role": "system", "content": "Return JSON only, follow schema."},
        {"role": "user", "content": customer_text},
    ],
    ActionDecision,
)
```

---

## 4  Best Practices & Tips

**Prompting**: Keep user prompts natural; enforce JSON via **system message** + **schema**.

**Required Fields**: Handle missing fields: Gemini → `rsp.parsed is None`; OpenAI → `try/except`.

**Enums & Literals**: Use `enum.Enum` or `Literal[...]`; both providers respect them.

**Schema Changes**: Version your schemas; update client code immutably to avoid breaking callers.

**Token Limits**: Large schemas can raise token counts. Favour `Literal`/`Enum` + concise descriptions.

---

## 5  Troubleshooting

### Gemini returns `None`

* Check that **all required fields** truly exist in the answer.
* Simplify the prompt or add clarifying sentences: *"If any value is unknown, set it to null."*

### OpenAI raises `ValidationError`

* Inspect `completion.choices[0].message.content` to see raw output; adjust prompt or schema.
* For multi‑turn chats, always repeat **"Respond with JSON only"** in the system role before the parse step.
