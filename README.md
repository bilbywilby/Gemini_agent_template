# agent-framework — blank Gemini gem template

A resilient async orchestration layer over the unified `google-genai` SDK
(Gemini API on Vertex AI / Gemini Enterprise Agent Platform). This repo is
a **template**: `examples/blank_gem.py` is the starting point for a new
gem — copy it, register your real tools, and fill in the system prompt.

## Why this exists

The `vertexai.generative_models` module (and the rest of the legacy Vertex
AI generative modules) was deprecated 2025-06-24 and its removal date was
2026-06-24. This template is built exclusively on `google-genai`, the SDK
Google now directs all new work to, so it doesn't inherit that removal.

## Layout

```
src/agent_framework/
    __init__.py       public API surface
    agent.py           AgentLoop / AgentConfig / Turn — the ReAct loop
    vertex_client.py   async google-genai wrapper (retry + circuit breaker applied here)
    resilience.py       AsyncCircuitBreaker (TOCTOU-safe, lock-guarded state machine)
    retry.py            RetryPolicy (exponential backoff + full jitter, breaker-aware)
    tools.py             ToolDefinition / ToolRegistry (manual dispatch — AFC is disabled)
    observability.py     AgentMetrics (bounded deque) + TelemetryHooks (decoupled protocol)
examples/blank_gem.py    minimal working gem — copy this to start a new one
tests/                    zero-network pytest suite (resilience, retry, tool dispatch)
```

## Install

```bash
pip install -e ".[dev]"
```

## Auth

This template talks to Gemini via Vertex AI using Application Default
Credentials — no API key is embedded anywhere in the code:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
```

## Run the blank gem

```bash
python examples/blank_gem.py
```

## Test

```bash
pytest
```

The suite is entirely offline — no calls to Gemini are made. Model-call
paths are exercised through `VertexClient`'s retry/circuit-breaker
composition using injected fakes, not live network requests.

## Design notes worth knowing before you extend this

- **Automatic function calling is deliberately disabled.** Tool execution
  goes through `ToolRegistry.dispatch`, not the SDK's internal AFC loop, so
  every tool call gets this framework's timeout, error-surfacing, and
  telemetry — not just the SDK's own retry behavior.
- **Function responses use `role="user"`.** The Gemini `Content` schema has
  no separate `function` role; replies to `function_call` parts are sent
  back as `role="user"` messages containing `function_response` parts.
- **`AgentLoop` does not own conversation storage.** `run()` takes and
  returns `history` explicitly — persist it however fits your gem (DB row,
  session cache, request/response body).
- **Circuit-breaker waits don't consume retry attempts, but are separately
  bounded** (`RetryPolicy.max_circuit_open_waits`) so a circuit that never
  recovers can't block `run()` forever.
