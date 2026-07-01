"""
vertex_client.py — Thin async wrapper around google.genai.Client.

Uses the unified `google-genai` SDK (`pip install google-genai`), not the
deprecated `vertexai.generative_models` module — Google deprecated that
module on 2025-06-24 with removal effective 2026-06-24. This wrapper only
ever talks to `client.aio.models`, so every call in this framework is
already async-native rather than a thread-wrapped sync call.

Circuit breaking and retry are applied here, at the single chokepoint every
model call passes through, rather than in agent.py — that keeps AgentLoop
free of transport concerns and means any caller of VertexClient.generate
gets resilience for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types as genai_types

from .resilience import AsyncCircuitBreaker
from .retry import RetryPolicy


@dataclass(frozen=True)
class GenerationResult:
    """Normalized result of a single generate_content call."""

    text: str | None
    function_calls: list[genai_types.FunctionCall]
    raw_response: genai_types.GenerateContentResponse
    finish_reason: str | None


class VertexClient:
    """
    Async Gemini client for the Gemini API on Vertex AI (Gemini Enterprise
    Agent Platform), wired with circuit breaking and retry.

    Args:
        project: GCP project ID for quota/billing attribution.
        location: Vertex region, e.g. 'us-central1'. Use 'global' for
            models only available on the global endpoint.
        model: Default model ID used when a call site doesn't override it.
        circuit_breaker: Injected so multiple VertexClient instances can
            optionally share one breaker's state (e.g. per-project rather
            than per-client). A fresh breaker is created if omitted.
        retry_policy: Injected for the same reason; a sane default is used
            if omitted.
    """

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash",
        circuit_breaker: AsyncCircuitBreaker | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.project = project
        self.location = location
        self.default_model = model
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._breaker = circuit_breaker or AsyncCircuitBreaker(
            name=f"vertex_client:{project}:{location}",
            failure_threshold=5,
            open_timeout=30.0,
        )
        self._retry = retry_policy or RetryPolicy()

    async def aclose(self) -> None:
        """Release the underlying async HTTP client's connections."""
        await self._client.aio.aclose()

    async def generate(
        self,
        contents: list[genai_types.Content],
        *,
        model: str | None = None,
        system_instruction: str | None = None,
        tools: genai_types.Tool | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> GenerationResult:
        """
        Issue one generate_content call, guarded by retry + circuit breaker.

        automatic_function_calling is always disabled: tool execution is the
        caller's (AgentLoop's) responsibility so that every tool call flows
        through this framework's own dispatch, timeout, and telemetry path
        instead of the SDK's internal AFC loop.
        """
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=[tools] if tools is not None else None,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

        async def _call() -> genai_types.GenerateContentResponse:
            return await self._client.aio.models.generate_content(
                model=model or self.default_model,
                contents=contents,
                config=config,
            )

        response = await self._retry.run(self._breaker, _call)
        return self._to_result(response)

    @staticmethod
    def _to_result(response: genai_types.GenerateContentResponse) -> GenerationResult:
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = (
            candidate.finish_reason.value
            if candidate is not None and candidate.finish_reason is not None
            else None
        )
        function_calls = list(response.function_calls or [])
        return GenerationResult(
            text=response.text,
            function_calls=function_calls,
            raw_response=response,
            finish_reason=finish_reason,
        )

    @staticmethod
    def user_content(text: str) -> genai_types.Content:
        """Build a role='user' Content block — the correct role for turns
        originating from the human or from tool-result replay, per the
        Gemini Content schema."""
        return genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=text)])

    @staticmethod
    def model_content(*parts: genai_types.Part) -> genai_types.Content:
        """Build a role='model' Content block for assistant turns being
        replayed back into history (e.g. a prior function_call turn)."""
        return genai_types.Content(role="model", parts=list(parts))

    @staticmethod
    def function_response_content(results: list[Any]) -> genai_types.Content:
        """
        Build the role='user' Content block carrying function_response
        parts. Per the Gemini API, function responses are sent back with
        role='user', not role='function' or role='model' — there is no
        separate function role in this schema.
        """
        return genai_types.Content(role="user", parts=[r.to_function_response_part() for r in results])
