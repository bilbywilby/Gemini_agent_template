"""
agent.py — ReAct-style orchestration loop over VertexClient + ToolRegistry.

The loop is stateless with respect to conversation storage: callers inject
history (a list[Content]) into `run()` and get the updated history back on
the Turn, rather than AgentLoop owning a session. This makes it trivial to
persist/resume conversations from any store (DB row, request payload, ...)
without coupling AgentLoop to a particular persistence mechanism.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from google.genai import types as genai_types

from .observability import AgentMetrics, NullTelemetryHooks, TelemetryHooks
from .tools import ToolRegistry
from .vertex_client import VertexClient


class AgentError(RuntimeError):
    """Raised when the loop cannot complete a turn (exhausted iterations,
    unrecoverable transport failure after retry, etc.)."""


@dataclass(frozen=True)
class AgentConfig:
    """
    Args:
        model: Overrides VertexClient.default_model for this loop's calls.
        system_instruction: Persistent system prompt for every call in the turn.
        max_tool_iterations: Upper bound on model->tool->model round trips
            within a single `run()` call, so a model that keeps requesting
            tools cannot loop forever.
        parallel_tool_calls: Passed through to ToolRegistry.dispatch.
        temperature: Sampling temperature for every call in the turn.
    """

    model: str | None = None
    system_instruction: str | None = None
    max_tool_iterations: int = 6
    parallel_tool_calls: bool = True
    temperature: float = 0.2


@dataclass
class Turn:
    """Result of one AgentLoop.run() call."""

    final_response: str | None
    history: list[genai_types.Content]
    tool_iterations: int
    finish_reason: str | None


class AgentLoop:
    """
    Orchestrates a single logical turn: send history to Gemini, execute any
    requested tool calls, replay results, repeat until the model returns a
    plain text response or max_tool_iterations is reached.
    """

    def __init__(
        self,
        client: VertexClient,
        registry: ToolRegistry,
        config: AgentConfig | None = None,
        metrics: AgentMetrics | None = None,
        telemetry: TelemetryHooks | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.config = config or AgentConfig()
        self.metrics = metrics or AgentMetrics()
        self.telemetry = telemetry or NullTelemetryHooks()

    async def run(
        self,
        message: str,
        history: list[genai_types.Content] | None = None,
    ) -> Turn:
        """
        Run one turn starting from `message`, appended to `history` (an
        empty list is used when history is None — callers own persistence
        of the returned Turn.history for the next call).
        """
        working_history: list[genai_types.Content] = list(history or [])
        working_history.append(VertexClient.user_content(message))

        tool = self.registry.to_genai_tool()
        finish_reason: str | None = None

        for iteration in range(self.config.max_tool_iterations):
            span = self.telemetry.start_span("agent.generate", iteration=iteration)
            start = time.monotonic()
            try:
                result = await self.client.generate(
                    working_history,
                    model=self.config.model,
                    system_instruction=self.config.system_instruction,
                    tools=tool,
                    temperature=self.config.temperature,
                )
            except Exception as exc:
                self.metrics.record("agent.generate", (time.monotonic() - start) * 1000, False)
                self.telemetry.end_span(span, success=False)
                raise AgentError(f"generation failed on iteration {iteration}: {exc}") from exc

            self.metrics.record("agent.generate", (time.monotonic() - start) * 1000, True)
            self.telemetry.end_span(span, success=True, finish_reason=result.finish_reason)
            finish_reason = result.finish_reason

            if not result.function_calls:
                # Plain text turn: append the model's reply and stop.
                if result.text is not None:
                    working_history.append(
                        VertexClient.model_content(genai_types.Part.from_text(text=result.text))
                    )
                return Turn(
                    final_response=result.text,
                    history=working_history,
                    tool_iterations=iteration,
                    finish_reason=finish_reason,
                )

            # Replay the model's function_call turn, then dispatch and
            # replay the function_response turn, before looping again.
            call_parts = [genai_types.Part(function_call=fc) for fc in result.function_calls]
            working_history.append(VertexClient.model_content(*call_parts))

            tool_span = self.telemetry.start_span("agent.tool_dispatch", iteration=iteration)
            tool_start = time.monotonic()
            tool_results = await self.registry.dispatch(
                result.function_calls, parallel=self.config.parallel_tool_calls
            )
            any_error = any(r.is_error for r in tool_results)
            self.metrics.record(
                "agent.tool_dispatch", (time.monotonic() - tool_start) * 1000, not any_error
            )
            self.telemetry.end_span(tool_span, success=not any_error)

            working_history.append(VertexClient.function_response_content(tool_results))

        raise AgentError(
            f"exceeded max_tool_iterations={self.config.max_tool_iterations} without a final response"
        )
