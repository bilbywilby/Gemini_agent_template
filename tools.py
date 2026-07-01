"""
tools.py — Tool definitions and dispatch for the Gemini agent loop.

Automatic function calling (AFC) in google-genai is deliberately not used
here. AFC hides tool execution inside the SDK's own retry/turn loop, which
means it bypasses this framework's circuit breaker, retry policy, and
telemetry hooks entirely — a tool call that trips the breaker would just
silently retry inside the SDK instead of surfacing through our resilience
layer. Function declarations are still built with google.genai.types so the
wire format matches what Gemini expects; execution is dispatched manually
by AgentLoop with automatic_function_calling disabled.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from google.genai import types as genai_types


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a single tool invocation, ready to fold back into history."""

    name: str
    call_id: str | None
    output: Any
    is_error: bool = False

    def to_function_response_part(self) -> genai_types.Part:
        """
        Build the Part Gemini expects as the reply to a function_call.

        Errors are surfaced inside the `response` payload (as an `error`
        key) rather than raised past this point — Gemini expects a
        function_response part either way, and folding the error into the
        payload lets the model see and react to the failure instead of the
        turn just dying.
        """
        payload: dict[str, Any] = (
            {"error": str(self.output)} if self.is_error else {"result": self.output}
        )
        return genai_types.Part(
            function_response=genai_types.FunctionResponse(
                name=self.name,
                response=payload,
                id=self.call_id,
            )
        )


@dataclass(frozen=True)
class ToolDefinition:
    """
    A single callable tool exposed to the model.

    Args:
        name: Must match `^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$` per the Gemini
            function-calling spec.
        description: Sent to the model verbatim — this is the primary
            signal the model uses to decide when to call the tool, so it
            should state precisely what the tool does and does not do.
        parameters_schema: OpenAPI-subset JSON Schema dict, as required by
            FunctionDeclaration.parameters.
        handler: Async or sync callable executing the tool. Sync callables
            are run in a thread via asyncio.to_thread so a blocking handler
            cannot stall the agent's event loop.
        timeout_seconds: Per-call wall-clock timeout enforced by the
            registry, independent of any timeout the handler itself sets.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[..., Any]
    timeout_seconds: float = 30.0

    def to_function_declaration(self) -> genai_types.FunctionDeclaration:
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )


class ToolRegistry:
    """Holds ToolDefinitions and dispatches calls the model requests."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def to_genai_tool(self) -> genai_types.Tool | None:
        """A single Tool bundling every registered FunctionDeclaration, or
        None if nothing is registered (so callers can omit `tools=` cleanly
        rather than sending an empty declarations list)."""
        if not self._tools:
            return None
        return genai_types.Tool(
            function_declarations=[t.to_function_declaration() for t in self._tools.values()]
        )

    async def _invoke_one(self, call: genai_types.FunctionCall) -> ToolResult:
        tool = self._tools.get(call.name or "")
        if tool is None:
            return ToolResult(
                name=call.name or "<unknown>",
                call_id=call.id,
                output=f"no tool registered with name '{call.name}'",
                is_error=True,
            )

        kwargs: dict[str, Any] = dict(call.args or {})
        try:
            async with asyncio.timeout(tool.timeout_seconds):
                if inspect.iscoroutinefunction(tool.handler):
                    output = await tool.handler(**kwargs)
                else:
                    output = await asyncio.to_thread(tool.handler, **kwargs)
        except TimeoutError:
            return ToolResult(
                name=tool.name,
                call_id=call.id,
                output=f"tool '{tool.name}' timed out after {tool.timeout_seconds}s",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001 - must surface to the model, not crash the turn
            return ToolResult(name=tool.name, call_id=call.id, output=str(exc), is_error=True)

        return ToolResult(name=tool.name, call_id=call.id, output=output, is_error=False)

    async def dispatch(
        self,
        calls: list[genai_types.FunctionCall],
        *,
        parallel: bool = True,
    ) -> list[ToolResult]:
        """
        Execute every requested call and return results in the same order
        the model requested them — order matters because each
        function_response must be matched back to its function_call by
        position/id when replayed into history.

        parallel=True runs independent tool calls concurrently via
        asyncio.gather; set False when tools share mutable state and must
        not interleave.
        """
        if not calls:
            return []
        if parallel:
            return list(await asyncio.gather(*(self._invoke_one(c) for c in calls)))
        return [await self._invoke_one(c) for c in calls]
