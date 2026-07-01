"""
agent_framework — Production-grade async agentic framework for Gemini via
the unified google-genai SDK (Gemini API on Vertex AI / Gemini Enterprise
Agent Platform).

Supported Python: 3.10+
Required: google-genai >= 1.0.0

Example:
    >>> import asyncio
    >>> from agent_framework import AgentLoop, VertexClient, ToolRegistry, AgentConfig
    >>> async def run():
    ...     client = VertexClient(project="my-gcp-project", location="us-central1")
    ...     agent = AgentLoop(client=client, registry=ToolRegistry(), config=AgentConfig())
    ...     turn = await agent.run("What is the capital of France?")
    ...     print(turn.final_response)
    ...     await client.aclose()
    >>> asyncio.run(run())
"""

from __future__ import annotations

from typing import TYPE_CHECKING


def _check_dependencies() -> None:
    """Validate required dependencies at import time before any submodules load."""
    try:
        import google.genai  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "agent_framework requires google-genai. Install with: pip install google-genai"
        ) from e


# Execute the dependency check immediately to guard all subsequent imports.
_check_dependencies()

# Single-source version lookup.
from ._version import __version__
from .agent import AgentConfig, AgentError, AgentLoop, Turn
from .observability import AgentMetrics, LatencyRecord, NullTelemetryHooks, TelemetryHooks
from .resilience import AsyncCircuitBreaker, CircuitOpenError, CircuitState
from .retry import RetryExhaustedError, RetryPolicy
from .tools import ToolDefinition, ToolRegistry, ToolResult
from .vertex_client import GenerationResult, VertexClient

if TYPE_CHECKING:
    from google import genai  # noqa: F401

__all__ = [
    "__version__",
    "AgentConfig",
    "AgentError",
    "AgentLoop",
    "AgentMetrics",
    "AsyncCircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "GenerationResult",
    "LatencyRecord",
    "NullTelemetryHooks",
    "RetryExhaustedError",
    "RetryPolicy",
    "TelemetryHooks",
    "ToolDefinition",
    "ToolRegistry",
    "ToolResult",
    "Turn",
    "VertexClient",
]
