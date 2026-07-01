"""
observability.py — Metrics and telemetry, decoupled from resilience.py.

AgentMetrics is a plain in-process recorder (bounded deque, no I/O) so it
can be read synchronously from a health-check endpoint. TelemetryHooks is a
separate Protocol so that wiring an external backend (OpenTelemetry, Cloud
Trace, Datadog, ...) never requires resilience.py or agent.py to import a
telemetry SDK directly — they only ever call hook methods on whatever
implementation was injected, including a no-op default.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LatencyRecord:
    """A single timed operation, e.g. one model call or one tool call."""

    operation: str
    duration_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentMetrics:
    """
    Bounded in-memory latency/error tracker.

    Uses deque(maxlen=...) rather than an unbounded list so a long-running
    process (e.g. a Cloud Run service handling many turns) has a fixed
    memory ceiling for metrics regardless of uptime.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._records: deque[LatencyRecord] = deque(maxlen=maxlen)
        self._total_calls = 0
        self._total_errors = 0

    def record(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        **metadata: Any,
    ) -> None:
        self._records.append(
            LatencyRecord(operation=operation, duration_ms=duration_ms, success=success, metadata=metadata)
        )
        self._total_calls += 1
        if not success:
            self._total_errors += 1

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_errors(self) -> int:
        return self._total_errors

    @property
    def error_rate(self) -> float:
        return 0.0 if self._total_calls == 0 else self._total_errors / self._total_calls

    def recent(self, n: int = 50) -> list[LatencyRecord]:
        return list(self._records)[-n:]

    def p50_latency_ms(self, operation: str | None = None) -> float | None:
        return self._percentile(0.50, operation)

    def p95_latency_ms(self, operation: str | None = None) -> float | None:
        return self._percentile(0.95, operation)

    def _percentile(self, pct: float, operation: str | None) -> float | None:
        values = sorted(
            r.duration_ms for r in self._records if operation is None or r.operation == operation
        )
        if not values:
            return None
        idx = min(len(values) - 1, int(len(values) * pct))
        return values[idx]


@runtime_checkable
class TelemetryHooks(Protocol):
    """
    Minimal span/event interface agent.py and tools.py call into. Any
    concrete backend (OpenTelemetry, a no-op stub, a test spy) needs only
    satisfy this shape — no inheritance required.
    """

    def start_span(self, name: str, **attributes: Any) -> Any:
        """Return an opaque span/context handle passed to end_span."""
        ...

    def end_span(self, handle: Any, *, success: bool, **attributes: Any) -> None:
        ...

    def record_event(self, name: str, **attributes: Any) -> None:
        ...


class NullTelemetryHooks:
    """No-op TelemetryHooks implementation — the default when nothing is wired."""

    def start_span(self, name: str, **attributes: Any) -> Any:
        return None

    def end_span(self, handle: Any, *, success: bool, **attributes: Any) -> None:
        return None

    def record_event(self, name: str, **attributes: Any) -> None:
        return None
