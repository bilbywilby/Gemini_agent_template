"""
resilience.py — Async circuit breaker for outbound Gemini / tool calls.

Design notes:
    - State transitions are guarded by a single asyncio.Lock to close a
      TOCTOU window: without the lock, two concurrent callers can both read
      state as HALF_OPEN and both be admitted as the "trial" call, which
      defeats the purpose of half-open (only one trial call should be live
      at a time).
    - The breaker is a plain async callable: `await breaker(fn, *args, **kwargs)`.
      This lets it wrap both `asyncio.to_thread(...)` targets and native
      coroutine functions without a different code path for each.
    - Failure counting only counts exceptions raised by the wrapped call,
      not cancellation. A cancelled call is neither a success nor a failure
      of the downstream system.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, name: str, opened_at: float, retry_after: float) -> None:
        self.name = name
        self.opened_at = opened_at
        self.retry_after = retry_after
        super().__init__(
            f"circuit '{name}' is open; retry after {retry_after:.1f}s"
        )


@dataclass
class _BreakerStats:
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    opened_at: float | None = None
    last_error: BaseException | None = None
    total_calls: int = 0
    total_failures: int = 0
    total_rejections: int = 0


class AsyncCircuitBreaker:
    """
    A three-state (CLOSED / OPEN / HALF_OPEN) async circuit breaker.

    Args:
        name: Identifier used in error messages and metrics.
        failure_threshold: Consecutive failures in CLOSED state before
            tripping to OPEN.
        open_timeout: Seconds to remain OPEN before allowing a single
            HALF_OPEN trial call.
        half_open_success_threshold: Consecutive successes in HALF_OPEN
            required to close the circuit again.
        failure_predicate: Optional callable to decide whether a raised
            exception counts as a circuit failure (e.g. to exclude 4xx
            client errors from tripping the breaker while counting 5xx and
            network errors). Defaults to counting every exception.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        open_timeout: float = 30.0,
        half_open_success_threshold: int = 1,
        failure_predicate: Callable[[BaseException], bool] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if half_open_success_threshold < 1:
            raise ValueError("half_open_success_threshold must be >= 1")

        self.name = name
        self.failure_threshold = failure_threshold
        self.open_timeout = open_timeout
        self.half_open_success_threshold = half_open_success_threshold
        self._failure_predicate = failure_predicate or (lambda _exc: True)

        self._state = CircuitState.CLOSED
        self._stats = _BreakerStats()
        self._lock = asyncio.Lock()
        # Only one HALF_OPEN trial call may be in flight at a time.
        self._half_open_trial_in_flight = False

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> _BreakerStats:
        return self._stats

    async def _admit(self) -> bool:
        """
        Decide whether to admit a call, transitioning OPEN -> HALF_OPEN when
        the timeout has elapsed. Returns True if the call may proceed.
        Must be called under self._lock.
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            assert self._stats.opened_at is not None
            elapsed = time.monotonic() - self._stats.opened_at
            if elapsed < self.open_timeout:
                return False
            self._state = CircuitState.HALF_OPEN
            self._half_open_trial_in_flight = False

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_trial_in_flight:
                return False
            self._half_open_trial_in_flight = True
            return True

        return False

    async def _on_success(self) -> None:
        async with self._lock:
            self._stats.consecutive_failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._stats.consecutive_successes += 1
                self._half_open_trial_in_flight = False
                if self._stats.consecutive_successes >= self.half_open_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._stats.opened_at = None
                    self._stats.consecutive_successes = 0
            else:
                self._stats.consecutive_successes += 1

    async def _on_failure(self, exc: BaseException) -> None:
        async with self._lock:
            self._stats.total_failures += 1
            self._stats.last_error = exc
            self._stats.consecutive_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                # Trial call failed — reopen immediately, reset the timer.
                self._half_open_trial_in_flight = False
                self._state = CircuitState.OPEN
                self._stats.opened_at = time.monotonic()
                self._stats.consecutive_failures = 0
                return

            self._stats.consecutive_failures += 1
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._stats.opened_at = time.monotonic()

    async def __call__(
        self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute `fn(*args, **kwargs)` through the breaker.

        Raises CircuitOpenError without calling fn if the circuit rejects
        the call. Propagates whatever fn raises after recording it.
        """
        async with self._lock:
            admitted = await self._admit()
            if not admitted:
                self._stats.total_rejections += 1
                opened_at = self._stats.opened_at or time.monotonic()
                retry_after = max(0.0, self.open_timeout - (time.monotonic() - opened_at))
                raise CircuitOpenError(self.name, opened_at, retry_after)
            self._stats.total_calls += 1

        try:
            result = await fn(*args, **kwargs)
        except asyncio.CancelledError:
            # Cancellation is not a circuit failure; release the half-open
            # trial slot so a future call can retry.
            async with self._lock:
                self._half_open_trial_in_flight = False
            raise
        except BaseException as exc:  # noqa: BLE001 - breaker must see everything
            if self._failure_predicate(exc):
                await self._on_failure(exc)
            else:
                await self._on_success()
            raise
        else:
            await self._on_success()
            return result

    def reset(self) -> None:
        """Force the breaker back to CLOSED. Intended for tests only."""
        self._state = CircuitState.CLOSED
        self._stats = _BreakerStats()
        self._half_open_trial_in_flight = False
