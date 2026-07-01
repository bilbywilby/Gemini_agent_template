"""
retry.py — Circuit-breaker-aware exponential backoff with full jitter.

A plain retry decorator is unsafe to compose with a circuit breaker: if the
breaker opens mid-retry-loop, a naive retry will keep sleeping and retrying
against a circuit it can never pass, burning the full retry budget on
guaranteed CircuitOpenError rejections instead of failing fast. RetryPolicy
special-cases CircuitOpenError: it does not consume a retry attempt for it,
and instead sleeps for the breaker's own `retry_after` hint (bounded by the
policy's own max_delay) before trying again.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

from .resilience import CircuitOpenError

T = TypeVar("T")


class RetryExhaustedError(RuntimeError):
    """Raised when every attempt failed. Chains the final underlying error."""


@dataclass(frozen=True)
class RetryPolicy:
    """
    Exponential backoff with full jitter (AWS-style: delay = uniform(0, cap)).

    Args:
        max_attempts: Total attempts including the first, non-retry call.
        base_delay: Seconds for the first backoff interval before jitter.
        max_delay: Upper bound on any single sleep, including the
            circuit-breaker retry_after wait.
        multiplier: Exponential growth factor applied per attempt.
        retryable_exceptions: Exception types that trigger a retry. Any
            other exception propagates immediately on first occurrence.
        max_circuit_open_waits: Upper bound on how many times this call will
            wait out an open circuit before giving up. Circuit-open waits do
            not consume `max_attempts`, so without an independent cap a
            circuit that never recovers would make `run()` block forever.
    """

    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 20.0
    multiplier: float = 2.0
    retryable_exceptions: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
        OSError,
    )
    max_circuit_open_waits: int = 3

    def _backoff_seconds(self, attempt: int) -> float:
        """Full jitter: uniform(0, min(max_delay, base * multiplier**attempt))."""
        cap = min(self.max_delay, self.base_delay * (self.multiplier ** attempt))
        return random.uniform(0.0, cap)

    async def run(
        self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Call `fn(*args, **kwargs)`, retrying on `retryable_exceptions` and on
        CircuitOpenError (which does not count against max_attempts).

        Raises RetryExhaustedError wrapping the last exception once
        max_attempts is exhausted for a retryable error. Non-retryable
        exceptions propagate unmodified on their first occurrence.
        """
        last_exc: BaseException | None = None
        attempt = 0
        circuit_waits = 0

        while attempt < self.max_attempts:
            try:
                return await fn(*args, **kwargs)
            except CircuitOpenError as exc:
                # Does not consume an attempt: retrying against an open
                # circuit is not the same failure mode the policy exists to
                # smooth over, and burning attempts here just guarantees
                # RetryExhaustedError immediately after the breaker recovers.
                # Bounded separately by max_circuit_open_waits so a circuit
                # that never closes cannot block run() forever.
                circuit_waits += 1
                last_exc = exc
                if circuit_waits > self.max_circuit_open_waits:
                    break
                wait = min(exc.retry_after, self.max_delay)
                await asyncio.sleep(wait)
                continue
            except self.retryable_exceptions as exc:
                last_exc = exc
                attempt += 1
                if attempt >= self.max_attempts:
                    break
                await asyncio.sleep(self._backoff_seconds(attempt))

        assert last_exc is not None
        raise RetryExhaustedError(
            f"exhausted {self.max_attempts} attempts: {last_exc!r}"
        ) from last_exc
