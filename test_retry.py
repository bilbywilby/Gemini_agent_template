from __future__ import annotations

import pytest

from agent_framework.resilience import AsyncCircuitBreaker, CircuitOpenError
from agent_framework.retry import RetryExhaustedError, RetryPolicy


@pytest.mark.asyncio
async def test_succeeds_without_retry() -> None:
    policy = RetryPolicy(max_attempts=3, base_delay=0.001, max_delay=0.01)

    async def ok() -> str:
        return "ok"

    assert await policy.run(ok) == "ok"


@pytest.mark.asyncio
async def test_retries_then_succeeds() -> None:
    policy = RetryPolicy(max_attempts=3, base_delay=0.001, max_delay=0.01)
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient")
        return "ok"

    assert await policy.run(flaky) == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_exhausts_and_raises() -> None:
    policy = RetryPolicy(max_attempts=2, base_delay=0.001, max_delay=0.01)

    async def always_fails() -> str:
        raise ConnectionError("permanent")

    with pytest.raises(RetryExhaustedError):
        await policy.run(always_fails)


@pytest.mark.asyncio
async def test_non_retryable_propagates_immediately() -> None:
    policy = RetryPolicy(max_attempts=5, base_delay=0.001, max_delay=0.01)

    async def type_error() -> str:
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        await policy.run(type_error)


@pytest.mark.asyncio
async def test_circuit_open_waits_do_not_consume_attempts_but_are_bounded() -> None:
    breaker = AsyncCircuitBreaker("t", failure_threshold=1, open_timeout=100.0)

    async def boom() -> str:
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        await breaker(boom)

    policy = RetryPolicy(max_attempts=5, base_delay=0.001, max_delay=0.01, max_circuit_open_waits=1)

    async def blocked() -> str:
        return "unreachable"

    with pytest.raises(RetryExhaustedError) as exc_info:
        await policy.run(breaker, blocked)
    assert isinstance(exc_info.value.__cause__, CircuitOpenError)
