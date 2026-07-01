from __future__ import annotations

import asyncio

import pytest

from agent_framework.resilience import AsyncCircuitBreaker, CircuitOpenError, CircuitState


async def _ok() -> str:
    return "ok"


async def _boom() -> str:
    raise ConnectionError("boom")


@pytest.mark.asyncio
async def test_closed_circuit_passes_calls() -> None:
    breaker = AsyncCircuitBreaker("test", failure_threshold=3, open_timeout=0.05)
    assert await breaker(_ok) == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_trips_open_after_threshold() -> None:
    breaker = AsyncCircuitBreaker("test", failure_threshold=2, open_timeout=1.0)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            await breaker(_boom)
    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        await breaker(_ok)


@pytest.mark.asyncio
async def test_half_open_recovers_to_closed() -> None:
    breaker = AsyncCircuitBreaker("test", failure_threshold=1, open_timeout=0.02)
    with pytest.raises(ConnectionError):
        await breaker(_boom)
    assert breaker.state == CircuitState.OPEN

    await asyncio.sleep(0.03)
    assert await breaker(_ok) == "ok"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_trial_failure_reopens() -> None:
    breaker = AsyncCircuitBreaker("test", failure_threshold=1, open_timeout=0.02)
    with pytest.raises(ConnectionError):
        await breaker(_boom)

    await asyncio.sleep(0.03)
    with pytest.raises(ConnectionError):
        await breaker(_boom)
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_only_one_half_open_trial_admitted() -> None:
    breaker = AsyncCircuitBreaker("test", failure_threshold=1, open_timeout=0.01)
    with pytest.raises(ConnectionError):
        await breaker(_boom)
    await asyncio.sleep(0.02)

    async def _slow_ok() -> str:
        await asyncio.sleep(0.05)
        return "ok"

    results = await asyncio.gather(
        breaker(_slow_ok), breaker(_ok), return_exceptions=True
    )
    open_errors = [r for r in results if isinstance(r, CircuitOpenError)]
    assert len(open_errors) == 1
