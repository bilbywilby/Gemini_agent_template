from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_framework.tools import ToolDefinition, ToolRegistry


def _call(name: str, call_id: str, **args: object) -> SimpleNamespace:
    """Duck-typed stand-in for google.genai.types.FunctionCall — ToolRegistry
    only reads .name, .args, and .id, so a real FunctionCall isn't required
    for these tests."""
    return SimpleNamespace(name=name, args=args, id=call_id)


@pytest.mark.asyncio
async def test_dispatch_sync_handler() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="add",
            description="Adds two integers.",
            parameters_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            handler=lambda a, b: a + b,
        )
    )

    results = await registry.dispatch([_call("add", "1", a=2, b=3)])
    assert len(results) == 1
    assert results[0].output == 5
    assert not results[0].is_error


@pytest.mark.asyncio
async def test_dispatch_async_handler_in_parallel() -> None:
    order: list[str] = []

    async def slow(tag: str) -> str:
        await asyncio.sleep(0.02)
        order.append(tag)
        return tag

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="slow", description="d", parameters_schema={"type": "object"}, handler=slow)
    )

    start = asyncio.get_event_loop().time()
    results = await registry.dispatch(
        [_call("slow", "1", tag="a"), _call("slow", "2", tag="b")], parallel=True
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert {r.output for r in results} == {"a", "b"}
    assert elapsed < 0.035  # both ran concurrently, not serially (2 * 0.02)


@pytest.mark.asyncio
async def test_unknown_tool_surfaces_as_error_not_exception() -> None:
    registry = ToolRegistry()
    results = await registry.dispatch([_call("missing", "1")])
    assert results[0].is_error
    assert "no tool registered" in str(results[0].output)


@pytest.mark.asyncio
async def test_handler_exception_surfaces_as_error_not_exception() -> None:
    def broken(**_: object) -> None:
        raise RuntimeError("handler exploded")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="broken", description="d", parameters_schema={"type": "object"}, handler=broken)
    )
    results = await registry.dispatch([_call("broken", "1")])
    assert results[0].is_error
    assert "handler exploded" in str(results[0].output)


@pytest.mark.asyncio
async def test_timeout_surfaces_as_error() -> None:
    async def hangs() -> None:
        await asyncio.sleep(10)

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="hangs", description="d", parameters_schema={"type": "object"}, handler=hangs, timeout_seconds=0.01
        )
    )
    results = await registry.dispatch([_call("hangs", "1")])
    assert results[0].is_error
    assert "timed out" in str(results[0].output)


def test_duplicate_registration_rejected() -> None:
    registry = ToolRegistry()
    tool = ToolDefinition(name="x", description="d", parameters_schema={"type": "object"}, handler=lambda: None)
    registry.register(tool)
    with pytest.raises(ValueError):
        registry.register(tool)
