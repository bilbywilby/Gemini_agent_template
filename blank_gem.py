"""
examples/blank_gem.py — Starting point for a new Gemini "gem" built on
agent_framework. Copy this file, rename the tool(s), and fill in real logic.

Requires environment configured for Vertex AI auth (Application Default
Credentials) and:
    export GOOGLE_CLOUD_PROJECT=your-project-id
    export GOOGLE_CLOUD_LOCATION=us-central1
"""

from __future__ import annotations

import asyncio
import os

from agent_framework import AgentConfig, AgentLoop, ToolDefinition, ToolRegistry, VertexClient


def echo(text: str) -> str:
    """Example synchronous tool handler — replace with real logic."""
    return f"echo: {text}"


async def main() -> None:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echoes back the given text, prefixed with 'echo: '.",
            parameters_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to echo."}},
                "required": ["text"],
            },
            handler=echo,
        )
    )

    client = VertexClient(project=project, location=location, model="gemini-2.5-flash")
    agent = AgentLoop(
        client=client,
        registry=registry,
        config=AgentConfig(
            system_instruction="You are a blank gem template. Use the echo tool when asked to echo something.",
        ),
    )

    try:
        turn = await agent.run("Use the echo tool to echo 'hello gem'.")
        print(turn.final_response)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
