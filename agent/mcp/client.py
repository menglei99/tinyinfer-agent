"""Synchronous wrapper around an MCP stdio client session.

LangGraph nodes are synchronous; the MCP SDK is async. This module spins up
a stdio session against our own `agent.mcp.server`, executes a small batch
of tool calls, and tears it down — all behind a blocking API.

For long-lived clients (e.g. a UI streaming many tool calls) a persistent
async client is the right shape. For our pipeline node — "build then run
tests, twice per pipeline" — spinning up a session per node is simpler and
keeps the LangGraph code synchronous.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolCallResult:
    name: str
    ok: bool
    payload: dict


def _server_params(python_exe: Optional[str] = None) -> StdioServerParameters:
    return StdioServerParameters(
        command=python_exe or sys.executable,
        args=["-m", "agent.mcp.server"],
    )


@asynccontextmanager
async def _session(python_exe: Optional[str] = None):
    params = _server_params(python_exe)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _parse_payload(content) -> dict:
    """Decode FastMCP tool output. FastMCP wraps a JSON object as text content."""
    if not content:
        return {}
    item = content[0]
    text = getattr(item, "text", None)
    if text is None:
        return {"raw": repr(item)}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


async def _call_many(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    python_exe: Optional[str] = None,
) -> list[ToolCallResult]:
    out: list[ToolCallResult] = []
    async with _session(python_exe) as session:
        for name, args in calls:
            res = await session.call_tool(name, args)
            payload = _parse_payload(res.content)
            ok = bool(payload.get("ok", True)) and not res.isError
            out.append(ToolCallResult(name=name, ok=ok, payload=payload))
    return out


def call_tools(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    python_exe: Optional[str] = None,
) -> list[ToolCallResult]:
    """Blocking entry point. Runs the supplied (tool_name, args) sequence
    against a freshly-spawned MCP server, returns all results.
    """
    # Use asyncio.run if no loop is running; otherwise fall back to a fresh loop.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_call_many(calls, python_exe=python_exe))
    # Running inside an existing loop is unusual for LangGraph sync nodes
    # but we handle it defensively.
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(_call_many(calls, python_exe=python_exe))
    finally:
        new_loop.close()
