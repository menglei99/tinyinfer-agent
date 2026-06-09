"""MCP stdio client session 的同步封装。

LangGraph 节点是同步的；MCP SDK 是 async 的。这个模块对我们自己的
`agent.mcp.server` 拉起一个 stdio session，跑一小批 tool call，再 teardown
—— 全部躲在一个阻塞 API 后面。

如果是长寿命的 client（比如 UI 流式跑很多 tool call），用持久 async client
更合适。对我们的 pipeline 节点 ——"build 一次 + run tests 一次，每个 pipeline
执行两次"—— 每个节点新拉一个 session 更简单，也保持 LangGraph 代码同步。
"""

from __future__ import annotations

import asyncio
import json
import os
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
    # MCP SDK 在 StdioServerParameters.env=None 时只转发一个最小"安全"env
    # 子集 —— 那会把 TINYINFER_DOCKER_IMAGE、TINYINFER_PROJECT_DIR、
    # TINYINFER_BUILD_DIR、LangSmith creds 等全部切掉。这里显式把父进程的
    # env 透传过去，server 子进程看到的配置才和 agent 进程一致。
    return StdioServerParameters(
        command=python_exe or sys.executable,
        args=["-m", "agent.mcp.server"],
        env={k: v for k, v in os.environ.items() if v is not None},
    )


@asynccontextmanager
async def _session(python_exe: Optional[str] = None):
    params = _server_params(python_exe)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _parse_payload(content) -> dict:
    """解码 FastMCP 的 tool 输出。FastMCP 把 JSON object 包成 text content。"""
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
    """阻塞式入口。对一个临时拉起的 MCP server 跑给定的 (tool_name, args)
    序列，返回所有结果。
    """
    # 没有 running loop 就用 asyncio.run；否则 fallback 到一个 fresh loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_call_many(calls, python_exe=python_exe))
    # 在已有 loop 里跑对 LangGraph sync 节点来说不寻常，但做个防御
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(_call_many(calls, python_exe=python_exe))
    finally:
        new_loop.close()
