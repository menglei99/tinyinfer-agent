"""MCP server 的 smoke 测试。

不拉子进程（慢、Windows CI 上 flaky）。直接调底下 tool 实现。MCP wrapper
只是 agent.tools.cmake_driver 的薄壳，这样高覆盖、低成本。
"""

import sys
from pathlib import Path


def test_mcp_server_module_imports():
    """The server module must import cleanly so `python -m agent.mcp.server` works."""
    import importlib

    mod = importlib.import_module("agent.mcp.server")
    assert hasattr(mod, "mcp")
    assert hasattr(mod, "main")


def test_mcp_tools_registered():
    """FastMCP exposes registered tools — verify ours are there."""
    from agent.mcp.server import mcp

    # FastMCP API: list registered tools via the internal manager.
    import asyncio

    async def _names():
        tools = await mcp.list_tools()
        return {t.name for t in tools}

    names = asyncio.run(_names())
    expected = {"cmake_available", "cmake_configure", "cmake_build", "ctest_run", "list_tests"}
    missing = expected - names
    assert not missing, f"missing tools: {missing}"


def test_cmake_available_returns_bool():
    """Even without cmake installed, the tool should return cleanly."""
    from agent.mcp.server import cmake_available

    out = cmake_available.fn() if hasattr(cmake_available, "fn") else cmake_available()
    assert "available" in out
    assert isinstance(out["available"], bool)
