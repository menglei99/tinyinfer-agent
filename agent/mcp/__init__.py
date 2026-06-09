"""MCP 集成包。

Server 在 `agent.mcp.server`，是 stdio MCP 协议的入口。我们故意不在 package
import 时拉它，避免 `python -m agent.mcp.server` 时撞到 "already in sys.modules"
的 runtime warning。

用法：
    python -m agent.mcp.server
    tinyinfer-agent mcp-server
"""
