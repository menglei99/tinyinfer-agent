"""MCP integration package.

The server lives in `agent.mcp.server` and is the entry point for the
stdio MCP protocol. We deliberately avoid importing it at package import
time so that running `python -m agent.mcp.server` doesn't trigger the
"already in sys.modules" runtime warning.

Usage:
    python -m agent.mcp.server
    tinyinfer-agent mcp-server
"""
