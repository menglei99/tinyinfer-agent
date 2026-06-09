"""可选的 LangSmith 集成。

最简单的开启路径（推荐）：

    pip install langsmith
    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY=...
    export LANGSMITH_PROJECT=tinyinfer-agent     # 可选

这几个 env var 设上之后 LangGraph 会自动 instrument；我们不用自己 wrap 也不用
monkey-patch。这个 module 只做两件事：

  1. 判断当前配置是否激活了 LangSmith。
  2. 在 CLI 启动时打一行 status banner，让用户知道 trace 会不会进 dashboard。

LANGSMITH_API_KEY 没设时全是 no-op —— TraceWriter 的 JSONL trace 仍然正常工作。
"""

from __future__ import annotations

import os


def is_enabled() -> bool:
    """LangSmith env 都齐 + SDK 装上时返回 True。"""
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    try:
        import langsmith  # noqa: F401
    except ImportError:
        return False
    return True


def status_line() -> str:
    """给 CLI banner 用的一行 status。

    没配置时返回 "disabled"，否则返回当前 project 名（让用户知道去 dashboard
    哪里看）。
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        return "disabled (set LANGSMITH_TRACING=true + LANGSMITH_API_KEY to enable)"
    try:
        import langsmith  # noqa: F401
    except ImportError:
        return "disabled (langsmith package not installed; run `pip install langsmith`)"

    if (os.getenv("LANGSMITH_TRACING", "").lower() not in ("1", "true", "yes")):
        return "key set but LANGSMITH_TRACING is not 'true' — auto-tracing is OFF"
    project = os.getenv("LANGSMITH_PROJECT", "default")
    return f"enabled (project={project})"


__all__ = ["is_enabled", "status_line"]
