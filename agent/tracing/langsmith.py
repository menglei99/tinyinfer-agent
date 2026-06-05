"""Optional LangSmith integration.

The simplest activation path — recommended:

    pip install langsmith
    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY=...
    export LANGSMITH_PROJECT=tinyinfer-agent     # optional

LangGraph auto-instruments when these env vars are set; we don't have to wrap
or monkey-patch anything ourselves. This module exists only to:

  1. Detect whether the activation is configured.
  2. Print a one-line status banner at CLI startup so the user knows whether
     traces will appear in their dashboard.

If LANGSMITH_API_KEY is unset, everything is a no-op — JSONL tracing via
TraceWriter still works.
"""

from __future__ import annotations

import os


def is_enabled() -> bool:
    """True iff LangSmith env vars are configured and the SDK is importable."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    try:
        import langsmith  # noqa: F401
    except ImportError:
        return False
    return True


def status_line() -> str:
    """One-line status for the CLI banner.

    Returns "disabled" when the user hasn't set things up, otherwise a string
    naming the active project so they know where to look in the dashboard.
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
