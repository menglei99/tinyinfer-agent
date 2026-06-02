"""tinyinfer-toolchain MCP server.

Exposes the C++ build & test toolchain over Model Context Protocol so any
MCP-aware client (Claude Desktop, Cursor, our own agent's analyze loop) can
drive cmake/ctest with the same wire format.

Run as a stdio server:

    python -m agent.mcp.server

Or via CLI:

    tinyinfer-agent mcp-server

Register with Claude Desktop by adding the snippet from `mcp_config.json.example`
to your `claude_desktop_config.json`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from agent.tools import cmake_driver


# Project-aware default: assume the server is launched from repo root.
# Override via TINYINFER_PROJECT_DIR / TINYINFER_BUILD_DIR env vars.
_DEFAULT_PROJECT = os.getenv("TINYINFER_PROJECT_DIR", "tinyinfer")
_DEFAULT_BUILD = os.getenv("TINYINFER_BUILD_DIR", "tinyinfer/build")


mcp = FastMCP("tinyinfer-toolchain")


def _resolve(path_str: str) -> Path:
    """Resolve a path, allowing both absolute and repo-relative forms."""
    p = Path(path_str)
    return p if p.is_absolute() else Path.cwd() / p


def _result_to_dict(r: cmake_driver.CommandResult) -> dict:
    return {
        "ok": r.ok,
        "returncode": r.returncode,
        "stdout_tail": _tail(r.stdout, 4000),
        "stderr_tail": _tail(r.stderr, 4000),
        "cmd": r.cmd,
    }


def _tail(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return "...[truncated]...\n" + s[-n:]


# ---------- tools ----------


@mcp.tool()
def cmake_available() -> dict:
    """Check whether cmake is installed on the host PATH.

    Use this before any other tool to bail out early on a host without cmake.
    """
    return {"available": cmake_driver.cmake_available()}


@mcp.tool()
def cmake_configure(
    project_dir: Optional[str] = None,
    build_dir: Optional[str] = None,
) -> dict:
    """Run `cmake -S <project_dir> -B <build_dir> -DCMAKE_BUILD_TYPE=Release`.

    Args:
        project_dir: path to the C++ project (defaults to env or 'tinyinfer').
        build_dir:   path for cmake build artefacts.
    Returns: {ok, returncode, stdout_tail, stderr_tail, cmd}
    """
    proj = _resolve(project_dir or _DEFAULT_PROJECT)
    bld = _resolve(build_dir or _DEFAULT_BUILD)
    return _result_to_dict(cmake_driver.configure(proj, bld))


@mcp.tool()
def cmake_build(
    build_dir: Optional[str] = None,
    target: Optional[str] = None,
) -> dict:
    """Run `cmake --build <build_dir> --config Release [--target <target>]`.

    Args:
        build_dir: path for cmake build artefacts.
        target:    optional single target name (e.g. 'test_matmul_baseline').
    """
    bld = _resolve(build_dir or _DEFAULT_BUILD)
    return _result_to_dict(cmake_driver.build(bld, target=target))


@mcp.tool()
def ctest_run(
    build_dir: Optional[str] = None,
    test_filter: Optional[str] = None,
) -> dict:
    """Run `ctest --test-dir <build_dir> --output-on-failure [-R <test_filter>]`.

    Use test_filter (a regex) to narrow which tests run, e.g. 'matmul'.
    """
    bld = _resolve(build_dir or _DEFAULT_BUILD)
    return _result_to_dict(cmake_driver.ctest(bld, test_filter=test_filter))


@mcp.tool()
def list_tests(build_dir: Optional[str] = None) -> dict:
    """List ctest test names registered in <build_dir>.

    Useful for discovering which generated tests are currently in the build.
    """
    from agent.tools.cmake_driver import _run  # internal helper, OK for our own server.

    bld = _resolve(build_dir or _DEFAULT_BUILD)
    res = _run(["ctest", "--test-dir", str(bld), "-N"])
    # Parse "Test #N: name" lines.
    names: list[str] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("Test #"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                names.append(parts[1].strip())
    return {"ok": res.ok, "tests": names, "raw_tail": _tail(res.stdout, 1500)}


# ---------- entrypoint ----------


def main() -> None:
    """Entry point used by `python -m agent.mcp.server` and CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
