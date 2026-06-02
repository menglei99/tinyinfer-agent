"""CLI entry point.

Usage:
    tinyinfer-agent analyze --diff demo/diffs/sample_matmul.diff
    tinyinfer-agent analyze --diff demo/diffs/sample_matmul.diff --mock
    tinyinfer-agent build-and-test
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Windows + GBK locale chokes on rich's unicode glyphs. Force UTF-8 stdout
# before rich imports its Console.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.graph import build_graph
from agent.llm import get_llm_client
from agent.state import AgentState
from agent.tools import cmake_driver
from agent.tracing import TraceWriter

app = typer.Typer(help="tinyinfer-agent — AI inference framework regression test agent")
console = Console(force_terminal=True, legacy_windows=False)


@app.command()
def analyze(
    diff: Path = typer.Option(..., "--diff", "-d", exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("demo/generated_tests"), "--output-dir", "-o", help="Where to write generated .cpp"
    ),
    trace: Optional[Path] = typer.Option(None, "--trace", "-t", help="Trace file path"),
    mock: bool = typer.Option(False, "--mock", help="Force mock LLM regardless of env"),
    show_report: bool = typer.Option(True, "--show-report/--no-report"),
    execute: bool = typer.Option(
        False, "--execute/--no-execute",
        help="Drive cmake configure/build/ctest via the MCP toolchain after generation",
    ),
):
    """Analyse a git diff and emit regression test C++ source."""

    if mock:
        os.environ["LLM_PROVIDER"] = "mock"

    llm = get_llm_client()
    provider = os.getenv("LLM_PROVIDER", "mock")
    console.print(Panel.fit(f"LLM provider: [bold]{provider}[/bold]", title="setup"))

    diff_text = diff.read_text(encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)

    with TraceWriter(trace) as tw:
        tw.write("run.start", diff_path=str(diff), provider=provider, execute=execute)

        graph = build_graph(llm=llm)
        initial: AgentState = {
            "diff": diff_text,
            "diff_path": str(diff),
            "trace_id": tw.trace_id,
            "critic_iterations": 0,
            "execute_requested": execute,
        }

        # Stream events to the trace; LangGraph emits one update per node.
        final_state: AgentState = initial
        for chunk in graph.stream(initial, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                tw.write("node.update", node=node_name, keys=list(node_update.keys()))
                # Merge so we have the running state for emitting files.
                _merge_state(final_state, node_update)

        # Emit generated test files to disk.
        emitted = _emit_tests(final_state, output_dir)
        for path in emitted:
            tw.write("file.emit", path=str(path))

        tw.write("run.end", num_tests=len(final_state.get("generated_tests", [])))

    if show_report:
        report_md = final_state.get("report_markdown", "")
        if report_md:
            console.print(Markdown(report_md))

    console.print(
        f"\n[green]OK[/green] wrote {len(emitted)} file(s) to "
        f"[cyan]{output_dir}[/cyan], trace at [cyan]{trace or 'traces/'}[/cyan]"
    )


@app.command("build-and-test")
def build_and_test(
    project_dir: Path = typer.Option(Path("tinyinfer"), "--project"),
    build_dir: Path = typer.Option(Path("tinyinfer/build"), "--build-dir"),
):
    """Configure, build, and run the C++ test suite (requires cmake on PATH)."""
    if not cmake_driver.cmake_available():
        console.print("[red]cmake not found on PATH[/red]")
        raise typer.Exit(1)

    cfg = cmake_driver.configure(project_dir, build_dir)
    if not cfg.ok:
        console.print(Panel(cfg.stderr or cfg.stdout, title="cmake configure failed"))
        raise typer.Exit(cfg.returncode)
    console.print("[green]OK[/green] configure ok")

    bld = cmake_driver.build(build_dir)
    if not bld.ok:
        console.print(Panel(bld.stderr or bld.stdout, title="build failed"))
        raise typer.Exit(bld.returncode)
    console.print("[green]OK[/green] build ok")

    res = cmake_driver.ctest(build_dir)
    console.print(res.stdout)
    if not res.ok:
        console.print("[red]FAIL tests failed[/red]")
        raise typer.Exit(res.returncode)
    console.print("[green]OK[/green] ctest ok")


@app.command("mcp-server")
def mcp_server_cmd():
    """Launch the tinyinfer-toolchain MCP stdio server.

    Run this manually or register it with Claude Desktop / Cursor via
    `mcp_config.json.example`. The server speaks Model Context Protocol over
    stdin/stdout — it should be launched by an MCP client, not interactively.
    """
    # Heads-up to anyone who runs it from a shell by accident.
    console.print(
        "[yellow]MCP server starting on stdio — this is meant to be launched by an MCP client.[/yellow]",
        end="\n",
    )
    from agent.mcp.server import main as mcp_main

    mcp_main()


# ---------- helpers ----------


def _merge_state(running: dict, update: dict) -> None:
    for k, v in update.items():
        if isinstance(v, list) and isinstance(running.get(k), list):
            running[k] = running[k] + v
        else:
            running[k] = v


def _emit_tests(state: dict, output_dir: Path) -> list[Path]:
    """Write each unique cpp_source to disk. One file per (op, suffix)."""
    seen: dict[str, str] = {}
    for tc in state.get("generated_tests", []) or []:
        if not tc.cpp_source:
            continue
        suffix = ""
        if isinstance(tc.inputs, dict):
            raw = tc.inputs.get("file_suffix")
            if isinstance(raw, str) and raw:
                suffix = f"_{raw}"
        key = f"test_{tc.op_name}{suffix}_generated.cpp"
        seen[key] = tc.cpp_source
    out = []
    for fname, src in seen.items():
        path = output_dir / fname
        path.write_text(src, encoding="utf-8")
        out.append(path)
    return out


def main() -> None:  # for `python -m agent.cli`
    app()


if __name__ == "__main__":
    sys.exit(app() or 0)
