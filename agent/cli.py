"""CLI 入口。

用法：
    tinyinfer-agent analyze --diff demo/diffs/sample_matmul.diff
    tinyinfer-agent analyze --diff demo/diffs/sample_matmul.diff --mock
    tinyinfer-agent build-and-test
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Windows + GBK locale 会被 rich 的 unicode glyph 噎住。在 rich 拉 Console 之前
# 强制 UTF-8 stdout。
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

app = typer.Typer(help="tinyinfer-agent —— AI 推理框架回归测试 agent")
console = Console(force_terminal=True, legacy_windows=False)


@app.command()
def analyze(
    diff: Path = typer.Option(..., "--diff", "-d", exists=True, readable=True),
    output_dir: Path = typer.Option(
        Path("demo/generated_tests"), "--output-dir", "-o", help="生成的 .cpp 写到哪"
    ),
    trace: Optional[Path] = typer.Option(None, "--trace", "-t", help="trace 文件路径"),
    mock: bool = typer.Option(False, "--mock", help="不管 env，强制用 mock LLM"),
    show_report: bool = typer.Option(True, "--show-report/--no-report"),
    execute: bool = typer.Option(
        False, "--execute/--no-execute",
        help="生成完之后通过 MCP toolchain 跑 cmake configure/build/ctest",
    ),
    rag: bool = typer.Option(
        True, "--rag/--no-rag",
        help="构造 hybrid retriever，把 critic / generator ground 到检索到的 corpus 片段上。",
    ),
):
    """解析一个 git diff，输出回归测试 C++ 源码。"""

    if mock:
        os.environ["LLM_PROVIDER"] = "mock"

    llm = get_llm_client()
    provider = os.getenv("LLM_PROVIDER", "mock")
    console.print(Panel.fit(f"LLM provider: [bold]{provider}[/bold]", title="setup"))

    from agent.tracing.langsmith import status_line as _ls_status

    console.print(f"LangSmith: [dim]{_ls_status()}[/dim]")

    retriever = None
    if rag:
        try:
            # 懒 import，让 base 安装（没装 [rag] extra）在 module load 时不挂掉
            from agent.rag import build_default_retriever
            from agent.llm import get_embedding_client

            retriever = build_default_retriever(embedder=get_embedding_client())
            console.print(
                f"[green]OK[/green] retriever built ({retriever.corpus_size} docs)"
            )
        except ImportError as exc:
            console.print(f"[yellow]RAG disabled: {exc}[/yellow]")

    diff_text = diff.read_text(encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)

    with TraceWriter(trace) as tw:
        tw.write("run.start", diff_path=str(diff), provider=provider, execute=execute, rag=bool(retriever))

        graph = build_graph(llm=llm, retriever=retriever)
        initial: AgentState = {
            "diff": diff_text,
            "diff_path": str(diff),
            "trace_id": tw.trace_id,
            "critic_iterations": 0,
            "execute_requested": execute,
        }

        # 把每个节点的事件流写进 trace；LangGraph 每个节点 emit 一个 update
        final_state: AgentState = initial
        for chunk in graph.stream(initial, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                tw.write("node.update", node=node_name, keys=list(node_update.keys()))
                # merge 一下让 emitting 文件时拿到完整 running state
                _merge_state(final_state, node_update)

        # 把生成的测试文件落盘
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
    """跑 cmake configure / build / ctest（需要 cmake 在 PATH 上）。"""
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
    """启动 tinyinfer-toolchain MCP stdio server。

    手动跑或者通过 `mcp_config.json.example` 注册到 Claude Desktop / Cursor。
    Server 在 stdin/stdout 上讲 Model Context Protocol —— 应该被 MCP client
    拉起来，不要在终端里交互式跑。
    """
    # 给从 shell 误启动的人一个提醒
    console.print(
        "[yellow]MCP server starting on stdio — 这个应该是被 MCP 客户端拉起来跑的。[/yellow]",
        end="\n",
    )
    from agent.mcp.server import main as mcp_main

    mcp_main()


# ---------- helper ----------


def _merge_state(running: dict, update: dict) -> None:
    for k, v in update.items():
        if isinstance(v, list) and isinstance(running.get(k), list):
            running[k] = running[k] + v
        else:
            running[k] = v


def _emit_tests(state: dict, output_dir: Path) -> list[Path]:
    """每段 unique cpp_source 落盘，每个 (op, suffix) 一个文件。"""
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


def main() -> None:  # 给 `python -m agent.cli` 用
    app()


if __name__ == "__main__":
    sys.exit(app() or 0)
