# Devlog 01 — Wrapping the C++ toolchain in an MCP server

## Why now

Week 1 left `agent/tools/cmake_driver.py` as a thin Python wrapper around
`subprocess.run`. The Agent could call into it. But nothing else could.

That's the wrong shape for a project whose pitch is "production-grade AI
application engineering". In a real org:

- Engineers want to drive the same `cmake_configure / cmake_build / ctest_run`
  from Claude Desktop or Cursor while they're code-reviewing.
- The Agent should call these tools through a formal protocol so we can later
  swap implementations (local subprocess → remote build farm) without touching
  agent code.
- We don't want to invent yet another tool format on top of LangChain's. MCP
  is the standard.

So Week 2 starts by lifting the C++ toolchain into an MCP stdio server.

## What I built

`agent/mcp/server.py` exposes 5 tools via `FastMCP`:

| Tool | Wraps |
|------|-------|
| `cmake_available` | `shutil.which('cmake')` |
| `cmake_configure` | `cmake -S … -B … -DCMAKE_BUILD_TYPE=Release` |
| `cmake_build`     | `cmake --build … [--target …]` |
| `ctest_run`       | `ctest --test-dir … -R …` |
| `list_tests`      | `ctest -N` + parse |

Three things I deliberately did:

1. **Defaults via env vars** (`TINYINFER_PROJECT_DIR`, `TINYINFER_BUILD_DIR`)
   so the Claude Desktop config can pin them once and the LLM can call tools
   with no arguments.
2. **Tail-truncate `stdout` / `stderr`** in the tool result. Build logs can
   blow MCP message limits; a 4 KB tail is enough to debug.
3. **Same underlying primitive as the agent** (`agent.tools.cmake_driver`).
   The Python agent and the MCP-launched Claude Desktop share one
   implementation — no drift.

## Smoke test

I exercised the server two ways:

1. **In-process** (via pytest): import the FastMCP instance, call
   `await mcp.list_tools()`, and call individual tool functions directly. This
   gives high coverage without the cost of spawning subprocesses on Windows
   CI.
2. **Real stdio handshake**: spawn `python -m agent.mcp.server` as a subprocess
   from a Python MCP client, run `initialize` → `list_tools` → `call_tool`.
   This proves the JSON-RPC layer works end-to-end.

Both passed.

## Surprises / gotchas

- **`mcp.tool()` doesn't wrap the function** in this version of the SDK.
  My first instinct (`cmake_available.fn()`) was wrong — the decorated
  callable is the same object as the original. The test now uses the
  fallback path: just call it like a normal function.
- **`agent/mcp/__init__.py` had `from .server import main`** at first.
  Then `python -m agent.mcp.server` would emit a `RuntimeWarning` because
  `agent.mcp.server` was already in `sys.modules` before the `-m` runner
  touched it. Removed the import; left a docstring instead.
- **FastMCP logs to stderr by default** (good — stdout is reserved for
  protocol). I left default logging on; the messages are quiet and tagged
  with the request type.

## What this gives the project

For the agent loop:
- A clean way to gate "execute the generated tests" behind a protocol.
  Future work: an `execute_tests_node` that calls these MCP tools and writes
  `ExecutionResult` records into `AgentState`.

For the resume / interview story:
- "I wrote a custom MCP server that wraps the C++ toolchain. Now the same
  cmake / ctest tools can be driven by my Agent, by Claude Desktop, or by
  any future MCP-aware IDE." This is the **portability story** that I can't
  tell with a hand-rolled JSON-RPC of my own.

For Week 3:
- Same shape will be reused for a `tinyinfer-knowledge` MCP server that
  exposes the operator-doc retriever as MCP tools. RAG without a custom
  protocol.

## Open questions / TODO

- Should the agent's own execute path go through the MCP client too, or
  keep using `cmake_driver` directly? Going through MCP is purer but adds
  a stdio subprocess hop on every run. I'll measure latency before deciding.
- `list_tests` parses `ctest -N` output by string matching. Brittle if ctest
  output changes between versions. Replace with `ctest -N --json` when
  cmake ≥ 3.21 is the floor.
- No streaming yet — `cmake_build` of a real project takes minutes. MCP
  supports incremental progress notifications; worth wiring up later.

## Status

- [x] MCP server in `agent/mcp/server.py`
- [x] CLI subcommand `tinyinfer-agent mcp-server`
- [x] `mcp_config.json.example` for Claude Desktop registration
- [x] `docs/MCP.md` with quickstart and tool reference
- [x] Smoke test in `tests/test_mcp_server.py`
- [x] Real stdio handshake verified
- [ ] (next) Wire `execute_tests_node` into the graph so MCP execution is
      a normal pipeline step, not a separate CLI command.
