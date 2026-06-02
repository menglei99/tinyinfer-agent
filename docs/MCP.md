# MCP integration

`agent/mcp/server.py` exposes the local C++ toolchain as an MCP stdio server.
That means any MCP-aware client can build and test `tinyinfer/` using the same
portable tool API:

- `cmake_available`
- `cmake_configure`
- `cmake_build`
- `ctest_run`
- `list_tests`

## Why this matters

This is one of the core engineering stories of the project.
The same build/test functionality is now usable by:

- this repository's own LangGraph agent
- Claude Desktop
- Cursor
- any future IDE or orchestrator that speaks MCP

That is much stronger than keeping these actions buried inside one Python app.

## Start the server manually

```bash
cd D:/tinyinfer-agent
.venv/Scripts/python.exe -m agent.mcp.server
```

Or via the CLI wrapper:

```bash
tinyinfer-agent mcp-server
```

The server speaks JSON-RPC over stdio, so it should normally be launched by an
MCP client rather than run interactively in a terminal.

## Configure Claude Desktop

Use `mcp_config.json.example` as a template and copy the `tinyinfer-toolchain`
entry into your `claude_desktop_config.json`.

Important fields:

- `command`: absolute path to this repo's virtualenv Python
- `args`: `[-m, agent.mcp.server]`
- `TINYINFER_PROJECT_DIR`: absolute path to `tinyinfer/`
- `TINYINFER_BUILD_DIR`: absolute path to `tinyinfer/build/`

## Tool reference

### `cmake_available()`
Returns whether `cmake` is installed on the host PATH.
Use this first so the client can fail fast on machines without cmake.

### `cmake_configure(project_dir?, build_dir?)`
Equivalent to:

```bash
cmake -S <project_dir> -B <build_dir> -DCMAKE_BUILD_TYPE=Release
```

### `cmake_build(build_dir?, target?)`
Equivalent to:

```bash
cmake --build <build_dir> --config Release [--target <target>]
```

### `ctest_run(build_dir?, test_filter?)`
Equivalent to:

```bash
ctest --test-dir <build_dir> --output-on-failure [-R <regex>]
```

### `list_tests(build_dir?)`
Runs `ctest -N` and returns the discovered test names.
Useful after generated tests are written to disk.

## Current limitation

The server can orchestrate build/test, but this host currently does **not**
have `cmake` installed on PATH. So `cmake_available()` returns false until you
install cmake locally.

Once cmake is installed, the same server will be able to configure, build, and
run the generated GTest suites without any code changes.
