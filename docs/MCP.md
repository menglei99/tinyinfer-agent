# MCP 集成

`agent/mcp/server.py` 把本地 C++ 工具链暴露成一个 MCP stdio server。
也就是说，任何 MCP-aware 的客户端都可以用同一套可移植的工具 API 来
构建和测试 `tinyinfer/`：

- `cmake_available`
- `cmake_configure`
- `cmake_build`
- `ctest_run`
- `list_tests`

## 为什么这件事重要

这是本项目核心的工程故事之一。
同一套 build/test 能力现在可以被以下客户端复用：

- 本仓库自己的 LangGraph agent
- Claude Desktop
- Cursor
- 未来任何会说 MCP 的 IDE 或编排器

这比把这些动作藏在一个 Python 应用内部要强得多。

## 手动起 server

```bash
cd D:/tinyinfer-agent
.venv/Scripts/python.exe -m agent.mcp.server
```

或者用 CLI 包装：

```bash
tinyinfer-agent mcp-server
```

server 走 stdio 上的 JSON-RPC，所以正常情况下应该由 MCP 客户端拉起来，
而不是在终端里交互式跑。

## 配置 Claude Desktop

用 `mcp_config.json.example` 当模板，把 `tinyinfer-toolchain` 那段
拷到你的 `claude_desktop_config.json` 里。

关键字段：

- `command`：本仓库 venv 里 Python 的绝对路径
- `args`：`[-m, agent.mcp.server]`
- `TINYINFER_PROJECT_DIR`：`tinyinfer/` 的绝对路径
- `TINYINFER_BUILD_DIR`：`tinyinfer/build/` 的绝对路径

## 工具说明

### `cmake_available()`
返回宿主机 PATH 上有没有 `cmake`。
客户端先调用这个就能在没装 cmake 的机器上提前 fail。

### `cmake_configure(project_dir?, build_dir?)`
等价于：

```bash
cmake -S <project_dir> -B <build_dir> -DCMAKE_BUILD_TYPE=Release
```

### `cmake_build(build_dir?, target?)`
等价于：

```bash
cmake --build <build_dir> --config Release [--target <target>]
```

### `ctest_run(build_dir?, test_filter?)`
等价于：

```bash
ctest --test-dir <build_dir> --output-on-failure [-R <regex>]
```

### `list_tests(build_dir?)`
跑 `ctest -N` 然后返回发现的测试名列表。
生成的测试落地之后调用它可以确认它们被 cmake 正确发现。

## 当前限制

server 本身可以编排 build/test，但当前开发机 **没装** `cmake`。
所以 `cmake_available()` 在本机会返回 false，
`execute_tests_node` 走 SKIP 路径并把"未执行 + 原因"写到 report 里。

新机器装上 cmake 之后，同一份 server 代码不用改就能 configure、build、
跑生成出来的 GTest 套件。
