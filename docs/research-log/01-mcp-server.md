# Devlog 01 —— 把 C++ 工具链包成一个 MCP server

## 为什么现在做

Week 1 留下的 `agent/tools/cmake_driver.py` 是包了一层
`subprocess.run` 的 Python 函数。Agent 自己能调，**别人调不了**。

对一个号称"production-grade AI 应用工程"的项目来说，这个形状不对。
在真实组织里：

- 工程师在 code review 的时候，想直接在 Claude Desktop 或 Cursor 里
  调 `cmake_configure / cmake_build / ctest_run`。
- Agent 调这些工具应该走正式协议，将来可以把实现整组替换
  （本地 subprocess → 远端 build farm）而不动 agent 代码。
- 不想在 LangChain 自己的工具格式之上又发明一个。MCP 就是那个标准。

所以 Week 2 第一件事就是把 C++ 工具链提到 MCP stdio server 里。

## 做了什么

`agent/mcp/server.py` 通过 `FastMCP` 暴露 5 个工具：

| 工具 | 包的命令 |
|------|---------|
| `cmake_available` | `shutil.which('cmake')` |
| `cmake_configure` | `cmake -S … -B … -DCMAKE_BUILD_TYPE=Release` |
| `cmake_build`     | `cmake --build … [--target …]` |
| `ctest_run`       | `ctest --test-dir … -R …` |
| `list_tests`      | `ctest -N` + 解析 |

刻意做的三件事：

1. **默认值走环境变量**（`TINYINFER_PROJECT_DIR`、`TINYINFER_BUILD_DIR`），
   这样 Claude Desktop 的配置文件只需要钉一次，LLM 调工具的时候
   可以不传参。
2. **截尾 stdout / stderr**。Build 日志能撑爆 MCP message size limit；
   4 KB 尾部足够 debug。
3. **跟 agent 共用同一份底层实现**（`agent.tools.cmake_driver`）。
   Python agent 和 MCP 拉起来的 Claude Desktop 共享一套实现 —— 没有漂移。

## Smoke test

我从两个角度验过 server：

1. **同进程内**（pytest 走）：import `FastMCP` 实例，`await mcp.list_tools()`，
   直接调每个工具函数。覆盖率高、不用在 Windows CI 上 spawn 子进程。
2. **真实 stdio 握手**：从一个 Python MCP client spawn
   `python -m agent.mcp.server` 子进程，跑 `initialize` → `list_tools` → `call_tool`。
   证明 JSON-RPC 这层端到端 work。

两个都过。

## 踩到的坑

- **`mcp.tool()` 在当前 SDK 里不 wrap 函数。** 我的第一反应
  （`cmake_available.fn()`）是错的 —— 被装饰的 callable 跟原函数是
  同一个对象。测试现在走的是回退路径：当普通函数调就行。
- **`agent/mcp/__init__.py` 最初有 `from .server import main`。** 然后
  `python -m agent.mcp.server` 会报 `RuntimeWarning`，因为 `agent.mcp.server`
  在 `-m` runner 启动之前就已经在 `sys.modules` 里了。删了那个 import，
  留个 docstring 就行。
- **FastMCP 默认日志写 stderr**（对的 —— stdout 留给协议）。我没关默认日志，
  消息很安静，按请求类型 tag。

## 对项目的价值

对 agent 主循环：
- 有了一个干净的方式把"执行生成的测试"用协议门控起来。
  后续：写一个 `execute_tests_node` 调这些 MCP 工具，把 `ExecutionResult`
  写回 `AgentState`。

对简历 / 面试故事：
- "我写了一个自定义 MCP server，把 C++ 工具链包了起来。现在同一套
  cmake / ctest 工具可以被我的 Agent、Claude Desktop、或者任何未来的
  MCP-aware IDE 驱动。" 这是**可移植性故事**，自己手写 JSON-RPC 是讲不出来的。

对 Week 3：
- 同样的形状会复用到一个 `tinyinfer-knowledge` MCP server 上，把算子文档
  检索器暴露成 MCP 工具。RAG 不用自定义协议。

## 待办 / 未决

- agent 自己的执行路径要不要也走 MCP client？还是继续直接调
  `cmake_driver`？走 MCP 更纯，但每次跑都加一个 stdio 子进程 hop。
  先量延迟再决定。
- `list_tests` 现在按字符串匹配解析 `ctest -N` 输出。脆。
  以后 ctest 输出格式一变就崩。等 cmake ≥ 3.21 是地板了换成 `ctest -N --json`。
- 没做流式 —— 真项目 `cmake_build` 要几分钟。MCP 支持增量进度通知，
  后面加上。

## 状态

- [x] MCP server 在 `agent/mcp/server.py`
- [x] CLI 子命令 `tinyinfer-agent mcp-server`
- [x] `mcp_config.json.example` 给 Claude Desktop 注册用
- [x] `docs/MCP.md` 含快速上手和工具参考
- [x] `tests/test_mcp_server.py` 里的 smoke test
- [x] 真实 stdio 握手验过
- [ ]（下一步）把 `execute_tests_node` 接进 graph，让 MCP 执行成为
       流水线里的一个普通节点，不是单独的 CLI 命令。
