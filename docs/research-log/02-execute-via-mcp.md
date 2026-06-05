# Devlog 02 —— 闭环：execute_tests_node 走 MCP

## 为什么重要

Devlog 01 立起来了 MCP stdio server，但只有 Claude Desktop 受益。
Agent 自己依然在 `write_report` 之后停下，从不真跑它生成的测试。
形状不对：**搞 MCP server 的全部理由**就是让 agent 也走同一套协议。

这条 devlog 把环闭上：
**parse_diff → generate → critic → install → execute → report**。

## 加了什么

| 组件 | 职责 |
|-----------|------|
| `agent/mcp/client.py` | 同步 wrapper：起一个 MCP stdio session，跑一批工具调用，收掉 |
| `install_tests_node` | 把生成的 `.cpp` 拷到 `tinyinfer/tests/`，下次 configure 时 cmake 的 `file(GLOB)` 就能扫到 |
| `execute_tests_node` | 走 `cmake_available → cmake_configure → cmake_build → ctest_run`，全程通过 MCP server。把结果记成 `ExecutionResult` |
| `AgentState` 里的 `execute_requested` 字段 | 让 CLI 控制是否要执行这步；`--execute` 把它翻成 true |
| `_execute_router` 条件边 | flag false 时跳过 `execute_tests`，默认无 key demo 保持快 |
| `--execute / --no-execute` CLI 旗标 | 把控制权交给用户 |
| markdown report 里的 Execution 段 | 显示 pass/skip/fail 和 stderr 尾部 |

## 两个值得说明的设计选择

### 1. 同步 wrapper 包异步 MCP client

LangGraph 节点是同步的。MCP SDK 是异步的。我把每次 `execute_tests`
调用包在 `asyncio.run(...)` 里。每次调用拉起一个新的 stdio 子进程
（Windows 上握手大约 3 秒），对我们的流水线来说足够 —— 这个节点
每次跑最多 fire 一次。

如果是流式 UI 每秒触发多个 tool call，正确的形状是常驻的 async client。
我们还没到那一步。

### 2. cmake 不在时优雅 skip

我现在跑代码这台机器 PATH 上没 cmake。图依然干净跑完：
`cmake_available` 返回 false，节点记一条单独的 `<toolchain>` SKIP
`ExecutionResult`，report 里展示出来。**没有异常、没有半截执行、
没有惊吓。**

研究玩具和真正能交给同事的工具，差别就在这里。一个因为缺依赖就崩溃的
流水线，是比"我没跑，原因如下"更差的交付物。

## Trace 证据

`--execute` 跑一次 sample matmul diff 会产生 7 个节点 update：

```
parse_diff → route_skill → generate_tests → critic
                                          → install_tests
                                          → execute_tests
                                          → write_report
```

`--no-execute` 的 run 干净地省掉 `execute_tests` —— 条件边 work。

## 踩到的坑 / 意外

- 我最初计划从 agent 调 MCP server **只是**为了证明可移植性。
  实际上这意味着 agent 和 Claude Desktop 共享一套工具实现 —— 这是
  真实的生产收益，不是炫技。故事自己就写好了。
- `install_tests_node` 即便 `--no-execute` 也会跑。这是故意的：
  就算你不想 build，通常也希望生成的文件落到 cmake 可见的位置，
  方便单独跑 `tinyinfer-agent build-and-test`。如果将来这个判断
  错了，加个门控就行。
- FastMCP 把工具返回包成带 JSON 字符串的 `TextContent`。客户端适配器
  得 `json.loads` 之后 LangGraph 才能继续。

## 给后面铺路

下周的 perf 和 memory skill 会通过**同一个 MCP server**获取数据 ——
`ctest_run` 加 `--output-junit`，再加新工具 `run_benchmark` 和
`run_asan`。执行路径已经通用，需要长的只是工具集。

## 状态

- [x] `agent/mcp/client.py`
- [x] `install_tests_node` 和 `execute_tests_node`
- [x] 由 `execute_requested` 驱动的条件边
- [x] `--execute` CLI 旗标
- [x] report 里的 Execution 段
- [x] `test_execute_path.py`（stdio 子进程那条测试打 slow mark）
- [x] 17/17 pytest 过
- [ ]（下一步）等 cmake 装好之后：真正的 demo，绿色 ctest
