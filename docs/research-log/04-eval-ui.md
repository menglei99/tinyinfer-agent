# Devlog 04 —— Eval、UI

## Fault-injection benchmark（Week 2 遗留项，终于落地）

`benchmark/seeds/` 下 10 条手写 diff 种子，每条一个故障，分布在我们有
oracle 的三个 op 上：

- 4 条 matmul 故障：内层 k loop 差一、右操作数索引颠倒、漏掉 zero-init、
  累加器降位到 fp16
- 3 条 softmax 故障：丢掉 max 减、归一化分母错、exp 在减之前
- 3 条 layernorm 故障：丢掉 eps、用 n-1 算方差、整数截断后的 mean

harness 加载每条种子，用 mock 模式跑 agent graph，再调用
`_fault_caught_heuristic` 判断生成测试名/rationale 是否字面上命中了
这个故障对应的关键词组。

### 为什么用启发式

"真" oracle 应该是：把 mutation 应用到干净 checkout，重新编译，跑生成的测试，
看是否 fail。这需要本机有 cmake + C++ 编译器 —— 目前没有，留给新机器。
启发式是有意的替身：它回答"agent 至少瞄准了正确维度吗"，
而不是"测试会捕获故障吗"。两个问题都有用，后者严格包含前者。

启发式的接口在 `_fault_caught_heuristic(generated_tests, expected)`。
换成 compile+run oracle 就是替换一个函数。

### Mock baseline

`python -m benchmark.harness --seeds benchmark/seeds --mock --no-rag` 报告
mock 模式下 10/10。这是预期的：确定性的默认 shape 集合本来就覆盖了
每条故障影响的维度。Mock baseline 回答的是"流水线路由 + 产出正确 op
吗" —— 作为 `parse_diff` 和 skill 注册表的回归 guard 有用。

有意思的数字来自对同一份 harness 跑：

- 真实 LLM 下 `SELF_CONSISTENCY_N=0` vs `N=3`（投票有用吗？）
- 有 RAG vs 没 RAG（检索上下文能扩展覆盖吗？）

两组 ablation 已经在本机用 Qwen 跑过，结果见 devlog 05。

## Streamlit UI

`agent/ui/app.py`。三栏：input（粘 diff 或选 sample）、tests（每个生成的 `.cpp`
一个 expander）、critic（verdict 徽章 + coverage report + RAG context）。

跑法：

```bash
pip install -e ".[ui]"
streamlit run agent/ui/app.py
```

我在意的约束：

- App 复用 `build_graph` 和 `_collect_final_state`，行为跟 CLI 完全一致。
- 没把执行路径（cmake/ctest）接进 UI —— 让它专注于测试生成。
- 没用后台线程。流水线是同步且短的（mock < 30s；带 self-consistency
  的真 LLM 几分钟）。状态 spinner 直接用 `st.status`。
- 模块级 `_running_under_streamlit()` 守卫，防止 pytest 里
  `import agent.ui.app` 时启动 Streamlit。

本机 Windows + Python 3.13 上 dev server 起得干净。完整交互流走查
留给新机器，让人真的能点。

## CI

按用户要求移除了。本地 `pytest tests/` 依然能跑。将来要恢复，工作流可以从
同样的 matrix（Linux + Windows × Python 3.12/3.13）按需重新生成。

## Docker compose

`docker-compose.yml` 暴露两组可选服务：

- `qdrant` —— 单容器，默认分组（无 profile）。从嵌入式 Chroma 升级到
  "真" server 时的简单路径。
- `etcd` + `minio` + `milvus-standalone` —— 放在 `milvus` profile 下
  （`docker compose --profile milvus up -d`）。只对特别要 Milvus 栈的用户
  有意义。

两组都还没在本机 `up` 过 —— 默认 Chroma 路径根本不需要 Docker。Compose
是给新机器的菜谱。

## LangSmith shim

`agent/tracing/langsmith.py` 不再是 `TraceWriter.write` 的 wrapper。
LangGraph 在 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` 设好时
自动 instrument —— 我们这边不需要 monkey-patch 任何东西。这个模块现在
只是在 CLI 启动时打印一行状态 banner，告诉用户 trace 会不会进 dashboard。

如果用户以后想自托管可观察性，Langfuse 是显然的备选（开源、在 Docker 里跑、
有 LangChain callback handler）。它需要一个新 shim —— 环境变量魔法这个
办法只对 LangSmith 有效。

## 留给新机器

见 `docs/DEFERRED_VALIDATION.md`。简短列表：

- 真实 LLM 的 ablation 数字进一步收紧（多次重跑求平均）
- Docker compose 真的把 Milvus standalone 起起来
- 完整 Streamlit 交互验收
- 通过 MCP 跑 C++ build/test（Week 2 的 SKIP 路径还在）
- LangSmith dashboard 真打开看

以上每一项的代码都在，缺的只是 wall-clock 上真去跑一次。
