# 架构

## 1. 为什么是这个形状

给推理框架写回归测试 Agent，跟"让 LLM 写 pytest"那种通用项目**不是一回事**。
独有的约束有四条：

1. **数值正确性需要可信的 oracle**
   —— LLM 不能自己编 expected 值
2. **覆盖维度是领域特定的**
   —— shape × dtype × edge × hardware × layout，不只是代码路径
3. **工具链是 C++**（cmake / ctest / ASan / perf），不是 pytest
4. **失效模式很微妙**
   —— fp32 最大误差 `1e-7` 可能没事，`1e-3` 可能就是真的 regression

下面的架构就是为这四条约束服务的。

## 2. 流水线

```
git diff
   │
   ▼
parse_diff ─── 识别出被改的算子（先 regex，LLM 兜底）
   │
   ▼
route_skill ── 选要跑哪些测试维度
   │            （numerical / performance / memory / hardware）
   │
   ▼
retrieve_context ── BM25 + dense embedding + RRF 融合
   │                 从 corpus（算子语义 / 源码注释 / 历史 fault rationale）里
   │                 召回 top-k 给后续 critic 做 grounding
   ▼
generate_tests
   │   ┌──────────────────────────────────────────────┐
   │   │  对每个 (op, skill)：                         │
   │   │     LLM → 提议 shapes / cases                │
   │   │     numpy/ONNX oracle → 算 reference 值      │
   │   │     renderer → 生成 GTest C++ 源码           │
   │   └──────────────────────────────────────────────┘
   ▼
critic ────── 结构化规则先跑，LLM peer-review 后跑
   │            verdict.passed = false →
   │              （Reflexion）把 missing dims 写成 lesson
   │              回到 generate_tests 再来一轮
   ▼
write_report ── 输出 markdown summary
   │
   ▼
（可选）install_tests + execute_tests
       通过 MCP server 把生成的 .cpp 放进 cmake 树
       并跑 cmake configure / build / ctest，把 ExecutionResult 写回 state
```

## 3. 关键设计决策

### LLM 选 shape，oracle 选数值

最重要的一条规则。LLM 擅长**测什么**（shape、edge、dtype 组合），
但**期望数值**它给得不靠谱。我们严格分开：

```python
# Skill 问 LLM：
shapes = llm.plan_shapes(op_name)            # 创造性

# Oracle 算出数值：
expected = numpy_or_onnx_reference(shape)    # 确定性

# Renderer 组装：
emit_gtest(shape, expected)                  # 机械
```

这就是所谓"基于证据的测试生成（evidence-grounded test generation）"模式。
也正因为如此，mock 模式也能产出真实可跑的测试 —— 数值始终来自 numpy。

### 状态用带 reducer 的 TypedDict

LangGraph state 是流经各节点的一个 dict。list 字段挂了一个 **concat reducer**
而不是覆盖语义，这样 critic loop 里多轮迭代可以累计测试用例，
不会把上一轮的工作丢掉。Reflexion 的 lessons 也走同一套 reducer 机制。

### Critic 分两层

- **结构化 critic**（确定性、快）—— 先跑。规则便宜
  （"测试集里包含 thin shape 吗？"）。可以独立给出 verdict。
- **LLM critic**（慢、可选）—— 只在结构化通过后跑。
  抓结构化规则不知道的维度，且看到 RAG 检索回来的 corpus 片段做 grounding。

如果 LLM critic 解析失败，回退到结构化 verdict。避免一次 LLM 抽风
就把流水线卡死。

### 工具暴露成 MCP server，不是私有 Python 函数

`agent/tools/cmake_driver.py` 暴露 `configure / build / ctest` 是纯 Python 函数。
但是 `agent/mcp/server.py` 把这些工具再包装一层 MCP stdio server，
这样 **Claude Desktop / Cursor / 本仓库的 Agent 共用同一份工具实现**，
不会两个客户端写两套，也方便以后换实现（subprocess → 远端 build farm）。

## 4. 模块地图

| 模块 | 职责 |
|--------|------|
| `agent/state.py` | Pydantic models + TypedDict state |
| `agent/llm/` | LLM client（DeepSeek/OpenAI/Anthropic/Qwen/Mock）+ Embedding client（DashScope/Mock） |
| `agent/rag/` | 向量库（Numpy/Chroma/Qdrant/Milvus）+ BM25 + RRF 混合检索 + 语料 + 嵌入磁盘缓存 |
| `agent/skills/` | 每个测试维度一个模块（numerical/perf/memory） |
| `agent/tools/` | 参考 oracle、C++ renderer、cmake driver |
| `agent/graph/` | LangGraph 节点 + StateGraph builder |
| `agent/mcp/` | 自定义 MCP server + 同步客户端封装 |
| `agent/tracing/` | JSONL trace writer + LangSmith 可选 shim |
| `agent/ui/` | Streamlit UI |
| `agent/cli.py` | Typer CLI |
| `benchmark/` | Fault-injection benchmark + 对比脚本 |
| `tinyinfer/` | 被测的 C++ 系统 |

## 5. 扩展点

- **加新算子**：在 `tinyinfer/` 里加 header + impl + baseline 测试，
  在 `agent/tools/oracle.py` 里加一个 oracle，
  在 `agent/skills/numerical.py` 里加一个 generator 分支。
- **加新 skill**：继承 `Skill`，在 `agent/skills/__init__.py` 注册，
  在 `route_skill_node` 里加路由规则。
- **加新 LLM provider**：在 `agent/llm/client.py` 里加一个类，
  在 `get_llm_client()` 里加一个分支。
- **加新向量库后端**：在 `agent/rag/store.py` 里加一个 `*VectorStore` 类，
  在 `get_vector_store()` 里加分支即可，调用方不用动。
