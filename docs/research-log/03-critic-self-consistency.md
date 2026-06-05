# Devlog 03 —— 带检索上下文的 Critic + self-consistency

**目标**：让 critic 的 verdict 比单一 `passed` 布尔值更丰富；
让测试生成器在 LLM 单次响应抽风时也能扛住。

## Coverage report

`CriticVerdict` 现在带一个开放 schema 的 `coverage_report: dict` 字段，
跟原有的 `passed` / `missing_dimensions` / `feedback` 并列。当前它记录：

```python
{
  "structural_pass": bool,
  "groups_checked": int,           # 看到的 (op, skill) 桶个数
  "total_tests": int,
  "missing_count": int,
  "context_docs_used": int,        # critic 实际用了几个 RAG 命中
  "per_skill": {"matmul_fp32/numerical": ["missing_dim_a", ...]},
  # 当 LLM critic 真跑了，会出现这些额外字段：
  "llm_passed": bool,
  "llm_missing": [str, ...],
}
```

schema 故意开放。以后加新维度（perf budget tightness、memory-skill ASan
兼容性等等）直接加 key 就行 —— 不用 model migration。报告渲染器读到什么
显示什么，没的字段静默跳过。

## 为什么是两个 critic 而不是一个

critic 是个三明治：先跑便宜的基于规则的结构 critic，结构通过后再可选跑 LLM critic。
结构 critic 守住地板 —— **每一个**测试集都要查 shape variety、stability case
等等。LLM critic 加上品味：它看 top-3 检索回来的 corpus 片段加测试 summary，
确认 OR 提议额外维度。

拆开保留两个属性：

- 单测**确定性**（结构 critic 是纯函数）。
- 成本**紧致性**（结构 critic 说测试集至少最小完整了，才花一次 LLM 调用）。

## RAG grounding

`state["retrieved_docs"]` 非空时，critic 把 top-3 文档（按 retriever score）
作为 "Relevant context from corpus" 块前置到 LLM user prompt 上。
封 3 个是为了让 prompt 预算可预测；retriever 已经按 score 排好。

检索失败**不会**向上传递。`retrieve_context_node` 吞掉异常，把错误字符串
写进 state.error —— 其余流水线表现得像 RAG 没开一样。这条规则是为新机器
交接而设计的：错配的 Milvus 不应该阻塞 agent 生成测试。

## Self-consistency

Numerical-skill 的 shape planning 是 LLM 噪声经典出现的地方：问"给我 4-6 个
matmul shape"，每次返回的集合都会略不一样。Self-consistency 缓解这个：
画 N 个独立计划（同 prompt、temperature 拉开），**多数投票**保留的 shape。

激活：`SELF_CONSISTENCY_N=3` 环境变量。默认关闭（`N=0`/`1`），因为：

- 三倍 LLM 成本（每个 op 三次调用，而不是一次）。
- 手写的 default shape 集合已经覆盖了 ~95% 的价值；SC 主要在 LLM 提议
  不寻常 shape 时有效。
- 行为变化够大，希望用户显式 opt-in。

投票逻辑：

```python
threshold = ceil(N / 2)
保留 canonical key（例如 ("matmul", m, k, n)）在 >= threshold 次草稿里出现过的 shape
```

平局打破规则：first-seen 顺序。这样给定 LLM 响应序列时输出确定 —— 对测试很重要。

## 比老 critic 多抓到的

- **Shape 分布漂移** —— LLM 偶尔提议 `(m=999, k=1, n=999)`，投票把它丢掉。
- **Schema 噪声** —— N 次里某次 JSON 格式坏掉，其余次数把它投掉。
- **跨算子的 critic 反馈** —— 新 prompt 看到"softmax 稳定性 case：large
  positive logits、near-one-hot inputs"这种检索片段，critic 的
  `missing_dimensions` 更具体。

## 局限

- 结构 critic 的 per-op 维度表是手写的。长期来说我们想让 corpus 直接
  种子化 critic 应该查的维度 —— 但需要更结构化的 corpus schema。
- benchmark 里的 "fault_likely_caught" 启发式用的是字面重叠（测试名
  含期望关键词）。这是对真问题"编译并跑这个测试能让种子故障 fail 吗"的
  粗代理。真答案需要本机有 cmake。

## 待办

- 真 LLM 的 A/B：结构 only vs 结构+LLM critic 在 benchmark 上的准确率差异
  —— 需要 API 预算。
- Self-consistency 的成本 vs 准确率曲线 —— 需要 API 预算和新机器。
- Reflexion loop：把上一轮 `coverage_report` 反喂给下一轮生成的
  "lessons learned" prompt。设计好了，本轮没实现。
