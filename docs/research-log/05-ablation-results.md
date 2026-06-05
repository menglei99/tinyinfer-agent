# Devlog 05 —— Fault-injection ablation 结果

Week 3 的计划里包含 fault-injection benchmark 和四组 ablation。这条 devlog
记下实际数字、意外发现，以及它说明 agent 里哪些部分在真正做工。

## 跑了哪些配置

`benchmark/seeds/` 下 10 条手写 diff 种子，分布在 matmul / softmax /
layernorm 三个 op 上。四组配置，按"运动部件"由少到多排：

| 配置 | LLM | RAG | Self-consistency |
|---|---|---|---|
| `mock_norag`  | mock（确定性） | off | N=0 |
| `qwen_norag`  | Qwen 2.5 plus  | off | N=0 |
| `qwen_rag`    | Qwen 2.5 plus  | on（BM25+dense+RRF，Chroma） | N=0 |
| `qwen_sc3`    | Qwen 2.5 plus  | on                          | N=3 |

`mock_norag` 是结构地板 —— 它回答"流水线的确定性骨架能否产出名义上覆盖
每条故障维度的测试集？"真 LLM 配置层层加码：真实的 Qwen shape 提议、
检索 corpus 作为 critic grounding，最后是 N=3 的 self-consistency 投票。

## 头条表格

| 配置 | Accuracy | Avg tests/seed | Critic pass rate | Avg critic iters | Avg missing dims | Avg RAG docs |
|---|---|---|---|---|---|---|
| mock_norag | 10/10 (100%) | 7.20 | 1.00 | 1.00 | 0.00 | 0 |
| qwen_norag | 10/10 (100%) | 8.60 | 0.30 | 1.70 | **2.60** | 0 |
| qwen_rag   | 10/10 (100%) | 8.80 | 0.40 | 1.80 | **0.60** | 5 |
| qwen_sc3   | 10/10 (100%) | 7.80 | 0.30 | 1.80 | **0.70** | 5 |

随时复跑这张表：

```bash
python -m benchmark.compare results/*.json --md results/ablation.md
```

## 意外：accuracy 饱和了，但 critic 数据亮起来了

二值的"测试集是否覆盖了故障维度"在**所有**配置下都 10/10，包括 mock baseline。
这不是因为 agent 完美 —— 是因为手写的 `default_shape_cases` 集合已经把
每条种子故障触及的维度命名了一遍，`_fault_caught_heuristic` 那条启发式
只查这些名字。

accuracy 这列因此没什么信息量。真故事在 **critic 数据**里。

### 发现 1：RAG 把 critic 幻觉砍掉 4×

头条数字：**qwen_norag 的 LLM critic 平均每个种子声称少了 2.60 个维度；
qwen_rag 是 0.60**。同样的 Qwen 模型，同样的 prompt，唯一区别是
critic prompt 里追加了 top-5 检索到的 corpus 片段。

没有 RAG 上下文时，LLM critic 自信地标出"少 aligned shape 变种"、
"少 batched 2D 路径"、"少 zero-input edge case" —— 但这些**已经**在
结构 critic 的维度表里，**而且**已经被生成的测试覆盖。LLM 在幻觉空缺，
因为它没有 grounding。

带 RAG 上下文之后，critic 看到具体的 corpus 片段 —— *"softmax test
shapes: 1D basic, 1D very short (n=2 or 3, degenerate edge), batched 2D…
aligned lengths"* —— 然后把它的提议拿去和文档里说的对一遍。"缺失维度"
从 2.6 降到 0.6。

这是整次 ablation 里最强的信号：**用真实 corpus 把 LLM critic ground 住，
才是 RAG 真正买到的东西**，而不是"找出更多 bug"。

### 发现 2：Self-consistency 让测试集更稳，但没扩展覆盖

qwen_norag 和 qwen_rag 都平均 ~8.7 个测试/种子。qwen_sc3 降到 **7.8** ——
更少的测试，不是更多。投票把 Qwen 只在三次草稿之一里提议的 shape 滤掉了。
这是设计就这样：SC=3 在降噪。需要 CI 跨 run 可复现时有用；
不是想要新覆盖的时候的头部杠杆。

critic pass rate（30% / 30% / 40% / 30%）跟着 SC 几乎不动。
Self-consistency 作用在*生成器*上，不在*critic*上；critic 的判决依赖
测试集质量，而那只小幅改善 —— 因为确定性的默认 case 已经占主导。

### 发现 3：Critic 循环值它的钱

mock critic 总是 iter 1 就 PASS（确定性结构通过）。每个 Qwen 配置都
平均 1.7-1.8 次 critic 迭代 —— 也就是大约 70% 的种子走了第二轮
`generate_tests → critic`，因为第一次 verdict 是 FAIL。这正是 critic
loop 设计要做的：抓住不完整的第一次尝试，重新生成。

critic pass rate 在所有 Qwen 配置上稳定在 30-40%，说明 LLM critic
**始终严格**。即便走第二轮，仍然有 ~6-7 / 10 个种子能找到剩余空缺。
这是真实质量信号 —— 不是 bug。

## 什么没动

Accuracy。10 个种子配字面重叠启发式，每个配置都顶到天花板。下一步是更严的
oracle：把 mutation 应用到干净 checkout，build，跑生成测试，看结果 fail
还是 pass。这需要本机 cmake —— 留给新机器。ablation 表会在那之后变得
有意思得多。

## 成本

每次 benchmark run（10 种子）的近似数字：

| 配置 | LLM 调用 | Wall clock |
|---|---|---|
| mock_norag | 0 | ~3s |
| qwen_norag | ~25 | ~12 min |
| qwen_rag   | ~25 + 30 次 embedding（之后命中缓存） | ~15 min |
| qwen_sc3   | ~50 | ~30 min |

Self-consistency 把 matmul plan cost 三倍化。4 matmul + 6 vector 种子加起来
每次 benchmark 多 12 个 LLM 调用 —— corpus 长到 30 个种子的时候不可忽略。

## 收获

1. **RAG 的价值集中在 critic。** `critic_missing_count` 从 2.60 → 0.60，
   就靠 prompt 里加一段上下文 —— 这是整个项目里最可量化的胜利。
2. **Self-consistency 让测试集更稳定，不改 accuracy。** 想要 CI 确定性
   值得开；想要新覆盖不值它 2× LLM 成本。
3. **`fault_caught` accuracy 列被启发式饱和了，不是被 agent。**
   要么收紧启发式（compile + run），要么扩散种子。两件事都在 roadmap 上。

## 本机复跑

```bash
# 1. 配置
echo "LLM_PROVIDER=qwen" >> .env
echo "EMBEDDING_PROVIDER=dashscope" >> .env
echo "DASHSCOPE_API_KEY=..." >> .env   # 你的 key

# 2. 跑四组
python -m benchmark.harness --seeds benchmark/seeds --mock --no-rag --json results/mock_norag.json
LLM_PROVIDER=qwen python -m benchmark.harness --seeds benchmark/seeds --no-rag --json results/qwen_norag.json
LLM_PROVIDER=qwen EMBEDDING_PROVIDER=dashscope python -m benchmark.harness --seeds benchmark/seeds --json results/qwen_rag.json
SELF_CONSISTENCY_N=3 LLM_PROVIDER=qwen EMBEDDING_PROVIDER=dashscope python -m benchmark.harness --seeds benchmark/seeds --json results/qwen_sc3.json

# 3. 对比
python -m benchmark.compare results/*.json --md results/ablation.md
```

总 wall time：约 1 小时。总 token 成本：~150 LLM 调用 + ~60 embedding。
DashScope 免费 tier 里够用。
