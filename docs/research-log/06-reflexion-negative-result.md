# Devlog 06 —— Reflexion：我们以为会怎样，结果如何

## 想法

critic 返回 FAIL，给出 `missing_dimensions=["square_2d", ...]` 和一句
LLM feedback；按今天的实现，下一轮 `generate_tests` 没有上次抱怨的
任何记忆。shape planner 重发同一份 prompt 给 LLM，LLM 提议类似的
shape 集合，critic 再找同样的 spec —— critic loop 跑了 2 轮，
大多数种子最后依然 FAIL。

Reflexion（出处见
[原论文](https://arxiv.org/abs/2303.11366)）把这个环闭上：把 critic 的
抱怨抓成自然语言"lesson"，下一轮喂回生成器的 prompt。假设是
critic 检测出的空缺会在 iter 2 被填上，而不是重复出现。

## 实现

三块小东西：

1. **State。** `AgentState.reflexion_lessons: Annotated[list[str], _merge_list]`
   —— 通过 concat reducer 跨迭代累计。
2. **Critic。** FAIL 时发一条 lesson，形如
   `"Lesson #2: the previous test set was missing square_2d, k1_outer. Feedback: ..."`。
   lesson 通过 state 留给下一轮。
3. **生成器。** `NumericalSkill._plan_shapes` 读 `self._lessons`，把它们作为
   "Lessons from prior iterations" 块前置到 LLM user prompt 上。

通过 `REFLEXION` 环境变量门控（默认 on）。14 个新测试覆盖：门控行为、
lesson 字符串格式、critic FAIL 时发、critic PASS 时不发、planner 拿到时注入、
planner 没拿到时省略。

## 结果：Reflexion 在这个 benchmark 上没起作用

同样的 10 种子，真 Qwen + RAG，唯一差别是 `REFLEXION=on`。头条：

| 配置 | Critic pass rate | Avg missing dims | Avg tests/seed |
|---|---|---|---|
| qwen_rag             | 0.40 | 0.60 | 8.80 |
| qwen_rag_reflexion   | **0.20** | **0.90** | 8.20 |

pass rate 降了。每种子展开看：

| 种子 | qwen_rag verdict | reflexion verdict |
|---|---|---|
| 01_matmul_offbyone_inner       | PASS (iters=2, 14 tests) | FAIL (iters=2, 13 tests) |
| 02_matmul_swapped_indices      | PASS (iters=2, 14 tests) | FAIL (iters=2, 12 tests) |
| 03_matmul_uninit_accumulator   | PASS (iters=1, 11 tests) | PASS (iters=1, 11 tests) |
| 04_matmul_fp16_accumulator     | PASS (iters=1, 13 tests) | PASS (iters=1, 10 tests) |
| 05–10 softmax/layernorm        | FAIL (iters=2)            | FAIL (iters=2)            |

Reflexion 只可能影响 4 个 matmul 种子（见下面"Scope 问题"）。这 4 个里，
2 个从 PASS 翻到 FAIL。

## 为什么？三个诚实的假设

### 1. Scope 问题（确凿）

Reflexion 通过 `NumericalSkill._plan_shapes` 投递 lesson，而那个方法
只对 matmul 触发。softmax 和 layernorm 用静态的 `default_shape_cases`，
完全不调 LLM planner。所以 10 个种子里有 6 个机械上不在 Reflexion 的
影响范围 —— lesson 落到 state 里没人读。

这就解释了天花板：~60% 的 benchmark FAIL 在构造上就 Reflexion 触不到。
要解决需要把 LLM 增强的 shape planning 也接到 softmax/layernorm，
这是比本 devlog 范围更大的改动。

### 2. Critic 在迭代之间目标移动（很可能）

Qwen LLM critic 在 temperature=0.2 下不是确定性的。iter 1 它标
"missing square_2d"；lesson 给到生成器；iter 2 加上 square_2d shape；
iter 2 的 critic 看到的 square_2d 已被满足，但用新鲜眼光扫整个测试集，
找到别的新问题。lesson 追上了上一个 critic 的抱怨，下一个 critic 不认。

更严格、更确定性的 critic prompt（或者跨迭代复用同一个 critic 调用）
能减弱这个效应。本轮不做。

### 3. n=10 太小，统计上不够（也确凿）

matmul 4 个种子。两个翻盘。在温度噪声下，2 个种子的摆动差不多是
1σ。把每个配置跑三次取平均能告诉我们 Reflexion 到底是略差、中性，
还是其实略好但运气不好。

我没跑这三次。每次 ~45 分钟，三倍取样 ~2.25 小时的 LLM 调用 ——
而当前主结果（"RAG 让 missing_count 降 4×"）已经在手了。
Reflexion 的三倍取样进新机器列表。

## 这教会了我们什么

- **负向结果也是结果。** Reflexion 是教科书技术。接进来 ~50 行
  代码 + 14 个测试。benchmark 说："在这个 setup、这个种子数、这个
  critic prompt 下，它没量化提升。" 这本身值得知道。
- **Reflexion 救不了它够不到的东西。** 故障在 ablation 设计上和技术
  本身一样多 —— 大多数种子根本不经过 Reflexion 影响的 LLM 规划路径。
- **critic 才是主导变量。** `qwen_rag` 和 `qwen_rag_reflexion` 都是
  ~2 次迭代 × ~6-7 FAIL verdict。把 pass rate 钉在 30% 附近的是 critic，
  不是生成器。下一个高杠杆改动**不是**"更聪明的生成器反馈"而是
  "更受约束的 critic prompt"或"更紧的维度注册表"。

## 如何复现 / 关掉

Reflexion **默认开**。要跟无 Reflexion run 对比：

```bash
REFLEXION=off LLM_PROVIDER=qwen EMBEDDING_PROVIDER=dashscope \
    python -m benchmark.harness --seeds benchmark/seeds --json results/qwen_rag.json

REFLEXION=on LLM_PROVIDER=qwen EMBEDDING_PROVIDER=dashscope \
    python -m benchmark.harness --seeds benchmark/seeds --json results/qwen_rag_reflexion.json

python -m benchmark.compare results/qwen_rag.json results/qwen_rag_reflexion.json
```

## 下一步候选

1. **把 LLM shape planning 扩到 softmax/layernorm。** 它们当前只用默认。
   照 matmul `_plan_shapes` 镜像一份，这样 Reflexion（和普通 Qwen）能
   影响 10/10 而不是 4/10 种子。
2. **更紧的 critic prompt。** 现在 LLM critic 返回
   `missing_dimensions: list[str]` 是自由文本。把它约束到固定词汇表
   （结构 critic 的维度名）能让"缺失"声明可操作，并减少迭代间的 critic
   漂移。
3. **temperature=0 下三倍取样**消除对比里的噪声。3× 成本，3× 信噪比。
