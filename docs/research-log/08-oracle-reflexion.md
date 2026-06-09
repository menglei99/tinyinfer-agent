# Devlog 08 —— Oracle-feedback reflexion + planner schema 扩展：实测发现 lesson 反馈没起效，胜利在别处

## 起点

[Devlog 07](07-real-oracle.md) 末尾留了三个候选下一步。其中"oracle-feedback reflexion"
听起来最像论文里那一条：让 oracle（apply mutation → build → ctest）跑完后，
如果 generated tests 全过（漏抓了 fault），把这个"oracle 反馈"喂回 generator
重生成一轮。比 critic-feedback reflexion（[devlog 06](06-reflexion-negative-result.md)）
强多了——失败的是**真实测试结果**而不是 LLM critic 的猜测。

这条 devlog 把这件事做了。**结论先放上来**：oracle-reflexion 的机制干净落地了
（~150 行 + 9 个单测），但实测对照实验里**它从未被触发**。真正让 ablation
accuracy 涨上去的是另一个改动——**扩 LLM planner 的 JSON schema** 让 LLM 能
直接 propose 非 shape 维度。这件事比 reflexion 直接得多。

## 机制：怎么做的

两块东西：

### 1. ORACLE_MISS_LESSON 通用反馈

`benchmark/harness.py` 加一条字符串常量：

```python
ORACLE_MISS_LESSON = (
    "Oracle 反馈：你这一轮生成的测试在 mutated source 上全部通过了，"
    "也就是说有一个真实 bug 被漏过了。仅靠 input shape 的多样性没能触发它。"
    "请重新考虑除 shape 之外的维度：output buffer 初值（预填非零 vs 零初始化）、"
    "输入/输出 alias、非默认 stride、NaN / Inf / denormal 输入、"
    "以及 API 合约本身（哪些参数必须被写入，哪些只读）。"
    "在下一轮 shape plan 里至少加一个 case 覆盖以上任一维度。"
)
```

设计选择：**故意不**把 mutation diff 的具体内容回灌给 generator。那等于给 LLM
**抄答案**——它直接能从 diff 里读出 bug 是啥。lesson 是"泛化的反馈"：你漏了
一个 bug，请往这些方向找。具体是哪种漏给 LLM 自己想。

这是 oracle-reflexion 跟 critic-reflexion 真正不同的点：critic 反馈的是"我觉得
你漏了 X"，oracle 反馈的是"测试实际没抓到 fault"——后者是**地面真相**，不依赖
critic 自己的猜测。

### 2. Harness 二次循环

`FaultInjectionHarness._run_one_inner` 改成：

```
seed → run agent attempt 1 → install → cmake build → ctest
  if caught:        done (attempts=1)
  if build_failed:  done (attempts=1)      # infra 问题，重跑也没用
  if missed AND oracle_reflexion:
    把 ORACLE_MISS_LESSON 注入新 state.reflexion_lessons
    run agent attempt 2 → install (覆盖第一轮) → build → ctest
    用第二次的判决作为最终结果，标记 attempts=2 / oracle_reflexion_used=True
```

注入路径直接复用现有 reflexion 机制：`AgentState.reflexion_lessons` 是个有
reducer 的 list，`generate_tests_node` 已经会把 `state["reflexion_lessons"]`
传给 `NumericalSkill(lessons=...)`。所以"用 oracle 反馈触发 reflexion"和
"用 critic 反馈触发 reflexion"走的是同一条 lesson channel，只是 lesson 来源
不同。

`tests/test_oracle_reflexion.py` 9 个单测 mock 掉 cmake_driver，锁定：
关 oracle_reflexion 时不重跑；missed 触发重跑且 lesson 注入到第二次 state；
两次都 missed 时如实报；第一次就抓到时跳过重跑；build_failed 时也跳过。

## 但是：planner schema 才是真的瓶颈

实现完之后开始想：generator 拿到 lesson 后能干什么？看 `NumericalSkill._plan_shapes`
的 prompt：

```
Schema per shape: {"m": int, "k": int, "n": int}
```

LLM 在这里能 propose 的字段**只有 `m / k / n`**。lesson 说"考虑 output buffer
初值、aliasing、stride、NaN"，但 LLM 在 JSON 里根本表达不出来——schema 里没有
`c_init`，没有 `aliased: bool`，没有 `nan_inputs: bool`。任何 lesson 都被
schema 锁死。

这是个**实现完才发现的设计问题**：oracle-reflexion 的机制 OK，但当前 generator
的可达"测试维度"只有 input shape，lesson 给的方向再正确也没载体。

所以补了一刀：**扩 planner schema** 让 LLM 能 propose `c_init`：

```python
def _shape_planner_user("matmul_fp32"):
    return (
        ...
        'Schema per shape: {"m": int, "k": int, "n": int, "c_init": [float,...] (可选)}\n'
        '其中 "c_init" 是可选字段，长度必须等于 m*n。设上之后我们会用 caller-owned '
        'c[] 预填这些值再调 pointer API——专门测 API 合约 / accumulator-mode 类 bug。\n'
        'Propose 4-6 shapes。如果你判断要测 output buffer 初值、aliasing、stride、NaN '
        '等非 shape 维度，请把对应 case 用 c_init 表达。'
    )
```

`_generate_matmul` 收到 LLM 提议的 case 时，如果带 `c_init` 字段且长度对，就
透传给 renderer 的 pointer-API 路径（[devlog 07](07-real-oracle.md) 已经落地）。

外加一个开关 `PREFILLED_OUTPUT_BUFFER_DEFAULT=off` 把 default 里的
`prefilled_output_buffer_2x2` case 临时关掉，专门验证"LLM 自己能不能基于 prompt
提议 c_init case"。

## 对照实验

模型：Qwen 3.7-plus（DashScope）。Embedding：DashScope（dense + BM25 + RRF）。
跑 10 个 seed，每个 seed agent 跑完 + real oracle（docker build + ctest）。

| 配置 | Schema | Oracle-Reflexion | Default Prefill Case | Accuracy |
|---|---|---|---|---|
| Group A | 扩了，可提 c_init | OFF | OFF | **10/10** (但 2 个 fallback heuristic) |
| Group B | 扩了，可提 c_init | ON  | OFF | **10/10** (全部真 oracle) |

| Seed | Group A oracle_mode | Group A build | Group A tests | Group B oracle_mode | Group B tests |
|---|---|---|---|---|---|
| 01 matmul off-by-one      | real_fallback_heuristic | skipped (build_failed) | 14 | real | 16 |
| 02 matmul swapped indices | real_fallback_heuristic | skipped (build_failed) | 18 | real | 17 |
| 03 matmul uninit acc      | real | ok | **17** | real | 17 |
| 04 matmul fp16 acc        | real | ok | 18 | real | 18 |
| 05–10 softmax/layernorm   | real | ok | 6 each | real | 6 each |

**两个非常重要的观察**：

### 观察 1：oracle-reflexion 在两组实验里**从未被触发**

Group B 全部 `attempts=1`、`oracle_reflexion_used=False`。也就是说：LLM 在
扩 schema 后，**第一次 attempt 就抓到了所有 fault**，根本没机会触发"missed →
re-generate"路径。

这是把诚实的话说出来：**oracle-reflexion 的机制干净，但今天的 benchmark 上
它没有作用**。胜利来自 schema 扩展。

### 观察 2：LLM 在新 schema 下也会 hallucinate 不合法 cpp

Group A 的 seed 01 / 02 走 fallback：

```
... <brace-enclosed initializer list>
gmake: *** [Makefile:146: all] Error 2
```

LLM 偶尔会 propose 一个 `c_init` 列表长度不对（不等于 `m * n`），或者 a/b 数组
shape 跟 `m/k/n` 对不上。这让 `cpp_renderer.render_matmul_test` 渲染出语法
错误（编译失败而不是逻辑错误）。当前实现 `_generate_matmul` 检查 `c_init` 长度
但不检查每个 case 的整体一致性。

这是 Qwen 3.7-plus 这个**生成时**的随机性，不是 oracle-reflexion 的问题。但它
带出来一个**真正值得做的下一步**：

**Build-failed 也应该触发 oracle-reflexion**。"你提议的 case 编不过"也是一个
真实的 oracle 反馈，跟"测试漏抓 fault"一样值得喂回 generator 重生成。当前实现
故意把 build_failed 排除掉（reasoning 是"infra 问题，重跑也没用"），但 LLM
hallucinate 出错代码这个场景下，**重跑反而有用**——LLM 看到"你刚才的提议编不过"
会去掉那个 broken case。

## 这件事告诉我们的

### 1. 实现 + 测试 ≠ 验证有效

Oracle-reflexion 9 个单测全过、机制完美、设计也对（lesson 不泄漏 mutation
内容，是"泛化反馈"），但在当前 benchmark 上**一次都没触发**。
"代码工作了"和"代码影响了 ablation 数字"是两件事。

### 2. 扩 LLM 的可达表达空间比扩反馈机制更直接

Reflexion 是"再给 LLM 一次说话机会"。但如果 LLM 当前 schema 只能说"3 个整数 m, k, n"，
再给多少次机会它都只能说 shape。**先让 LLM 能表达更多维度（schema 扩展），
再考虑给它更多反馈（reflexion）**。顺序很重要。

### 3. LLM 在 generative 任务里的随机性需要 oracle 兜底

Group A 跑出 "build_failed → fallback heuristic" —— 启发式恰好命中关键词，
表面 accuracy 是 10/10，但其中 2 个 seed 其实没真验过。Group B 重跑一次同样
配置就全部 build OK 了。这是温度 > 0 的天然代价。

要让 ablation 数字稳定，要么 temperature=0（损失多样性），要么跑 N 次取平均
（成本 N 倍），要么把 build_failed 也接进 reflexion 让 LLM 自纠错（最有意思）。

## 下一步候选

- **把 build_failed 接进 oracle-reflexion**：当前 `_run_one_inner` 在
  `build_status == "ok"` 才触发重跑；改成在 `build_failed` 时也触发，lesson
  改成 `"上一轮你提议的某个 case 让 cpp 编译失败：<stderr tail>。请重新
  考虑 shape 合法性，特别是 c_init 长度必须等于 m*n。"`。
- **planner 输出做 schema 校验**：在 `_generate_matmul` 里增加更严格的 case
  校验（c_init 长度、a/b shape 跟 m/k/n 一致），不合法的 case 直接 drop，让
  LLM 不能撞穿 cpp_renderer。
- **温度 0 跑一遍**：消除 random noise，看 oracle-reflexion 在确定性 LLM
  下是不是更容易被触发（应该不会，因为机制本身没问题，问题在 schema）。
- **更难的 seed**：当前 10 个 seed 都是 input-shape 维度的 bug。等扩到 30 seed
  涵盖 aliasing / stride / NaN 输入，oracle-reflexion 的"提示 LLM 往这些方向找"
  才真有用武之地。

## 复现

```bash
docker build -t tinyinfer-builder docker/builder/

# Group A：扩 schema，关 default prefill case，关 oracle-reflexion
TINYINFER_DOCKER_IMAGE=tinyinfer-builder TINYINFER_BUILD_DIR=tinyinfer/build-docker \
  PREFILLED_OUTPUT_BUFFER_DEFAULT=off LLM_PROVIDER=qwen DASHSCOPE_MODEL=qwen3.7-plus \
  EMBEDDING_PROVIDER=dashscope REFLEXION=off \
  python -m benchmark.harness --seeds benchmark/seeds --oracle real \
    --json results/real_qwen37_rag_no_prefill.json

# Group B：同上 + oracle-reflexion
TINYINFER_DOCKER_IMAGE=tinyinfer-builder TINYINFER_BUILD_DIR=tinyinfer/build-docker \
  PREFILLED_OUTPUT_BUFFER_DEFAULT=off LLM_PROVIDER=qwen DASHSCOPE_MODEL=qwen3.7-plus \
  EMBEDDING_PROVIDER=dashscope REFLEXION=off \
  python -m benchmark.harness --seeds benchmark/seeds --oracle real --oracle-reflexion \
    --json results/real_qwen37_rag_no_prefill_with_oracle_refl.json
```

Wall time：A ~15 min, B ~15 min（因为没触发 reflexion，B 没真的"二次跑"）。

## 总结

Oracle-feedback reflexion 是个**正确但今天没起作用的机制**。真正让 ablation
脱困的是"扩 LLM 的可达维度"——schema 加一个 `c_init` 可选字段，加上 prompt
里"如果你判断要测 output buffer 初值..."这一段，LLM (Qwen 3.7-plus) 就自己
主动提议了带 c_init 的 case，seed 03 被真正抓到（17 个测试，包含 LLM 主动加的
buffer-init case）。

**机制 vs. 表达力**：当生成器的表达力受限时，再多反馈也没用——先扩表达，
再加反馈。这条 devlog 是这个原则的实证。
