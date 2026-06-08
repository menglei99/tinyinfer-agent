# Devlog 07 —— 真实 oracle：accuracy 终于解锁，但揭示了一个新盲区

## 起点

[Devlog 05](05-ablation-results.md) 的头号问题：accuracy 列在所有 5 个配置上都饱和到 10/10 (100%)。原因是当时的 `_fault_caught_heuristic` 只看测试**名字/rationale** 是否含有 fault 维度关键词——而生成器的 `default_shape_cases` 已经把这些关键词命名了一遍。这条二值信号毫无区分能力，整张表只剩 critic 数据。

[Devlog 05 自己点名](05-ablation-results.md#%E4%B8%8B%E4%B8%80%E6%AD%A5)：**下一步是真 oracle —— 把 mutation 应用到 clean checkout，build，跑生成测试，看 fail/pass。需要本机 cmake。**

Devlog 06（reflexion 负向结果）也总结说："accuracy 列被启发式饱和了，不是被 agent。要么收紧启发式（compile + run），要么扩散种子。"

这条 devlog 把这件事做了。

## 实现

两块：

1. **`agent/tools/mutation.py`** —— in-process 的 unified-diff applier。解析 hunk + apply + 自动回滚（context manager），fuzz=5 行容错。不依赖 GNU patch（Windows / docker 容器内都没装）。
2. **`benchmark/harness.py`** —— 新增 `FAULT_ORACLE=real|heuristic` 切换。real 路径流程：
   - agent 走完 pipeline（生成 + install 到 `tinyinfer/tests/`）
   - 一次性 configure + build（拉 googletest，建 build dir）
   - 应用 mutation `with mutation.apply_diff(seed_diff, root):`
   - 增量 build（cmake 检测到只有 src 一个文件变了）
   - `ctest -R "test_<op>_fp32.*_generated"` 只跑 agent 生成的（避开 baseline 噪音）
   - 解析 ctest 输出，**至少一个 generated test FAIL → fault caught**

Build 全部走 [Docker builder](../DOCKER_BUILDER.md)（task #1 的产物），所以本机无 MSVC 也能跑。

每 seed 增量 build + ctest 的额外开销 ~5s，docker 启动 ~1s × 2 = 2s。10 seed 实际加 ~70s wall time。

## 结果：accuracy 解锁了，但 ablation 维度上**所有配置都一样**

5 个配置都用 real oracle 重跑：

| Config | Accuracy | Avg tests/seed | Critic pass rate | Avg critic iters | Avg missing dims | Avg RAG docs |
|---|---|---|---|---|---|---|
| real_mock_norag           | 9/10 (90%) | 7.20 | 1.00 | 1.00 | 0.00 | 0 |
| real_qwen_norag           | 9/10 (90%) | 8.40 | 0.40 | 1.70 | **1.90** | 0 |
| real_qwen_rag             | 9/10 (90%) | 8.20 | 0.40 | 1.70 | **0.90** | 5 |
| real_qwen_sc3             | 9/10 (90%) | 8.00 | 0.30 | 1.80 | 0.80 | 5 |
| real_qwen_rag_reflexion   | 9/10 (90%) | 8.80 | 0.30 | 1.70 | 0.80 | 5 |

两个事情同时是真的：

- **Accuracy 不再饱和到 100%**（每个配置都掉到 90%）—— 启发式信号已破除。
- **但 5 个配置 accuracy 完全一致**。每个配置都漏同一个 seed：`03_matmul_uninit_accumulator`。

## 为什么所有人都漏 seed 03

Seed 03 的 mutation：

```diff
-            float acc = 0.0f;
+            // BUG: accumulator not reset between output cells; reads stale c[].
+            float acc = c[i * n + j];
```

这个 bug 只有在**调用方传非零初值的 `c[]` 数组**时才会显式表现。如果 `c` 是 0 初始化的，`acc = 0 + sum(a*b) = sum(a*b)`，跟正确实现一致。

而 agent 生成的 `test_matmul_fp32_generated`，每一个 case 都长这样：

```cpp
std::vector<float> c(m * n);   // ← 默认 0 初始化
ASSERT_TRUE(tinyinfer::matmul_fp32(a.data(), b.data(), c.data(), m, k, n));
EXPECT_NEAR(c[0], expected[0], ...);
```

`std::vector<float> c(m*n)` 默认全 0。所以 mutation 后跑这些测试，结果还是对的——bug 没被触发。

这是个**生成器架构层面的盲区**：当前的 numerical skill 只 vary **input 张量的 shape/values**，不 vary **output buffer 的初值**。任何 LLM、任何 RAG、任何 SC、任何 reflexion 都改不了这件事，因为它们只能影响 _planner_ 的 shape 选择，影响不了 `cpp_renderer.render_matmul_test` 对 output buffer 的写法。

## 这件事告诉我们的

### 1. Real oracle 暴露了 heuristic 装聋作哑的故障类型

旧 heuristic 在 seed 03 上其实 PASS 了（关键词 `square / rect / aligned` 都有命中）。换成 real oracle 之后，"啊原来这些命名维度跟实际 fault 的触发条件根本不在一个轴上"——这是用启发式根本看不见的事。

这是整个 oracle upgrade 最有面试价值的 finding：**oracle 不只是把 10/10 拉到 9/10。它告诉你 agent 的系统性盲区在哪。**

### 2. RAG 在 critic 数据上的胜利依然存在

`qwen_norag` 平均 1.90 个 missing dim，`qwen_rag` 0.90 —— 2.1× 改善。devlog 05 旧数据是 4.3× (2.60 → 0.60)。方向一致，幅度小一点。差异可能来自：
- 模型升级：旧测用 qwen-2.5-plus，本次用 qwen3.6-plus。新模型本身 hallucination 更少，所以 RAG 边际收益变小
- 温度噪声：n=10 + 单 run，1σ 摆动可以解释 0.3-0.5 的差异

Critic 数据（不是 accuracy）依然是检验 RAG 价值的主信号。Devlog 05 的结论站得住。

### 3. SC=3 + Reflexion 在 accuracy 上**仍然无效**

跟 devlog 06 一致：SC 把 missing dim 从 0.90 微降到 0.80，但 accuracy 没动。Reflexion 也没动 accuracy。原因还是 devlog 06 已说过——生成器层面的微调改不了根本盲区。

### 4. Real oracle 不能加 30% accuracy 给你

很多人以为换 oracle 数字会立刻好看。事实是 5 个配置都 90%——**oracle 的精确化让原本饱和的指标变得有信号，但信号说的是"agent 整体能力有上限"**，不是哪个 ablation 杠杆更优秀。

要让 accuracy 区分开，下一步是：
1. **架构改动**：让生成器也 vary output buffer init（不是 vary shape）。这能让 seed 03 至少在某些配置下被抓到。
2. **种子扩展**：加更多需要"非标准调用约定"的 fault（uninit 是其中一种；还有 alias 输入输出、传 stride != n、传 NaN 输入等）。30 个 seed 时 accuracy 分布才能有差。

## 复现

```bash
docker build -t tinyinfer-builder docker/builder/

# 五个配置
TINYINFER_DOCKER_IMAGE=tinyinfer-builder TINYINFER_BUILD_DIR=tinyinfer/build-docker \
    LANGSMITH_TRACING=false REFLEXION=off \
    python -m benchmark.harness --seeds benchmark/seeds --mock --no-rag \
        --oracle real --json results/real_mock_norag.json

TINYINFER_DOCKER_IMAGE=tinyinfer-builder TINYINFER_BUILD_DIR=tinyinfer/build-docker \
    LANGSMITH_TRACING=false REFLEXION=off LLM_PROVIDER=qwen \
    python -m benchmark.harness --seeds benchmark/seeds --no-rag \
        --oracle real --json results/real_qwen_norag.json

# ... and so on for qwen_rag / qwen_sc3 / qwen_rag_reflexion

python -m benchmark.compare results/real_*.json --md results/ablation_real.md
```

Wall time per config（增量 build 已建好的 docker image 之后）：
- mock_norag: ~80 s
- qwen_norag: ~15 min
- qwen_rag:   ~16 min
- qwen_sc3:   ~30 min (3× LLM call)
- qwen_rag_reflexion: ~17 min

总：~80 min。

## 下一步

- 给 `cpp_renderer` 加 `c_init: "zero" | "garbage" | "ones"` 参数，让 numerical skill 在某些 shape case 里测非零 init buffer。预期能让 seed 03 在所有配置下都被抓到（顺便也可能在某些 LLM 配置上被漏到，因为 LLM planner 想得到 shape 但想不到 buffer init）。
- 把 oracle 报告的"failed test names"做成 second-pass critic 信号：oracle 跑完看到哪些 dim 没抓到，反馈回生成器（real-oracle 版 reflexion）。这条比 critic-feedback reflexion 强多了——失败的是**真实测试**，不是 LLM 的猜测。
- benchmark 从 10 扩到 30 seed，让 accuracy 分布有空间分化。

## 后续（同一天补的）：output-buffer init 维度落地

把上面"下一步"第 1 条做了：

**改动**（~50 行 + 4 个新单测）：
- `agent/tools/cpp_renderer.py::render_matmul_test` 加 `c_init: np.ndarray | None`。无值走老的 vector overload；有值切换到 pointer API 并预填 `std::vector<float> c = {…非零值…}`。
- `agent/tools/oracle.py::default_shape_cases("matmul_fp32")` 加一个 `prefilled_output_buffer_2x2`，`c_init=[2.5, -1.75, 3.125, -0.5]`。
- `agent/skills/numerical.py::_generate_matmul` 把 `sc.inputs.get("c_init")` 透传给 renderer。
- `agent/graph/nodes.py::_coverage_misses` 加一个 dim：`output_buffer_init_case`（结构 critic 检查是否有 prefilled/output_buffer/buffer_init 关键词的 test）。

**结果**（同 mock_norag 配置，real oracle 重跑）：

| Config         | 之前 | 加 buffer-init dim 之后 |
|---|---|---|
| real_mock_norag | 9/10 (90%) | **10/10 (100%)** ← seed 03 被抓 |

ctest 输出：

```
03_matmul_uninit_accumulator.diff: op=matmul_fp32 tests=10 build=ok
  (failed tests: test_matmul_fp32_generated)
```

mutation 把 `acc = 0.0f` 改成 `acc = c[i*n+j]`，正确实现下 acc 被 `c[i*n+j] = acc` 覆盖、c_init 的初值无关；buggy 实现下 c_init 的非零值被读进 acc，最终 `c[i*n+j] = c_init + sum(a*b) ≠ sum(a*b)`，EXPECT_NEAR 直接 fail。

**这件事对 ablation 信号的影响**：accuracy 列重新饱和到 10/10（所有配置都会跑这个 default case）。这是好事——agent 真的变强了，而不是 oracle 装聋作哑。要恢复 ablation 区分度，下一步要么把这个 case 改成 LLM-only（让 mock 不带、LLM 才提议），要么扩 30 seed 覆盖更多"非标准调用约定"的 fault 类型（alias 输入输出、传 stride、传 NaN）。这是 devlog 06 也碰到过的同一问题：**累积下来，accuracy 的信号取决于 seed 集是否仍能跑赢生成器的当前能力上限**。

**真正的胜利**：之前所有 5 个配置都漏的故障类型，现在 mock baseline 都能抓到。Agent 在 inference-framework regression test 这一类真实 bug 上**学了一个新维度**。整个改动通过现有 LangGraph node、cmake driver、docker builder 一路打通，零修改新依赖。

