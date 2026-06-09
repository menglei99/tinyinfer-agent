# 简历 + 面试讲稿（tinyinfer-agent）

岗位定位：LLM 应用 / Agent 工程 / AI 应用开发。中文版。

---

## 一、简历 bullet

放在简历的"项目经历"段。3-4 行版本（适合一页简历）：

> **tinyinfer-agent**｜AI 推理框架回归测试自动化 Agent（个人项目，4-6 周，[链接])
> - 用 **LangGraph** 编排 `parse_diff → route → retrieve_context → generate_tests → critic → install → execute` 主循环，配合自建 **MCP server** 把 C++ build/ctest 暴露给 agent 和 Claude Desktop 共用；本机无 MSVC 时通过 docker 容器透明跑通 cmake/build/ctest 全链路。
> - **Hybrid RAG**（BM25 + dense embedding + Reciprocal Rank Fusion，Chroma 后端）给 critic 做 grounding，把 LLM critic 的"幻觉缺失维度"从 2.60 降到 0.90（2× 改善）。Self-consistency、Reflexion、Oracle-feedback reflexion 三套反馈机制全部 ablation 验证。
> - **Real fault-injection oracle**：自实现 unified-diff applier + docker build + ctest 解析，把 mutation 应用到干净源码后真实编译并执行 agent 生成的 GTest，让 ablation accuracy 从启发式饱和的 100% 真正脱困，并由此发现一个所有 5 个配置都漏的架构盲区（output buffer 初值未 vary），随后用 ~50 行修复让 mock baseline 直接 9/10 → 10/10。
> - 8 篇中文 devlog（含两条"负向结果"诚实复盘）+ 150 个 pytest + Streamlit UI + LangSmith trace 集成。

精简单行（用于 LinkedIn 头条 / 极简简历）：

> 用 LangGraph + Hybrid RAG（BM25+dense+RRF）+ 自建 MCP server 做了个 AI 推理框架回归测试 agent；自实现 docker-based real oracle 跑通 mutation→build→ctest 闭环，从中发现 5 个 LLM 配置共同的架构盲区并 ~50 行修复。8 篇中文 devlog + 150 pytest。

---

## 二、30 秒电梯版

> 我做了一个 LangGraph 编排的 AI 推理框架回归测试 agent。它读 C++ 算子的 git diff，自动生成 GTest 数值精度 / 性能 / 内存三类测试，用 numpy 做 reference oracle 保证数值可信。
>
> 区别于通用 coding agent，它把"推理框架回归测试"这一个垂直场景做透：MCP server 把 C++ build/ctest 暴露给 agent 共用，docker builder 解锁本机没编译器的环境；hybrid RAG（BM25 + 稠密 embedding + RRF）让 LLM critic 不再幻觉缺失维度；真 fault-injection oracle（apply mutation → docker build → ctest）让 ablation 脱离启发式饱和。
>
> 项目最强的不是哪个 SOTA 数字，而是 8 篇中文 devlog 里有两条**负向结果**——reflexion 没起效、oracle-reflexion 一次都没触发——我把"为什么"分析清楚，最后发现真正瓶颈是**LLM 的可达表达空间**，扩 schema 比加反馈机制有效得多。

---

## 三、5 分钟项目 Tour

按 4 段话讲，每段 ~1 分钟。

### 段 1：问题 + 切入角度（~1 min）

> 通用 coding agent 满大街都是，但**AI 推理框架的回归测试**几乎没人做。推理框架改一个算子要同时验：数值精度（对照 reference）、性能延迟、内存安全、跨硬件一致性——这五个维度普通测试不会管，框架里却是一等公民。
>
> 我做的 agent 把这五个维度作为一等公民，输入是 C++ 算子的 git diff，输出是直接落盘的 GTest C++ 源码。一个核心设计：**LLM 选 shape，oracle 决定 value**。LLM 提议 "测哪种 shape / 边界 case"，但具体数值是 numpy 算出来的——保证生成的测试本身是 evidence-grounded，不是 LLM 在编。

### 段 2：架构 + MCP 解耦（~1.5 min）

> 主循环用 **LangGraph** StateGraph，七个节点：
> `parse_diff → route_skill → retrieve_context → generate_tests → critic → install_tests → execute_tests → write_report`。
>
> Critic 是带反馈环的：critic FAIL 就重新生成，最多两轮。三个 skill（numerical / perf / memory）按 diff 关键词路由（SIMD intrinsic → perf，allocator 改动 → memory）。
>
> 比较有意思的工程决策是 **MCP 解耦**：把本地 cmake/build/ctest 抽成一个 MCP stdio server，agent 和 Claude Desktop / Cursor 共用同一套 wire format。这件事让"agent 不是 toolchain 的唯一入口"——任何 MCP 客户端都能调。
>
> 然后是 **docker builder**：开发机没 MSVC、cmake configure 会挂在 `CMAKE_CXX_COMPILER not set`。我在 `cmake_driver` 内部加了一个切换层，env 设上 `TINYINFER_DOCKER_IMAGE` 就把每条命令包成 `docker run --rm -v <repo>:/work` 跑在 gcc:13 容器里。MCP server、LangGraph 节点零修改。踩过两个坑：Git Bash 把 `/work` 当 Windows 路径误翻译；MCP SDK `StdioServerParameters.env=None` 时只继承一个安全 env 子集，导致子进程拿不到我设的 docker 镜像名。

### 段 3：RAG + critic + oracle（~1.5 min）

> RAG 用 **Hybrid 检索**：BM25（词面）+ DashScope 稠密 embedding（语义）+ Reciprocal Rank Fusion 融合。corpus 是手写的 ONNX 算子语义 + src 注释 + 历史 fault 总结，~50 条。默认后端 Chroma（嵌入式 sqlite，Windows 友好），Qdrant / Milvus 可选。RAG 主要价值在 critic：把 critic 的"幻觉缺失维度数"从 2.60 降到 0.90，2× 改善。
>
> Oracle 是这个项目最值得讲的部分。最早的 fault-caught 判定是**启发式**——看测试名 / rationale 里有没有 fault 关键词。十个 seed 全部"caught"，ablation accuracy 饱和到 100%，五个 LLM 配置毫无区分。
>
> 我后来做了**真 oracle**：自己写了一个 minimal unified-diff applier（带 fuzz=5 行容错），把 mutation 应用到干净源码 → docker build → ctest `-R` 只跑 agent 生成的测试 → 解析 `***Failed` 看哪些 fail → 用 context manager 保证源码回滚。
>
> 结果意外：5 个配置 accuracy 都从 100% 掉到 90%，**全员漏同一个 seed**——`acc = c[i*n+j]`（accumulator 不初始化）。根因是 generator 架构里从来没 vary 过 output buffer 的初值，`std::vector<float> c(m*n)` 默认 0 init 把这类 API contract bug 完全藏起来。我加了 50 行：`cpp_renderer` 加 `c_init` 走 pointer API、`oracle.py` 加一个 prefilled case、structural critic 加新维度。mock baseline 直接 9/10 → 10/10。

### 段 4：负向结果 + 教训（~1 min）

> 项目有两条**诚实的负向结果**写进了 devlog：
>
> 第一条是 **critic-feedback Reflexion**——critic FAIL 的反馈做成 lesson 喂回 generator。教科书做法，~50 行 + 14 个测试。实际跑下来：accuracy 几乎不变。devlog 06 分析三个原因——scope（lesson 只影响 4/10 seed）、critic 跨轮目标漂移、n=10 噪声大。
>
> 第二条更狠：**oracle-feedback Reflexion**——oracle 报 missed 时把"漏抓"信号回灌给 generator。机制干净、9 个单测全过，但跑真实 ablation 时**一次都没触发**。原因是我先扩了 LLM planner 的 schema 让它能 propose `c_init`，结果 LLM 第一次就抓到了所有 fault，根本没机会"miss → retry"。
>
> 教训：**扩 LLM 的可达表达空间比加反馈机制直接得多**。当生成器只能说 `{m, k, n}` 时，再多 lesson 也没载体。先扩表达，再加反馈——顺序很重要。这条 devlog 8 是项目最深的工程 insight。

---

## 四、30 分钟深挖版（要点 + 应答）

按面试官可能切入的角度分块。每块给"核心点"和"被追问时的回答"。

### 0. 自我介绍 + 一句话 elevator pitch（1 min）

照 30s 电梯版。

### 1. 为什么不是通用 coding agent？为什么选这个题（3 min）

**核心点**：
- 通用 agent 满大街，差异化在垂直 vertical
- 推理框架回归测试是一类**特殊**的测试任务：需要数值精度对 reference / 需要延迟 budget / 需要内存安全维度，普通 unittest 不管
- 这些维度是推理框架开发的"一等公民"，agent 把它们做成 first-class skill

**被追问"为啥不直接用 Anthropic / OpenAI 的 SOTA 通用 agent + tool use"**：
- 通用 agent 不知道"数值精度容差"这个维度——它会让 LLM 直接写 EXPECT_EQ
- 我把维度做成结构（skill + critic 的 dim 检查），LLM 提议 shape，oracle 出 value，整个测试是 evidence-grounded
- 这是"agent 工程"而不是"prompt 工程"——结构决定了 LLM 出错的边界

### 2. LangGraph 选型（3 min）

**核心点**：
- LangGraph = StateGraph，节点之间的 state 是 typed dict，list 字段有 reducer
- critic loop 的"回头边"靠 conditional_edges 实现
- 优点：可观测（每个节点 emit update），可序列化 state，跟 LangSmith trace 天然兼容
- 选 LangGraph 不选 autogen / Pydantic AI：因为我要的是**确定性 pipeline + 局部 LLM 调用**，不是 multi-agent 对话

**被追问"如果重做会用啥"**：
- 现在的 graph 其实可以用纯 Python + Pydantic state 写，工作量没差太多
- 选 LangGraph 的真正好处是 LangSmith trace 一行不写就有；如果不在意 trace 平台，自写 state machine 更轻

### 3. MCP server 设计（3 min）

**核心点**：
- 把 cmake configure/build/ctest 抽成 5 个 MCP tool，通过 stdio 暴露
- 任何 MCP 客户端（包括 Claude Desktop / Cursor / 我的 agent）都能用同一套 wire format
- 解耦的好处：agent 不是 toolchain 的唯一入口；Claude Desktop 可以独立用这套 tool 测试
- 实现踩坑：MCP SDK 默认 `StdioServerParameters.env=None` 只继承安全 env 子集，导致子进程看不到我设的 docker 镜像名——手动透传 `os.environ` 才修

**被追问"MCP vs 直接调 cmake_driver 的区别"**：
- 直接调 driver = LangGraph 节点和 toolchain 耦合
- 走 MCP = toolchain 是一个独立 stdio server，多个 client 共享
- 跨 IDE / 工具的可重用是真实的工程价值

### 4. RAG：为什么 hybrid，为什么 RRF（3 min）

**核心点**：
- BM25 抓精确 token（op 名字、type 名字、error 关键词），dense 抓语义 paraphrase
- 单独用任一个都会漏：BM25 不懂 paraphrase，dense 把精确 token 模糊掉
- 用 **Reciprocal Rank Fusion** 不用调 score threshold——只看 rank
- 公式：`score(d) = w_bm25 / (k + rank_bm25) + w_dense / (k + rank_dense)`，k=60，w_bm25=0.4
- corpus 故意保留 ASCII（rank-bm25 按空白分词，中文会让召回掉一半）

**被追问"为什么不只用 dense embedding"**：
- 看 devlog 02 的对比：纯 dense 在"matmul k1 outer product"这种短精确 query 上 rank 不稳
- BM25 在精确 token 上 rank 鲁棒；dense 在 paraphrase 上鲁棒；融合互补
- ablation 数字：RAG 把 critic 幻觉缺失维度从 2.60 降到 0.90（2.9× 改善）

### 5. Critic 设计：结构 critic + LLM critic + Reflexion（5 min）

**核心点**：
- **两层 critic**：结构 critic 是确定性规则（每个 (op, skill) 该覆盖哪些 dim），LLM critic 看检索到的 corpus 片段做 grounding
- 结构先跑（便宜），失败直接 FAIL；结构通过再调 LLM critic（贵但严）
- Critic loop max 2 轮——避免 LLM 卡循环
- **Reflexion**：critic FAIL 的反馈做成 lesson 字符串，下一轮 generator 的 prompt 前缀里挂上

**被追问"Reflexion 为啥负向结果"**：
- devlog 06 列三个原因（scope / 漂移 / n 小）
- 但真正根本原因（写在 devlog 08）：当生成器只能 propose `{m, k, n}` 时，再多 lesson 也只能影响 shape 选择，影响不了 buffer init / aliasing 这些维度
- 教训：先扩生成器表达力（schema），再加反馈机制
- 改了 schema 让 LLM 能 propose `c_init`，**问题在 30 秒内就解了**——LLM 看 prompt 提示自然就用了新维度

### 6. Self-consistency（2 min）

**核心点**：
- 抽 N 份独立 shape plan，多数投票留下 ≥ ceil(N/2) 出现的 shape
- 用 latency / $ 换"对单次 hallucinate outlier 的鲁棒性"
- ablation：SC=3 把 critic missing 从 0.90 降到 0.80（边际改善），但 LLM 调用 3x

**被追问"SC vs 直接降 temperature"**：
- temperature=0 的代价是丢多样性，可能漏掉非典型 shape
- SC 保留多样性，靠投票滤掉异常
- 我两个都跑了，结论是 SC 在我的 benchmark 上微改善，温度 0 没区别——可能 prompt 已经足够引导

### 7. Real oracle 怎么做的（5 min）

**核心点**：
- 三步：apply mutation → docker build → ctest
- **mutation applier 自己写**，不依赖 GNU patch（容器内也没装）。150 行 + 8 个单测
- unified diff 解析：单文件多 hunk，fuzz=5 行容错（seeds 写时和现在 src 有偏移）
- context manager：apply 进 try，revert 在 finally，**保证半途出错也不污染源码**
- docker build：增量编译，每个 seed 大概 5 秒
- ctest -R 只跑 agent 生成的（避免 baseline 被 mutation 影响 = 假阳性）
- 解析 `***Failed` 行抽出失败测试名

**关键发现讲故事**：
- 5 个配置都 9/10，**全员漏 seed 03**
- 不是 LLM 不够好，是 generator 架构盲区
- 加 50 行修复后 mock baseline 直接拉满
- 这就是"oracle 不是用来涨数字，是用来暴露盲区"

**被追问"为什么不直接用 git apply"**：
- seed 的 hash 是占位 `1111111` 不真实
- git apply 严格 hash 校验会拒绝
- patch 命令容器没装；GNU patch 也不在 base image
- 自己写 minimal applier 更可控（fuzz 行为完全在我手里）

**被追问"context manager 的 revert 怎么处理 OS error"**：
- finally 块里 try/except；任何 restore 失败 raise RuntimeError 大声报错
- 因为污染源码是**最糟的结果**——必须吵醒用户

### 8. Oracle-feedback Reflexion（4 min）

**核心点**：
- 这是 task #6，比 critic-feedback reflexion 强：失败的是真实测试 fail/pass，不是 LLM critic 的猜测
- 机制：oracle 报 missed → 注入 `ORACLE_MISS_LESSON` 到 state → 重跑一次 graph → 再 oracle
- lesson **故意不**包含 mutation diff 内容（那是抄答案）——只说"你漏了，请考虑 shape 之外的维度"
- 9 个单测覆盖三态行为
- **实测：一次都没触发**

**为啥负向结果**：
- 上面 Reflexion 段已经说了——schema 不够表达
- 扩 schema 后 LLM 第一次就抓到了
- 这条 devlog 是项目最深的工程 insight

**被追问"那 oracle-reflexion 是不是白做"**：
- 不是。机制本身是对的：当 seed 集扩到 30 个、覆盖 alias / stride / NaN 这些维度时，LLM 第一次可能漏，那时 oracle-reflexion 真有用
- 现在只是"benchmark 太简单，触发不到"
- 投资在机制上的成本（9 个单测 + 通用 lesson 设计）已经在那里

### 9. 为什么写 8 篇中文 devlog（2 min）

**核心点**：
- 工程项目的真正价值是"复现性 + 推理过程"，不只是最终代码
- 每个决策有 trade-off，写下来强迫自己想清楚
- 两条**负向结果**（Reflexion + Oracle-Reflexion）也写——这是诚实
- 面试官能从 devlog 里直接看到我怎么思考、踩什么坑、得什么教训

### 10. 还有时间的话讲什么

- **LangSmith trace**：3 个 env var 就开，wrap_openai 让 LLM 调用变 LLM span
- **Streamlit UI**：跟 CLI 共用同一份 LangGraph 构造，行为一致
- **Docker compose**：Qdrant + Milvus profile，DEFERRED VALIDATION 没本机起过
- **Benchmark 设计**：10 个 seed 故意手写覆盖三个算子三类故障

---

## 五、被问到弱点 / 可改进点的回答

主动承认这几点，比被发现强：

1. **Seed 只有 10 个，温度噪声大**。30 seed 在 ROADMAP 上待做。
2. **Reflexion 是负向结果，我留着它不删**。因为机制本身没错，是 benchmark 触不到。
3. **没在真实开源项目（NCNN / mlx）上跑过**。stretch goal，没做完。
4. **MSVC 路径没本机验证**。我所有 build 都走 docker，Windows 原生 cmake 路径理论上能走但没试过。
5. **C++ side 只有 3 个算子**（matmul/softmax/layernorm）。conv、quantize 在 ROADMAP 上 stretch。

---

## 六、千万别说的话

- "我做了一个 SOTA agent" —— 项目不大，别吹
- "用了 LangGraph 所以更好" —— 选型有 trade-off，要讲为啥
- 跳过 Reflexion 负向结果 —— 直接讲，这是项目最值钱的内容之一
- "Anthropic / OpenAI 的 agent 都做得很烂" —— 没必要踩别人
