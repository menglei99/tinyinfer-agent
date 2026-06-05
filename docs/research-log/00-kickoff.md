# Devlog 00 —— 立项：为什么给推理框架做回归测试 Agent

## 动机

我做 AI 推理框架的 C++（自动驾驶栈）。日常最痛的事是
**改完算子之后要做回归测试**。一个高级工程师典型的流程是：

1. 读 diff，搞清楚数值层面可能改了什么
2. 手写一堆 shape 变化、dtype 组合、edge case
3. 用 PyTorch / numpy 算出期望值
4. 把期望值嵌成 GTest 的 `EXPECT_NEAR`
5. 性能和内存维度再做一遍

这活儿大部分是机械的，但需要深领域知识：**你得知道哪些维度才是重要的**。
不懂这一点，通用测试生成工具要么过度测试（慢），要么漏掉那条真正会产生
漂移的回归（静默失败）。

LLM agent 在这里看起来很有戏。但简单的"让 LLM 写 GTest"过不了信任关：

- LLM 可能编出错的期望数值 → 假信心
- LLM 可能不知道哪个 op 哪种 shape edge 重要
- LLM 可能产生编不过你 ABI 的代码
- 没有可复现性，没有审计 trail

**假设**：如果在 LLM（创造性 shape planning）和确定性工具
（numpy/ONNX oracle、C++ renderer）之间做好分工，
能搞出一个高级框架工程师真的愿意用的东西。

## 我不打算做的事

- 不是通用代码生成 agent（Cursor / Aider 已经在了）
- 不是 SWE-bench 求解器（有 DM-Code-Agent 等等）
- 不是 LLM 套个对话壳
- **不是**想替代高级工程师 —— 是给他们做增强

## Day 1 立的设计规则

1. **LLM 选 shape，oracle 选数值。** 不许例外。
2. **Mock LLM 模式必须产出真实可跑的测试。** 这强迫架构在 LLM 蠢的时候
   也能成立，同时让无 key demo 成为可能。
3. **每个节点更新都写 JSONL trace。** 杜绝"再问一遍模型来 debug"。
4. **覆盖是结构化的，不是手挥。** 一个带显式维度
   （`shape_variety`、`numerical_stability` 等）的 critic 才是判断
   测试集"是否足够"的唯一方法。

## MVP 里有什么

- `LangGraph` StateGraph 五个节点：parse → route → generate → critic → report。
  Critic 最多回环两次。
- `numerical` skill 驱动 numpy oracle，目前处理 `matmul_fp32` 和
  `softmax_fp32`（后者暂时只生成 disabled 测试，因为 C++ 算子还是 TODO）。
- `tinyinfer` C++ 项目，一个真算子 + 一个手写 baseline 测试，证明
  测试基础设施 work。
- 不带 key 也能跑完整条 graph 的 `mock` LLM。
- JSONL trace + `analyze` CLI。

## Week 1 学到的

- **TypedDict 里的 `Annotated[list, reducer]`** 是 LangGraph 做状态聚合的
  最干净写法。不加它的话，critic loop 每次 update 会把上轮测试用例
  全冲掉。
- **两层 critic**（确定性结构规则在前，LLM 在后）比纯 LLM 可靠很多。
  结构 critic 抓得住 LLM 最容易瞎说的情况（声称覆盖了实际不存在的东西）。
- **别让 LLM 直接吐 GTest C++。** Python renderer 一百行解决，纯机械。
  让 LLM 待在 shapes 的 JSON 模式里就好。

## 下一步（Week 2）

- 加 `perf` 和 `memory` skill —— 这俩才是真正体现领域专长的地方。
- 自定义 **MCP server** 包 cmake/ctest。这是跨工具生态的卖点：
  同一份 MCP server 能被 Claude Desktop、Cursor 或任何 MCP-aware IDE 用。
- 做一个 10 条的 **fault-injection benchmark** —— 已知坏 commit，
  "正确"生成的测试应该 fail。这是简历上最重要的数据点。

## 还没想清楚的问题

- 性能 regression 测试应该用固定机器的 baseline 还是相对阈值？
  CI 机器抖动大。
- quantize 算子，ONNX Runtime 的参考值可能跟 PyTorch 在量化舍入上不一致。
  谁是"那个"oracle？
- 怎么让 LLM critic 可靠地指出 **dtype** 覆盖缺失？结构规则在这里有点弱，
  因为 dtype 通常隐式藏在 op 签名里。

Week 4 跑完第一次 ablation 后再回头看这几条。
