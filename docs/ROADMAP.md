# Roadmap

目标：一个 4-6 周的个人项目，能拿来面试讲 AI 应用开发。

## Week 1 —— MVP 地基 ✅

- [x] 项目脚手架、pyproject、gitignore
- [x] LangGraph StateGraph：parse_diff → route → generate → critic → report
- [x] 一个 skill（numerical）端到端打通，配 numpy oracle
- [x] 一个算子（`matmul_fp32`）C++ 实现 + baseline GTest
- [x] LLM 抽象 + mock provider（无 key demo 可跑）
- [x] JSONL trace writer
- [x] CLI：`analyze` 和 `build-and-test`
- [x] 示例 diff + Python smoke test
- [x] README、ARCHITECTURE、ROADMAP、devlog 00

**Demo 命令（不需要 API key）：**
```bash
python -m agent.cli analyze --diff demo/diffs/sample_matmul.diff --mock
```

## Week 2 —— 多 skill + MCP

- [x] 实现 `softmax_fp32` C++ 算子（带手写 baseline）
- [x] 加第三个算子（`layernorm_fp32`）
- [x] Performance skill：基于 chrono 的 regression budget GTest 用例
- [x] Memory skill：guard-band 压力测试（与 ASan 兼容）
- [x] diff 内容感知的路由（allocator 改动 → memory，hot path 改动 → perf）
- [x] **自定义 MCP server**（`agent/mcp/server.py`）
  - tools：`cmake_configure`、`cmake_build`、`ctest_run`、`compile_only`
  - 通过 `langchain_mcp_adapters` 或 stdio loop 接入
- [x] 手写的 10 条 fault-injection benchmark + 基线准确率

## Week 3 —— RAG + 高级 critic

- [x] Hybrid retriever（BM25 + embedding + RRF），corpus 包含：
  - 手写的 ONNX 算子语义
  - tinyinfer 源码注释
  - 历史 fault-injection 的归纳总结
- [x] 可选的 Milvus 后端（extra：`pip install -e ".[rag-milvus]"`） —— 默认实际用 Chroma
- [x] 带 retrieved context 的 LLM critic，多维 coverage report
- [x] Self-Consistency：N 路生成 + 多数投票（环境变量控制）
- [x] Devlog 02（RAG）和 devlog 03（critic + self-consistency）

## Week 4 —— Eval、UI、CI、发布

- [ ] fault-injection benchmark 扩展到 30 个 commit
- [x] 跑 ablation：有/无 critic、有/无 RAG（已在本机用 Qwen 完成 5 种配置对比，见 devlog 05）
- [x] Streamlit UI：diff 输入 + 测试预览 + critic 反馈
- [x] `langsmith` 集成（环境变量即开即用 + `wrap_openai` 把 LLM 调用变成 LLM span）
- [ ] GitHub Actions：lint + mypy + pytest *（应用户要求暂时不做）*
- [x] Docker compose：Qdrant + 可选 Milvus profile *（已写，未在本机起服务）*
- [ ] 2 分钟 demo 视频；技术博客（中文）
- [x] Devlog 04（eval + UI）

## Stretch（Week 5-6）

- [x] **Docker 编译器后端**：本机无 MSVC/g++ 时用 docker 容器跑 cmake/build/ctest。
       MCP server / LangGraph 节点零修改，docker 切换在 `cmake_driver` 内部完成。
       详见 [`docs/DOCKER_BUILDER.md`](DOCKER_BUILDER.md)。
- [x] **真实 fault-injection oracle**：apply mutation → docker build → ctest，
       让 ablation accuracy 列脱离启发式饱和。结果在 devlog 07：5 个配置都掉到 90%
       （都漏 seed 03 uninit accumulator），暴露 agent **共同的架构盲区**。
- [ ] 通过 ONNX Runtime 给 fused / quantized 算子做参考 oracle
- [ ] 跨硬件：触发 ARM 交叉编译的测试调用
- [x] **Reflexion loop**：critic FAIL 的反馈喂回 generator 作为 lesson —— 已实现，
       通过 `REFLEXION` 环境变量门控；ablation devlog 06 表明在当前 benchmark
       上没显著提升（scope 问题 + critic 漂移），保留作进一步探索。
- [x] **Output-buffer init 维度**：让 numerical skill 在某些 case 测非零 init buffer，
       让 seed 03 类的 uninit-accumulator fault 被抓到。落地后 mock_norag accuracy
       从 9/10 → 10/10 —— agent 真的学了一个新维度。devlog 07 末尾补记。
- [ ] 真实开源项目目标（NCNN 或 mlx）：在真实 PR 上 demo
- [ ] 简历 bullet + 面试讲稿

## 故意不做的事

- 通用代码生成（这是回归测试 agent，不是 coding agent）
- Streamlit 以外的 Web 前端
- 多用户 / 鉴权 / 云
- 实时流式 UX（等 LangGraph events 接通了再说）
