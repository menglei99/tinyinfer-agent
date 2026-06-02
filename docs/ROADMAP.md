# Roadmap

Target: 4-6 week solo project, presentable for AI 应用开发 interviews.

## Week 1 — MVP foundation ✅ (current state)

- [x] Project scaffolding, pyproject, gitignore
- [x] LangGraph StateGraph: parse_diff → route → generate → critic → report
- [x] One skill (numerical) end-to-end with numpy oracle
- [x] One operator (`matmul_fp32`) in C++ with baseline GTest
- [x] LLM abstraction with mock provider (key-less demo works)
- [x] JSONL trace writer
- [x] CLI: `analyze` and `build-and-test`
- [x] Sample diffs + Python smoke tests
- [x] README, ARCHITECTURE, ROADMAP, devlog 00

**Demo command (works today, no API key):**
```bash
python -m agent.cli analyze --diff demo/diffs/sample_matmul.diff --mock
```

## Week 2 — Multi-skill + MCP

- [x] Implement `softmax_fp32` C++ operator (with hand-written baseline)
- [x] Add a third operator (`layernorm_fp32`)
- [x] Performance skill: emit chrono-based regression budgets as GTest cases
- [x] Memory skill: emit guard-band stress tests (ASan-compatible)
- [x] Diff-content-aware routing (allocator change → memory, hot path change → perf)
- [x] **Custom MCP server** (`agent/mcp/server.py`)
  - tools: `cmake_configure`, `cmake_build`, `ctest_run`, `compile_only`
  - integrate via `langchain_mcp_adapters` or stdio loop
- [ ] Hand-written 10-test fault-injection benchmark; baseline accuracy

## Week 3 — RAG + advanced critic

- [ ] Hybrid retriever (BM25 + embedding + RRF) over:
  - ONNX op spec
  - tinyinfer source comments
  - past "fault-injection" rationales
- [ ] Optional Milvus backend (extra: `pip install -e ".[rag]"`)
- [ ] LLM critic with retrieved context, multi-dimensional coverage report
- [ ] Self-Consistency: N-way generation with majority shape selection
- [ ] Devlog 02 (RAG) and devlog 03 (critic + self-consistency)

## Week 4 — Eval, UI, CI, release

- [ ] Expand fault-injection benchmark to 30 commits
- [ ] Run ablation: with/without critic, with/without RAG
- [ ] Streamlit UI: diff input + test preview + critic feedback
- [ ] `langsmith` integration (dual-track observability)
- [ ] GitHub Actions: lint + mypy + pytest + benchmark on PR
- [ ] Docker compose: agent + Milvus
- [ ] 2-minute demo video; tech blog post (Chinese + English)
- [ ] Devlog 04 (final ablation + reflection)

## Stretch (Week 5-6)

- [ ] Real reference oracle via ONNX Runtime for fused / quantized ops
- [ ] Cross-hardware: emit ARM cross-compile test invocation
- [ ] Reflexion loop: failed test runs feed back as lessons
- [ ] Real open-source target (NCNN or mlx): demo on a real PR
- [ ] Resume bullet + interview talking-points doc

## Deliberately out of scope

- General code generation (this is a regression-test agent, not a coding agent)
- Web frontend beyond Streamlit
- Multi-user / auth / cloud
- Real-time streaming UX (defer until LangGraph events are plumbed properly)
