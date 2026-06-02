# Architecture

## 1. Why this shape

Building a regression test agent for an inference framework is **not** the same
as a generic "LLM writes pytest" project. The unique constraints:

1. **Numerical correctness needs a trusted oracle**
   — the LLM must not invent expected values
2. **Coverage dimensions are domain-specific**
   — shape × dtype × edge × hardware × layout, not just code paths
3. **Toolchain is C++** (cmake / ctest / ASan / perf), not pytest
4. **Failure modes are subtle**
   — fp32 max diff `1e-7` may be fine, `1e-3` may be a real regression

The architecture below is shaped by these constraints.

## 2. Pipeline

```
git diff
   │
   ▼
parse_diff ─── identify changed operators (regex first, LLM as fallback)
   │
   ▼
route_skill ── pick which testing dimensions apply
   │            (numerical / performance / memory / hardware)
   ▼
generate_tests
   │   ┌──────────────────────────────────────────────┐
   │   │  for each (op, skill):                       │
   │   │     LLM → propose shapes / cases             │
   │   │     numpy/ONNX oracle → compute references   │
   │   │     renderer → emit GTest C++ source         │
   │   └──────────────────────────────────────────────┘
   ▼
critic ────── structural rules first, then LLM peer-review
   │            verdict.passed = false  →  loop back to generate_tests
   ▼
write_report ── emit markdown summary
   │
   ▼
(optional) build_and_test ── invoke cmake/ctest, write ExecutionResult
```

## 3. Key design decisions

### LLM picks shapes, oracle picks values

The single most important rule. The LLM is good at **what to test**
(shapes, edges, dtype combinations). It is unreliable at **expected numeric
values**. We strictly separate the two:

```python
# Skill asks LLM:
shapes = llm.plan_shapes(op_name)            # creative

# Oracle computes values:
expected = numpy_or_onnx_reference(shape)    # deterministic

# Renderer assembles:
emit_gtest(shape, expected)                  # mechanical
```

This is the "evidence-grounded test generation" pattern; it's also why mock
mode produces real, correct tests — the values come from numpy regardless.

### State is a TypedDict with reducer

LangGraph state is a single dict that flows between nodes. List fields use a
reducer that **concats** instead of overwriting, so the critic loop can
accumulate tests across iterations rather than throwing away work.

### Critic is two-tier

- **Structural critic** (deterministic, fast) — runs first. Cheap rules
  ("does test set include a thin shape?"). Produces a verdict on its own.
- **LLM critic** (slow, optional) — runs only when structural passes.
  Catches dimensions structural rules don't know about.

If the LLM critic fails to parse, we fall back to the structural verdict.
This avoids one LLM hiccup blocking the pipeline.

### MVP keeps tools as Python functions

`agent/tools/cmake_driver.py` exposes `configure / build / ctest` as plain
functions. **Week 2** lifts these into an MCP server so the same tools can
be reused by Claude Desktop, Cursor, or any MCP-aware IDE.

## 4. Module map

| Module | Role |
|--------|------|
| `agent/state.py` | Pydantic models + TypedDict state |
| `agent/llm/` | LLM client abstraction (DeepSeek/OpenAI/Anthropic/Mock) |
| `agent/skills/` | One module per testing dimension |
| `agent/tools/` | Reference oracle, C++ renderer, cmake driver |
| `agent/graph/` | LangGraph nodes + StateGraph builder |
| `agent/tracing/` | JSONL trace writer |
| `agent/cli.py` | Typer CLI |
| `tinyinfer/` | C++ system under test |

## 5. Extension points

- **New operator**: add header + impl + baseline test in `tinyinfer/`,
  add an oracle in `agent/tools/oracle.py`, add a generator branch in
  `agent/skills/numerical.py`.
- **New skill**: subclass `Skill`, register in `agent/skills/__init__.py`,
  add routing rule in `route_skill_node`.
- **New LLM provider**: add a class in `agent/llm/client.py` and a branch
  in `get_llm_client()`.
