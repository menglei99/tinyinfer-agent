# Devlog 00 — Kickoff: why a regression test agent for inference frameworks

## Motivation

I work on AI inference frameworks in C++ (autonomous driving stack).
The single most painful day-to-day task is **regression testing after an
operator change**. A senior engineer typically:

1. Reads the diff to figure out what could have changed numerically
2. Hand-writes shape variations, dtype combos, edge cases
3. Computes expected values via PyTorch / numpy
4. Embeds them as `EXPECT_NEAR` in GTest
5. Repeats for performance and memory dimensions

This work is largely mechanical, but it requires deep domain knowledge:
*you have to know which dimensions matter*. Without that knowledge, generic
test-generation tools either over-test (slow) or miss the regression that
actually matters (silent drift).

LLM agents look promising here, but a naive "ask LLM to write GTest" approach
fails the trust bar:

- LLM may invent expected numeric values → false confidence
- LLM may not know which shape edges matter for which op
- LLM may emit code that doesn't compile against your ABI
- No reproducibility, no audit trail

**Hypothesis:** with the right division of labour between LLM (creative
shape planning) and deterministic tooling (numpy/ONNX oracle, C++ renderer),
we can build something a senior framework engineer would actually trust.

## What I'm not building

- Not a generic code-generation agent (Cursor / Aider already exist)
- Not a SWE-bench solver (DM-Code-Agent and others handle that)
- Not a chatbot wrapper around an LLM
- **Not** trying to replace the senior engineer — augment them

## Design rules I committed to on day 1

1. **LLM picks shapes; oracle picks values.** No exceptions.
2. **Mock LLM mode must produce real, runnable tests.** This forces the
   architecture to be valid even when the LLM is dumb. It also makes
   key-less demoing possible.
3. **Every node update is traced as JSONL.** No "ask the model again to
   debug" loops.
4. **Coverage is structured, not hand-waved.** A critic with explicit
   dimensions (`shape_variety`, `numerical_stability`, ...) is the only way
   to know if the test set is "good".

## What's in the MVP

- `LangGraph` StateGraph with five nodes: parse → route → generate →
  critic → report. Critic can loop back up to twice.
- `numerical` skill that drives a numpy oracle, handles `matmul_fp32`
  and `softmax_fp32` (the latter as a forward-looking generator since
  the C++ op is still TODO).
- `tinyinfer` C++ project with one real operator and a hand-written
  baseline test that proves the test infra works.
- `mock` LLM that walks the full graph without a key.
- JSONL trace + `analyze` CLI.

## What I learned in week 1

- **`Annotated[list, reducer]`** in TypedDict is the cleanest way to do
  state aggregation in LangGraph. Without it, the critic loop wipes out
  earlier-iteration test cases on each update.
- **Two-tier critic** (deterministic structural rules first, LLM second)
  is more reliable than LLM-only. The structural critic catches the cases
  the LLM most often hallucinates around (claiming coverage that isn't
  actually present).
- **Don't ask the LLM to emit raw GTest C++.** The Python renderer is
  100 lines and is mechanical. Let the LLM stay in JSON-mode for shapes.

## Next (week 2)

- Add `perf` and `memory` skills — these are where the real
  domain expertise shows up.
- Custom **MCP server** wrapping cmake/ctest. This is the cross-tool
  ecosystem play: the same MCP server can be used by Claude Desktop,
  Cursor, or any other MCP-aware IDE.
- Build a 10-test **fault-injection benchmark** — known-bad commits where
  the "right" generated test should fail. This is the single most
  important data point for the resume.

## Open questions

- Should performance regressions be tested with a fixed-machine baseline
  or a relative threshold? CI machines are noisy.
- For quantize ops, ONNX Runtime's reference may differ from PyTorch's
  by quant rounding. Which is "the" oracle?
- How do I get the LLM critic to call out missing **dtype** coverage
  reliably? Structural rules struggle here because the dtype is implicit
  in the op signature.

I'll revisit these after the first ablation in week 4.
