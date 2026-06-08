"""LangGraph nodes.

Each node takes the current AgentState and returns a partial state update.
Keep them small, side-effect-light, and individually testable.
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from agent.llm import LLMClient
from agent.skills import get_skill
from agent.state import (
    AgentState,
    ChangedOp,
    CriticVerdict,
    ExecutionResult,
    SkillKind,
)


# ---------- parse_diff ----------

# Files we care about. Diff hunks elsewhere are ignored.
_OP_FILE_RE = re.compile(
    r"^\+\+\+ b/(?P<path>tinyinfer/(src|include/tinyinfer)/(?P<stem>[a-zA-Z0-9_]+)\.(cpp|hpp))",
    re.MULTILINE,
)


def parse_diff_node(state: AgentState, llm: LLMClient | None = None) -> dict[str, Any]:
    diff = state.get("diff", "") or ""
    if not diff:
        return {"changed_ops": [], "error": "empty diff"}

    seen: OrderedDict[str, ChangedOp] = OrderedDict()
    for match in _OP_FILE_RE.finditer(diff):
        path = match.group("path")
        stem = match.group("stem")
        op_name = stem if stem.endswith("_fp32") else f"{stem}_fp32"

        if op_name not in seen:
            seen[op_name] = ChangedOp(
                name=op_name,
                file_path=path,
                summary="modification detected by static diff parse",
            )

    # If regex found nothing, optionally consult LLM (skipped here for determinism).
    return {"changed_ops": list(seen.values())}


# ---------- route_skill ----------


# Keyword sets used by the routing heuristic. Keep these short and high-signal —
# anything ambiguous (e.g. "loop") would route everything to everything.
_PERF_KEYWORDS = (
    "simd", "vectorise", "vectorize", "unroll", "tile", "blocked",
    "fastpath", "fast path", "cache", "loop reorder",
    "hot path", "intrinsic", "avx", "neon",
    "#pragma omp", "#pragma unroll", "#pragma gcc",
    "_mm_", "_mm256_", "_mm512_",
)
_MEMORY_KEYWORDS = (
    "malloc(", "free(", "new[", "delete[", "new ", "delete ",
    "memcpy", "memset", "memmove",
    "allocator", "alloc(", "realloc",
    ".resize(", ".reserve(",
)


def _detect_skills_from_diff(diff: str) -> list[SkillKind]:
    """Heuristic: scan the +-prefixed diff lines for skill-trigger keywords.

    NUMERICAL is always on — correctness is the floor. PERF / MEMORY are
    additive based on what the diff actually touches.
    """
    skills: list[SkillKind] = [SkillKind.NUMERICAL]
    if not diff:
        return skills

    # Only consider added/removed code, not the surrounding context.
    plus_lines = [
        ln.lower()
        for ln in diff.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
        or ln.startswith("-") and not ln.startswith("---")
    ]
    haystack = "\n".join(plus_lines)
    if any(kw in haystack for kw in _PERF_KEYWORDS):
        skills.append(SkillKind.PERFORMANCE)
    if any(kw in haystack for kw in _MEMORY_KEYWORDS):
        skills.append(SkillKind.MEMORY)
    return skills


def route_skill_node(state: AgentState) -> dict[str, Any]:
    """Pick which Skills to run for the changed ops.

    Always routes to NUMERICAL. PERF / MEMORY are added when the diff
    body contains skill-relevant keywords (eg. SIMD intrinsics → perf,
    allocator changes → memory).
    """
    diff = state.get("diff", "") or ""
    return {"selected_skills": _detect_skills_from_diff(diff)}


# ---------- retrieve_context ----------


def retrieve_context_node(state: AgentState, retriever=None) -> dict[str, Any]:
    """Pull a few relevant corpus docs to ground the rest of the pipeline.

    The retriever is supplied by the graph builder via partial(); it can be
    None (RAG disabled or build failed) — in which case we return an empty list
    and downstream nodes degrade gracefully.

    The query is built from changed_ops names and summaries — small, focused,
    and cheap to embed.
    """
    if retriever is None:
        return {"retrieved_docs": []}

    ops = state.get("changed_ops") or []
    if not ops:
        return {"retrieved_docs": []}

    # Build one query string per pipeline run. The op summary is short ("modification
    # detected by static diff parse") so we mainly rely on the op name and the
    # diff body's first 400 characters as a topical hint.
    parts: list[str] = []
    for op in ops:
        parts.append(op.name)
        if op.summary:
            parts.append(op.summary)
    diff = state.get("diff", "") or ""
    if diff:
        parts.append(diff[:400])
    query = " ".join(parts)

    try:
        hits = retriever.retrieve(query, top_k=5)
    except Exception as exc:  # never fail the pipeline because of RAG
        return {"retrieved_docs": [], "error": f"retrieve_context: {exc}"}

    return {"retrieved_docs": [h.to_dict() for h in hits]}


# ---------- generate_tests ----------


def generate_tests_node(state: AgentState, llm: LLMClient | None = None) -> dict[str, Any]:
    changed_ops = state.get("changed_ops", []) or []
    selected = state.get("selected_skills", []) or []
    existing = state.get("generated_tests", []) or []
    lessons = state.get("reflexion_lessons") or []

    # Dedup against earlier critic-loop iterations: keep only test names we
    # haven't already produced for the same op. Without this the
    # list-concat reducer accumulates duplicates across iterations.
    seen_keys: set[tuple[str, str]] = {(t.op_name, t.test_name) for t in existing}

    new_tests: list = []
    for kind in selected:
        try:
            skill = get_skill(kind, llm=llm, lessons=lessons)
        except NotImplementedError:
            continue
        for op in changed_ops:
            for tc in skill.generate(op):
                key = (tc.op_name, tc.test_name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                new_tests.append(tc)

    return {"generated_tests": new_tests}


# ---------- critic ----------

_CRITIC_SYSTEM = """\
You are a strict reviewer of AI inference framework regression tests.
Verify that the proposed test set covers: shape variety (square/rectangular/m=1),
edge shapes, and numerical-stability cases when applicable.
Respond ONLY with JSON: {"passed": bool, "missing_dimensions": [str,...], "feedback": str}
"""


def critic_node(state: AgentState, llm: LLMClient | None = None) -> dict[str, Any]:
    tests = state.get("generated_tests", []) or []
    iterations = int(state.get("critic_iterations") or 0) + 1
    docs = state.get("retrieved_docs") or []

    def _result(verdict: CriticVerdict) -> dict[str, Any]:
        """Pack the critic update, including a Reflexion lesson when the
        verdict failed (so the next generate_tests iteration knows what to
        target). The lesson reducer concats across iterations.
        """
        out: dict[str, Any] = {
            "critic_verdict": verdict,
            "critic_iterations": iterations,
        }
        if not verdict.passed and _reflexion_enabled():
            lesson = _build_reflexion_lesson(iterations, verdict)
            if lesson:
                out["reflexion_lessons"] = [lesson]
        return out

    if not tests:
        verdict = CriticVerdict(
            passed=False,
            missing_dimensions=["no_tests_generated"],
            feedback="Generation produced zero tests.",
            coverage_report=_build_coverage_report(tests, [], docs, structural_pass=False),
        )
        return _result(verdict)

    # Deterministic structural check first — cheap and reliable.
    structural = _structural_critic(tests)
    base_report = _build_coverage_report(
        tests, structural.missing_dimensions, docs, structural_pass=structural.passed
    )

    if not structural.passed:
        structural.coverage_report = base_report
        return _result(structural)

    # Optional LLM critic on top of structural pass.
    if llm is None:
        structural.coverage_report = base_report
        return _result(structural)

    try:
        summary = "\n".join(f"- {t.op_name}::{t.test_name} ({t.rationale})" for t in tests)
        # Inject retrieved corpus snippets as grounding for the critic. We cap
        # at top-3 to keep prompt size in check; the retriever already returned
        # them ranked.
        rag_block = ""
        if docs:
            top = docs[:3]
            lines = []
            for d in top:
                doc_id = d.get("doc", {}).get("doc_id", "")
                text = d.get("doc", {}).get("text", "")
                lines.append(f"- [{doc_id}] {text}")
            rag_block = "\nRelevant context from corpus:\n" + "\n".join(lines) + "\n"

        resp = llm.complete(
            system=_CRITIC_SYSTEM,
            user=f"{rag_block}Proposed tests:\n{summary}\n\nReview them.",
            json_mode=True,
        )
        payload = json.loads(resp.text)
        verdict = CriticVerdict(
            passed=bool(payload.get("passed", True)),
            missing_dimensions=list(payload.get("missing_dimensions", [])),
            feedback=str(payload.get("feedback", "")),
            coverage_report={
                **base_report,
                "llm_passed": bool(payload.get("passed", True)),
                "llm_missing": list(payload.get("missing_dimensions", [])),
            },
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        verdict = structural
        verdict.coverage_report = base_report

    return _result(verdict)


def _reflexion_enabled() -> bool:
    """True unless REFLEXION env var is explicitly off (default: on)."""
    v = (os.getenv("REFLEXION", "on") or "on").lower().strip()
    return v not in ("0", "false", "off", "no")


def _build_reflexion_lesson(iteration: int, verdict: CriticVerdict) -> str:
    """Human-readable lesson string for the next generator pass.

    The generator prepends these to its planner prompt. Keep the format
    consistent — the LLM does much better with a structured "Lesson #N:"
    prefix than with raw missing-dimension lists.
    """
    parts: list[str] = [f"Lesson #{iteration}:"]
    missing = [m for m in (verdict.missing_dimensions or []) if m]
    if missing:
        parts.append("the previous test set was missing " + ", ".join(missing) + ".")
    if verdict.feedback:
        parts.append("Feedback: " + verdict.feedback.strip())
    if len(parts) == 1:
        # Don't emit empty lessons.
        return ""
    return " ".join(parts)


def _build_coverage_report(
    tests: list,
    missing: list[str],
    docs: list[dict],
    *,
    structural_pass: bool,
) -> dict[str, Any]:
    """Open-schema dict for the critic verdict's coverage_report field.

    Captures: how many (op, skill) groups were scored, total / missing
    dimensions, how many corpus docs were used to ground the LLM critic, and
    the per-skill miss list — useful for the report renderer + downstream
    metrics.
    """
    groups: dict[tuple, list] = {}
    for t in tests:
        groups.setdefault((t.op_name, t.skill), []).append(t)

    per_skill: dict[str, list[str]] = {}
    for entry in missing:
        # entry shape from _structural_critic: "op_name/skill_value:dim_name"
        parts = entry.split(":", 1)
        if len(parts) == 2:
            per_skill.setdefault(parts[0], []).append(parts[1])
        else:
            per_skill.setdefault("_other", []).append(entry)

    return {
        "structural_pass": bool(structural_pass),
        "groups_checked": len(groups),
        "total_tests": len(tests),
        "missing_count": len(missing),
        "context_docs_used": len(docs),
        "per_skill": per_skill,
    }


def _structural_critic(tests: list) -> CriticVerdict:
    """Rule-based critic: verify the test set covers the basic dimensions.

    Cheap, deterministic, and the foundation an LLM critic builds on. We
    dispatch per (op, skill) group so each combination owns a small set of
    dimensions appropriate to it — matmul/numerical asks for shape variety,
    perf asks for an aligned case, memory asks for a guard-band case, etc.
    """
    if not tests:
        return CriticVerdict(
            passed=False,
            missing_dimensions=["no_tests_generated"],
            feedback="No tests in this set.",
        )

    missing: list[str] = []
    groups: dict[tuple, list] = {}
    for t in tests:
        groups.setdefault((t.op_name, t.skill), []).append(t)

    for (op_name, skill), bucket in groups.items():
        names_lower = [t.test_name.lower() for t in bucket]
        miss = _coverage_misses(op_name, skill, names_lower)
        skill_tag = skill.value if hasattr(skill, "value") else str(skill)
        for m in miss:
            missing.append(f"{op_name}/{skill_tag}:{m}")

    if missing:
        return CriticVerdict(
            passed=False,
            missing_dimensions=missing,
            feedback="Structural critic: coverage incomplete.",
        )
    return CriticVerdict(
        passed=True, missing_dimensions=[], feedback="Structural critic: coverage adequate."
    )


def _coverage_misses(op_name: str, skill: SkillKind, names_lower: list[str]) -> list[str]:
    """Return the list of missing coverage dimensions for one (op, skill) group."""

    def any_tag(*tags: str) -> bool:
        return any(any(t in n for t in tags) for n in names_lower)

    miss: list[str] = []

    if skill == SkillKind.NUMERICAL:
        if op_name == "matmul_fp32":
            if not any_tag("square"):
                miss.append("square_shape")
            if not any_tag("rect"):
                miss.append("rectangular_shape")
            if not any_tag("vector", "thin", "1x"):
                miss.append("m1_or_n1_shape")
            if not any_tag("stability", "overflow", "underflow", "nan", "inf"):
                miss.append("numerical_stability_cases")
            if not any_tag("outer_product", "_k1", "k1_"):
                miss.append("k1_outer_product_shape")
            if not any_tag("aligned"):
                miss.append("hardware_aligned_shape")
            if not any_tag("prefilled", "output_buffer", "buffer_init"):
                miss.append("output_buffer_init_case")
        elif op_name in ("softmax_fp32", "layernorm_fp32"):
            if not any_tag("basic"):
                miss.append("basic_shape")
            if not any_tag("thin", "vector"):
                miss.append("thin_shape")
            if not any_tag("batch", "2d"):
                miss.append("batched_shape")
            if not any_tag("stability", "overflow", "underflow", "near_constant"):
                miss.append("numerical_stability_cases")
            if not any_tag("aligned"):
                miss.append("hardware_aligned_shape")

    elif skill == SkillKind.PERFORMANCE:
        if not any_tag("small"):
            miss.append("small_shape_budget")
        if not any_tag("aligned", "medium"):
            miss.append("aligned_or_medium_shape_budget")

    elif skill == SkillKind.MEMORY:
        if not any_tag("guard"):
            miss.append("guard_band_case")
        if not any_tag("repeat", "alloc"):
            miss.append("repeated_alloc_case")

    return miss


# ---------- install_tests ----------


def _project_test_dir() -> Path:
    """Where cmake's tests/test_*.cpp glob looks. Overridable via env var."""
    override = os.getenv("TINYINFER_PROJECT_DIR")
    base = Path(override) if override else Path("tinyinfer")
    return base / "tests"


def install_tests_node(state: AgentState) -> dict[str, Any]:
    """Copy generated cpp into the C++ project's tests/ directory.

    cmake's file(GLOB) is evaluated at configure time, so dropping new files
    here is enough — execute_tests_node will re-configure before building.

    Tests are grouped by (op_name, file_suffix). A per-skill suffix lets one
    op land multiple files (eg. test_matmul_fp32_generated.cpp +
    test_matmul_fp32_perf_generated.cpp) without overwriting each other.
    """
    tests = state.get("generated_tests", []) or []
    if not tests:
        return {"installed_test_paths": []}

    target_dir = _project_test_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    by_group: dict[tuple[str, str], str] = {}
    for tc in tests:
        if not tc.cpp_source:
            continue
        suffix = ""
        if isinstance(tc.inputs, dict):
            raw = tc.inputs.get("file_suffix")
            if isinstance(raw, str) and raw:
                suffix = f"_{raw}"
        by_group[(tc.op_name, suffix)] = tc.cpp_source

    installed: list[str] = []
    for (op_name, suffix), src in by_group.items():
        path = target_dir / f"test_{op_name}{suffix}_generated.cpp"
        path.write_text(src, encoding="utf-8")
        installed.append(str(path))
    return {"installed_test_paths": installed}


# ---------- execute_tests ----------


def execute_tests_node(state: AgentState) -> dict[str, Any]:
    """Drive cmake configure/build/ctest through the MCP server.

    Going through MCP (not a direct cmake_driver call) is the point: the
    same protocol is consumed by Claude Desktop / Cursor, so the agent is
    not the only entry point to the toolchain.

    On hosts without cmake we record a SKIPPED ExecutionResult instead of
    failing — partial pipelines are more useful than red ones.
    """
    from agent.mcp.client import call_tools  # local import: optional dep at run time

    project = os.getenv("TINYINFER_PROJECT_DIR", "tinyinfer")
    build = os.getenv("TINYINFER_BUILD_DIR", "tinyinfer/build")

    # Probe first; if no cmake, return a skipped result and let the report explain.
    probe = call_tools([("cmake_available", {})])
    if not probe or not probe[0].payload.get("available"):
        payload = probe[0].payload if probe else {}
        mode = payload.get("mode", "host")
        image = payload.get("image")
        if mode == "docker":
            reason = (
                f"docker mode requested (image={image!r}) but docker/image not reachable; "
                f"build it with `docker build -t {image} docker/builder/`."
            )
        else:
            reason = "cmake not available on host PATH; execution skipped."
        return {
            "execution_results": [
                ExecutionResult(
                    test_name="<toolchain>",
                    compiled=False,
                    ran=False,
                    passed=False,
                    stderr=reason,
                )
            ]
        }

    calls = [
        ("cmake_configure", {"project_dir": project, "build_dir": build}),
        ("cmake_build", {"build_dir": build}),
        ("ctest_run", {"build_dir": build}),
    ]
    results = call_tools(calls)

    cfg, bld, ct = results
    if not cfg.ok:
        return {
            "execution_results": [
                ExecutionResult(
                    test_name="<configure>",
                    compiled=False, ran=False, passed=False,
                    stdout=cfg.payload.get("stdout_tail", ""),
                    stderr=cfg.payload.get("stderr_tail", ""),
                )
            ]
        }
    if not bld.ok:
        return {
            "execution_results": [
                ExecutionResult(
                    test_name="<build>",
                    compiled=False, ran=False, passed=False,
                    stdout=bld.payload.get("stdout_tail", ""),
                    stderr=bld.payload.get("stderr_tail", ""),
                )
            ]
        }
    return {
        "execution_results": [
            ExecutionResult(
                test_name="<ctest>",
                compiled=True,
                ran=True,
                passed=ct.ok,
                stdout=ct.payload.get("stdout_tail", ""),
                stderr=ct.payload.get("stderr_tail", ""),
            )
        ]
    }


# ---------- write_report ----------


def write_report_node(state: AgentState) -> dict[str, Any]:
    tests = state.get("generated_tests", []) or []
    critic = state.get("critic_verdict")
    iters = int(state.get("critic_iterations") or 0)
    ops = state.get("changed_ops", []) or []

    lines: list[str] = []
    lines.append("# tinyinfer-agent — regression test report\n")
    lines.append(f"Trace ID: `{state.get('trace_id', 'n/a')}`\n")

    # Loud banner if the critic exited unsatisfied. We still emit the file
    # (better partial coverage than no coverage), but the human reviewer
    # must see this before they commit.
    if critic is not None and not critic.passed:
        lines.append("> [!WARNING]")
        lines.append(f"> **Critic did NOT pass after {iters} iteration(s).**")
        if critic.missing_dimensions:
            missing = ", ".join(critic.missing_dimensions)
            lines.append(f"> Missing dimensions: `{missing}`")
        lines.append("> Generated tests are written to disk anyway — review before committing.\n")

    lines.append("## Changed operators\n")
    if ops:
        for op in ops:
            lines.append(f"- **{op.name}** in `{op.file_path}` — {op.summary}")
    else:
        lines.append("_(none detected)_")
    lines.append("")

    lines.append(f"## Generated tests ({len(tests)})\n")
    for t in tests:
        lines.append(f"- `{t.op_name}::{t.test_name}` — {t.rationale}")
    lines.append("")

    if critic is not None:
        status = "PASS" if critic.passed else "FAIL"
        lines.append(f"## Critic — {status}\n")
        lines.append(f"- iterations: {iters}")
        if critic.missing_dimensions:
            lines.append(f"- missing: {', '.join(critic.missing_dimensions)}")
        if critic.feedback:
            lines.append(f"- feedback: {critic.feedback}")
        cov = critic.coverage_report or {}
        if cov:
            lines.append(
                f"- coverage: {cov.get('total_tests', 0)} tests across "
                f"{cov.get('groups_checked', 0)} (op,skill) groups; "
                f"{cov.get('missing_count', 0)} missing dimension(s); "
                f"context docs used: {cov.get('context_docs_used', 0)}"
            )
        lines.append("")

    docs = state.get("retrieved_docs") or []
    if docs:
        lines.append(f"## RAG context ({len(docs)} doc(s))\n")
        for d in docs[:5]:
            doc = d.get("doc", {})
            score = d.get("score", 0.0)
            text = doc.get("text", "")
            short = text if len(text) <= 160 else text[:157] + "..."
            lines.append(f"- `{doc.get('doc_id','')}` (score={score:.3f}) — {short}")
        lines.append("")

    installed = state.get("installed_test_paths") or []
    if installed:
        lines.append("## Installed test files\n")
        for p in installed:
            lines.append(f"- `{p}`")
        lines.append("")

    exec_results = state.get("execution_results") or []
    if exec_results:
        lines.append("## Execution (via MCP toolchain)\n")
        for r in exec_results:
            status = "PASS" if r.passed else ("SKIP" if not r.ran else "FAIL")
            lines.append(f"- `{r.test_name}` — **{status}**, compiled={r.compiled}, ran={r.ran}")
            if not r.passed and (r.stdout or r.stderr):
                tail = (r.stderr or r.stdout)[-600:]
                lines.append(f"  ```\n  {tail}\n  ```")
        lines.append("")

    return {"report_markdown": "\n".join(lines)}
