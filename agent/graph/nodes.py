"""LangGraph 节点。

每个节点拿当前 AgentState、返回一个 partial state update。保持小、副作用少、
单独可测试。
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

# 我们关心的文件。别的 hunk 直接忽略。
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

    # regex 没找到时也可以选择问 LLM（这里保留确定性，不调）
    return {"changed_ops": list(seen.values())}


# ---------- route_skill ----------


# 路由启发式用的关键词集。保持短而高信号——任何模糊词（比如 "loop"）会把所有
# 东西都路由到所有 skill。
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
    """启发式：扫 diff 里 +/- 行有没有触发 skill 的关键词。

    NUMERICAL 始终开着 —— 正确性是底线。PERF / MEMORY 按 diff 实际触及的内容
    增量加上。
    """
    skills: list[SkillKind] = [SkillKind.NUMERICAL]
    if not diff:
        return skills

    # 只看新加 / 删掉的代码，不看周围 context
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
    """选给改动的算子跑哪些 Skill。

    始终路由到 NUMERICAL。当 diff body 里含 skill 相关关键词时再加 PERF / MEMORY
    （比如 SIMD intrinsic → perf，allocator 改动 → memory）。
    """
    diff = state.get("diff", "") or ""
    return {"selected_skills": _detect_skills_from_diff(diff)}


# ---------- retrieve_context ----------


def retrieve_context_node(state: AgentState, retriever=None) -> dict[str, Any]:
    """拉几条相关的 corpus 文档给 pipeline 剩下的步骤当 ground。

    retriever 由 graph builder 通过 partial() 注入；可以是 None（RAG 关了或者
    build 失败）—— 这种情况返回空 list，下游节点会优雅 degrade。

    query 用 changed_ops 的名字 + summary 拼出来 —— 短、聚焦、embed 便宜。
    """
    if retriever is None:
        return {"retrieved_docs": []}

    ops = state.get("changed_ops") or []
    if not ops:
        return {"retrieved_docs": []}

    # 每次 pipeline 跑构造一条 query。op summary 短（"modification detected by
    # static diff parse"）所以主要靠 op 名字加 diff body 前 400 字符当 topical hint。
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
    except Exception as exc:  # RAG 任何失败都不让 pipeline 挂
        return {"retrieved_docs": [], "error": f"retrieve_context: {exc}"}

    return {"retrieved_docs": [h.to_dict() for h in hits]}


# ---------- generate_tests ----------


def generate_tests_node(state: AgentState, llm: LLMClient | None = None) -> dict[str, Any]:
    changed_ops = state.get("changed_ops", []) or []
    selected = state.get("selected_skills", []) or []
    existing = state.get("generated_tests", []) or []
    lessons = state.get("reflexion_lessons") or []

    # 跟前几轮 critic loop 的输出去重：只保留同一 op 下还没产生过的 test 名字。
    # 不这么做的话 list-concat reducer 会跨迭代累积重复 case。
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
你是一个严格的 AI 推理框架回归测试 reviewer。
验证提议的 test 集是否覆盖：shape 多样性（square/rectangular/m=1）、
edge shape、以及在适用时的数值稳定性 case。
只返回 JSON：{"passed": bool, "missing_dimensions": [str,...], "feedback": str}
"""


def critic_node(state: AgentState, llm: LLMClient | None = None) -> dict[str, Any]:
    tests = state.get("generated_tests", []) or []
    iterations = int(state.get("critic_iterations") or 0) + 1
    docs = state.get("retrieved_docs") or []

    def _result(verdict: CriticVerdict) -> dict[str, Any]:
        """打包 critic update；verdict 失败时附带一条 Reflexion lesson
        （让下一轮 generate_tests 知道往哪个方向找）。lesson reducer 跨迭代 concat。
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

    # 先跑确定性的结构 check —— 便宜可靠
    structural = _structural_critic(tests)
    base_report = _build_coverage_report(
        tests, structural.missing_dimensions, docs, structural_pass=structural.passed
    )

    if not structural.passed:
        structural.coverage_report = base_report
        return _result(structural)

    # 结构 PASS 后再跑可选的 LLM critic
    if llm is None:
        structural.coverage_report = base_report
        return _result(structural)

    try:
        summary = "\n".join(f"- {t.op_name}::{t.test_name} ({t.rationale})" for t in tests)
        # 把检索到的 corpus 片段当 grounding 喂给 critic。最多 top-3，控制 prompt
        # 长度；retriever 已经排序好了。
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
    """REFLEXION env 显式 off 时返回 False，默认 on。"""
    v = (os.getenv("REFLEXION", "on") or "on").lower().strip()
    return v not in ("0", "false", "off", "no")


def _build_reflexion_lesson(iteration: int, verdict: CriticVerdict) -> str:
    """给下一轮 generator 一条 human-readable lesson 字符串。

    generator 会把这些 prepend 到 planner prompt 前面。格式要一致 —— LLM 对
    带 "Lesson #N:" 前缀的结构化反馈反应明显比裸 missing-dimension list 好。
    """
    parts: list[str] = [f"Lesson #{iteration}:"]
    missing = [m for m in (verdict.missing_dimensions or []) if m]
    if missing:
        parts.append("the previous test set was missing " + ", ".join(missing) + ".")
    if verdict.feedback:
        parts.append("Feedback: " + verdict.feedback.strip())
    if len(parts) == 1:
        # 不发空 lesson
        return ""
    return " ".join(parts)


def _build_coverage_report(
    tests: list,
    missing: list[str],
    docs: list[dict],
    *,
    structural_pass: bool,
) -> dict[str, Any]:
    """critic verdict 的 coverage_report 字段，schema 开放的 dict。

    抓取：评分覆盖了多少 (op, skill) 组、total / missing 维度数、grounding LLM
    critic 用了多少 corpus 文档、per-skill 的 miss 列表 —— 这些都给 report
    renderer 和下游指标用。
    """
    groups: dict[tuple, list] = {}
    for t in tests:
        groups.setdefault((t.op_name, t.skill), []).append(t)

    per_skill: dict[str, list[str]] = {}
    for entry in missing:
        # _structural_critic 出来的 entry 形如 "op_name/skill_value:dim_name"
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
    """规则式 critic：验证测试集是否覆盖基础维度。

    便宜、确定性，是 LLM critic 的地基。按 (op, skill) 分组 dispatch，每个组合
    管自己那一小撮维度 —— matmul/numerical 要 shape 多样性，perf 要一个 aligned
    case，memory 要一个 guard-band case，依此类推。
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
    """返回一个 (op, skill) 组里缺失的覆盖维度。"""

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
    """cmake 的 tests/test_*.cpp glob 找的目录。可以用 env 覆盖。"""
    override = os.getenv("TINYINFER_PROJECT_DIR")
    base = Path(override) if override else Path("tinyinfer")
    return base / "tests"


def install_tests_node(state: AgentState) -> dict[str, Any]:
    """把生成的 cpp 拷到 C++ 项目的 tests/ 目录下。

    cmake 的 file(GLOB) 在 configure 时评估，所以新文件落在这里就够 ——
    execute_tests_node 会在 build 前重 configure。

    测试按 (op_name, file_suffix) 分组。per-skill 的 suffix 让同一个 op
    可以落多个文件（例如 test_matmul_fp32_generated.cpp +
    test_matmul_fp32_perf_generated.cpp）而不互相覆盖。
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
    """通过 MCP server 驱动 cmake configure/build/ctest。

    走 MCP 而不是直接 call cmake_driver 是关键：同一套协议同时被 Claude Desktop
    / Cursor 用，agent 不是 toolchain 的唯一入口。

    宿主没 cmake 时记一条 SKIPPED 的 ExecutionResult 而不是 fail —— 部分 pipeline
    总比全红 pipeline 有用。
    """
    from agent.mcp.client import call_tools  # 本地 import：runtime 才需要的可选依赖

    project = os.getenv("TINYINFER_PROJECT_DIR", "tinyinfer")
    build = os.getenv("TINYINFER_BUILD_DIR", "tinyinfer/build")

    # 先 probe；没 cmake 就返回 skipped 让 report 解释清楚
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

    # critic 没满意时打个大横幅。文件仍然落盘（部分覆盖比没覆盖好），
    # 但人来 review 时必须看到这个 banner 才能 commit。
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
