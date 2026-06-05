"""Numerical correctness skill.

Strategy:
  1. Ask the LLM for shape variety it considers useful.
  2. Fall back to hand-curated shape cases when the LLM is unsure or in mock mode.
  3. Compute reference outputs via numpy oracle (deterministic).
  4. Render GTest C++ source with embedded inputs + expected outputs.

The LLM picks the *shapes*. The oracle decides the *values*. This split is
what makes generated tests trustworthy.

Self-consistency (opt-in via SELF_CONSISTENCY_N env): draw N independent shape
plans from the LLM and majority-vote the survivors. Trades latency/$ for
robustness against a single hallucinated outlier.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Optional

import numpy as np

from agent.llm import LLMClient
from agent.skills.base import Skill
from agent.state import ChangedOp, SkillKind, TestCase
from agent.tools import cpp_renderer, oracle


_SHAPE_PLANNER_SYSTEM = """\
You are a senior inference-framework test engineer. Given a changed operator,
propose a list of input shapes that maximise regression coverage.
Cover: smallest non-trivial, asymmetric shapes, m=1 / n=1 thin paths, larger
batch, and any numerical edge cases relevant to this operator.
Respond ONLY with JSON of the form: {"shapes": [{...}, {...}]} — schema follows the operator.
"""


def _shape_planner_user(op_name: str) -> str:
    if op_name == "matmul_fp32":
        return (
            "Operator: matmul_fp32 (row-major fp32 GEMM, C[M,N] = A[M,K] * B[K,N])\n"
            "Schema per shape: {\"m\": int, \"k\": int, \"n\": int}\n"
            "Propose 4-6 shapes."
        )
    if op_name == "softmax_fp32":
        return (
            "Operator: softmax_fp32\n"
            "Schema per shape: {\"shape\": [int,...], \"axis\": int}\n"
            "Propose 3-5 shapes including a numerical-stability case."
        )
    return f"Operator: {op_name}\nPropose a small set of shapes as JSON."


class NumericalSkill(Skill):
    kind = SkillKind.NUMERICAL
    activation_keywords = ["matmul", "conv", "softmax", "layernorm", "quantize"]

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        rng_seed: int = 42,
        *,
        lessons: Optional[list[str]] = None,
    ):
        self.llm = llm
        self.rng = np.random.default_rng(rng_seed)
        # Reflexion lessons accumulated by previous critic loops in this run.
        # The generator prepends them to the LLM planner prompt so the LLM
        # can target the gaps the critic just complained about.
        self._lessons: list[str] = list(lessons or [])
        # Self-consistency: number of independent LLM shape-plans to draw and
        # majority-vote over. 0/1 -> no SC (single call). >=2 -> N draws,
        # keep shapes that appear in >= ceil(N/2) draws.
        try:
            self._sc_n = max(0, int(os.getenv("SELF_CONSISTENCY_N", "0")))
        except ValueError:
            self._sc_n = 0

    def required_coverage_dimensions(self) -> list[str]:
        return ["shape_variety", "edge_shape", "numerical_stability"]

    def _plan_shapes(self, op_name: str) -> list[dict]:
        """Ask LLM for shapes; fall back to defaults on parse failure.

        With SELF_CONSISTENCY_N >= 2, we sample N independent plans and keep
        only shapes (matched by tuple key) that appear in at least majority of
        plans. Robust to a single hallucinated outlier.

        When `self._lessons` is non-empty, those reflexion lessons are
        prepended to the user prompt — the LLM is told what the critic
        complained about last round so the next plan can target the gap.
        """
        if self.llm is None:
            return []

        n_draws = self._sc_n if self._sc_n >= 2 else 1
        all_shapes: list[dict] = []
        # Build the lesson preamble once per call. Format:
        #     Lessons from prior iterations:
        #     - Lesson #1: ...
        #     - Lesson #2: ...
        # Empty when lessons list is empty -> caller behaviour unchanged.
        lesson_block = ""
        if self._lessons:
            lines = "\n".join(f"- {ln}" for ln in self._lessons)
            lesson_block = (
                "Lessons from prior iterations (incorporate them; do not repeat the same gap):\n"
                + lines
                + "\n\n"
            )
        user = lesson_block + _shape_planner_user(op_name)

        for _ in range(n_draws):
            try:
                resp = self.llm.complete(
                    system=_SHAPE_PLANNER_SYSTEM,
                    user=user,
                    json_mode=True,
                )
                payload = json.loads(resp.text)
                shapes = payload.get("shapes", []) or []
                if isinstance(shapes, list):
                    all_shapes.extend(s for s in shapes if isinstance(s, dict))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        if n_draws == 1:
            return all_shapes

        return _majority_vote_shapes(all_shapes, op_name=op_name, n_draws=n_draws)

    def generate(self, op: ChangedOp) -> list[TestCase]:
        if not oracle.has_oracle(op.name):
            return []

        cases: list[TestCase] = []

        if op.name == "matmul_fp32":
            cases.extend(self._generate_matmul(op))
        elif op.name == "softmax_fp32":
            cases.extend(self._generate_softmax(op))
        elif op.name == "layernorm_fp32":
            cases.extend(self._generate_layernorm(op))
        return cases

    # ---------- per-op generators ----------

    def _generate_matmul(self, op: ChangedOp) -> list[TestCase]:
        planned = self._plan_shapes(op.name)
        # Always include the hand-curated baseline cases — guarantees coverage
        # even when the LLM hallucinates.
        shape_cases = oracle.default_shape_cases(op.name, rng=self.rng)

        for s in planned:
            try:
                m, k, n = int(s["m"]), int(s["k"]), int(s["n"])
            except (KeyError, ValueError, TypeError):
                continue
            if m <= 0 or k <= 0 or n <= 0 or m * k * n > 4096:
                continue
            shape_cases.append(
                oracle.ShapeCase(
                    name=f"llm_{m}x{k}_{k}x{n}",
                    inputs={
                        "a": self.rng.standard_normal((m, k)).tolist(),
                        "b": self.rng.standard_normal((k, n)).tolist(),
                        "m": m, "k": k, "n": n,
                    },
                    rationale="LLM-proposed shape",
                )
            )

        out: list[TestCase] = []
        suite = "MatmulGenerated"
        body_lines: list[str] = []
        for sc in shape_cases:
            a = np.asarray(sc.inputs["a"], dtype=np.float32)
            b = np.asarray(sc.inputs["b"], dtype=np.float32)
            ref = oracle.compute("matmul_fp32", {"a": a, "b": b})
            body_lines.append(
                cpp_renderer.render_matmul_test(
                    suite=suite,
                    case_name=sc.name,
                    a=a,
                    b=b,
                    expected=ref.output,
                    m=sc.inputs["m"],
                    k=sc.inputs["k"],
                    n=sc.inputs["n"],
                    rationale=sc.rationale,
                )
            )
            out.append(
                TestCase(
                    op_name=op.name,
                    skill=self.kind,
                    test_name=sc.name,
                    cpp_source="",
                    rationale=sc.rationale,
                    inputs={"shape": ref.shape_signature},
                    expected_outputs={"sum": float(ref.output.sum())},
                )
            )

        rendered = cpp_renderer.assemble_file(
            op_name=op.name, suite=suite, body="\n".join(body_lines)
        )
        for tc in out:
            tc.cpp_source = rendered.cpp_source
        return out

    def _generate_layernorm(self, op: ChangedOp) -> list[TestCase]:
        shape_cases = oracle.default_shape_cases(op.name, rng=self.rng)
        out: list[TestCase] = []
        suite = "LayernormGenerated"
        body_lines: list[str] = []
        for sc in shape_cases:
            x = np.asarray(sc.inputs["x"], dtype=np.float32)
            eps = float(sc.inputs.get("eps", 1e-5))
            ref = oracle.compute("layernorm_fp32", {"x": x, "eps": eps})
            body_lines.append(
                cpp_renderer.render_layernorm_test(
                    suite=suite,
                    case_name=sc.name,
                    x=x,
                    eps=eps,
                    expected=ref.output,
                    rationale=sc.rationale,
                )
            )
            out.append(
                TestCase(
                    op_name=op.name,
                    skill=self.kind,
                    test_name=sc.name,
                    cpp_source="",
                    rationale=sc.rationale,
                )
            )

        rendered = cpp_renderer.assemble_file(
            op_name=op.name, suite=suite, body="\n".join(body_lines)
        )
        for tc in out:
            tc.cpp_source = rendered.cpp_source
        return out

    def _generate_softmax(self, op: ChangedOp) -> list[TestCase]:
        shape_cases = oracle.default_shape_cases(op.name, rng=self.rng)
        out: list[TestCase] = []
        suite = "SoftmaxGenerated"
        body_lines: list[str] = []
        for sc in shape_cases:
            x = np.asarray(sc.inputs["x"], dtype=np.float32)
            axis = int(sc.inputs.get("axis", -1))
            ref = oracle.compute("softmax_fp32", {"x": x, "axis": axis})
            body_lines.append(
                cpp_renderer.render_softmax_test(
                    suite=suite,
                    case_name=sc.name,
                    x=x,
                    axis=axis,
                    expected=ref.output,
                    rationale=sc.rationale,
                )
            )
            out.append(
                TestCase(
                    op_name=op.name,
                    skill=self.kind,
                    test_name=sc.name,
                    cpp_source="",
                    rationale=sc.rationale,
                )
            )

        rendered = cpp_renderer.assemble_file(
            op_name=op.name, suite=suite, body="\n".join(body_lines)
        )
        for tc in out:
            tc.cpp_source = rendered.cpp_source
        return out


# ---------- self-consistency helpers ----------


def _shape_key(shape: dict, op_name: str) -> Optional[tuple]:
    """Canonical hashable key for one proposed shape, per op.

    Returns None if the shape is malformed for the op — caller drops it.
    """
    try:
        if op_name == "matmul_fp32":
            return ("matmul", int(shape["m"]), int(shape["k"]), int(shape["n"]))
        if op_name == "softmax_fp32":
            s = shape.get("shape", shape.get("x"))
            if not isinstance(s, (list, tuple)):
                return None
            return ("softmax", tuple(int(x) for x in s), int(shape.get("axis", -1)))
        if op_name == "layernorm_fp32":
            s = shape.get("shape", shape.get("x"))
            if not isinstance(s, (list, tuple)):
                return None
            return ("layernorm", tuple(int(x) for x in s))
    except (KeyError, ValueError, TypeError):
        return None
    return None


def _majority_vote_shapes(
    shapes: list[dict], *, op_name: str, n_draws: int
) -> list[dict]:
    """Keep shapes that appear in >= ceil(n_draws/2) of the LLM draws.

    Tie-break by first-seen order so the output is deterministic for a given
    sequence of LLM responses.
    """
    threshold = math.ceil(n_draws / 2)
    counts: Counter[tuple] = Counter()
    first_seen: dict[tuple, dict] = {}
    for s in shapes:
        key = _shape_key(s, op_name)
        if key is None:
            continue
        counts[key] += 1
        first_seen.setdefault(key, s)
    return [first_seen[k] for k, c in counts.items() if c >= threshold]
