"""数值正确性 skill。

策略：
  1. 问 LLM 它觉得有用的 shape variety。
  2. LLM 不确定或者 mock 模式时 fallback 到手写的 shape case。
  3. 用 numpy oracle 算 reference output（确定性）。
  4. render GTest C++ 源码，把 input + expected output 嵌进去。

LLM 选 *shape*，oracle 决定 *value*。这个 split 是生成测试可信的关键。

Self-consistency（通过 SELF_CONSISTENCY_N env opt-in）：从 LLM 抽 N 份独立的
shape plan，多数投票留下幸存的。用延迟 / 钱换"对单个 hallucinate outlier 的
鲁棒性"。
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
你是资深的推理框架测试工程师。给定一个改动过的算子，请提议一组让回归覆盖
最大的输入 shape。
覆盖：最小非平凡 shape、非对称 shape、m=1 / n=1 的 thin path、更大的 batch、
以及和该算子相关的数值边界 case。
只用 JSON 回答：{"shapes": [{...}, {...}]} —— schema 跟着具体的算子。
"""


def _shape_planner_user(op_name: str) -> str:
    if op_name == "matmul_fp32":
        return (
            "Operator: matmul_fp32 (row-major fp32 GEMM, C[M,N] = A[M,K] * B[K,N])\n"
            "Schema per shape: {\"m\": int, \"k\": int, \"n\": int, \"c_init\": [float,...] (可选)}\n"
            "其中 \"c_init\" 是可选字段，长度必须等于 m*n。设上之后我们会用 caller-owned "
            "c[] 预填这些值再调 pointer API——专门测 API 合约 / accumulator-mode 类 bug。\n"
            "Propose 4-6 shapes。如果你判断要测 output buffer 初值、aliasing、stride、NaN "
            "等非 shape 维度，请把对应 case 用 c_init 表达。"
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
        # 本轮 critic loop 累积下来的 Reflexion lesson。generator 把这些 prepend
        # 到 LLM planner prompt 上，让 LLM 针对上一轮 critic 抱怨的 gap 重新 plan。
        self._lessons: list[str] = list(lessons or [])
        # Self-consistency：抽多少份独立 LLM shape-plan 然后多数投票。
        # 0/1 -> 不开 SC（单次 call）；>=2 -> N 次抽样，留下出现 >= ceil(N/2)
        # 次的 shape。能扛单次 hallucinate outlier。
        try:
            self._sc_n = max(0, int(os.getenv("SELF_CONSISTENCY_N", "0")))
        except ValueError:
            self._sc_n = 0

    def required_coverage_dimensions(self) -> list[str]:
        return ["shape_variety", "edge_shape", "numerical_stability"]

    def _plan_shapes(self, op_name: str) -> list[dict]:
        """问 LLM 拿 shape；解析失败就 fallback 到 default。

        SELF_CONSISTENCY_N >= 2 时抽 N 份独立 plan，只保留（tuple key 相等的）
        在多数 plan 都出现过的 shape。能扛单次 hallucinate outlier。

        `self._lessons` 非空时，这些 reflexion lesson 会被 prepend 到 user
        prompt 上 —— 等于告诉 LLM 上一轮 critic 抱怨了啥，让下一轮 plan 针对
        那个 gap。
        """
        if self.llm is None:
            return []

        n_draws = self._sc_n if self._sc_n >= 2 else 1
        all_shapes: list[dict] = []
        # 每次 call 构造一次 lesson preamble。格式：
        #     Lessons from prior iterations:
        #     - Lesson #1: ...
        #     - Lesson #2: ...
        # lesson 列表为空时这块也为空 -> caller 行为不变。
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
        # 始终带上手工默认 cases —— LLM 出幻觉也能保底覆盖。
        shape_cases = oracle.default_shape_cases(op.name, rng=self.rng)

        for s in planned:
            try:
                m, k, n = int(s["m"]), int(s["k"]), int(s["n"])
            except (KeyError, ValueError, TypeError):
                continue
            if m <= 0 or k <= 0 or n <= 0 or m * k * n > 4096:
                continue
            # LLM 可选地提议 c_init（长度 m*n 的 list[float]）来测 API 合约
            # / accumulator-mode 类 bug。校验长度，类型错就丢掉这个 case。
            llm_c_init = s.get("c_init")
            extra_inputs: dict = {}
            name_suffix = ""
            if isinstance(llm_c_init, list) and len(llm_c_init) == m * n:
                try:
                    extra_inputs["c_init"] = [float(v) for v in llm_c_init]
                    name_suffix = "_cinit"
                except (TypeError, ValueError):
                    pass
            shape_cases.append(
                oracle.ShapeCase(
                    name=f"llm_{m}x{k}_{k}x{n}{name_suffix}",
                    inputs={
                        "a": self.rng.standard_normal((m, k)).tolist(),
                        "b": self.rng.standard_normal((k, n)).tolist(),
                        "m": m, "k": k, "n": n,
                        **extra_inputs,
                    },
                    rationale=(
                        "LLM-proposed shape, preloaded c[] (output-buffer-init dim)"
                        if "c_init" in extra_inputs else "LLM-proposed shape"
                    ),
                )
            )

        out: list[TestCase] = []
        suite = "MatmulGenerated"
        body_lines: list[str] = []
        for sc in shape_cases:
            a = np.asarray(sc.inputs["a"], dtype=np.float32)
            b = np.asarray(sc.inputs["b"], dtype=np.float32)
            ref = oracle.compute("matmul_fp32", {"a": a, "b": b})
            c_init_list = sc.inputs.get("c_init")
            c_init = (
                np.asarray(c_init_list, dtype=np.float32)
                if c_init_list is not None
                else None
            )
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
                    c_init=c_init,
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
