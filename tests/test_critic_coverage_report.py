"""Tests for the critic upgrade: retrieved-context grounding + coverage_report."""

import json

from agent.graph.nodes import critic_node
from agent.llm import LLMClient
from agent.llm.client import LLMResponse
from agent.state import SkillKind, TestCase


class CapturingLLM:
    """LLM stub that records the last prompt and returns a scripted JSON verdict."""

    def __init__(self, payload: dict | None = None):
        self.last_system = ""
        self.last_user = ""
        self._payload = payload or {"passed": True, "missing_dimensions": [], "feedback": "ok"}

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        self.last_system = system
        self.last_user = user
        return LLMResponse(text=json.dumps(self._payload))


def _matmul_full_coverage_tests() -> list[TestCase]:
    return [
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="square_2x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="rect_3x4_4x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="thin_1x8_8x4", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="stability_large_magnitude", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="outer_product_k1", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="aligned_16x32_32x64", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="prefilled_output_buffer_2x2", cpp_source="x"),
    ]


def test_critic_includes_retrieved_docs_in_llm_prompt():
    llm = CapturingLLM()
    docs = [
        {"doc": {"doc_id": "fault.matmul.offbyone_inner", "text": "matmul off-by-one in k loop"}, "score": 0.9},
        {"doc": {"doc_id": "op.matmul.numerics", "text": "matmul numerical stability"}, "score": 0.8},
    ]
    state = {
        "generated_tests": _matmul_full_coverage_tests(),
        "retrieved_docs": docs,
        "critic_iterations": 0,
    }
    critic_node(state, llm=llm)
    assert "Relevant context from corpus" in llm.last_user
    assert "fault.matmul.offbyone_inner" in llm.last_user
    assert "op.matmul.numerics" in llm.last_user


def test_critic_runs_without_retrieved_docs():
    llm = CapturingLLM()
    state = {"generated_tests": _matmul_full_coverage_tests(), "critic_iterations": 0}
    out = critic_node(state, llm=llm)
    # No "Relevant context" header when nothing was retrieved.
    assert "Relevant context" not in llm.last_user
    assert out["critic_verdict"].passed


def test_coverage_report_populated_on_success():
    out = critic_node(
        {"generated_tests": _matmul_full_coverage_tests(), "critic_iterations": 0},
        llm=None,  # structural-only path
    )
    cov = out["critic_verdict"].coverage_report
    assert cov["structural_pass"] is True
    assert cov["total_tests"] == 7
    assert cov["groups_checked"] == 1
    assert cov["missing_count"] == 0
    assert cov["context_docs_used"] == 0


def test_coverage_report_populated_on_structural_failure():
    bad = [
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL, test_name="square_2x2", cpp_source="x"),
    ]
    out = critic_node({"generated_tests": bad, "critic_iterations": 0}, llm=None)
    cov = out["critic_verdict"].coverage_report
    assert cov["structural_pass"] is False
    assert cov["missing_count"] > 0
    # per_skill keys are "op/skill" strings as emitted by _structural_critic.
    assert any(k.startswith("matmul_fp32/") for k in cov["per_skill"])


def test_coverage_report_counts_context_docs():
    docs = [{"doc": {"doc_id": f"d{i}", "text": f"t{i}"}, "score": 0.1 * i} for i in range(4)]
    out = critic_node(
        {
            "generated_tests": _matmul_full_coverage_tests(),
            "retrieved_docs": docs,
            "critic_iterations": 0,
        },
        llm=None,
    )
    cov = out["critic_verdict"].coverage_report
    assert cov["context_docs_used"] == 4


def test_coverage_report_includes_llm_outcome():
    """When the LLM critic runs and parses, its verdict shows up in coverage_report."""
    llm = CapturingLLM(
        payload={"passed": False, "missing_dimensions": ["overflow_test"], "feedback": "incomplete"}
    )
    docs = [{"doc": {"doc_id": "d1", "text": "t1"}, "score": 0.5}]
    out = critic_node(
        {
            "generated_tests": _matmul_full_coverage_tests(),
            "retrieved_docs": docs,
            "critic_iterations": 0,
        },
        llm=llm,
    )
    cov = out["critic_verdict"].coverage_report
    assert cov["llm_passed"] is False
    assert "overflow_test" in cov["llm_missing"]
