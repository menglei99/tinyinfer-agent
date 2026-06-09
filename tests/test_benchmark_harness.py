"""fault-injection benchmark harness 的测试。"""

from pathlib import Path

import pytest

from benchmark.harness import FaultInjectionHarness, _fault_caught_heuristic

REPO = Path(__file__).resolve().parent.parent
SEEDS = REPO / "benchmark" / "seeds"


def test_seeds_directory_has_ten_diffs():
    diffs = sorted(SEEDS.glob("*.diff"))
    assert len(diffs) == 10


def test_each_seed_is_parseable_diff():
    for p in sorted(SEEDS.glob("*.diff")):
        text = p.read_text(encoding="utf-8")
        assert text.startswith("diff --git")
        assert "+++ b/tinyinfer/" in text


def test_fault_caught_heuristic_matches_keyword():
    class T:
        def __init__(self, op_name, test_name, rationale=""):
            self.op_name = op_name
            self.test_name = test_name
            self.rationale = rationale

    tests = [T("softmax_fp32", "stability_large_values_overflow", "overflow case")]
    caught, detail = _fault_caught_heuristic(
        tests, {"op": "softmax_fp32", "dim_groups": [("stability", "overflow")]}
    )
    assert caught
    assert "overflow" in detail or "stability" in detail


def test_fault_caught_heuristic_misses_when_op_absent():
    class T:
        def __init__(self, op_name, test_name, rationale=""):
            self.op_name = op_name
            self.test_name = test_name
            self.rationale = rationale

    tests = [T("matmul_fp32", "square_2x2")]
    caught, detail = _fault_caught_heuristic(
        tests, {"op": "softmax_fp32", "dim_groups": [("stability",)]}
    )
    assert not caught
    assert "no tests targeted" in detail


@pytest.mark.slow
def test_harness_run_one_mock(monkeypatch):
    """Run the full graph (mock LLM, no RAG) on one seed. Marked slow because
    it builds + streams the LangGraph pipeline."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from agent.llm import get_llm_client

    harness = FaultInjectionHarness(SEEDS, llm=get_llm_client(), retriever=None)
    result = harness.run_one(SEEDS / "05_softmax_missing_max_subtract.diff")
    assert result.op_detected == "softmax_fp32"
    assert result.tests_generated > 0
    # softmax default cases always include the overflow stability case, so the
    # missing-max-subtract fault must be flagged as caught.
    assert result.fault_likely_caught


@pytest.mark.slow
def test_harness_run_all_mock_baseline(monkeypatch):
    """Full benchmark in mock mode. The deterministic default shape sets cover
    every seeded fault dimension, so the mock baseline should be high."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from agent.llm import get_llm_client

    harness = FaultInjectionHarness(SEEDS, llm=get_llm_client(), retriever=None)
    report = harness.run_all()
    assert report.total == 10
    # Mock baseline: every op's default shape set covers its fault dimension.
    assert report.accuracy >= 0.8
