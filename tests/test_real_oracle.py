"""Real-oracle unit tests for the fault-injection harness.

We don't need (or want) docker running for these — every cmake_driver call
is mocked so we can verify the harness's call shape, ctest parsing, and
fall-back behaviour without spending 30 seconds rebuilding tinyinfer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.tools.cmake_driver import CommandResult
from benchmark import harness as harness_mod
from benchmark.harness import (
    FaultInjectionHarness,
    _fault_caught_real,
    _parse_failed_test_names,
    _real_oracle_test_filter,
)


# ---------- ctest output parser ----------


def test_parse_failed_test_names_extracts_failures():
    sample = (
        "Test project /work/build-docker\n"
        "    Start 1: test_matmul_baseline\n"
        "1/3 Test #1: test_matmul_baseline ............   Passed    0.01 sec\n"
        "    Start 2: test_matmul_fp32_generated\n"
        "2/3 Test #2: test_matmul_fp32_generated ......***Failed    0.02 sec\n"
        "    Start 3: test_softmax_fp32_generated\n"
        "3/3 Test #3: test_softmax_fp32_generated .....   Passed    0.01 sec\n"
        "33% tests passed, 1 tests failed out of 3\n"
    )
    assert _parse_failed_test_names(sample) == ["test_matmul_fp32_generated"]


def test_parse_failed_test_names_empty_when_all_pass():
    sample = (
        "1/2 Test #1: test_a ..........   Passed    0.01 sec\n"
        "2/2 Test #2: test_b ..........   Passed    0.01 sec\n"
        "100% tests passed, 0 tests failed out of 2\n"
    )
    assert _parse_failed_test_names(sample) == []


def test_real_oracle_test_filter_targets_only_generated():
    # Should NOT match `test_matmul_baseline` — baseline can fail under mutation
    # too, but baseline isn't an agent-produced test.
    pattern = _real_oracle_test_filter("matmul_fp32")
    import re

    assert re.match(pattern, "test_matmul_fp32_generated")
    assert re.match(pattern, "test_matmul_fp32_perf_generated")
    assert re.match(pattern, "test_matmul_fp32_memory_generated")
    assert not re.match(pattern, "test_matmul_baseline")
    assert not re.match(pattern, "test_softmax_fp32_generated")


# ---------- _fault_caught_real ----------


_TINY_DIFF = """\
diff --git a/src.txt b/src.txt
--- a/src.txt
+++ b/src.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_mutated
 line3
"""


def _make_seed(tmp_path: Path) -> Path:
    seed = tmp_path / "seed.diff"
    seed.write_text(_TINY_DIFF, encoding="utf-8")
    return seed


def _make_project(tmp_path: Path) -> Path:
    """Create a tmp project dir with the file the diff targets."""
    (tmp_path / "src.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    return tmp_path


def test_real_oracle_detects_failed_generated_test(tmp_path):
    project = _make_project(tmp_path)
    seed = _make_seed(tmp_path)
    expected = {"op": "matmul_fp32", "dim_groups": [], "fault": "x"}

    build_ok = CommandResult(cmd=["cmake", "--build"], returncode=0, stdout="", stderr="")
    ctest_fail = CommandResult(
        cmd=["ctest"],
        returncode=8,  # ctest non-zero on test failure
        stdout=(
            "1/2 Test #1: test_matmul_fp32_generated ..***Failed   0.05 sec\n"
            "2/2 Test #2: test_matmul_fp32_perf_generated ..   Passed   0.01 sec\n"
        ),
        stderr="",
    )

    with patch("agent.tools.cmake_driver.build", return_value=build_ok), \
         patch("agent.tools.cmake_driver.ctest", return_value=ctest_fail):
        caught, detail, status, failed = _fault_caught_real(
            expected, seed, project_dir=project, build_dir=project / "build",
        )

    assert caught is True
    assert status == "ok"
    assert failed == ["test_matmul_fp32_generated"]
    assert "test_matmul_fp32_generated" in detail


def test_real_oracle_marks_missed_when_all_tests_pass(tmp_path):
    project = _make_project(tmp_path)
    seed = _make_seed(tmp_path)
    expected = {"op": "matmul_fp32", "dim_groups": [], "fault": "x"}

    build_ok = CommandResult(cmd=["cmake", "--build"], returncode=0, stdout="", stderr="")
    ctest_pass = CommandResult(
        cmd=["ctest"], returncode=0,
        stdout="1/1 Test #1: test_matmul_fp32_generated ..   Passed   0.01 sec\n",
        stderr="",
    )

    with patch("agent.tools.cmake_driver.build", return_value=build_ok), \
         patch("agent.tools.cmake_driver.ctest", return_value=ctest_pass):
        caught, detail, status, failed = _fault_caught_real(
            expected, seed, project_dir=project, build_dir=project / "build",
        )

    assert caught is False
    assert status == "ok"
    assert failed == []
    assert "all generated tests passed" in detail


def test_real_oracle_reports_build_failure(tmp_path):
    project = _make_project(tmp_path)
    seed = _make_seed(tmp_path)
    expected = {"op": "matmul_fp32", "dim_groups": [], "fault": "x"}

    build_fail = CommandResult(
        cmd=["cmake", "--build"], returncode=2, stdout="", stderr="undefined reference",
    )

    with patch("agent.tools.cmake_driver.build", return_value=build_fail), \
         patch("agent.tools.cmake_driver.ctest") as ctest_mock:
        caught, detail, status, failed = _fault_caught_real(
            expected, seed, project_dir=project, build_dir=project / "build",
        )

        # ctest must NOT run when build failed.
        ctest_mock.assert_not_called()

    assert caught is False
    assert status == "build_failed"
    assert "undefined reference" in detail


def test_real_oracle_restores_source_after_run(tmp_path):
    project = _make_project(tmp_path)
    seed = _make_seed(tmp_path)
    expected = {"op": "matmul_fp32", "dim_groups": [], "fault": "x"}

    captured_during_build: dict = {}

    def fake_build(*args, **kwargs):
        captured_during_build["src"] = (project / "src.txt").read_text(encoding="utf-8")
        return CommandResult(cmd=["cmake"], returncode=0, stdout="", stderr="")

    fake_ctest = CommandResult(
        cmd=["ctest"], returncode=0,
        stdout="1/1 Test #1: test_x_fp32_generated .. Passed 0.01 sec\n",
        stderr="",
    )

    with patch("agent.tools.cmake_driver.build", side_effect=fake_build), \
         patch("agent.tools.cmake_driver.ctest", return_value=fake_ctest):
        _fault_caught_real(expected, seed, project_dir=project, build_dir=project / "build")

    # During the build phase, the file MUST have been mutated.
    assert "line2_mutated" in captured_during_build["src"]
    # After _fault_caught_real returns, the file MUST be restored.
    assert (project / "src.txt").read_text(encoding="utf-8") == "line1\nline2\nline3\n"


# ---------- harness oracle switch ----------


def test_harness_defaults_to_heuristic(monkeypatch, tmp_path):
    monkeypatch.delenv("FAULT_ORACLE", raising=False)
    h = FaultInjectionHarness(tmp_path)
    assert h.oracle_mode == "heuristic"


def test_harness_respects_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("FAULT_ORACLE", "real")
    h = FaultInjectionHarness(tmp_path)
    assert h.oracle_mode == "real"


def test_harness_constructor_arg_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAULT_ORACLE", "heuristic")
    h = FaultInjectionHarness(tmp_path, oracle="real")
    assert h.oracle_mode == "real"


def test_harness_rejects_unknown_oracle(tmp_path):
    with pytest.raises(ValueError, match="unknown oracle mode"):
        FaultInjectionHarness(tmp_path, oracle="bogus")


def test_real_oracle_falls_back_when_cmake_unavailable(monkeypatch, tmp_path):
    """If cmake/docker isn't reachable, the harness records a heuristic
    result with oracle_mode="real_fallback_heuristic" rather than crashing.
    """
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    seed = seeds / "01_matmul_offbyone_inner.diff"
    seed.write_text("dummy", encoding="utf-8")  # never actually applied in fallback path

    h = FaultInjectionHarness(seeds, oracle="real")

    # Stub the graph so we don't need real LLM / RAG.
    from agent.state import SkillKind, TestCase

    fake_tests = [
        TestCase(
            op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
            test_name="aligned_16x32", cpp_source="// x", rationale="aligned",
        ),
    ]

    class StubGraph:
        def stream(self, initial, stream_mode="updates"):
            yield {"parse_diff": {"changed_ops": []}}
            yield {"generate_tests": {"generated_tests": fake_tests}}

    h._graph = StubGraph()
    monkeypatch.setattr("agent.tools.cmake_driver.cmake_available", lambda: False)

    result = h.run_one(seed)
    assert result.oracle_mode == "real_fallback_heuristic"
    assert result.build_status == "skipped"
    # Heuristic should still hit on "aligned" keyword (matmul dim_groups include "aligned").
    assert result.fault_likely_caught is True
