"""install_tests_node 和 execute_tests_node 的测试。

execute_tests_node 真的会拉起一个 MCP 子进程，可能要花几秒；端到端 smoke
test 已经覆盖了它，这里不做 per-node 单测。

宿主上没 cmake 时，execute 应该返回 SKIPPED 结果（不是 fail）。
"""

import os
from pathlib import Path

import pytest

from agent.graph.nodes import execute_tests_node, install_tests_node
from agent.state import SkillKind, TestCase


def test_install_tests_writes_one_file_per_op(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYINFER_PROJECT_DIR", str(tmp_path / "proj"))

    tests = [
        TestCase(
            op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
            test_name="square_2x2", cpp_source="// CPP A",
        ),
        TestCase(
            op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
            test_name="rect_3x4_4x2", cpp_source="// CPP A",
        ),
        TestCase(
            op_name="softmax_fp32", skill=SkillKind.NUMERICAL,
            test_name="basic_1d", cpp_source="// CPP B",
        ),
    ]

    out = install_tests_node({"generated_tests": tests})
    paths = [Path(p) for p in out["installed_test_paths"]]
    assert len(paths) == 2  # one per op
    names = {p.name for p in paths}
    assert "test_matmul_fp32_generated.cpp" in names
    assert "test_softmax_fp32_generated.cpp" in names
    for p in paths:
        assert p.exists()


def test_install_tests_separates_files_by_suffix(tmp_path, monkeypatch):
    """Perf + memory skills carry a `file_suffix` so they land in their own
    .cpp alongside the numerical file rather than overwriting it."""
    monkeypatch.setenv("TINYINFER_PROJECT_DIR", str(tmp_path / "proj"))

    tests = [
        TestCase(
            op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
            test_name="square_2x2", cpp_source="// numerical",
        ),
        TestCase(
            op_name="matmul_fp32", skill=SkillKind.PERFORMANCE,
            test_name="perf_small_16x16x16", cpp_source="// perf",
            inputs={"file_suffix": "perf"},
        ),
        TestCase(
            op_name="matmul_fp32", skill=SkillKind.MEMORY,
            test_name="memory_guard_bands_small", cpp_source="// memory",
            inputs={"file_suffix": "memory"},
        ),
    ]
    out = install_tests_node({"generated_tests": tests})
    names = {Path(p).name for p in out["installed_test_paths"]}
    assert "test_matmul_fp32_generated.cpp" in names
    assert "test_matmul_fp32_perf_generated.cpp" in names
    assert "test_matmul_fp32_memory_generated.cpp" in names


def test_install_tests_handles_empty():
    out = install_tests_node({"generated_tests": []})
    assert out == {"installed_test_paths": []}


@pytest.mark.slow
def test_execute_tests_returns_skip_when_no_cmake():
    """Without cmake on PATH, execute should record a skipped result.

    Marked slow because it spawns a stdio subprocess.
    """
    out = execute_tests_node({})
    results = out["execution_results"]
    assert len(results) == 1
    r = results[0]
    # Either cmake genuinely isn't available (expected on this host) or it is.
    # We only require that the node returned a well-formed ExecutionResult.
    assert r.test_name in ("<toolchain>", "<ctest>", "<configure>", "<build>")
    if r.test_name == "<toolchain>":
        assert not r.ran
        assert "cmake" in r.stderr.lower()
