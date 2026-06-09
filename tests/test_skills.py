"""perf / memory skill 的 smoke 测试。"""

from agent.skills import get_skill
from agent.state import ChangedOp, SkillKind


def _op(name: str) -> ChangedOp:
    return ChangedOp(name=name, file_path=f"tinyinfer/src/{name.replace('_fp32','')}.cpp", summary="x")


def test_perf_skill_emits_chrono_cases_for_matmul():
    skill = get_skill(SkillKind.PERFORMANCE)
    tests = skill.generate(_op("matmul_fp32"))
    assert tests, "perf skill should emit at least one case for matmul"
    src = tests[0].cpp_source
    # chrono-based timing, regression budget assertion, op call.
    assert "std::chrono::steady_clock" in src
    assert "EXPECT_LT" in src
    assert "tinyinfer::matmul_fp32" in src
    # File suffix routes install into a separate .cpp.
    assert all(tc.inputs.get("file_suffix") == "perf" for tc in tests)


def test_perf_skill_emits_cases_for_softmax_and_layernorm():
    perf = get_skill(SkillKind.PERFORMANCE)
    for op_name in ("softmax_fp32", "layernorm_fp32"):
        tests = perf.generate(_op(op_name))
        assert tests, f"perf skill should emit cases for {op_name}"
        assert "EXPECT_LT" in tests[0].cpp_source


def test_memory_skill_emits_guard_band_for_matmul():
    skill = get_skill(SkillKind.MEMORY)
    tests = skill.generate(_op("matmul_fp32"))
    assert tests
    src = tests[0].cpp_source
    assert "kGuard" in src
    assert "guards_intact" in src
    assert "tinyinfer::matmul_fp32" in src
    assert all(tc.inputs.get("file_suffix") == "memory" for tc in tests)


def test_memory_skill_returns_empty_for_unknown_op():
    skill = get_skill(SkillKind.MEMORY)
    assert skill.generate(_op("unknown_op_fp32")) == []


def test_numerical_skill_supports_layernorm():
    skill = get_skill(SkillKind.NUMERICAL)
    tests = skill.generate(_op("layernorm_fp32"))
    assert tests
    src = tests[0].cpp_source
    assert "tinyinfer::layernorm_fp32" in src
    assert "EXPECT_NEAR" in src


def test_numerical_skill_softmax_now_emits_active_test():
    """Renderer used to emit DISABLED_ tests for softmax. Now it must call
    the real C++ op and compare to oracle output."""
    skill = get_skill(SkillKind.NUMERICAL)
    tests = skill.generate(_op("softmax_fp32"))
    assert tests
    src = tests[0].cpp_source
    assert "DISABLED_" not in src
    assert "GTEST_SKIP" not in src
    assert "tinyinfer::softmax_fp32" in src
