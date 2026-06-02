from agent.graph.nodes import (
    _detect_skills_from_diff,
    _structural_critic,
    parse_diff_node,
    route_skill_node,
)
from agent.state import SkillKind, TestCase


def test_parse_diff_extracts_matmul_op():
    diff = """\
diff --git a/tinyinfer/src/matmul.cpp b/tinyinfer/src/matmul.cpp
--- a/tinyinfer/src/matmul.cpp
+++ b/tinyinfer/src/matmul.cpp
@@ -1,1 +1,1 @@
-old
+new
"""
    out = parse_diff_node({"diff": diff})
    ops = out["changed_ops"]
    assert len(ops) == 1
    assert ops[0].name == "matmul_fp32"
    assert "matmul.cpp" in ops[0].file_path


def test_parse_diff_handles_empty():
    out = parse_diff_node({"diff": ""})
    assert out["changed_ops"] == []
    assert "error" in out


def test_route_skill_defaults_to_numerical():
    out = route_skill_node({"diff": "diff --git a/foo b/foo\n+++ b/foo\n+x"})
    assert SkillKind.NUMERICAL in out["selected_skills"]


def test_route_skill_promotes_perf_on_simd_keywords():
    diff = "+++ b/x.cpp\n+// SIMD intrinsic unroll on hot path\n+__m256 v;\n"
    skills = _detect_skills_from_diff(diff)
    assert SkillKind.PERFORMANCE in skills
    assert SkillKind.NUMERICAL in skills


def test_route_skill_promotes_memory_on_allocator_keywords():
    diff = "+++ b/x.cpp\n+std::memcpy(dst, src, n);\n+x.resize(64);\n"
    skills = _detect_skills_from_diff(diff)
    assert SkillKind.MEMORY in skills
    assert SkillKind.NUMERICAL in skills


def test_route_skill_keeps_only_numerical_when_diff_is_clean():
    diff = "+++ b/x.cpp\n+int a = 1;\n+return a;\n"
    skills = _detect_skills_from_diff(diff)
    assert skills == [SkillKind.NUMERICAL]


def test_structural_critic_passes_with_full_matmul_coverage():
    tests = [
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="square_2x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="rect_3x4_4x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="thin_1x8_8x4", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="stability_large_magnitude", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="outer_product_k1", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="aligned_16x32_32x64", cpp_source="x"),
    ]
    v = _structural_critic(tests)
    assert v.passed


def test_structural_critic_flags_missing_thin_shape():
    tests = [
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="square_2x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="rect_3x4_4x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="stability_large_magnitude", cpp_source="x"),
    ]
    v = _structural_critic(tests)
    assert not v.passed
    assert any("m1_or_n1_shape" in d for d in v.missing_dimensions)


def test_structural_critic_flags_missing_stability():
    tests = [
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="square_2x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="rect_3x4_4x2", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.NUMERICAL,
                 test_name="thin_1x8_8x4", cpp_source="x"),
    ]
    v = _structural_critic(tests)
    assert not v.passed
    assert any("numerical_stability_cases" in d for d in v.missing_dimensions)


def test_structural_critic_softmax_full_coverage():
    tests = [
        TestCase(op_name="softmax_fp32", skill=SkillKind.NUMERICAL,
                 test_name="basic_1d_n8", cpp_source="x"),
        TestCase(op_name="softmax_fp32", skill=SkillKind.NUMERICAL,
                 test_name="thin_1d_n3", cpp_source="x"),
        TestCase(op_name="softmax_fp32", skill=SkillKind.NUMERICAL,
                 test_name="batch_2d_axis_last_4x16", cpp_source="x"),
        TestCase(op_name="softmax_fp32", skill=SkillKind.NUMERICAL,
                 test_name="stability_large_values_overflow", cpp_source="x"),
        TestCase(op_name="softmax_fp32", skill=SkillKind.NUMERICAL,
                 test_name="aligned_1d_n32", cpp_source="x"),
    ]
    v = _structural_critic(tests)
    assert v.passed


def test_structural_critic_perf_dimensions():
    tests = [
        TestCase(op_name="matmul_fp32", skill=SkillKind.PERFORMANCE,
                 test_name="perf_small_16x16x16", cpp_source="x"),
    ]
    v = _structural_critic(tests)
    # Missing "aligned" or "medium".
    assert not v.passed
    assert any("aligned_or_medium" in d for d in v.missing_dimensions)

    tests.append(
        TestCase(op_name="matmul_fp32", skill=SkillKind.PERFORMANCE,
                 test_name="perf_aligned_32x32x32", cpp_source="x")
    )
    v = _structural_critic(tests)
    assert v.passed


def test_structural_critic_memory_dimensions():
    tests = [
        TestCase(op_name="matmul_fp32", skill=SkillKind.MEMORY,
                 test_name="memory_guard_bands_small", cpp_source="x"),
        TestCase(op_name="matmul_fp32", skill=SkillKind.MEMORY,
                 test_name="memory_repeat_alloc_medium", cpp_source="x"),
    ]
    v = _structural_critic(tests)
    assert v.passed
