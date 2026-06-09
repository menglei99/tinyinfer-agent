"""render_matmul_test 里 c_init 指针 API 路径的测试。

覆盖 devlog 07 发现的架构盲区：vector 重载 wrapper 内部把 c[] 零初始化，
盖住了 accumulator-mode bug（seed 03）。当 c_init 被设上时，renderer 必须
切到 pointer API 并预填 c[]。
"""

from __future__ import annotations

import numpy as np

from agent.tools import cpp_renderer, oracle
from agent.skills.numerical import NumericalSkill
from agent.state import ChangedOp


def test_renderer_default_uses_vector_overload():
    """No c_init -> vector API; doesn't pre-declare c[]."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    expected = a @ b
    src = cpp_renderer.render_matmul_test(
        suite="MatmulGenerated", case_name="square_2x2",
        a=a, b=b, expected=expected, m=2, k=2, n=2,
        rationale="smallest non-trivial square",
    )
    assert "auto c = tinyinfer::matmul_fp32(a, b" in src
    assert "std::vector<float> c =" not in src
    assert "c.data()" not in src


def test_renderer_c_init_switches_to_pointer_api():
    """c_init present -> pointer API + caller-allocated c[] with non-zero values."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    expected = a @ b
    c_init = np.array([2.5, -1.75, 3.125, -0.5], dtype=np.float32)

    src = cpp_renderer.render_matmul_test(
        suite="MatmulGenerated", case_name="prefilled_output_buffer_2x2",
        a=a, b=b, expected=expected, m=2, k=2, n=2,
        rationale="output buffer prefilled with non-zero values",
        c_init=c_init,
    )

    # Pointer API call
    assert "tinyinfer::matmul_fp32(a.data(), b.data(), c.data()" in src
    # Caller-allocated c[] with the actual non-zero values
    assert "std::vector<float> c = {" in src
    assert "2.5f" in src
    assert "-1.75f" in src
    # Default vector path must not also appear
    assert "auto c = tinyinfer::matmul_fp32(a, b" not in src


def test_default_shape_cases_include_prefilled_output_buffer():
    cases = oracle.default_shape_cases("matmul_fp32")
    names = [c.name for c in cases]
    assert "prefilled_output_buffer_2x2" in names

    pf = next(c for c in cases if c.name == "prefilled_output_buffer_2x2")
    assert "c_init" in pf.inputs
    assert all(v != 0.0 for v in pf.inputs["c_init"])


def test_numerical_skill_generates_pointer_api_test_for_matmul():
    """End-to-end through NumericalSkill: the rendered file contains both the
    vector-overload tests AND a pointer-API test for the prefilled case."""
    skill = NumericalSkill(llm=None)
    op = ChangedOp(name="matmul_fp32", file_path="tinyinfer/src/matmul.cpp", summary="x")
    tests = skill.generate(op)
    assert tests, "skill must produce at least one matmul test"

    full_cpp = tests[0].cpp_source  # all cases share the same assembled file
    # The new dim is present
    assert "TEST(MatmulGenerated, prefilled_output_buffer_2x2)" in full_cpp
    # And it uses the pointer API
    assert "matmul_fp32(a.data(), b.data(), c.data()" in full_cpp
    # Older cases still use the vector overload
    assert "auto c = tinyinfer::matmul_fp32(a, b" in full_cpp
