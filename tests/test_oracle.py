import numpy as np

from agent.tools import oracle


def test_matmul_oracle_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)
    res = oracle.compute("matmul_fp32", {"a": a, "b": b})

    expected = a @ b
    np.testing.assert_allclose(res.output, expected, atol=1e-5)
    assert res.shape_signature.startswith("a(3, 4)")


def test_softmax_oracle_sums_to_one_per_axis():
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    res = oracle.compute("softmax_fp32", {"x": x, "axis": -1})
    sums = res.output.sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones(2), atol=1e-6)


def test_softmax_oracle_stable_on_large_logits():
    x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    res = oracle.compute("softmax_fp32", {"x": x, "axis": -1})
    assert np.isfinite(res.output).all()
    assert abs(res.output.sum() - 1.0) < 1e-6


def test_default_shape_cases_for_matmul_have_variety():
    cases = oracle.default_shape_cases("matmul_fp32")
    names = {c.name for c in cases}
    assert "square_2x2" in names
    assert any("rect" in n for n in names)
    assert any("vector" in n or "thin" in n for n in names)


def test_layernorm_oracle_normalises_to_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 16)).astype(np.float32)
    res = oracle.compute("layernorm_fp32", {"x": x, "eps": 1e-5})
    # Per-row mean ~ 0 and var ~ 1 (within eps bias).
    mean = res.output.mean(axis=-1)
    var = res.output.var(axis=-1)
    np.testing.assert_allclose(mean, np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(var, np.ones(3), atol=1e-3)


def test_default_shape_cases_for_softmax_have_variety():
    cases = oracle.default_shape_cases("softmax_fp32")
    names = {c.name for c in cases}
    assert any("basic" in n for n in names)
    assert any("thin" in n for n in names)
    assert any("batch" in n or "2d" in n for n in names)
    assert any("stability" in n for n in names)
    assert any("aligned" in n for n in names)


def test_default_shape_cases_for_layernorm_have_variety():
    cases = oracle.default_shape_cases("layernorm_fp32")
    names = {c.name for c in cases}
    assert any("basic" in n for n in names)
    assert any("thin" in n for n in names)
    assert any("batch" in n or "2d" in n for n in names)
    assert any("stability" in n for n in names)
    assert any("aligned" in n for n in names)
