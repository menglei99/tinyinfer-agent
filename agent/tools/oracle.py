"""Reference oracle: numpy as ground truth for operator outputs.

Why numpy and not ONNX Runtime for MVP:
  - matmul / softmax / layernorm have direct numpy equivalents
  - keeps deps light, runs anywhere
ONNX Runtime is wired in for ops that don't have a clean numpy form
(quantize, fused conv) — see Week 3 in ROADMAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class OracleResult:
    output: np.ndarray
    inputs: dict
    shape_signature: str
    notes: str = ""


# Each registered op takes a dict of named np arrays and returns OracleResult.
_OracleFn = Callable[[dict], OracleResult]
_REGISTRY: dict[str, _OracleFn] = {}


def register(op_name: str):
    def deco(fn: _OracleFn) -> _OracleFn:
        _REGISTRY[op_name] = fn
        return fn

    return deco


def has_oracle(op_name: str) -> bool:
    return op_name in _REGISTRY


def compute(op_name: str, inputs: dict) -> OracleResult:
    if op_name not in _REGISTRY:
        raise KeyError(f"No reference oracle registered for op {op_name!r}")
    return _REGISTRY[op_name](inputs)


# ---------- registered oracles ----------


@register("matmul_fp32")
def _matmul(inputs: dict) -> OracleResult:
    a = np.asarray(inputs["a"], dtype=np.float32)
    b = np.asarray(inputs["b"], dtype=np.float32)
    out = (a @ b).astype(np.float32)
    return OracleResult(
        output=out,
        inputs={"a": a, "b": b},
        shape_signature=f"a{tuple(a.shape)}@b{tuple(b.shape)}->c{tuple(out.shape)}",
    )


@register("softmax_fp32")
def _softmax(inputs: dict) -> OracleResult:
    x = np.asarray(inputs["x"], dtype=np.float32)
    axis = int(inputs.get("axis", -1))
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    out = (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32)
    return OracleResult(
        output=out, inputs={"x": x, "axis": axis}, shape_signature=f"x{tuple(x.shape)}@axis{axis}"
    )


@register("layernorm_fp32")
def _layernorm(inputs: dict) -> OracleResult:
    x = np.asarray(inputs["x"], dtype=np.float32)
    eps = float(inputs.get("eps", 1e-5))
    # LayerNorm reduces over the last axis (rows for 2D, the vector itself for 1D).
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = ((x - mean) / np.sqrt(var + eps)).astype(np.float32)
    return OracleResult(
        output=out,
        inputs={"x": x, "eps": eps},
        shape_signature=f"x{tuple(x.shape)}@eps{eps:g}",
    )


# ---------- shape generators ----------


@dataclass
class ShapeCase:
    name: str
    inputs: dict = field(default_factory=dict)
    rationale: str = ""


def default_shape_cases(op_name: str, *, rng: np.random.Generator | None = None) -> list[ShapeCase]:
    """Hand-curated shape variety per op. Used when LLM declines or in mock mode."""
    rng = rng or np.random.default_rng(seed=42)

    if op_name == "matmul_fp32":
        return [
            ShapeCase(
                name="square_2x2",
                inputs={
                    "a": rng.standard_normal((2, 2)).tolist(),
                    "b": rng.standard_normal((2, 2)).tolist(),
                    "m": 2, "k": 2, "n": 2,
                },
                rationale="smallest non-trivial square",
            ),
            ShapeCase(
                name="rect_3x4_4x2",
                inputs={
                    "a": rng.standard_normal((3, 4)).tolist(),
                    "b": rng.standard_normal((4, 2)).tolist(),
                    "m": 3, "k": 4, "n": 2,
                },
                rationale="non-square shape, asymmetric m vs n",
            ),
            ShapeCase(
                name="vector_1xk_kx1",
                inputs={
                    "a": rng.standard_normal((1, 5)).tolist(),
                    "b": rng.standard_normal((5, 1)).tolist(),
                    "m": 1, "k": 5, "n": 1,
                },
                rationale="degenerate row/col vectors -> scalar",
            ),
            ShapeCase(
                name="thin_1x8_8x4",
                inputs={
                    "a": rng.standard_normal((1, 8)).tolist(),
                    "b": rng.standard_normal((8, 4)).tolist(),
                    "m": 1, "k": 8, "n": 4,
                },
                rationale="m=1 path frequently has separate optimisation",
            ),
            # ---- numerical stability cases ----
            ShapeCase(
                name="stability_large_magnitude",
                inputs={
                    "a": (rng.standard_normal((2, 4)) * 1e4).tolist(),
                    "b": (rng.standard_normal((4, 2)) * 1e4).tolist(),
                    "m": 2, "k": 4, "n": 2,
                },
                rationale="large magnitudes; products approach fp32 range upper edge",
            ),
            ShapeCase(
                name="stability_small_magnitude",
                inputs={
                    "a": (rng.standard_normal((2, 4)) * 1e-4).tolist(),
                    "b": (rng.standard_normal((4, 2)) * 1e-4).tolist(),
                    "m": 2, "k": 4, "n": 2,
                },
                rationale="small magnitudes; risk of denormal flush-to-zero",
            ),
            ShapeCase(
                name="stability_long_reduction_k128",
                inputs={
                    "a": rng.standard_normal((1, 128)).tolist(),
                    "b": rng.standard_normal((128, 1)).tolist(),
                    "m": 1, "k": 128, "n": 1,
                },
                rationale="long k reduction; accumulation error grows with k",
            ),
            ShapeCase(
                name="outer_product_k1",
                inputs={
                    "a": rng.standard_normal((17, 1)).tolist(),
                    "b": rng.standard_normal((1, 33)).tolist(),
                    "m": 17, "k": 1, "n": 33,
                },
                rationale="k=1 outer-product path often has specialised kernels",
            ),
            ShapeCase(
                name="aligned_16x32_32x64",
                inputs={
                    "a": rng.standard_normal((16, 32)).tolist(),
                    "b": rng.standard_normal((32, 64)).tolist(),
                    "m": 16, "k": 32, "n": 64,
                },
                rationale="hardware-aligned tile sizes for SIMD/vectorised kernels",
            ),
            # ---- API-contract case: caller-owned c[] is preloaded with
            # non-zero values. A correct implementation MUST overwrite each
            # c[i*n+j], not treat it as an accumulator. Catches the
            # `acc = c[i*n+j]` class of bugs (seed 03 in our benchmark).
            ShapeCase(
                name="prefilled_output_buffer_2x2",
                inputs={
                    "a": rng.standard_normal((2, 2)).tolist(),
                    "b": rng.standard_normal((2, 2)).tolist(),
                    # Distinct non-zero garbage so coincidental cancellations
                    # with sum(a*b) are very unlikely.
                    "c_init": [2.5, -1.75, 3.125, -0.5],
                    "m": 2, "k": 2, "n": 2,
                },
                rationale=(
                    "output buffer prefilled with non-zero values; "
                    "guards against accumulator-mode bugs that read c[] as initial state"
                ),
            ),
        ]

    if op_name == "softmax_fp32":
        return [
            ShapeCase(
                name="basic_1d_n8",
                inputs={"x": rng.standard_normal(8).astype(np.float32).tolist(), "axis": -1},
                rationale="1D simplest case",
            ),
            ShapeCase(
                name="thin_1d_n3",
                inputs={"x": rng.standard_normal(3).astype(np.float32).tolist(), "axis": -1},
                rationale="very short vector — degenerate edge",
            ),
            ShapeCase(
                name="batch_2d_axis_last_4x16",
                inputs={"x": rng.standard_normal((4, 16)).astype(np.float32).tolist(), "axis": -1},
                rationale="batched softmax along last axis",
            ),
            ShapeCase(
                name="stability_large_values_overflow",
                inputs={"x": [1000.0, 1001.0, 1002.0], "axis": -1},
                rationale="numerical stability on large logits",
            ),
            ShapeCase(
                name="stability_negative_inf_one_hot",
                inputs={"x": [0.0, 0.0, 100.0, 0.0], "axis": -1},
                rationale="near-one-hot logits stress the softmax tail",
            ),
            ShapeCase(
                name="aligned_1d_n32",
                inputs={"x": rng.standard_normal(32).astype(np.float32).tolist(), "axis": -1},
                rationale="hardware-aligned length for SIMD softmax",
            ),
        ]

    if op_name == "layernorm_fp32":
        return [
            ShapeCase(
                name="basic_1d_n4",
                inputs={"x": rng.standard_normal(4).astype(np.float32).tolist(), "eps": 1e-5},
                rationale="smallest non-trivial 1D case",
            ),
            ShapeCase(
                name="thin_1d_n2",
                inputs={"x": [0.0, 1.0], "eps": 1e-5},
                rationale="length-2 vector — degenerate edge",
            ),
            ShapeCase(
                name="batch_2d_4x16",
                inputs={"x": rng.standard_normal((4, 16)).astype(np.float32).tolist(), "eps": 1e-5},
                rationale="batched layernorm along last axis",
            ),
            ShapeCase(
                name="stability_large_magnitude",
                inputs={"x": (rng.standard_normal(8) * 1e4).astype(np.float32).tolist(), "eps": 1e-5},
                rationale="large magnitudes stress mean/var accumulation",
            ),
            ShapeCase(
                name="stability_near_constant",
                inputs={"x": [1.0, 1.0, 1.0, 1.0001], "eps": 1e-5},
                rationale="near-constant input — variance ≈ 0 stresses eps path",
            ),
            ShapeCase(
                name="aligned_1d_n32",
                inputs={"x": rng.standard_normal(32).astype(np.float32).tolist(), "eps": 1e-5},
                rationale="hardware-aligned length for SIMD layernorm",
            ),
        ]

    return []
