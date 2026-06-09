"""tinyinfer-agent 的默认 RAG corpus。

三类来源拼成一个 Document 列表：

1. 手写的 ONNX 风格算子语义。我们不导入完整 ONNX 规范文本 —— 太长、大部分
   是 boilerplate，而真正有用的几句话每条都很短。
2. 从 `tinyinfer/src/*.cpp` 抽出来的注释。这些保住了 in-tree 代码"为啥"，LLM
   提议测试时用作 grounding。
3. 历史 fault-injection 的 rationale —— 每个 benchmark seed 一段短描述，说明
   想抓什么类型的 bug。

corpus 文本**保留 ASCII**（英文），因为 rank-bm25 按空白分词；混入中文会
silently 让 BM25 召回退化。换中文 tokenizer（jieba）是未来工作，挂在
DEFERRED_VALIDATION.md 里。
"""

from __future__ import annotations

import re
from pathlib import Path

from agent.rag.store import Document


_OP_SEMANTICS: list[tuple[str, str]] = [
    ("matmul.gemm", "matmul_fp32 computes C[M,N] = A[M,K] * B[K,N] in row-major fp32. Reductions over k accumulate in fp32 and are sensitive to ordering when k is large."),
    ("matmul.shapes", "matmul edge shapes: m=1 (gemv-like), k=1 (outer product), n=1 (matvec), and aligned tile sizes (multiples of 8/16/32 for SIMD/AVX kernels)."),
    ("matmul.numerics", "matmul numerical stability: large-magnitude inputs risk overflow in the running accumulator; small-magnitude inputs risk denormal flush-to-zero; long k reductions accumulate rounding error proportional to sqrt(k)."),
    ("matmul.layouts", "matmul kernels often reorder loops (i,k,j vs i,j,k) for cache locality. Loop reorders preserve the mathematical result but change the order of additions, which can shift the last bit of the fp32 output."),
    ("softmax.def", "softmax_fp32 over a length-n vector: y[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x)). The max subtraction is mandatory for numerical stability: without it, large logits like 1000 produce inf/nan."),
    ("softmax.shapes", "softmax test shapes: 1D basic, 1D very short (n=2 or 3, degenerate edge), batched 2D along the last axis, and aligned lengths for SIMD softmax kernels."),
    ("softmax.stability", "softmax stability cases: large positive logits (would overflow without max-subtract), near-one-hot inputs (one logit much larger than the rest), tiny logit gaps (saturation in exp)."),
    ("layernorm.def", "layernorm_fp32: y = (x - mean(x)) / sqrt(var(x) + eps). Mean and variance are computed along the last axis. Optional gamma/beta affine multiply at the end."),
    ("layernorm.eps", "The eps term inside the sqrt prevents division by zero when the variance is exactly 0 (constant input). Removing eps causes nan on near-constant inputs."),
    ("layernorm.shapes", "layernorm test shapes: 1D basic, 1D very short (n=2), batched 2D, near-constant (variance approximately 0 stresses eps path), large-magnitude (mean/var accumulation)."),
    ("conv.def", "conv2d_fp32: output[n,c_out,h_out,w_out] = bias[c_out] + sum over c_in,kh,kw of input[n,c_in,h_out*stride+kh-pad, w_out*stride+kw-pad] * kernel[c_out,c_in,kh,kw]."),
    ("conv.shapes", "conv2d test shapes: 1x1 kernel (pointwise), 3x3 stride-1 same-padding, 7x7 stride-2 valid-padding, depthwise (groups=c_in), grouped (groups>1)."),
    ("conv.bounds", "conv2d edge cases: stride+padding off-by-one is the most common bug. Output dims are floor((H + 2*pad - kh) / stride) + 1; verify on H=1 and H=stride to catch boundary errors."),
    ("quantize.int8", "int8 quantize: q[i] = round(x[i] / scale) + zero_point, clamped to [-128, 127]. Watch for asymmetric quantization, banker's rounding vs round-half-away-from-zero, and overflow on the scale*max input boundary."),
    ("quantize.shapes", "quantize test shapes: per-tensor scale, per-channel scale, dynamic range edges (max value, max value + 1 ULP), zero input, all-same input."),
    ("memory.aliasing", "operator memory tests: input and output buffers must not overlap unless the op declares in-place support. Add guard bytes around output buffers and verify they remain untouched after the op runs."),
    ("memory.alloc", "operator allocator tests: repeated allocate-run-free in a tight loop catches leaks and use-after-free. Combined with ASan, this catches most memory bugs introduced by allocator changes."),
    ("perf.budget", "performance regression tests use a chrono-based median latency budget. The budget is loose intentionally: the goal is to catch order-of-magnitude regressions, not to certify perf. A 10x slowdown trips the budget; a 5% noise does not."),
    ("perf.tile", "tile/block size changes (TILE=8 vs 16 vs 32) affect cache utilisation and SIMD vectorization. Always perf-test the aligned shape that matches the new tile size."),
    ("perf.simd", "SIMD intrinsics (_mm256_*, _mm512_*) lock the function to a specific ISA. Add a perf test for the aligned-multiple-of-vector-width shape so regressions on the SIMD path are caught."),
]


_FAULT_INJECTION_RATIONALES: list[tuple[str, str]] = [
    ("fault.matmul.offbyone_inner", "matmul off-by-one in the k loop bound (e.g., < k - 1 instead of < k) drops the last partial product. Caught by any non-trivial shape; smaller shapes show the largest relative error."),
    ("fault.matmul.swapped_indices", "swapping b[p*n+j] for b[j*n+p] transposes the right operand. Square shapes still produce a finite result so the test relies on numerical equality with the oracle to detect it."),
    ("fault.matmul.zero_init_missing", "removing the zero-init of the output buffer makes the result depend on whatever was in memory. Caught by running the test twice and comparing results, or by running on a buffer with poisoned bytes."),
    ("fault.matmul.k_accumulator_dtype", "downgrading the accumulator from fp32 to fp16 silently loses precision on long k reductions. Caught by k=128+ tests with random inputs."),
    ("fault.softmax.missing_max_subtract", "dropping the max-subtraction makes softmax overflow on large logits like [1000, 1001, 1002] yielding nan/inf. Caught by the stability_large_values_overflow case."),
    ("fault.softmax.wrong_normalize", "normalizing by n instead of by sum(exp(x - max)) violates softmax's probability-summing-to-one invariant. Caught by checking sum(y) == 1."),
    ("fault.softmax.exp_before_subtract", "calling exp(x[i]) before subtracting max overflows on large logits even if the subtraction happens later. Same fault class as missing_max_subtract."),
    ("fault.layernorm.no_eps", "removing eps from sqrt(var + eps) divides by zero on a constant input. Caught by the stability_near_constant case."),
    ("fault.layernorm.var_with_n_minus_one", "computing variance with (n-1) instead of n shifts the result slightly; not always caught by tolerance tests, but obvious on length-2 inputs."),
    ("fault.layernorm.mean_after_subtract", "subtracting the mean from x after recomputing it from y produces wrong output but is shape-correct. Detected by oracle comparison on any non-zero-mean input."),
]


def _scrape_source_comments(repo_root: Path) -> list[Document]:
    """从 tinyinfer/src/*.cpp 里抓短小的单行 `// ...` 注释。"""
    out: list[Document] = []
    src_dir = repo_root / "tinyinfer" / "src"
    if not src_dir.is_dir():
        return out
    pat = re.compile(r"//\s*(.+?)\s*$", re.MULTILINE)
    for cpp in sorted(src_dir.glob("*.cpp")):
        try:
            text = cpp.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, m in enumerate(pat.finditer(text)):
            comment = m.group(1).strip()
            # 跳过 "AUTO-GENERATED" 头、license boilerplate、单 token 注释 ——
            # 它们没有检索信号。
            if not comment or len(comment.split()) < 4:
                continue
            if comment.lower().startswith(("auto-generated", "do not edit")):
                continue
            out.append(
                Document(
                    doc_id=f"src.{cpp.stem}.{i}",
                    text=comment,
                    metadata={"source": "src_comment", "file": cpp.name},
                )
            )
    return out


def load_default_corpus(repo_root: Path | None = None) -> list[Document]:
    """构造默认的 seed corpus。

    repo_root 不传时根据当前文件位置推断 tinyinfer-agent repo 根。
    """
    if repo_root is None:
        # agent/rag/corpus.py -> agent/rag -> agent -> repo_root
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)

    docs: list[Document] = []

    for doc_id, text in _OP_SEMANTICS:
        docs.append(
            Document(doc_id=f"op.{doc_id}", text=text, metadata={"source": "op_semantics"})
        )

    for doc_id, text in _FAULT_INJECTION_RATIONALES:
        docs.append(
            Document(doc_id=doc_id, text=text, metadata={"source": "fault_rationale"})
        )

    docs.extend(_scrape_source_comments(repo_root))
    return docs


__all__ = ["load_default_corpus"]
