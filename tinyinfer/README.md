# tinyinfer

Mini operator library serving as the "system under test" for the Agent.

## Build & test

```bash
cd tinyinfer
cmake -S . -B build
cmake --build build --config Release -j
ctest --test-dir build --output-on-failure
```

## Operators (current)

| Operator | Status | File |
|----------|--------|------|
| `matmul_fp32` | ✅ baseline | `src/matmul.cpp` |
| `conv2d_fp32` | TODO | — |
| `softmax_fp32` | TODO | — |
| `layernorm_fp32` | TODO | — |
| `quantize_int8` | TODO | — |

## Adding a new operator

1. Header in `include/tinyinfer/<op>.hpp`
2. Impl in `src/<op>.cpp`(append to `add_library(tinyinfer ...)` list)
3. Hand-written baseline test in `tests/test_<op>_baseline.cpp`
4. Then let the Agent generate additional regression tests
