# tinyinfer

Mini 算子库，作为 Agent 的"被测系统"。

## 构建 & 测试

```bash
cd tinyinfer
cmake -S . -B build
cmake --build build --config Release -j
ctest --test-dir build --output-on-failure
```

## 当前算子

| 算子 | 状态 | 文件 |
|------|------|------|
| `matmul_fp32` | ✅ baseline + Agent 生成测试 | `src/matmul.cpp` |
| `softmax_fp32` | ✅ baseline + Agent 生成测试 | `src/softmax.cpp` |
| `layernorm_fp32` | ✅ baseline + Agent 生成测试 | `src/layernorm.cpp` |
| `conv2d_fp32` | TODO | — |
| `quantize_int8` | TODO | — |

## 添加新算子

1. 头文件放 `include/tinyinfer/<op>.hpp`
2. 实现放 `src/<op>.cpp`（追加到 `add_library(tinyinfer ...)` 列表）
3. 手写 baseline 测试放 `tests/test_<op>_baseline.cpp`
4. 然后让 Agent 生成额外的回归测试
