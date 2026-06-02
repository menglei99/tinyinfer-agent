#pragma once

#include <cstddef>
#include <vector>

namespace tinyinfer {

// 1D LayerNorm over a contiguous buffer of length n.
//   mean = sum(x) / n
//   var  = sum((x - mean)^2) / n
//   y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
// gamma / beta are optional; pass nullptr to skip affine (treated as 1 / 0).
// Returns false on null x/y or n == 0.
bool layernorm_fp32(const float* x, float* y, std::size_t n,
                    const float* gamma = nullptr,
                    const float* beta = nullptr,
                    float eps = 1e-5f);

// Convenience wrapper.
std::vector<float> layernorm_fp32(const std::vector<float>& x,
                                  float eps = 1e-5f);

}  // namespace tinyinfer
