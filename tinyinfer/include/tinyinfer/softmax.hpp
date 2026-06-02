#pragma once

#include <cstddef>
#include <vector>

namespace tinyinfer {

// Numerically stable 1D softmax over a contiguous buffer of length n.
// y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
// Returns false on null pointers or n == 0.
bool softmax_fp32(const float* x, float* y, std::size_t n);

// Convenience wrapper for std::vector.
std::vector<float> softmax_fp32(const std::vector<float>& x);

}  // namespace tinyinfer
