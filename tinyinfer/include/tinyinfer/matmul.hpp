#pragma once

#include <cstddef>
#include <vector>

namespace tinyinfer {

// Row-major 2D matmul: C[M,N] = A[M,K] * B[K,N]
// Returns false on shape mismatch.
bool matmul_fp32(const float* a, const float* b, float* c,
                 std::size_t m, std::size_t k, std::size_t n);

// Convenience wrapper for std::vector.
std::vector<float> matmul_fp32(const std::vector<float>& a,
                               const std::vector<float>& b,
                               std::size_t m, std::size_t k, std::size_t n);

}  // namespace tinyinfer
