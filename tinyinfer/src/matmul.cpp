#include "tinyinfer/matmul.hpp"

#include <stdexcept>

namespace tinyinfer {

bool matmul_fp32(const float* a, const float* b, float* c,
                 std::size_t m, std::size_t k, std::size_t n) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        return false;
    }

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (std::size_t p = 0; p < k; ++p) {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    return true;
}

std::vector<float> matmul_fp32(const std::vector<float>& a,
                               const std::vector<float>& b,
                               std::size_t m, std::size_t k, std::size_t n) {
    if (a.size() != m * k || b.size() != k * n) {
        throw std::invalid_argument("matmul_fp32: input size mismatch");
    }
    std::vector<float> c(m * n, 0.0f);
    matmul_fp32(a.data(), b.data(), c.data(), m, k, n);
    return c;
}

}  // namespace tinyinfer
