#include "tinyinfer/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tinyinfer {

bool softmax_fp32(const float* x, float* y, std::size_t n) {
    if (x == nullptr || y == nullptr || n == 0) {
        return false;
    }

    float m = x[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (x[i] > m) m = x[i];
    }

    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = std::exp(x[i] - m);
        s += y[i];
    }

    const float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        y[i] *= inv;
    }
    return true;
}

std::vector<float> softmax_fp32(const std::vector<float>& x) {
    if (x.empty()) {
        throw std::invalid_argument("softmax_fp32: empty input");
    }
    std::vector<float> y(x.size(), 0.0f);
    softmax_fp32(x.data(), y.data(), x.size());
    return y;
}

}  // namespace tinyinfer
