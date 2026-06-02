#include "tinyinfer/layernorm.hpp"

#include <cmath>
#include <stdexcept>

namespace tinyinfer {

bool layernorm_fp32(const float* x, float* y, std::size_t n,
                    const float* gamma, const float* beta, float eps) {
    if (x == nullptr || y == nullptr || n == 0) {
        return false;
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += x[i];
    const float mean = static_cast<float>(sum / static_cast<double>(n));

    double sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(x[i]) - mean;
        sq += d * d;
    }
    const float var = static_cast<float>(sq / static_cast<double>(n));
    const float inv_std = 1.0f / std::sqrt(var + eps);

    for (std::size_t i = 0; i < n; ++i) {
        float v = (x[i] - mean) * inv_std;
        if (gamma) v *= gamma[i];
        if (beta) v += beta[i];
        y[i] = v;
    }
    return true;
}

std::vector<float> layernorm_fp32(const std::vector<float>& x, float eps) {
    if (x.empty()) {
        throw std::invalid_argument("layernorm_fp32: empty input");
    }
    std::vector<float> y(x.size(), 0.0f);
    layernorm_fp32(x.data(), y.data(), x.size(), nullptr, nullptr, eps);
    return y;
}

}  // namespace tinyinfer
