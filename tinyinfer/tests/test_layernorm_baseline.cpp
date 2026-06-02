// Hand-written baseline test for layernorm_fp32.
#include "tinyinfer/layernorm.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace {

constexpr float kTol = 1e-4f;

TEST(LayernormBaseline, MeanZeroVarOne) {
    // x = [1, 2, 3, 4]; mean = 2.5; var = 1.25
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    auto y = tinyinfer::layernorm_fp32(x);

    const float mean = std::accumulate(y.begin(), y.end(), 0.0f) /
                       static_cast<float>(y.size());
    float var = 0.0f;
    for (float v : y) var += (v - mean) * (v - mean);
    var /= static_cast<float>(y.size());

    EXPECT_NEAR(mean, 0.0f, kTol);
    EXPECT_NEAR(var, 1.0f, 1e-2f);  // eps gives slight bias on variance.
}

TEST(LayernormBaseline, MatchesByHandSmall) {
    // x = [0, 1]; mean = 0.5; var = 0.25; std ≈ 0.5
    // y0 = (0 - 0.5) / sqrt(0.25 + 1e-5) ≈ -1
    // y1 = (1 - 0.5) / sqrt(0.25 + 1e-5) ≈ +1
    std::vector<float> x = {0.0f, 1.0f};
    auto y = tinyinfer::layernorm_fp32(x);

    EXPECT_NEAR(y[0], -1.0f, 1e-3f);
    EXPECT_NEAR(y[1], 1.0f, 1e-3f);
}

TEST(LayernormBaseline, NullPointersReturnFalse) {
    EXPECT_FALSE(tinyinfer::layernorm_fp32(nullptr, nullptr, 4));
}

TEST(LayernormBaseline, ZeroLengthReturnsFalse) {
    std::vector<float> x{1.0f};
    std::vector<float> y{0.0f};
    EXPECT_FALSE(tinyinfer::layernorm_fp32(x.data(), y.data(), 0));
}

}  // namespace
