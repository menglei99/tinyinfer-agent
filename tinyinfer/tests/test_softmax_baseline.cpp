// Hand-written baseline test for softmax_fp32.
#include "tinyinfer/softmax.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace {

constexpr float kTol = 1e-5f;

TEST(SoftmaxBaseline, ThreeElementMatchesByHand) {
    // x = [1, 2, 3]
    // expected = [e^-2, e^-1, 1] / (e^-2 + e^-1 + 1)
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    auto y = tinyinfer::softmax_fp32(x);

    ASSERT_EQ(y.size(), 3u);
    const float denom = std::exp(-2.0f) + std::exp(-1.0f) + 1.0f;
    EXPECT_NEAR(y[0], std::exp(-2.0f) / denom, kTol);
    EXPECT_NEAR(y[1], std::exp(-1.0f) / denom, kTol);
    EXPECT_NEAR(y[2], 1.0f / denom, kTol);

    // Probabilities must sum to 1.
    const float s = std::accumulate(y.begin(), y.end(), 0.0f);
    EXPECT_NEAR(s, 1.0f, kTol);
}

TEST(SoftmaxBaseline, StableOnLargeLogits) {
    // Without max-subtraction this would overflow to inf/nan.
    std::vector<float> x = {1000.0f, 1001.0f, 1002.0f};
    auto y = tinyinfer::softmax_fp32(x);

    for (float v : y) {
        EXPECT_TRUE(std::isfinite(v));
    }
    const float s = std::accumulate(y.begin(), y.end(), 0.0f);
    EXPECT_NEAR(s, 1.0f, kTol);
}

TEST(SoftmaxBaseline, NullPointersReturnFalse) {
    EXPECT_FALSE(tinyinfer::softmax_fp32(nullptr, nullptr, 4));
}

TEST(SoftmaxBaseline, ZeroLengthReturnsFalse) {
    std::vector<float> x{1.0f};
    std::vector<float> y{0.0f};
    EXPECT_FALSE(tinyinfer::softmax_fp32(x.data(), y.data(), 0));
}

}  // namespace
