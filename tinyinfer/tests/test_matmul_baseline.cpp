// Hand-written baseline test. Proves the test infra works before any
// Agent-generated tests land. Agent-generated tests live next to this one.
#include "tinyinfer/matmul.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace {

constexpr float kTol = 1e-5f;

TEST(MatmulBaseline, Identity2x2) {
    // I * x = x
    std::vector<float> a = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> b = {3.0f, 4.0f, 5.0f, 6.0f};
    auto c = tinyinfer::matmul_fp32(a, b, 2, 2, 2);

    ASSERT_EQ(c.size(), 4u);
    EXPECT_NEAR(c[0], 3.0f, kTol);
    EXPECT_NEAR(c[1], 4.0f, kTol);
    EXPECT_NEAR(c[2], 5.0f, kTol);
    EXPECT_NEAR(c[3], 6.0f, kTol);
}

TEST(MatmulBaseline, Rectangular_2x3_times_3x2) {
    std::vector<float> a = {1, 2, 3, 4, 5, 6};        // [2,3]
    std::vector<float> b = {7, 8, 9, 10, 11, 12};     // [3,2]
    auto c = tinyinfer::matmul_fp32(a, b, 2, 3, 2);

    // Reference computed by numpy:
    //   [[ 58,  64], [139, 154]]
    ASSERT_EQ(c.size(), 4u);
    EXPECT_NEAR(c[0],  58.0f, kTol);
    EXPECT_NEAR(c[1],  64.0f, kTol);
    EXPECT_NEAR(c[2], 139.0f, kTol);
    EXPECT_NEAR(c[3], 154.0f, kTol);
}

TEST(MatmulBaseline, NullPointersReturnFalse) {
    EXPECT_FALSE(tinyinfer::matmul_fp32(nullptr, nullptr, nullptr, 1, 1, 1));
}

}  // namespace
