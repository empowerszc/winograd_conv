// winograd_matrices.hpp - Winograd transform matrices
//
// This file defines the G (weight), B^T (input), and A^T (output)
// transform matrices for F(2,2,3,3) and F(4,4,3,3).
//
// These matrices come from the Winograd minimal filtering algorithm
// (Lavin & Fast, 2016). ACL uses scaled integer versions to avoid
// floating-point error accumulation.
//
// Part of the winograd_conv project.

#pragma once

#include <array>

namespace winograd_conv {

// ============================================================================
// F(2,2,3,3) matrices
// ============================================================================
// Output tile: 2x2, Input tile: 4x4, GEMMs: 16
// All coefficients are {0, +/-1} -> transforms use only addition/subtraction,
// no multiplication needed.

// Weight transform: V = G * g * G^T  (3x3 kernel -> 4x4 Winograd domain)
// G is 4x3
struct F22_G {
    static constexpr int rows = 4;
    static constexpr int cols = 3;
    // Row 0: [1,   0,   0  ]  -> pick first row of kernel
    // Row 1: [0.5, 0.5, 0.5]  -> average of three rows
    // Row 2: [0.5,-0.5, 0.5]  -> first+last minus middle
    // Row 3: [0,   0,   1  ]  -> pick last row of kernel
    static constexpr float val[4][3] = {
        {1.0f,  0.0f,  0.0f},
        {0.5f,  0.5f,  0.5f},
        {0.5f, -0.5f,  0.5f},
        {0.0f,  0.0f,  1.0f}
    };
    static constexpr float norm = 1.0f;  // no normalization
};

// Input transform: U = B^T * d * B  (4x4 input -> 4x4 Winograd domain)
// B^T is 4x4
struct F22_Bt {
    static constexpr int rows = 4;
    static constexpr int cols = 4;
    // Coefficients are all +/-1 -> only add/subtract, no multiply
    static constexpr float val[4][4] = {
        { 1.0f,  0.0f, -1.0f,  0.0f},
        { 0.0f,  1.0f,  1.0f,  0.0f},
        { 0.0f, -1.0f,  1.0f,  0.0f},
        { 0.0f,  1.0f,  0.0f, -1.0f}
    };
};

// Output transform: f = A^T * M * A  (4x4 Winograd -> 2x2 output)
// A is 2x4, A^T is 4x2
struct F22_A {
    static constexpr int rows = 2;  // output rows
    static constexpr int cols = 4;  // input cols
    static constexpr float val[2][4] = {
        {1.0f,  1.0f,  1.0f,  0.0f},
        {0.0f,  1.0f, -1.0f, -1.0f}
    };
};

// ============================================================================
// F(4,4,3,3) matrices
// ============================================================================
// Output tile: 4x4, Input tile: 6x6, GEMMs: 36
// G uses integer coefficients scaled by 24, normalization = 1/576 = 1/24^2.
// B^T uses coefficients {0, +/-1, +/-2, +/-4, +/-5}.
// A uses coefficients {1, -1, 2, 4, 8}.

// Weight transform: V = G * g * G^T / 576  (3x3 kernel -> 6x6 Winograd domain)
// G is 6x3 (scaled by 24x relative to standard Winograd G)
struct F44_G {
    static constexpr int rows = 6;
    static constexpr int cols = 3;
    static constexpr float val[6][3] = {
        { 6.0f,  0.0f,  0.0f},
        {-4.0f, -4.0f, -4.0f},
        {-4.0f,  4.0f, -4.0f},
        { 1.0f,  2.0f,  4.0f},
        { 1.0f, -2.0f,  4.0f},
        { 0.0f,  0.0f, 24.0f}
    };
    static constexpr float norm = 1.0f / 576.0f;  // 1/24^2
};

// Input transform: U = B^T * d * B  (6x6 input -> 6x6 Winograd domain)
// B^T is 6x6
// Coefficients {0, +/-1, +/-2, +/-4, +/-5} -> needs multiply for 2,4,5
struct F44_Bt {
    static constexpr int rows = 6;
    static constexpr int cols = 6;
    static constexpr float val[6][6] = {
        { 4.0f,  0.0f, -5.0f,  0.0f,  1.0f,  0.0f},
        { 0.0f, -4.0f, -4.0f,  1.0f,  1.0f,  0.0f},
        { 0.0f,  4.0f, -4.0f, -1.0f,  1.0f,  0.0f},
        { 0.0f, -2.0f, -4.0f,  2.0f,  1.0f,  0.0f},
        { 0.0f,  2.0f, -4.0f, -2.0f,  1.0f,  0.0f},
        { 0.0f,  0.0f, -5.0f,  0.0f,  0.0f,  1.0f}
    };
};

// Output transform: f = A^T * M * A  (6x6 Winograd -> 4x4 output)
// A is 4x6
struct F44_A {
    static constexpr int rows = 4;  // output rows
    static constexpr int cols = 6;  // input cols
    // Coefficients {1, -1, 2, 4, 8} -> polynomial evaluation pattern
    static constexpr float val[4][6] = {
        {1.0f,  1.0f,  1.0f,  1.0f,  1.0f, 0.0f},
        {0.0f,  1.0f, -1.0f,  2.0f, -2.0f, 0.0f},
        {0.0f,  1.0f,  1.0f,  4.0f,  4.0f, 0.0f},
        {0.0f,  1.0f, -1.0f,  8.0f, -8.0f, 1.0f}
    };
};

} // namespace winograd_conv
