// winograd_convolution.hpp - End-to-end Winograd convolution
//
// This file assembles the three Winograd steps:
//   1. Weight transform (one-time, at prepare stage)
//   2. Input transform + GEMM + Output transform (per forward pass)
//
// It also provides a reference direct convolution for verification.
//
// Part of the winograd_conv project.

#pragma once

#include <vector>
#include <cstddef>
#include "winograd_config.hpp"
#include "winograd_transforms.hpp"

namespace winograd_conv {

// ============================================================================
// Simple batched GEMM for Winograd domain
// ============================================================================
// Computes: M[n][oc] = sum_ic U[n][ic] * V[oc][ic]
// This is a naive but correct implementation. In production, this would
// be replaced by arm_gemm or a tuned SVE/AMX GEMM kernel.
//
// Parameters:
//   U: [n_tiles][IC] - transformed input (one element of the Winograd domain)
//   V: [OC][IC] - transformed weights (one element of the Winograd domain)
//   M: [n_tiles][OC] - output (one element of the Winograd domain)
//   n_tiles, OC, IC - dimensions

void winograd_gemm(
    const float* U,   // [n_tiles][IC]
    const float* V,    // [OC][IC]
    float* M,           // [n_tiles][OC]
    int n_tiles,
    int OC,
    int IC
);

// ============================================================================
// Direct (reference) convolution
// ============================================================================
// Standard 3x3 convolution with stride 1, pad 1.
// Used as ground truth for verification.
//
// Parameters:
//   src:   [N][IC][IH][IW] (NCHW format)
//   wei:   [OC][IC][3][3]
//   bias:  [OC] or nullptr
//   dst:   [N][OC][OH][OW]
//   N, IC, IH, IW, OC, OH, OW - dimensions
//   act_min, act_max - activation clamp range

void direct_convolution_3x3(
    const float* src,   // [N][IC][IH][IW]
    const float* wei,   // [OC][IC][3][3]
    const float* bias,  // [OC] or nullptr
    float* dst,         // [N][OC][OH][OW]
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    float act_min = -1e30f, float act_max = 1e30f
);

// ============================================================================
// Winograd convolution (F(2,2,3,3) or F(4,4,3,3))
// ============================================================================
// Full Winograd convolution pipeline.
//
// Steps:
//   1. Weight transform: V = G * g * G^T / norm  (call once)
//   2. For each batch:
//      a. Extract input tiles (with padding)
//      b. Input transform: U = B^T * d * B  (per tile)
//      c. GEMM: M[tile][oc] = U[tile][ic] * V[oc][ic]  (per Winograd element)
//      d. Output transform: f = A^T * M * A + bias + ReLU  (per tile)
//      e. Write output tile to dst
//
// Parameters:
//   src:   [N][IC][IH][IW] (NCHW format)
//   wei:   [OC][IC][3][3]
//   bias:  [OC] or nullptr
//   dst:   [N][OC][OH][OW]
//   N, IC, IH, IW, OC, OH, OW - dimensions
//   config - WinogradConfig (F22 or F44)
//   act_min, act_max - activation clamp range

void winograd_convolution(
    const float* src,
    const float* wei,
    const float* bias,
    float* dst,
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    const WinogradConfig& config,
    float act_min = -1e30f, float act_max = 1e30f
);

// Convenience wrappers
inline void winograd_convolution_f22(
    const float* src, const float* wei, const float* bias, float* dst,
    int N, int IC, int IH, int IW, int OC, int OH, int OW,
    float act_min = -1e30f, float act_max = 1e30f
) {
    winograd_convolution(src, wei, bias, dst, N, IC, IH, IW, OC, OH, OW,
                         WinogradConfig::F22_33(), act_min, act_max);
}

inline void winograd_convolution_f44(
    const float* src, const float* wei, const float* bias, float* dst,
    int N, int IC, int IH, int IW, int OC, int OH, int OW,
    float act_min = -1e30f, float act_max = 1e30f
) {
    winograd_convolution(src, wei, bias, dst, N, IC, IH, IW, OC, OH, OW,
                         WinogradConfig::F44_33(), act_min, act_max);
}

} // namespace winograd_conv
