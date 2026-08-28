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

// K-major ("kt") V variant: element (n,k) of B lives at V[k*OC + n], i.e. V
// is stored IC x OC -- the layout modern arm_gemm consumes natively, so the
// arm_gemm path binds it with zero per-call staging. Other backends adapt
// trivially (OpenBLAS switches from RowMajor(NoTrans,Trans) to
// RowMajor(NoTrans,NoTrans)).
//
// Note: the PIPELINE no longer produces kt data (its kt scatter cost ~16x
// write amplification on no-L3 DRAM; see tools/build_arm_gemm.md) -- it
// scatters row-major and lets the driver pack Bt per call. kt entries remain
// for callers that already hold B K-major.
void winograd_gemm_kt(
    const float* U,   // [n_tiles][IC]
    const float* V,   // [IC][OC] k-major
    float* M,         // [n_tiles][OC]
    int n_tiles,
    int OC,
    int IC
);

// Batched kt variant over nmulti CONSECUTIVE ts_idx slices with the pipeline's
// native strides (U: n_tiles*IC, V: IC*OC, M: n_tiles*OC between slices).
// Under USE_ARM_GEMM this folds several formerly separate GEMMs into a single
// arm_gemm object (GemmArgs.nmulti): object construction, kernel selection and
// B pretranspose all amortize over the batch.
void winograd_gemm_batched_kt(
    const float* U,   // [nmulti][n_tiles][IC]
    const float* V,   // [nmulti][IC][OC] k-major
    float* M,         // [nmulti][n_tiles][OC]
    int n_tiles,
    int OC,
    int IC,
    int nmulti
);

// Batched row-major variant -- what the pipeline itself uses (row-major V
// panels at stride OC*IC). arm_gemm packs all nmulti panels to Bt in one
// pass, then folds the slices into one GemmArgs.nmulti call.
void winograd_gemm_batched(
    const float* U,   // [nmulti][n_tiles][IC]
    const float* V,   // [nmulti][OC][IC]
    float* M,         // [nmulti][n_tiles][OC]
    int n_tiles,
    int OC,
    int IC,
    int nmulti
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
    float act_min = -1e30f, float act_max = 1e30f,
    Layout layout = Layout::NCHW
);

// ISA-dispatched transform functions (for testing/debugging)
void dispatch_weight_transform(
    const float* g, float* V, int channels,
    bool is_f44, ISALevel isa
);
void dispatch_input_transform(
    const float* d, float* U, int channels,
    bool is_f44, ISALevel isa
);
void dispatch_output_transform(
    const float* M, float* f, int channels,
    const float* bias, float act_min, float act_max,
    bool is_f44, ISALevel isa
);

// Convenience wrappers
inline void winograd_convolution_f22(
    const float* src, const float* wei, const float* bias, float* dst,
    int N, int IC, int IH, int IW, int OC, int OH, int OW,
    float act_min = -1e30f, float act_max = 1e30f,
    Layout layout = Layout::NCHW
) {
    winograd_convolution(src, wei, bias, dst, N, IC, IH, IW, OC, OH, OW,
                         WinogradConfig::F22_33(), act_min, act_max, layout);
}

inline void winograd_convolution_f44(
    const float* src, const float* wei, const float* bias, float* dst,
    int N, int IC, int IH, int IW, int OC, int OH, int OW,
    float act_min = -1e30f, float act_max = 1e30f,
    Layout layout = Layout::NCHW
) {
    winograd_convolution(src, wei, bias, dst, N, IC, IH, IW, OC, OH, OW,
                         WinogradConfig::F44_33(), act_min, act_max, layout);
}

} // namespace winograd_conv
