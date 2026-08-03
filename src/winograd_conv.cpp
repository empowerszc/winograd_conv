// winograd_conv.cpp - End-to-end Winograd convolution implementation
//
// This file implements the full Winograd convolution pipeline:
//   1. Weight transform (one-time)
//   2. Input tile extraction + input transform
//   3. Batched GEMM (simplified naive version)
//   4. Output transform + bias + ReLU
//
// It supports both F(2,2,3,3) and F(4,4,3,3) configurations.
//
// Part of the winograd_conv project.
// Based on ACL's Winograd implementation approach.

#include "winograd_convolution.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// Include ISA-specific transform implementations
#include "winograd_transforms.hpp"           // NEON (always available on AArch64)

#if defined(__ARM_FEATURE_SVE)
#include "winograd_transforms_sve.hpp"       // SVE
#endif

#if defined(__ARM_FEATURE_SME)
#include "winograd_transforms_sme.hpp"       // SME
#endif

namespace winograd_conv {

// ============================================================================
// ISA-dispatched transform wrappers
// ============================================================================
// These functions select between NEON/SVE/SME based on runtime ISA level.
// The actual transform implementations are in the corresponding headers.

// Weight transform dispatcher
static void dispatch_weight_transform(
    const float* g, float* V, int channels,
    bool is_f44, ISALevel isa
) {
    if (is_f44) {
        switch (isa) {
            case ISALevel::NEON:
                weight_transform_f44_neon(g, V, channels); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                weight_transform_f44_sve(g, V, channels); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                weight_transform_f44_sme(g, V, channels); break;
#endif
            default:
                weight_transform_f44_neon(g, V, channels); break;
        }
    } else {
        switch (isa) {
            case ISALevel::NEON:
                weight_transform_f22_neon(g, V, channels); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                weight_transform_f22_sve(g, V, channels); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                weight_transform_f22_sme(g, V, channels); break;
#endif
            default:
                weight_transform_f22_neon(g, V, channels); break;
        }
    }
}

// Input transform dispatcher
static void dispatch_input_transform(
    const float* d, float* U, int channels,
    bool is_f44, ISALevel isa
) {
    if (is_f44) {
        switch (isa) {
            case ISALevel::NEON:
                input_transform_f44_neon(d, U, channels); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                input_transform_f44_sve(d, U, channels); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                input_transform_f44_sme(d, U, channels); break;
#endif
            default:
                input_transform_f44_neon(d, U, channels); break;
        }
    } else {
        switch (isa) {
            case ISALevel::NEON:
                input_transform_f22_neon(d, U, channels); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                input_transform_f22_sve(d, U, channels); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                input_transform_f22_sme(d, U, channels); break;
#endif
            default:
                input_transform_f22_neon(d, U, channels); break;
        }
    }
}

// Output transform dispatcher
static void dispatch_output_transform(
    const float* M, float* f, int channels,
    const float* bias, float act_min, float act_max,
    bool is_f44, ISALevel isa
) {
    if (is_f44) {
        switch (isa) {
            case ISALevel::NEON:
                output_transform_f44_neon(M, f, channels, bias, act_min, act_max); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                output_transform_f44_sve(M, f, channels, bias, act_min, act_max); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                output_transform_f44_sme(M, f, channels, bias, act_min, act_max); break;
#endif
            default:
                output_transform_f44_neon(M, f, channels, bias, act_min, act_max); break;
        }
    } else {
        switch (isa) {
            case ISALevel::NEON:
                output_transform_f22_neon(M, f, channels, bias, act_min, act_max); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                output_transform_f22_sve(M, f, channels, bias, act_min, act_max); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                output_transform_f22_sme(M, f, channels, bias, act_min, act_max); break;
#endif
            default:
                output_transform_f22_neon(M, f, channels, bias, act_min, act_max); break;
        }
    }
}

// ============================================================================
// Simple batched GEMM
// ============================================================================
// M[n][oc] = sum_ic U[n][ic] * V[oc][ic]
//
// This is a naive triple-loop GEMM. In production, replace with:
// - arm_gemm SVE kernel (sve_hybrid_fp32_mla_6x4VL)
// - or arm_gemm SME kernel (sme_interleaved_nomerge_fp32_mopa_2VLx2VL)
// - or AMX tile instruction (tdpbf16ps) on x64
//
// The key insight: this GEMM is called n_multis times (16 for F(2,2,3,3),
// 36 for F(4,4,3,3)), once per Winograd domain element.

void winograd_gemm(
    const float* U,   // [n_tiles][IC]
    const float* V,    // [OC][IC]
    float* M,           // [n_tiles][OC]
    int n_tiles,
    int OC,
    int IC
) {
    // Naive GEMM: M = U * V^T
    // For each output tile and output channel, accumulate over input channels
    for (int t = 0; t < n_tiles; t++) {
        for (int oc = 0; oc < OC; oc++) {
            float sum = 0.0f;
            for (int ic = 0; ic < IC; ic++) {
                sum += U[t * IC + ic] * V[oc * IC + ic];
            }
            M[t * OC + oc] = sum;
        }
    }
}

// ============================================================================
// Direct (reference) 3x3 convolution with stride=1, pad=1
// ============================================================================
// Standard implementation for verification.
// Input format: NCHW (batch, channels, height, width)

void direct_convolution_3x3(
    const float* src,   // [N][IC][IH][IW]
    const float* wei,   // [OC][IC][3][3]
    const float* bias,  // [OC] or nullptr
    float* dst,         // [N][OC][OH][OW]
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    float act_min, float act_max
) {
    // Assume stride=1, pad=1, dilation=1, so OH=IH, OW=IW
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float sum = bias ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int ih = oh - 1 + kh;  // pad=1: ih = oh + kh - 1
                                int iw = ow - 1 + kw;

                                // Bounds check with zero padding
                                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                    float s = src[((n * IC + ic) * IH + ih) * IW + iw];
                                    float w = wei[((oc * IC + ic) * 3 + kh) * 3 + kw];
                                    sum += s * w;
                                }
                                // else: padding contributes 0
                            }
                        }
                    }

                    // Activation (clamp)
                    if (sum > act_max) sum = act_max;
                    if (sum < act_min) sum = act_min;

                    dst[((n * OC + oc) * OH + oh) * OW + ow] = sum;
                }
            }
        }
    }
}

// ============================================================================
// Winograd convolution
// ============================================================================

void winograd_convolution(
    const float* src,   // [N][IC][IH][IW] (NCHW)
    const float* wei,   // [OC][IC][3][3]
    const float* bias,  // [OC] or nullptr
    float* dst,         // [N][OC][OH][OW] (NCHW)
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    const WinogradConfig& config,
    float act_min, float act_max
) {
    const int TS = config.input_tile_rows;  // tile size = m + r - 1
    const int OT = config.output_tile_rows;  // output tile = m
    const int NM = config.n_multis;          // number of GEMMs = TS^2
    const bool is_f44 = (TS == 6);

    // Get runtime ISA level (can be set via set_isa_level() or env var)
    ISALevel isa = isa_level();

    // Check for environment variable override
    if (const char* env = std::getenv("WINOGRAD_ISA")) {
        isa = parse_isa(env);
    }

    int n_tile_rows = config.n_tile_rows(OH);
    int n_tile_cols = config.n_tile_cols(OW);
    int n_tiles = n_tile_rows * n_tile_cols;

    // ---- Step 1: Weight transform (one-time) ----
    // V[TS][TS][OC][IC] = G * g * G^T / norm
    // For each (oc, ic) pair, transform the 3x3 weight to TSxTS
    //
    // Layout: V[ts_row][ts_col][oc * IC + ic]
    // This is the "weight matrix" used by all GEMMs

    int V_size = TS * TS * OC * IC;
    std::vector<float> V(V_size, 0.0f);

    // For each output channel and input channel
    for (int oc = 0; oc < OC; oc++) {
        for (int ic = 0; ic < IC; ic++) {
            // Extract 3x3 weight for this (oc, ic)
            float g[3][3];
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    g[kh][kw] = wei[((oc * IC + ic) * 3 + kh) * 3 + kw];
                }
            }

            // Transform: V = G * g * G^T / norm
            // Step 1: Ww = G * g (row transform)
            float Ww[6][3];  // max TS=6, kw=3
            for (int i = 0; i < TS; i++) {
                for (int j = 0; j < 3; j++) {
                    float val = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        // Use the appropriate G matrix
                        float g_coef;
                        if (TS == 4) {
                            g_coef = F22_G::val[i][k];
                        } else {
                            g_coef = F44_G::val[i][k];
                        }
                        val += g_coef * g[k][j];
                    }
                    Ww[i][j] = val;
                }
            }

            // Step 2: V = Ww * G^T / norm (col transform)
            float norm = (TS == 4) ? F22_G::norm : F44_G::norm;
            for (int i = 0; i < TS; i++) {
                for (int j = 0; j < TS; j++) {
                    float val = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        float g_coef;
                        if (TS == 4) {
                            g_coef = F22_G::val[j][k];  // G^T[j][k] = G[k][j]... wait
                            // Actually V[i][j] = sum_k Ww[i][k] * G[j][k]
                            // Because V = Ww * G^T, and (G^T)[j][k] = G[k][j]
                            // So V[i][j] = sum_k Ww[i][k] * G[k][j]
                            g_coef = F22_G::val[j][k];
                        } else {
                            g_coef = F44_G::val[j][k];
                        }
                        val += Ww[i][k] * g_coef;
                    }
                    val *= norm;  // apply normalization
                    // Store: V[ts_row][ts_col][oc * IC + ic]
                    V[(i * TS + j) * OC * IC + oc * IC + ic] = val;
                }
            }
        }
    }

    // ---- Step 2: For each batch ----
    for (int n = 0; n < N; n++) {
        // ---- Step 2a: Input transform ----
        // For each tile, extract input tile (with padding), transform to U
        //
        // U[n_multis][n_tiles][IC]  (Winograd domain input)
        // Layout: U[ts_idx][tile_idx][ic]
        //   where ts_idx = ts_row * TS + ts_col (0..NM-1)
        //   and tile_idx = tile_row * n_tile_cols + tile_col

        int U_size = NM * n_tiles * IC;
        std::vector<float> U(U_size, 0.0f);

        for (int tr = 0; tr < n_tile_rows; tr++) {
            for (int tc = 0; tc < n_tile_cols; tc++) {
                int tile_idx = tr * n_tile_cols + tc;

                // Extract input tile [TS][TS][IC] from src (with zero padding)
                // tile starts at (tr * OT - 1, tc * OT - 1) in input space
                // (pad=1 means the first output pixel corresponds to input position -1)
                std::vector<float> d_tile(TS * TS * IC, 0.0f);

                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ih = tr * OT - 1 + ti;  // input row (with pad=1 offset)
                        int iw = tc * OT - 1 + tj;  // input col

                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            for (int ic = 0; ic < IC; ic++) {
                                d_tile[(ti * TS + tj) * IC + ic] =
                                    src[((n * IC + ic) * IH + ih) * IW + iw];
                            }
                        }
                        // else: padding = 0 (already zeroed)
                    }
                }

                // Transform: U_tile = B^T * d * B
                std::vector<float> U_tile(TS * TS * IC, 0.0f);

                dispatch_input_transform(d_tile.data(), U_tile.data(), IC,
                                         is_f44, isa);

                // Scatter U_tile into U[ts_idx][tile_idx][:]
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ts_idx = ti * TS + tj;
                        for (int ic = 0; ic < IC; ic++) {
                            U[(ts_idx * n_tiles + tile_idx) * IC + ic] =
                                U_tile[(ti * TS + tj) * IC + ic];
                        }
                    }
                }
            }
        }

        // ---- Step 2b: GEMM ----
        // For each Winograd domain element (ts_idx = 0..NM-1):
        //   M[ts_idx][tile_idx][oc] = U[ts_idx][tile_idx][ic] * V[ts_idx][oc][ic]
        //
        // This is NM independent GEMMs, each of size [n_tiles x OC x IC]

        int M_size = NM * n_tiles * OC;
        std::vector<float> M(M_size, 0.0f);

        for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
            // U_slice: [n_tiles][IC]
            const float* U_slice = U.data() + ts_idx * n_tiles * IC;
            // V_slice: [OC][IC]
            const float* V_slice = V.data() + ts_idx * OC * IC;
            // M_slice: [n_tiles][OC]
            float* M_slice = M.data() + ts_idx * n_tiles * OC;

            winograd_gemm(U_slice, V_slice, M_slice, n_tiles, OC, IC);
        }

        // ---- Step 2c: Output transform ----
        // For each tile: f[OT][OT][OC] = A^T * M_tile * A + bias + ReLU

        for (int tr = 0; tr < n_tile_rows; tr++) {
            for (int tc = 0; tc < n_tile_cols; tc++) {
                int tile_idx = tr * n_tile_cols + tc;

                // Gather M_tile[TS][TS][OC] from M[ts_idx][tile_idx][oc]
                std::vector<float> M_tile(TS * TS * OC, 0.0f);
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ts_idx = ti * TS + tj;
                        for (int oc = 0; oc < OC; oc++) {
                            M_tile[(ti * TS + tj) * OC + oc] =
                                M[(ts_idx * n_tiles + tile_idx) * OC + oc];
                        }
                    }
                }

                // Transform: f_tile = A^T * M_tile * A
                std::vector<float> f_tile(OT * OT * OC, 0.0f);

                dispatch_output_transform(M_tile.data(), f_tile.data(), OC,
                                          bias, act_min, act_max, is_f44, isa);

                // Add bias (per output channel)
                if (bias) {
                    for (int oi = 0; oi < OT; oi++) {
                        for (int oj = 0; oj < OT; oj++) {
                            for (int oc = 0; oc < OC; oc++) {
                                float v = f_tile[(oi * OT + oj) * OC + oc] + bias[oc];
                                if (v > act_max) v = act_max;
                                if (v < act_min) v = act_min;
                                f_tile[(oi * OT + oj) * OC + oc] = v;
                            }
                        }
                    }
                }

                // Write to output (with bounds checking for edge tiles)
                for (int oi = 0; oi < OT; oi++) {
                    for (int oj = 0; oj < OT; oj++) {
                        int oh = tr * OT + oi;
                        int ow = tc * OT + oj;
                        if (oh < OH && ow < OW) {
                            for (int oc = 0; oc < OC; oc++) {
                                dst[((n * OC + oc) * OH + oh) * OW + ow] =
                                    f_tile[(oi * OT + oj) * OC + oc];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace winograd_conv
