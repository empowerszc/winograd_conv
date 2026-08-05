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
#include <arm_neon.h>

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

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
void dispatch_weight_transform(
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
void dispatch_input_transform(
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
void dispatch_output_transform(
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
// Batched GEMM for Winograd domain
// ============================================================================
// M[n][oc] = sum_ic U[n][ic] * V[oc][ic]
// This is M[n_tiles x OC] = U[n_tiles x IC] * V[OC x IC]^T
// Called n_multis times (16 for F(2,2,3,3), 36 for F(4,4,3,3)).

void winograd_gemm(
    const float* U,   // [n_tiles][IC]
    const float* V,    // [OC][IC]
    float* M,           // [n_tiles][OC]
    int n_tiles,
    int OC,
    int IC
) {
#ifdef USE_OPENBLAS
    // M = U * V^T  (U: n_tiles×IC, V: OC×IC, result: n_tiles×OC)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_tiles, OC, IC,
                1.0f, U, IC, V, IC,
                0.0f, M, OC);
#else
    // Naive fallback: triple loop
    for (int t = 0; t < n_tiles; t++) {
        for (int oc = 0; oc < OC; oc++) {
            float sum = 0.0f;
            for (int ic = 0; ic < IC; ic++) {
                sum += U[t * IC + ic] * V[oc * IC + ic];
            }
            M[t * OC + oc] = sum;
        }
    }
#endif
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

    // ---- Step 1: Weight transform (one-time, via ISA dispatch) ----
    // V[TS*TS][OC*IC], layout: V[m][oc*IC+ic]
    int V_size = TS * TS * OC * IC;
    std::vector<float> V(V_size, 0.0f);

    for (int oc = 0; oc < OC; oc++) {
        // Rearrange wei[oc][IC][3][3] → g[3][3][IC] (channels-contiguous)
        std::vector<float> g(9 * IC);
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[(kh * 3 + kw) * IC + ic] =
                        wei[((oc * IC + ic) * 3 + kh) * 3 + kw];

        // Transform via ISA dispatch (NEON/SVE/SME)
        std::vector<float> V_oc(TS * TS * IC);
        dispatch_weight_transform(g.data(), V_oc.data(), IC, is_f44, isa);

        // Store into V[m][oc*IC+ic]
        for (int m = 0; m < TS * TS; m++)
            for (int ic = 0; ic < IC; ic++)
                V[m * OC * IC + oc * IC + ic] = V_oc[m * IC + ic];
    }

    // ---- Pre-allocate workspace (reused across batches) ----
    int U_size = NM * n_tiles * IC;
    int M_size = NM * n_tiles * OC;
    std::vector<float> U(U_size, 0.0f);
    std::vector<float> M_buf(M_size, 0.0f);
    std::vector<float> d_tile(TS * TS * IC, 0.0f);
    std::vector<float> U_tile(TS * TS * IC, 0.0f);
    std::vector<float> M_tile(TS * TS * OC, 0.0f);
    std::vector<float> f_tile(OT * OT * OC, 0.0f);

    // ---- Step 2: For each batch ----
    for (int n = 0; n < N; n++) {
        // ---- Step 2a: Input transform ----
        for (int tr = 0; tr < n_tile_rows; tr++) {
            for (int tc = 0; tc < n_tile_cols; tc++) {
                int tile_idx = tr * n_tile_cols + tc;

                // Extract input tile [TS][TS][IC] from src (with zero padding)
                std::fill(d_tile.begin(), d_tile.end(), 0.0f);
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ih = tr * OT - 1 + ti;
                        int iw = tc * OT - 1 + tj;
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            for (int ic = 0; ic < IC; ic++) {
                                d_tile[(ti * TS + tj) * IC + ic] =
                                    src[((n * IC + ic) * IH + ih) * IW + iw];
                            }
                        }
                    }
                }

                // Transform: U_tile = B^T * d * B
                dispatch_input_transform(d_tile.data(), U_tile.data(), IC,
                                         is_f44, isa);

                // Scatter U_tile into U[ts_idx][tile_idx][ic]
                // (both sides contiguous in ic → vectorize with NEON)
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ts_idx = ti * TS + tj;
                        float* dst_ptr = U.data() + (ts_idx * n_tiles + tile_idx) * IC;
                        const float* src_ptr = U_tile.data() + (ti * TS + tj) * IC;
                        int ic = 0;
                        for (; ic + 4 <= IC; ic += 4)
                            vst1q_f32(dst_ptr + ic, vld1q_f32(src_ptr + ic));
                        for (; ic < IC; ic++)
                            dst_ptr[ic] = src_ptr[ic];
                    }
                }
            }
        }

        // ---- Step 2b: GEMM ----
        for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
            const float* U_slice = U.data() + ts_idx * n_tiles * IC;
            const float* V_slice = V.data() + ts_idx * OC * IC;
            float* M_slice = M_buf.data() + ts_idx * n_tiles * OC;
            winograd_gemm(U_slice, V_slice, M_slice, n_tiles, OC, IC);
        }

        // ---- Step 2c: Output transform (includes bias + ReLU) ----
        for (int tr = 0; tr < n_tile_rows; tr++) {
            for (int tc = 0; tc < n_tile_cols; tc++) {
                int tile_idx = tr * n_tile_cols + tc;

                // Gather M_tile[TS][TS][OC] from M[ts_idx][tile_idx][oc]
                // (both sides contiguous in oc → vectorize with NEON)
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ts_idx = ti * TS + tj;
                        const float* src_ptr = M_buf.data() + (ts_idx * n_tiles + tile_idx) * OC;
                        float* dst_ptr = M_tile.data() + (ti * TS + tj) * OC;
                        int oc = 0;
                        for (; oc + 4 <= OC; oc += 4)
                            vst1q_f32(dst_ptr + oc, vld1q_f32(src_ptr + oc));
                        for (; oc < OC; oc++)
                            dst_ptr[oc] = src_ptr[oc];
                    }
                }

                // Transform: f_tile = A^T * M * A (bias + ReLU applied inside)
                dispatch_output_transform(M_tile.data(), f_tile.data(), OC,
                                          bias, act_min, act_max, is_f44, isa);

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
