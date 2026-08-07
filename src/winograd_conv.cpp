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
// Prevent OpenBLAS from spawning its own threads — conflicts with OpenMP
extern "C" void openblas_set_num_threads(int);
#endif

#ifdef USE_ARM_GEMM
#include <arm_gemm.hpp>
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
#if defined(USE_ARM_GEMM)
    // arm_gemm: use GemmHybrid with SVE kernel
    // Requires: -DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/arm_gemm
    // arm_gemm computes M = U * V^T (V is transposed)
    arm_gemm::GemmHybrid<arm_gemm::gemm_wide, float, float> gemm(
        n_tiles, OC, IC, false /* transpose A */, true /* transpose B */);
    gemm.matmul(M, U, V, 1.0f, 0.0f);
#elif defined(USE_OPENBLAS)
    // OpenBLAS: cblas_sgemm
    // Requires: -DUSE_OPENBLAS=ON
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_tiles, OC, IC,
                1.0f, U, IC, V, IC,
                0.0f, M, OC);
#else
    // Naive fallback: triple loop (no external dependency)
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
    const float* src,   // NCHW: [N][IC][IH][IW]  or  NHWC: [N][IH][IW][IC]
    const float* wei,   // [OC][IC][3][3]
    const float* bias,  // [OC] or nullptr
    float* dst,         // NCHW: [N][OC][OH][OW]  or  NHWC: [N][OH][OW][OC]
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    const WinogradConfig& config,
    float act_min, float act_max,
    Layout layout
) {
#ifdef USE_OPENBLAS
    // OpenBLAS must use 1 thread — our OpenMP parallelism is on tiles, not GEMM
    openblas_set_num_threads(1);
#endif

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

    // ---- Step 1: Weight transform (parallelized over OC) ----
    // V[TS*TS][OC*IC], layout: V[m][oc*IC+ic]
    int V_size = TS * TS * OC * IC;
    std::vector<float> V(V_size, 0.0f);

    #pragma omp parallel
    {
        // Per-thread buffers, reused across OC iterations
        std::vector<float> g(9 * IC);
        std::vector<float> V_oc(TS * TS * IC);

        #pragma omp for schedule(dynamic, 4)
        for (int oc = 0; oc < OC; oc++) {
            // Rearrange wei[oc][IC][3][3] → g[3][3][IC] (channels-contiguous)
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        g[(kh * 3 + kw) * IC + ic] =
                            wei[((oc * IC + ic) * 3 + kh) * 3 + kw];

            // Transform via ISA dispatch (NEON/SVE/SME)
            dispatch_weight_transform(g.data(), V_oc.data(), IC, is_f44, isa);

            // Store into V[m][oc*IC+ic]
            for (int m = 0; m < TS * TS; m++)
                for (int ic = 0; ic < IC; ic++)
                    V[m * OC * IC + oc * IC + ic] = V_oc[m * IC + ic];
        }
    }

    // ---- Pre-allocate workspace (reused across batches) ----
    int U_size = NM * n_tiles * IC;
    int M_size = NM * n_tiles * OC;
    std::vector<float> U(U_size, 0.0f);
    std::vector<float> M_buf(M_size, 0.0f);

    // ---- Step 2: For each batch ----
    for (int n = 0; n < N; n++) {
        // Single OpenMP region: input → GEMM → output (reduces fork/join overhead)
        #pragma omp parallel
        {
            // Allocate ALL per-thread buffers once, reuse across all 3 phases
            std::vector<float> d_tile(TS * TS * IC, 0.0f);
            std::vector<float> U_tile(TS * TS * IC, 0.0f);
            std::vector<float> M_tile(TS * TS * OC, 0.0f);
            std::vector<float> f_tile(OT * OT * OC, 0.0f);

            // ---- Phase 1: Input transform (parallelized over tiles) ----
            #pragma omp for collapse(2) schedule(dynamic, 2)
            for (int tr = 0; tr < n_tile_rows; tr++) {
                for (int tc = 0; tc < n_tile_cols; tc++) {
                    int tile_idx = tr * n_tile_cols + tc;

                    // Optimization B: only zero padding for edge tiles
                    // Interior tiles have all positions valid → no zero needed
                    bool is_edge = (tr == 0 || tr == n_tile_rows - 1 ||
                                    tc == 0 || tc == n_tile_cols - 1);
                    if (is_edge) {
                        memset(d_tile.data(), 0, TS * TS * IC * sizeof(float));
                    }

                    // Pre-compute valid row/col range for this tile
                    int ti_start = (tr == 0) ? 1 : 0;
                    int ti_end   = (tr == n_tile_rows - 1) ? TS - 1 : TS;
                    int tj_start = (tc == 0) ? 1 : 0;
                    int tj_end   = (tc == n_tile_cols - 1) ? TS - 1 : TS;

                    // Extract input tile [TS][TS][IC] — only valid rows/cols
                    for (int ti = ti_start; ti < ti_end; ti++) {
                        int ih = tr * OT - 1 + ti;
                        for (int tj = tj_start; tj < tj_end; tj++) {
                            int iw = tc * OT - 1 + tj;
                            if (layout == Layout::NHWC) {
                                const float* sp = src + ((n * IH + ih) * IW + iw) * IC;
                                float* dp = d_tile.data() + (ti * TS + tj) * IC;
                                int ic = 0;
                                for (; ic + 4 <= IC; ic += 4)
                                    vst1q_f32(dp + ic, vld1q_f32(sp + ic));
                                for (; ic < IC; ic++)
                                    dp[ic] = sp[ic];
                            } else {
                                for (int ic = 0; ic < IC; ic++)
                                    d_tile[(ti * TS + tj) * IC + ic] =
                                        src[((n * IC + ic) * IH + ih) * IW + iw];
                            }
                        }
                    }

                    // Transform: U_tile = B^T * d * B
                    dispatch_input_transform(d_tile.data(), U_tile.data(), IC,
                                             is_f44, isa);

                    // Scatter U_tile into U[ts_idx][tile_idx][ic]
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
            // implicit barrier after #pragma omp for — ensures U is complete before GEMM

            // ---- Phase 2: GEMM (parallelized over Winograd domain elements) ----
            #pragma omp for schedule(dynamic)
            for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
                const float* U_slice = U.data() + ts_idx * n_tiles * IC;
                const float* V_slice = V.data() + ts_idx * OC * IC;
                float* M_slice = M_buf.data() + ts_idx * n_tiles * OC;
                winograd_gemm(U_slice, V_slice, M_slice, n_tiles, OC, IC);
            }
            // implicit barrier — ensures M_buf is complete before output transform

            // ---- Phase 3: Output transform (parallelized over tiles) ----
            // nowait: skip implicit barrier — this is the last phase, no one waits
            #pragma omp for collapse(2) schedule(dynamic, 2) nowait
            for (int tr = 0; tr < n_tile_rows; tr++) {
                for (int tc = 0; tc < n_tile_cols; tc++) {
                    int tile_idx = tr * n_tile_cols + tc;

                    // Gather M_tile[TS][TS][OC] from M[ts_idx][tile_idx][oc]
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
                                if (layout == Layout::NHWC) {
                                    float* dp = dst + ((n * OH + oh) * OW + ow) * OC;
                                    const float* sp = f_tile.data() + (oi * OT + oj) * OC;
                                    int oc = 0;
                                    for (; oc + 4 <= OC; oc += 4)
                                        vst1q_f32(dp + oc, vld1q_f32(sp + oc));
                                    for (; oc < OC; oc++)
                                        dp[oc] = sp[oc];
                                } else {
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
        } // end single parallel region (1 fork/join per batch)
    }
}

} // namespace winograd_conv
