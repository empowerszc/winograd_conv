// winograd_transforms.hpp - Weight/Input/Output transform implementations
//
// This file implements the three Winograd transforms using NEON intrinsics.
// It supports both F(2,2,3,3) and F(4,4,3,3).
//
// Design based on ACL's implementation:
// - Weight transform: V = G * g * G^T  (along OC dimension, 4 floats per NEON op)
// - Input transform:  U = B^T * d * B  (along channel dimension, 4 floats per NEON op)
// - Output transform: f = A^T * M * A  (along OC dimension, with bias + ReLU)
//
// Part of the winograd_conv project.

#pragma once

#include <arm_neon.h>
#include <cstddef>
#include <cstring>
#include <vector>
#include "winograd_matrices.hpp"
#include "winograd_config.hpp"

namespace winograd_conv {

// ============================================================================
// Generic 1D matrix-vector transform using NEON
// ============================================================================
// Given a transform matrix M[out_size][in_size] and input data d[in_size][channels],
// compute output[out_size][channels] = M * d
//
// This is the core 1D transform used for both row and column transforms.
// It processes 4 channels at a time using float32x4_t (NEON Q registers).

template <int OUT_SIZE, int IN_SIZE>
void transform_1d_neon(
    const float matrix[OUT_SIZE][IN_SIZE],  // transform matrix
    const float* input,                       // input[in_size][channels], row-major
    float* output,                            // output[out_size][channels]
    int channels,
    int in_stride,    // stride between input rows (= channels for row transform)
    int out_stride     // stride between output rows
) {
    for (int o = 0; o < OUT_SIZE; o++) {
        // Initialize output[o][:] = matrix[o][0] * input[0][:]
        float* out_row = output + o * out_stride;
        const float* in_row = input;

        if (matrix[o][0] != 0.0f) {
            int c = channels;
            // Process 4 channels at a time
            for (; c >= 4; c -= 4) {
                float32x4_t acc = vld1q_f32(in_row);
                acc = vmulq_n_f32(acc, matrix[o][0]);
                vst1q_f32(out_row, acc);
                in_row += 4;
                out_row += 4;
            }
            // Tail: 2 channels
            if (c >= 2) {
                float32x2_t acc = vld1_f32(in_row);
                acc = vmul_n_f32(acc, matrix[o][0]);
                vst1_f32(out_row, acc);
                in_row += 2;
                out_row += 2;
                c -= 2;
            }
            // Tail: 1 channel
            for (; c > 0; c--) {
                *out_row = *in_row * matrix[o][0];
                in_row++;
                out_row++;
            }
        } else {
            // Coefficient is zero, just zero the output
            std::memset(output + o * out_stride, 0, channels * sizeof(float));
        }

        // Add remaining terms: output[o][:] += matrix[o][k] * input[k][:]
        for (int k = 1; k < IN_SIZE; k++) {
            if (matrix[o][k] == 0.0f) continue;

            float coef = matrix[o][k];
            out_row = output + o * out_stride;
            in_row = input + k * in_stride;

            int c = channels;
            // 4-channel FMA
            for (; c >= 4; c -= 4) {
                float32x4_t acc = vld1q_f32(out_row);
                float32x4_t inp = vld1q_f32(in_row);
                acc = vmlaq_n_f32(acc, inp, coef);  // acc += inp * coef
                vst1q_f32(out_row, acc);
                in_row += 4;
                out_row += 4;
            }
            // 2-channel
            if (c >= 2) {
                float32x2_t acc = vld1_f32(out_row);
                float32x2_t inp = vld1_f32(in_row);
                acc = vmla_n_f32(acc, inp, coef);
                vst1_f32(out_row, acc);
                in_row += 2;
                out_row += 2;
                c -= 2;
            }
            // 1-channel
            for (; c > 0; c--) {
                *out_row += *in_row * coef;
                in_row++;
                out_row++;
            }
        }
    }
}

// ============================================================================
// 2D transform: output = M * input * M^T
// ============================================================================
// Step 1: row transform: tmp = M * input (transform each column)
// Step 2: col transform: output = tmp * M^T (transform each row)
// M is OUT_SIZE x IN_SIZE, input is IN_SIZE x IN_SIZE, output is OUT_SIZE x OUT_SIZE

template <int OUT_SIZE, int IN_SIZE>
void transform_2d_neon(
    const float matrix[OUT_SIZE][IN_SIZE],
    const float* input,   // [IN_SIZE][IN_SIZE][channels]
    float* output,         // [OUT_SIZE][OUT_SIZE][channels]
    int channels,
    int channel_stride     // stride between elements (= channels for packed layout)
) {
    // Temporary buffer for intermediate result
    // tmp[OUT_SIZE][IN_SIZE][channels]
    std::vector<float> tmp(OUT_SIZE * IN_SIZE * channels);

    // Step 1: Row transform - for each column j, compute tmp[:][j] = M * input[:][j]
    for (int j = 0; j < IN_SIZE; j++) {
        const float* in_col = input + j * channel_stride;  // input[0][j][:]
        float* tmp_col = tmp.data() + j * channel_stride;   // tmp[0][j][:]
        transform_1d_neon<OUT_SIZE, IN_SIZE>(
            matrix, in_col, tmp_col, channels,
            IN_SIZE * channel_stride,   // in_stride: skip a full row
            IN_SIZE * channel_stride     // out_stride: same
        );
    }

    // Step 2: Col transform - for each row i, compute output[i][:] = tmp[i] * M^T
    for (int i = 0; i < OUT_SIZE; i++) {
        const float* tmp_row = tmp.data() + i * IN_SIZE * channel_stride;
        float* out_row = output + i * OUT_SIZE * channel_stride;

        transform_1d_neon<OUT_SIZE, IN_SIZE>(
            matrix, tmp_row, out_row, channels,
            channel_stride,      // in_stride: skip one element (= channels)
            channel_stride        // out_stride
        );
    }
}

// ============================================================================
// Weight transform: V = G * g * G^T
// ============================================================================
// Input:  g[3][3][channels] (kernel weights, channels = OC, processed in blocks of 4)
// Output: V[tile_size][tile_size][channels] (transformed weights)
// For F(2,2,3,3): G is 4x3, output is 4x4
// For F(4,4,3,3): G is 6x3, output is 6x6

template <int TILE_SIZE>
void weight_transform_neon(
    const float* g,       // [3][3][channels]
    float* V,             // [TILE_SIZE][TILE_SIZE][channels]
    int channels,
    const float G_matrix[TILE_SIZE][3],
    float normalization
) {
    // Step 1: Ww = G * g  (row transform: 3 -> TILE_SIZE, for each of 3 columns)
    alignas(64) float Ww[TILE_SIZE * 3 * 4];  // max 6*3*4 = 72
    // Process 4 channels at a time
    for (int c_block = 0; c_block < channels; c_block += 4) {
        int ch = std::min(4, channels - c_block);

        for (int j = 0; j < 3; j++) {
            const float* g_col = g + j * channels + c_block;
            float* Ww_col = Ww + j * 4;

            for (int i = 0; i < TILE_SIZE; i++) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int k = 0; k < 3; k++) {
                    float32x4_t g_val = vld1q_f32(g + k * 3 * channels + j * channels + c_block);
                    if (ch < 4) {
                        // Handle tail: load only valid elements
                        for (int c = 0; c < ch; c++) {
                            Ww[i * 3 * 4 + j * 4 + c] = 0.0f;
                            for (int k2 = 0; k2 < 3; k2++) {
                                Ww[i * 3 * 4 + j * 4 + c] +=
                                    G_matrix[i][k2] * g[k2 * 3 * channels + j * channels + c_block + c];
                            }
                        }
                    } else {
                        // Full 4-channel NEON
                        float32x4_t result = vdupq_n_f32(0.0f);
                        for (int k2 = 0; k2 < 3; k2++) {
                            float32x4_t gk = vld1q_f32(g + k2 * 3 * channels + j * channels + c_block);
                            result = vmlaq_n_f32(result, gk, G_matrix[i][k2]);
                        }
                        vst1q_f32(Ww + i * 3 * 4 + j * 4, result);
                    }
                }
            }
        }

        // Step 2: V = Ww * G^T / normalization (col transform)
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                float32x4_t result = vdupq_n_f32(0.0f);
                for (int k = 0; k < 3; k++) {
                    if (ch >= 4) {
                        float32x4_t wwk = vld1q_f32(Ww + i * 3 * 4 + k * 4);
                        result = vmlaq_n_f32(result, wwk, G_matrix[j][k]);
                    }
                }
                if (ch >= 4) {
                    // Apply normalization
                    if (normalization != 1.0f) {
                        result = vmulq_n_f32(result, normalization);
                    }
                    vst1q_f32(V + i * TILE_SIZE * channels + j * channels + c_block, result);
                } else {
                    for (int c = 0; c < ch; c++) {
                        float val = 0.0f;
                        for (int k = 0; k < 3; k++) {
                            val += Ww[i * 3 * 4 + k * 4 + c] * G_matrix[j][k];
                        }
                        if (normalization != 1.0f) val *= normalization;
                        V[i * TILE_SIZE * channels + j * channels + c_block + c] = val;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Input transform: U = B^T * d * B
// ============================================================================
// Input:  d[tile_size][tile_size][channels] (input tile from feature map, with padding)
// Output: U[tile_size][tile_size][channels] (Winograd domain)
//
// Uses transform_2d_neon with the appropriate B^T matrix.

template <int TILE_SIZE>
void input_transform_neon(
    const float* d,       // [TILE_SIZE][TILE_SIZE][channels]
    float* U,             // [TILE_SIZE][TILE_SIZE][channels]
    int channels,
    const float Bt_matrix[TILE_SIZE][TILE_SIZE]
) {
    transform_2d_neon<TILE_SIZE, TILE_SIZE>(
        Bt_matrix, d, U, channels, channels
    );
}

// ============================================================================
// Output transform: f = A^T * M * A  (+ bias + ReLU)
// ============================================================================
// Input:  M[tile_size][tile_size][channels] (Winograd domain result from GEMM)
// Output: f[output_tile][output_tile][channels] (output pixels)
//
// Uses transform_2d_neon with the appropriate A matrix.
// Then adds bias and applies ReLU (clamp to [min, max]).

template <int TILE_SIZE, int OUTPUT_TILE>
void output_transform_neon(
    const float* M,       // [TILE_SIZE][TILE_SIZE][channels]
    float* f,             // [OUTPUT_TILE][OUTPUT_TILE][channels]
    int channels,
    const float A_matrix[OUTPUT_TILE][TILE_SIZE],
    const float* bias,    // [channels] or nullptr
    float act_min,        // ReLU min (0 for ReLU, -inf for no activation)
    float act_max         // ReLU max (+inf for ReLU, alpha for BoundedReLU)
) {
    // Step 1+2: 2D transform using A
    transform_2d_neon<OUTPUT_TILE, TILE_SIZE>(
        A_matrix, M, f, channels, channels
    );

    // Step 3: Add bias + clamp per-channel (matches ACL output transform)
    float32x4_t vmin = vdupq_n_f32(act_min);
    float32x4_t vmax = vdupq_n_f32(act_max);

    for (int oi = 0; oi < OUTPUT_TILE; oi++) {
        for (int oj = 0; oj < OUTPUT_TILE; oj++) {
            float* fptr = f + (oi * OUTPUT_TILE + oj) * channels;
            int c = 0;
            // 4-channel NEON
            for (; c + 4 <= channels; c += 4) {
                float32x4_t v = vld1q_f32(fptr + c);
                if (bias) {
                    float32x4_t b = vld1q_f32(bias + c);
                    v = vaddq_f32(v, b);
                }
                v = vminq_f32(v, vmax);
                v = vmaxq_f32(v, vmin);
                vst1q_f32(fptr + c, v);
            }
            // Tail: scalar
            for (; c < channels; c++) {
                float v = fptr[c];
                if (bias) v += bias[c];
                if (v > act_max) v = act_max;
                if (v < act_min) v = act_min;
                fptr[c] = v;
            }
        }
    }
}

// ============================================================================
// Convenience wrappers for F(2,2,3,3) and F(4,4,3,3)
// ============================================================================

// F(2,2,3,3) weight transform
inline void weight_transform_f22_neon(
    const float* g, float* V, int channels
) {
    weight_transform_neon<4>(g, V, channels, F22_G::val, F22_G::norm);
}

// F(2,2,3,3) input transform
inline void input_transform_f22_neon(
    const float* d, float* U, int channels
) {
    input_transform_neon<4>(d, U, channels, F22_Bt::val);
}

// F(2,2,3,3) output transform
inline void output_transform_f22_neon(
    const float* M, float* f, int channels,
    const float* bias = nullptr,
    float act_min = -1e30f, float act_max = 1e30f
) {
    output_transform_neon<4, 2>(M, f, channels, F22_A::val, bias, act_min, act_max);
}

// F(4,4,3,3) weight transform
inline void weight_transform_f44_neon(
    const float* g, float* V, int channels
) {
    weight_transform_neon<6>(g, V, channels, F44_G::val, F44_G::norm);
}

// F(4,4,3,3) input transform
inline void input_transform_f44_neon(
    const float* d, float* U, int channels
) {
    input_transform_neon<6>(d, U, channels, F44_Bt::val);
}

// F(4,4,3,3) output transform
inline void output_transform_f44_neon(
    const float* M, float* f, int channels,
    const float* bias = nullptr,
    float act_min = -1e30f, float act_max = 1e30f
) {
    output_transform_neon<6, 4>(M, f, channels, F44_A::val, bias, act_min, act_max);
}

} // namespace winograd_conv
