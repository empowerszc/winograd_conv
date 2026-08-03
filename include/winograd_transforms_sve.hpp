// winograd_transforms_sve.hpp - SVE intrinsics 变换实现
//
// 用 ARM SVE (Scalable Vector Extension) intrinsics 实现 Winograd 三步变换。
// SVE 的核心优势：
//   1. 向量长度可伸缩（SVE-512 时 16 个 float/指令，vs NEON 的 4 个）
//   2. 谓词（predicate）自动处理 tail，无需降级到窄路径
//   3. whilelt 指令设置谓词，循环无分支
//
// 对应 ACL 的 sve_fp32_6x6.cpp 和 sme_fp32_mla_6x6.cpp（SME 版 = SVE + SMSTART）
//
// Part of the winograd_conv project.

#pragma once

#include <arm_sve.h>
#include <cstddef>
#include <cstring>
#include "winograd_matrices.hpp"
#include "winograd_config.hpp"

// Only compile if SVE is available
#if defined(__ARM_FEATURE_SVE)

namespace winograd_conv {

// ============================================================================
// SVE 向量长度辅助
// ============================================================================

// 获取当前 SVE 向量长度（float 元素数）
// SVE-128: 4, SVE-256: 8, SVE-512: 16
static inline int sve_count() {
    return svcntw();  // count of 32-bit words per vector
}

// SVE 谓词：设置 p0[i] = (start + i < total)
static inline svbool_t sve_whilelt(int start, int total) {
    return svwhilelt_b32_s32(start, total);
}

// ============================================================================
// SVE 1D 变换：output[OUT_SIZE][channels] = M * input[IN_SIZE][channels]
// ============================================================================
// 用 SVE 谓词自动处理 channels 不是 VL 整数倍的情况。
// 每条 fmla 处理 VL 个通道（SVE-512 时 16 个），无分支 tail 处理。

template <int OUT_SIZE, int IN_SIZE>
void transform_1d_sve(
    const float matrix[OUT_SIZE][IN_SIZE],
    const float* input,    // [IN_SIZE][channels], row-major, stride = in_stride
    float* output,         // [OUT_SIZE][channels], stride = out_stride
    int channels,
    int in_stride,
    int out_stride
) {
    for (int o = 0; o < OUT_SIZE; o++) {
        float* out_row = output + o * out_stride;

        // First non-zero coefficient
        bool first = true;
        for (int k = 0; k < IN_SIZE; k++) {
            if (matrix[o][k] == 0.0f) continue;

            float coef = matrix[o][k];
            const float* in_row = input + k * in_stride;
            int c = 0;

            if (first) {
                // Initialize: output = coef * input[k]
                svbool_t pg = sve_whilelt(c, channels);
                do {
                    svfloat32_t inp = svld1_f32(pg, in_row + c);
                    svfloat32_t result = svmul_n_f32_x(pg, inp, coef);
                    svst1_f32(pg, out_row + c, result);
                    c += sve_count();
                    pg = sve_whilelt(c, channels);
                } while (svptest_first(pg));
                first = false;
            } else {
                // Accumulate: output += coef * input[k]
                svbool_t pg = sve_whilelt(c, channels);
                do {
                    svfloat32_t inp = svld1_f32(pg, in_row + c);
                    svfloat32_t acc = svld1_f32(pg, out_row + c);
                    acc = svmla_n_f32_x(pg, acc, inp, coef);
                    svst1_f32(pg, out_row + c, acc);
                    c += sve_count();
                    pg = sve_whilelt(c, channels);
                } while (svptest_first(pg));
            }
        }

        // If all coefficients were zero, zero the output
        if (first) {
            int c = 0;
            svbool_t pg = sve_whilelt(c, channels);
            do {
                svst1_f32(pg, out_row + c, svdup_n_f32(0.0f));
                c += sve_count();
                pg = sve_whilelt(c, channels);
            } while (svptest_first(pg));
        }
    }
}

// ============================================================================
// SVE 2D 变换：output = M * input * M^T
// ============================================================================

template <int OUT_SIZE, int IN_SIZE, int CHANNELS>
void transform_2d_sve(
    const float matrix[OUT_SIZE][IN_SIZE],
    const float* input,   // [IN_SIZE][IN_SIZE][CHANNELS]
    float* output,         // [OUT_SIZE][OUT_SIZE][CHANNELS]
    int channel_stride
) {
    alignas(64) float tmp[OUT_SIZE * IN_SIZE * CHANNELS];

    // Step 1: Row transform (apply M to each column)
    for (int j = 0; j < IN_SIZE; j++) {
        transform_1d_sve<OUT_SIZE, IN_SIZE>(
            matrix,
            input + j * channel_stride,
            tmp + j * channel_stride,
            CHANNELS,
            IN_SIZE * channel_stride,
            IN_SIZE * channel_stride
        );
    }

    // Step 2: Col transform (apply M to each row, using same matrix)
    for (int i = 0; i < OUT_SIZE; i++) {
        transform_1d_sve<OUT_SIZE, IN_SIZE>(
            matrix,
            tmp + i * IN_SIZE * channel_stride,
            output + i * OUT_SIZE * channel_stride,
            CHANNELS,
            channel_stride,
            channel_stride
        );
    }
}

// ============================================================================
// SVE Weight transform: V = G * g * G^T / norm
// ============================================================================

template <int TILE_SIZE>
void weight_transform_sve(
    const float* g,       // [3][3][channels]
    float* V,             // [TILE_SIZE][TILE_SIZE][channels]
    int channels,
    const float G_matrix[TILE_SIZE][3],
    float normalization
) {
    // Step 1: Ww = G * g (row transform, for each column j of g)
    alignas(64) float Ww[6 * 3 * 16];  // max TILE_SIZE=6, kw=3, padded to 16
    int c = 0;
    svbool_t pg = sve_whilelt(c, channels);
    do {
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < TILE_SIZE; i++) {
                svfloat32_t result = svdup_n_f32(0.0f);
                for (int k = 0; k < 3; k++) {
                    svfloat32_t gk = svld1_f32(pg,
                        g + k * 3 * channels + j * channels + c);
                    result = svmla_n_f32_x(pg, result, gk, G_matrix[i][k]);
                }
                svst1_f32(pg, Ww + i * 3 * sve_count() + j * sve_count(), result);
            }
        }

        // Step 2: V = Ww * G^T / norm (col transform)
        for (int i = 0; i < TILE_SIZE; i++) {
            for (int j = 0; j < TILE_SIZE; j++) {
                svfloat32_t result = svdup_n_f32(0.0f);
                for (int k = 0; k < 3; k++) {
                    svfloat32_t wwk = svld1_f32(pg,
                        Ww + i * 3 * sve_count() + k * sve_count());
                    result = svmla_n_f32_x(pg, result, wwk, G_matrix[j][k]);
                }
                if (normalization != 1.0f) {
                    result = svmul_n_f32_x(pg, result, normalization);
                }
                svst1_f32(pg, V + i * TILE_SIZE * channels + j * channels + c, result);
            }
        }

        c += sve_count();
        pg = sve_whilelt(c, channels);
    } while (svptest_first(pg));
}

// ============================================================================
// SVE Input transform: U = B^T * d * B
// ============================================================================

template <int TILE_SIZE>
void input_transform_sve(
    const float* d,       // [TILE_SIZE][TILE_SIZE][channels]
    float* U,             // [TILE_SIZE][TILE_SIZE][channels]
    int channels,
    const float Bt_matrix[TILE_SIZE][TILE_SIZE]
) {
    transform_2d_sve<TILE_SIZE, TILE_SIZE>(
        Bt_matrix, d, U, channels, channels
    );
}

// ============================================================================
// SVE Output transform: f = A^T * M * A (+ bias + ReLU)
// ============================================================================

template <int TILE_SIZE, int OUTPUT_TILE>
void output_transform_sve(
    const float* M,       // [TILE_SIZE][TILE_SIZE][channels]
    float* f,             // [OUTPUT_TILE][OUTPUT_TILE][channels]
    int channels,
    const float A_matrix[OUTPUT_TILE][TILE_SIZE],
    const float* bias,    // [channels] or nullptr
    float act_min,
    float act_max
) {
    // 2D transform
    transform_2d_sve<OUTPUT_TILE, TILE_SIZE>(
        A_matrix, M, f, channels, channels
    );

    // Add bias + clamp per-channel (matches ACL output transform)
    svfloat32_t vmin = svdup_n_f32(act_min);
    svfloat32_t vmax = svdup_n_f32(act_max);

    for (int oi = 0; oi < OUTPUT_TILE; oi++) {
        for (int oj = 0; oj < OUTPUT_TILE; oj++) {
            float* fptr = f + (oi * OUTPUT_TILE + oj) * channels;
            int c = 0;
            svbool_t pg = sve_whilelt(c, channels);
            do {
                svfloat32_t v = svld1_f32(pg, fptr + c);
                if (bias) {
                    svfloat32_t b = svld1_f32(pg, bias + c);
                    v = svadd_f32_x(pg, v, b);
                }
                v = svmin_f32_x(pg, v, vmax);
                v = svmax_f32_x(pg, v, vmin);
                svst1_f32(pg, fptr + c, v);
                c += sve_count();
                pg = sve_whilelt(c, channels);
            } while (svptest_first(pg));
        }
    }
}

// ============================================================================
// Convenience wrappers for F(2,2,3,3) and F(4,4,3,3)
// ============================================================================

inline void weight_transform_f22_sve(const float* g, float* V, int channels) {
    weight_transform_sve<4>(g, V, channels, F22_G::val, F22_G::norm);
}
inline void input_transform_f22_sve(const float* d, float* U, int channels) {
    input_transform_sve<4>(d, U, channels, F22_Bt::val);
}
inline void output_transform_f22_sve(
    const float* M, float* f, int channels,
    const float* bias = nullptr, float act_min = -1e30f, float act_max = 1e30f
) {
    output_transform_sve<4, 2>(M, f, channels, F22_A::val, bias, act_min, act_max);
}

inline void weight_transform_f44_sve(const float* g, float* V, int channels) {
    weight_transform_sve<6>(g, V, channels, F44_G::val, F44_G::norm);
}
inline void input_transform_f44_sve(const float* d, float* U, int channels) {
    input_transform_sve<6>(d, U, channels, F44_Bt::val);
}
inline void output_transform_f44_sve(
    const float* M, float* f, int channels,
    const float* bias = nullptr, float act_min = -1e30f, float act_max = 1e30f
) {
    output_transform_sve<6, 4>(M, f, channels, F44_A::val, bias, act_min, act_max);
}

} // namespace winograd_conv

#endif // __ARM_FEATURE_SVE
