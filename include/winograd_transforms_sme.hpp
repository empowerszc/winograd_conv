// winograd_transforms_sme.hpp - SME 变换实现（真实 FMOPA 指令）
//
// 用 ARM SME (Scalable Matrix Extension) 实现 Winograd 输出变换。
// SME 的核心优势：
//   1. FMOPA 指令：一条做 VL×VL 外积累加 = 256 次 FMA（SVE-512 时）
//   2. ZA tile 寄存器：二维累加器，无需中间寄存器
//   3. 4 个 ZA tile 并行 → 4 行输出同时累加
//
// 输出变换：直接移植 ACL 53.1.0 的 sme_fp32_mopa_4x4_3x3.cpp 内联汇编
// 输入变换：SVE 指令 + SMSTART/SMSTOP（流式模式 SVE，与 ACL 一致）
// 权重变换：NEON（prepare 阶段，非热路径）
//
// Part of the winograd_conv project.
// Based on ACL's Winograd implementation (MIT license).

#pragma once

#include <arm_sve.h>
#include <arm_neon_sve.h>
#include <cstddef>
#include <cstring>
#include "winograd_matrices.hpp"
#include "winograd_config.hpp"

#if defined(__ARM_FEATURE_SME)

namespace winograd_conv {

// ============================================================================
// SME 指令编码宏
// ============================================================================

#define SME_SMSTART_ZA() \
    __asm__ __volatile__(".inst 0xd503477f" ::: "memory")

#define SME_SMSTOP() \
    __asm__ __volatile__(".inst 0xd503467f" ::: "memory")

// ============================================================================
// SME 输入变换：U = B^T * d * B
// ============================================================================
// SME 版输入变换 = SVE 变换 + SMSTART/SMSTOP 包裹
// 与 ACL 的 sme_fp32_mla_6x6.cpp 一致：流式模式下执行 SVE 指令

#include "winograd_transforms_sve.hpp"

template <int TILE_SIZE>
void input_transform_sme(
    const float* d,
    float* U,
    int channels,
    const float Bt_matrix[TILE_SIZE][TILE_SIZE]
) {
    SME_SMSTART_ZA();
    transform_2d_sve<TILE_SIZE, TILE_SIZE>(
        Bt_matrix, d, U, channels, channels
    );
    SME_SMSTOP();
}

// ============================================================================
// SME 输出变换（真实 FMOPA + MOVA 内联汇编）
// ============================================================================
// 直接移植自 ACL 53.1.0 sme_fp32_mopa_4x4_3x3.cpp
//
// 核心思路（vec trick）：
//   f = A^T * M * A  →  vec(f) = (A^T ⊗ A^T) * vec(M)
//   (A^T ⊗ A^T) 是 16×36 矩阵，用 FMOPA 外积累加实现
//
// 参数映射（ACL → 本项目）：
//   inptr           = M          ([6][6][channels] Winograd 域数据)
//   matrix_stride   = channels   (相邻 tile 元素的步长)
//   bptr            = bias       ([channels] 或 nullptr)
//   output          = f          ([4][4][channels] 输出)
//   output_row_stride = 4 * channels
//   output_col_stride = channels
//   n_channels      = channels

static inline void sme_output_transform_f44(
    const unsigned int n_channels,
    const float* inptr,
    const size_t matrix_stride,
    const float* bptr,
    float* const output,
    const size_t output_row_stride,
    const size_t output_col_stride,
    const float output_min,
    const float output_max
) {
    // A^T 矩阵的 Kronecker 系数（与 ACL 源码一致）
    static const float outer_terms[32] = {
         1, 1,  1, 1,
         0, 1, -1, 2,
         0, 1,  1, 4,
         0, 1, -1, 8,
         1, 0,  0, 0,
        -2, 0,  0, 0,
         4, 0,  0, 0,
        -8, 1,  0, 0
    };
    static const float inner_terms[24] = {
        1,  0, 0,  0,
        1,  1, 1,  1,
        1, -1, 1, -1,
        1,  2, 4,  8,
        1, -2, 4, -8,
        0,  0, 0,  1
    };

    struct Params {
        const float* outer_terms;
        const float* inner_terms;
        float act_min;
        float act_max;
        Params(const float* ot, const float* it, float mn, float mx)
          : outer_terms(ot), inner_terms(it), act_min(mn), act_max(mx) {}
    };
    Params params(outer_terms, inner_terms, output_min, output_max);

    __asm__ __volatile__(
      "ldr x20, [%x[params], %[ofs_ot]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p5.b\n"
      "ld1rw { z12.s }, p5/Z, [%x[params], %[ofs_amax]]\n"
      "ld1rw { z10.s }, p5/Z, [%x[params], %[ofs_amin]]\n"
      "pfalse p8.b\n"
      "ldr x8, [%x[params], %[ofs_it]]\n"
      "ld1w { z6.s }, p5/Z, [x20]\n"
      "ld1w { z7.s }, p5/Z, [x20, #1, MUL VL]\n"
      "ld1rqw { z9.s }, p5/Z, [x8]\n"
      "ld1rqw { z8.s }, p5/Z, [x8, #16]\n"
      "ld1rqw { z15.s }, p5/Z, [x8, #32]\n"
      "fmul z11.s, z9.s, z6.s[0]\n"
      "fmul z5.s, z9.s, z6.s[1]\n"
      "ld1rqw { z4.s }, p5/Z, [x8, #48]\n"
      "ld1rqw { z3.s }, p5/Z, [x8, #64]\n"
      "ld1rqw { z2.s }, p5/Z, [x8, #80]\n"
      "cbz %x[bptr], 1f\n"
      "ptrue p8.s\n"
      "1:\n"
      ".inst 0xc00800ff  // zero {zad0-zad7}\n"
      "fmov z1.s, #1.0\n"
      "mov x25, #0x0\n"
      "cntw x24\n"
      "cntw x23, ALL, MUL #2\n"
      "cntw x22, ALL, MUL #3\n"
      "whilelt p4.s, x25, %x[nchan]\n"
      "whilelt p3.s, x24, %x[nchan]\n"
      "ld1w { z31.s }, p4/Z, [%x[inptr], x25, LSL #2]\n"
      "ld1w { z30.s }, p3/Z, [%x[inptr], x24, LSL #2]\n"
      "whilelt p2.s, x23, %x[nchan]\n"
      "whilelt p1.s, x22, %x[nchan]\n"
      "ld1w { z29.s }, p2/Z, [%x[inptr], x23, LSL #2]\n"
      "add x21, %x[inptr], %x[mstride], LSL #2\n"
      "and p0.b, p5/Z, p8.b, p4.b\n"
      "ld1w { z28.s }, p1/Z, [%x[inptr], x22, LSL #2]\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x25, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p3.b\n"
      ".inst 0x8080b420  // fmopa za0.s, p5/M, p5/M, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x24, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p2.b\n"
      ".inst 0x8080b421  // fmopa za1.s, p5/M, p5/M, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x23, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p1.b\n"
      ".inst 0x8080b422  // fmopa za2.s, p5/M, p5/M, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x22, LSL #2]\n"
      ".inst 0x8080b423  // fmopa za3.s, p5/M, p5/M, z1.s, z0.s\n"
      "2:\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "mov x15, #0\n"
      "mov x14, #0xc\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "whilelt p0.s, x25, %x[nchan]\n"
      "add x20, %x[output], %x[ocs], LSL #2\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "add x8, %x[output], %x[ors], LSL #2\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z9.s, z6.s[2]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z9.s, z6.s[3]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z9.s, z7.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z9.s, z7.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z8.s, z6.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z8.s, z6.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z8.s, z6.s[2]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z8.s, z6.s[3]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z8.s, z7.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z8.s, z7.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z15.s, z6.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z15.s, z6.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z15.s, z6.s[2]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z15.s, z6.s[3]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z15.s, z7.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z15.s, z7.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z4.s, z6.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z4.s, z6.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z4.s, z6.s[2]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z4.s, z6.s[3]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z4.s, z7.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z4.s, z7.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z3.s, z6.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z3.s, z6.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z3.s, z6.s[2]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z3.s, z6.s[3]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z3.s, z7.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z3.s, z7.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z2.s, z6.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z2.s, z6.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z2.s, z6.s[2]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z2.s, z6.s[3]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
      ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"
      ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"
      "fmul z11.s, z2.s, z7.s[0]\n"
      ".inst 0x809bb4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z27.s\n"
      ".inst 0x809ab4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z26.s\n"
      ".inst 0x8099b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z25.s\n"
      ".inst 0x8098b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z24.s\n"
      "fmul z5.s, z2.s, z7.s[1]\n"
      ".inst 0x8097b560  // fmopa za0.s, p5/M, p5/M, z11.s, z23.s\n"
      ".inst 0x8096b561  // fmopa za1.s, p5/M, p5/M, z11.s, z22.s\n"
      ".inst 0x8095b562  // fmopa za2.s, p5/M, p5/M, z11.s, z21.s\n"
      ".inst 0x8094b563  // fmopa za3.s, p5/M, p5/M, z11.s, z20.s\n"
      "fmul z11.s, z9.s, z6.s[0]\n"
      ".inst 0x8093b4a0  // fmopa za0.s, p5/M, p5/M, z5.s, z19.s\n"
      ".inst 0x8092b4a1  // fmopa za1.s, p5/M, p5/M, z5.s, z18.s\n"
      ".inst 0x8091b4a2  // fmopa za2.s, p5/M, p5/M, z5.s, z17.s\n"
      ".inst 0x8090b4a3  // fmopa za3.s, p5/M, p5/M, z5.s, z16.s\n"
      "fmul z5.s, z9.s, z6.s[1]\n"
      // ---- MOVA: 从 ZA tile 读出 4×4 输出 + fmin/fmax (ReLU) + st1w ----
      ".inst 0xc082741f  // mova z31.s, p5/M, za0h.s[x15]\n"
      ".inst 0xc082541c  // mova z28.s, p5/M, za0h.s[x14]\n"
      "fmin z31.s, p5/M, z31.s, z10.s\n"
      ".inst 0xc082743b  // mova z27.s, p5/M, za0h.s[x15, #1]\n"
      "fmin z28.s, p5/M, z28.s, z10.s\n"
      ".inst 0xc0825438  // mova z24.s, p5/M, za0h.s[x14, #1]\n"
      "fmin z27.s, p5/M, z27.s, z10.s\n"
      "mov x13, #0x4\n"
      "mov x12, #0x8\n"
      ".inst 0xc082341e  // mova z30.s, p5/M, za0h.s[x13]\n"
      "fmin z24.s, p5/M, z24.s, z10.s\n"
      ".inst 0xc082141d  // mova z29.s, p5/M, za0h.s[x12]\n"
      "fmax z31.s, p5/M, z31.s, z12.s\n"
      "fmin z30.s, p5/M, z30.s, z10.s\n"
      ".inst 0xc082343a  // mova z26.s, p5/M, za0h.s[x13, #1]\n"
      "fmin z29.s, p5/M, z29.s, z10.s\n"
      "fmax z28.s, p5/M, z28.s, z12.s\n"
      ".inst 0xc0821439  // mova z25.s, p5/M, za0h.s[x12, #1]\n"
      "fmax z27.s, p5/M, z27.s, z12.s\n"
      "fmin z26.s, p5/M, z26.s, z10.s\n"
      ".inst 0xc0827457  // mova z23.s, p5/M, za0h.s[x15, #2]\n"
      "fmin z25.s, p5/M, z25.s, z10.s\n"
      "fmax z24.s, p5/M, z24.s, z12.s\n"
      ".inst 0xc0823456  // mova z22.s, p5/M, za0h.s[x13, #2]\n"
      "fmax z30.s, p5/M, z30.s, z12.s\n"
      "fmin z23.s, p5/M, z23.s, z10.s\n"
      ".inst 0xc0821455  // mova z21.s, p5/M, za0h.s[x12, #2]\n"
      "fmax z29.s, p5/M, z29.s, z12.s\n"
      "fmin z22.s, p5/M, z22.s, z10.s\n"
      ".inst 0xc0825454  // mova z20.s, p5/M, za0h.s[x14, #2]\n"
      "fmax z26.s, p5/M, z26.s, z12.s\n"
      "fmin z21.s, p5/M, z21.s, z10.s\n"
      ".inst 0xc0827473  // mova z19.s, p5/M, za0h.s[x15, #3]\n"
      "fmax z25.s, p5/M, z25.s, z12.s\n"
      "fmin z20.s, p5/M, z20.s, z10.s\n"
      ".inst 0xc0823472  // mova z18.s, p5/M, za0h.s[x13, #3]\n"
      "fmax z23.s, p5/M, z23.s, z12.s\n"
      "fmin z19.s, p5/M, z19.s, z10.s\n"
      ".inst 0xc0821471  // mova z17.s, p5/M, za0h.s[x12, #3]\n"
      "fmax z22.s, p5/M, z22.s, z12.s\n"
      "fmin z18.s, p5/M, z18.s, z10.s\n"
      ".inst 0xc0825470  // mova z16.s, p5/M, za0h.s[x14, #3]\n"
      "fmax z21.s, p5/M, z21.s, z12.s\n"
      "fmin z17.s, p5/M, z17.s, z10.s\n"
      "fmax z20.s, p5/M, z20.s, z12.s\n"
      "fmin z16.s, p5/M, z16.s, z10.s\n"
      // ---- Store row 0 ----
      "st1w { z31.s }, p0, [%x[output], x25, LSL #2]\n"
      "fmax z19.s, p5/M, z19.s, z12.s\n"
      "st1w { z30.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "fmax z18.s, p5/M, z18.s, z12.s\n"
      "st1w { z29.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "fmax z17.s, p5/M, z17.s, z12.s\n"
      "st1w { z28.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "fmax z16.s, p5/M, z16.s, z12.s\n"
      // ---- Store row 1 ----
      "st1w { z27.s }, p0, [x8, x25, LSL #2]\n"
      "add x8, x8, %x[ors], LSL #2\n"
      "st1w { z26.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z25.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z24.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      // ---- Store row 2 ----
      "st1w { z23.s }, p0, [x8, x25, LSL #2]\n"
      "add x8, x8, %x[ors], LSL #2\n"
      "st1w { z22.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z21.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z20.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      // ---- Store row 3 ----
      "st1w { z19.s }, p0, [x8, x25, LSL #2]\n"
      "st1w { z18.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z17.s }, p0, [x20, x25, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z16.s }, p0, [x20, x25, LSL #2]\n"
      // ---- Check: more channels for za1? ----
      "whilelt p0.s, x24, %x[nchan]\n"
      "b.none 3f\n"
      // ---- MOVA za1 ----
      ".inst 0xc082749f  // mova z31.s, p5/M, za1h.s[x15]\n"
      ".inst 0xc082349e  // mova z30.s, p5/M, za1h.s[x13]\n"
      "fmin z31.s, p5/M, z31.s, z10.s\n"
      ".inst 0xc082149d  // mova z29.s, p5/M, za1h.s[x12]\n"
      "fmin z30.s, p5/M, z30.s, z10.s\n"
      ".inst 0xc082549c  // mova z28.s, p5/M, za1h.s[x14]\n"
      "fmin z29.s, p5/M, z29.s, z10.s\n"
      ".inst 0xc08274bb  // mova z27.s, p5/M, za1h.s[x15, #1]\n"
      "fmin z28.s, p5/M, z28.s, z10.s\n"
      ".inst 0xc08234ba  // mova z26.s, p5/M, za1h.s[x13, #1]\n"
      "fmax z31.s, p5/M, z31.s, z12.s\n"
      "fmin z27.s, p5/M, z27.s, z10.s\n"
      ".inst 0xc08215b9  // mova z25.s, p5/M, za1h.s[x12, #1]\n"
      "fmax z30.s, p5/M, z30.s, z12.s\n"
      "fmin z26.s, p5/M, z26.s, z10.s\n"
      ".inst 0xc08254b8  // mova z24.s, p5/M, za1h.s[x14, #1]\n"
      "fmax z29.s, p5/M, z29.s, z12.s\n"
      "fmin z25.s, p5/M, z25.s, z10.s\n"
      ".inst 0xc08274d7  // mova z23.s, p5/M, za1h.s[x15, #2]\n"
      "fmax z28.s, p5/M, z28.s, z12.s\n"
      "fmin z24.s, p5/M, z24.s, z10.s\n"
      ".inst 0xc08234d6  // mova z22.s, p5/M, za1h.s[x13, #2]\n"
      "fmax z27.s, p5/M, z27.s, z12.s\n"
      "fmin z23.s, p5/M, z23.s, z10.s\n"
      ".inst 0xc08214d5  // mova z21.s, p5/M, za1h.s[x12, #2]\n"
      "fmax z26.s, p5/M, z26.s, z12.s\n"
      "fmin z22.s, p5/M, z22.s, z10.s\n"
      "add x20, %x[output], %x[ocs], LSL #2\n"
      ".inst 0xc08254d4  // mova z20.s, p5/M, za1h.s[x14, #2]\n"
      "fmax z25.s, p5/M, z25.s, z12.s\n"
      "fmin z21.s, p5/M, z21.s, z10.s\n"
      "add x8, %x[output], %x[ors], LSL #2\n"
      ".inst 0xc08274f3  // mova z19.s, p5/M, za1h.s[x15, #3]\n"
      "fmax z24.s, p5/M, z24.s, z12.s\n"
      "fmin z20.s, p5/M, z20.s, z10.s\n"
      ".inst 0xc08234f2  // mova z18.s, p5/M, za1h.s[x13, #3]\n"
      "fmax z23.s, p5/M, z23.s, z12.s\n"
      "fmin z19.s, p5/M, z19.s, z10.s\n"
      ".inst 0xc08214f1  // mova z17.s, p5/M, za1h.s[x12, #3]\n"
      "fmax z22.s, p5/M, z22.s, z12.s\n"
      "fmin z18.s, p5/M, z18.s, z10.s\n"
      ".inst 0xc08254f0  // mova z16.s, p5/M, za1h.s[x14, #3]\n"
      "fmax z21.s, p5/M, z21.s, z12.s\n"
      "fmin z17.s, p5/M, z17.s, z10.s\n"
      "fmax z20.s, p5/M, z20.s, z12.s\n"
      "fmin z16.s, p5/M, z16.s, z10.s\n"
      "st1w { z31.s }, p0, [%x[output], x24, LSL #2]\n"
      "fmax z19.s, p5/M, z19.s, z12.s\n"
      "st1w { z30.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "fmax z18.s, p5/M, z18.s, z12.s\n"
      "st1w { z29.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "fmax z17.s, p5/M, z17.s, z12.s\n"
      "st1w { z28.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "fmax z16.s, p5/M, z16.s, z12.s\n"
      "st1w { z27.s }, p0, [x8, x24, LSL #2]\n"
      "add x8, x8, %x[ors], LSL #2\n"
      "st1w { z26.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z25.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z24.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "st1w { z23.s }, p0, [x8, x24, LSL #2]\n"
      "add x8, x8, %x[ors], LSL #2\n"
      "st1w { z22.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z21.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z20.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "st1w { z19.s }, p0, [x8, x24, LSL #2]\n"
      "st1w { z18.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z17.s }, p0, [x20, x24, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z16.s }, p0, [x20, x24, LSL #2]\n"
      // ---- Check: more channels for za2? ----
      "whilelt p0.s, x23, %x[nchan]\n"
      "b.none 3f\n"
      // ---- MOVA za2 ----
      ".inst 0xc082751f  // mova z31.s, p5/M, za2h.s[x15]\n"
      ".inst 0xc082351e  // mova z30.s, p5/M, za2h.s[x13]\n"
      "fmin z31.s, p5/M, z31.s, z10.s\n"
      ".inst 0xc082151d  // mova z29.s, p5/M, za2h.s[x12]\n"
      "fmin z30.s, p5/M, z30.s, z10.s\n"
      ".inst 0xc082551c  // mova z28.s, p5/M, za2h.s[x14]\n"
      "fmin z29.s, p5/M, z29.s, z10.s\n"
      ".inst 0xc082753b  // mova z27.s, p5/M, za2h.s[x15, #1]\n"
      "fmin z28.s, p5/M, z28.s, z10.s\n"
      ".inst 0xc082353a  // mova z26.s, p5/M, za2h.s[x13, #1]\n"
      "fmax z31.s, p5/M, z31.s, z12.s\n"
      "fmin z27.s, p5/M, z27.s, z10.s\n"
      ".inst 0xc0821539  // mova z25.s, p5/M, za2h.s[x12, #1]\n"
      "fmax z30.s, p5/M, z30.s, z12.s\n"
      "fmin z26.s, p5/M, z26.s, z10.s\n"
      ".inst 0xc0825538  // mova z24.s, p5/M, za2h.s[x14, #1]\n"
      "fmax z29.s, p5/M, z29.s, z12.s\n"
      "fmin z25.s, p5/M, z25.s, z10.s\n"
      ".inst 0xc0827557  // mova z23.s, p5/M, za2h.s[x15, #2]\n"
      "fmax z28.s, p5/M, z28.s, z12.s\n"
      "fmin z24.s, p5/M, z24.s, z10.s\n"
      ".inst 0xc0823556  // mova z22.s, p5/M, za2h.s[x13, #2]\n"
      "fmax z27.s, p5/M, z27.s, z12.s\n"
      "fmin z23.s, p5/M, z23.s, z10.s\n"
      ".inst 0xc0821555  // mova z21.s, p5/M, za2h.s[x12, #2]\n"
      "fmax z26.s, p5/M, z26.s, z12.s\n"
      "fmin z22.s, p5/M, z22.s, z10.s\n"
      "add x20, %x[output], %x[ocs], LSL #2\n"
      ".inst 0xc0825554  // mova z20.s, p5/M, za2h.s[x14, #2]\n"
      "fmax z25.s, p5/M, z25.s, z12.s\n"
      "fmin z21.s, p5/M, z21.s, z10.s\n"
      "add x8, %x[output], %x[ors], LSL #2\n"
      ".inst 0xc0827573  // mova z19.s, p5/M, za2h.s[x15, #3]\n"
      "fmax z24.s, p5/M, z24.s, z12.s\n"
      "fmin z20.s, p5/M, z20.s, z10.s\n"
      ".inst 0xc0823572  // mova z18.s, p5/M, za2h.s[x13, #3]\n"
      "fmax z23.s, p5/M, z23.s, z12.s\n"
      "fmin z19.s, p5/M, z19.s, z10.s\n"
      ".inst 0xc0821571  // mova z17.s, p5/M, za2h.s[x12, #3]\n"
      "fmax z22.s, p5/M, z22.s, z12.s\n"
      "fmin z18.s, p5/M, z18.s, z10.s\n"
      ".inst 0xc0825570  // mova z16.s, p5/M, za2h.s[x14, #3]\n"
      "fmax z21.s, p5/M, z21.s, z12.s\n"
      "fmin z17.s, p5/M, z17.s, z10.s\n"
      "fmax z20.s, p5/M, z20.s, z12.s\n"
      "fmin z16.s, p5/M, z16.s, z10.s\n"
      "st1w { z31.s }, p0, [%x[output], x23, LSL #2]\n"
      "fmax z19.s, p5/M, z19.s, z12.s\n"
      "st1w { z30.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "fmax z18.s, p5/M, z18.s, z12.s\n"
      "st1w { z29.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "fmax z17.s, p5/M, z17.s, z12.s\n"
      "st1w { z28.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "fmax z16.s, p5/M, z16.s, z12.s\n"
      "st1w { z27.s }, p0, [x8, x23, LSL #2]\n"
      "add x8, x8, %x[ors], LSL #2\n"
      "st1w { z26.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z25.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z24.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "st1w { z23.s }, p0, [x8, x23, LSL #2]\n"
      "add x8, x8, %x[ors], LSL #2\n"
      "st1w { z22.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z21.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z20.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "st1w { z19.s }, p0, [x8, x23, LSL #2]\n"
      "st1w { z18.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z17.s }, p0, [x20, x23, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z16.s }, p0, [x20, x23, LSL #2]\n"
      // ---- Check: more channels for za3? ----
      "whilelt p0.s, x22, %x[nchan]\n"
      "b.none 3f\n"
      // ---- MOVA za3 + advance channel pointers + re-zero ZA + reload next batch ----
      "fmov z1.s, #1.0\n"
      ".inst 0xc082759f  // mova z31.s, p5/M, za3h.s[x15]\n"
      ".inst 0xc082359e  // mova z30.s, p5/M, za3h.s[x13]\n"
      "fmin z31.s, p5/M, z31.s, z10.s\n"
      ".inst 0xc082159d  // mova z29.s, p5/M, za3h.s[x12]\n"
      "fmin z30.s, p5/M, z30.s, z10.s\n"
      ".inst 0xc082559c  // mova z28.s, p5/M, za3h.s[x14]\n"
      "fmin z29.s, p5/M, z29.s, z10.s\n"
      ".inst 0xc08275bb  // mova z27.s, p5/M, za3h.s[x15, #1]\n"
      "fmin z28.s, p5/M, z28.s, z10.s\n"
      ".inst 0xc08235ba  // mova z26.s, p5/M, za3h.s[x13, #1]\n"
      "fmax z31.s, p5/M, z31.s, z12.s\n"
      "fmin z27.s, p5/M, z27.s, z10.s\n"
      ".inst 0xc08215b9  // mova z25.s, p5/M, za3h.s[x12, #1]\n"
      "fmax z30.s, p5/M, z30.s, z12.s\n"
      "fmin z26.s, p5/M, z26.s, z10.s\n"
      ".inst 0xc08255b8  // mova z24.s, p5/M, za3h.s[x14, #1]\n"
      "fmax z29.s, p5/M, z29.s, z12.s\n"
      "fmin z25.s, p5/M, z25.s, z10.s\n"
      ".inst 0xc08275d7  // mova z23.s, p5/M, za3h.s[x15, #2]\n"
      "fmax z28.s, p5/M, z28.s, z12.s\n"
      "fmin z24.s, p5/M, z24.s, z10.s\n"
      ".inst 0xc08235d6  // mova z22.s, p5/M, za3h.s[x13, #2]\n"
      "fmax z27.s, p5/M, z27.s, z12.s\n"
      "fmin z23.s, p5/M, z23.s, z10.s\n"
      ".inst 0xc08215d5  // mova z21.s, p5/M, za3h.s[x12, #2]\n"
      "fmax z26.s, p5/M, z26.s, z12.s\n"
      "fmin z22.s, p5/M, z22.s, z10.s\n"
      ".inst 0xc08255d4  // mova z20.s, p5/M, za3h.s[x14, #2]\n"
      "fmax z25.s, p5/M, z25.s, z12.s\n"
      "fmin z21.s, p5/M, z21.s, z10.s\n"
      "add x20, %x[output], %x[ocs], LSL #2\n"
      ".inst 0xc08275f3  // mova z19.s, p5/M, za3h.s[x15, #3]\n"
      "fmax z24.s, p5/M, z24.s, z12.s\n"
      "fmin z20.s, p5/M, z20.s, z10.s\n"
      "add x8, %x[output], %x[ors], LSL #2\n"
      ".inst 0xc08235f2  // mova z18.s, p5/M, za3h.s[x13, #3]\n"
      "fmax z23.s, p5/M, z23.s, z12.s\n"
      "fmin z19.s, p5/M, z19.s, z10.s\n"
      "incw x25, ALL, MUL #4\n"
      ".inst 0xc08215f1  // mova z17.s, p5/M, za3h.s[x12, #3]\n"
      "fmax z22.s, p5/M, z22.s, z12.s\n"
      "fmin z18.s, p5/M, z18.s, z10.s\n"
      "incw x24, ALL, MUL #4\n"
      ".inst 0xc08255f0  // mova z16.s, p5/M, za3h.s[x14, #3]\n"
      "fmax z21.s, p5/M, z21.s, z12.s\n"
      "fmin z17.s, p5/M, z17.s, z10.s\n"
      "incw x23, ALL, MUL #4\n"
      ".inst 0xc00800ff  // zero {zad0-zad7}\n"
      "fmax z20.s, p5/M, z20.s, z12.s\n"
      "fmin z16.s, p5/M, z16.s, z10.s\n"
      "add x21, %x[inptr], %x[mstride], LSL #2\n"
      "fmax z19.s, p5/M, z19.s, z12.s\n"
      "st1w { z31.s }, p0, [%x[output], x22, LSL #2]\n"
      "fmax z18.s, p5/M, z18.s, z12.s\n"
      "st1w { z30.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "fmax z17.s, p5/M, z17.s, z12.s\n"
      "st1w { z29.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "fmax z16.s, p5/M, z16.s, z12.s\n"
      "st1w { z28.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "st1w { z27.s }, p0, [x8, x22, LSL #2]\n"
      "add x8, x8, %x[ors], LSL #2\n"
      "st1w { z26.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z25.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z24.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "st1w { z23.s }, p0, [x8, x22, LSL #2]\n"
      "add x8, x8, %x[ors], LSL #2\n"
      "st1w { z22.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z21.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z20.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x8, %x[ocs], LSL #2\n"
      "st1w { z19.s }, p0, [x8, x22, LSL #2]\n"
      "st1w { z18.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z17.s }, p0, [x20, x22, LSL #2]\n"
      "add x20, x20, %x[ocs], LSL #2\n"
      "st1w { z16.s }, p0, [x20, x22, LSL #2]\n"
      "incw x22, ALL, MUL #4\n"
      // ---- Reload next batch of input + bias + FMOPA bias ----
      "whilelt p1.s, x22, %x[nchan]\n"
      "ld1w { z28.s }, p1/Z, [%x[inptr], x22, LSL #2]\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "whilelt p2.s, x23, %x[nchan]\n"
      "whilelt p3.s, x24, %x[nchan]\n"
      "ld1w { z30.s }, p3/Z, [%x[inptr], x24, LSL #2]\n"
      "whilelt p4.s, x25, %x[nchan]\n"
      "ld1w { z31.s }, p4/Z, [%x[inptr], x25, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p4.b\n"
      "ld1w { z29.s }, p2/Z, [%x[inptr], x23, LSL #2]\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x25, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p3.b\n"
      ".inst 0x8080b420  // fmopa za0.s, p5/M, p5/M, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x24, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p2.b\n"
      ".inst 0x8080b421  // fmopa za1.s, p5/M, p5/M, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x23, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p1.b\n"
      ".inst 0x8080b422  // fmopa za2.s, p5/M, p5/M, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x22, LSL #2]\n"
      ".inst 0x8080b423  // fmopa za3.s, p5/M, p5/M, z1.s, z0.s\n"
      "b.any 2b\n"
      "3:\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [bptr] "r" (bptr), [inptr] "r" (inptr), [mstride] "r" (matrix_stride),
        [nchan] "r" ((unsigned long) n_channels),
        [ofs_amax] "I" (offsetof(Params, act_max)),
        [ofs_amin] "I" (offsetof(Params, act_min)),
        [ofs_it] "I" (offsetof(Params, inner_terms)),
        [ofs_ot] "I" (offsetof(Params, outer_terms)),
        [output] "r" (output), [ocs] "r" (output_col_stride),
        [ors] "r" (output_row_stride), [params] "r" (&params)
      : "cc", "memory",
        "p0", "p1", "p2", "p3", "p4", "p5", "p8",
        "x8", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31"
    );
}

// ============================================================================
// 便捷包装函数
// ============================================================================

inline void input_transform_f44_sme(
    const float* d, float* U, int channels
) {
    input_transform_sme<6>(d, U, channels, F44_Bt::val);
}

inline void output_transform_f44_sme(
    const float* M, float* f, int channels,
    const float* bias = nullptr, float act_min = -1e30f, float act_max = 1e30f
) {
    sme_output_transform_f44(
        (unsigned int) channels,
        M,
        (size_t) channels,
        bias,
        f,
        (size_t)(4 * channels),
        (size_t) channels,
        act_min,
        act_max
    );
}

inline void input_transform_f22_sme(
    const float* d, float* U, int channels
) {
    input_transform_sme<4>(d, U, channels, F22_Bt::val);
}

// F(2,2) 输出变换：ACL 无 SME 版（仅有 NEON），回退到 SVE
inline void output_transform_f22_sme(
    const float* M, float* f, int channels,
    const float* bias = nullptr, float act_min = -1e30f, float act_max = 1e30f
) {
    output_transform_sve<4, 2>(M, f, channels, F22_A::val, bias, act_min, act_max);
}

// SME 权重变换 = NEON（prepare 阶段，非热路径）
#include "winograd_transforms.hpp"

inline void weight_transform_f22_sme(const float* g, float* V, int channels) {
    weight_transform_f22_neon(g, V, channels);
}

inline void weight_transform_f44_sme(const float* g, float* V, int channels) {
    weight_transform_f44_neon(g, V, channels);
}

} // namespace winograd_conv

#endif // __ARM_FEATURE_SME
