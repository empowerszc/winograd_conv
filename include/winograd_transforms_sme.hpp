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
#if defined(__ARM_NEON_SVE_BRIDGE) && __has_include(<arm_neon_sve_bridge.h>)
#include <arm_neon_sve_bridge.h>
#elif defined(__has_include)
// GCC spells the (unused here) NEON<->SVE bridge header arm_neon_sve.h
#if __has_include(<arm_neon_sve.h>) && defined(__GNUC__) && !defined(__clang__)
#include <arm_neon_sve.h>
#endif
#endif
#include <cstddef>
#include <cstring>
#include "winograd_matrices.hpp"
#include "winograd_config.hpp"

#if defined(__ARM_FEATURE_SME)

namespace winograd_conv {

// ============================================================================
// SME 指令编码宏（当前未被引用：输入变换为纯 SVE 不需要 streaming 模式，
// 输出变换的 SMSTART/SMSTOP 直接内联在其 asm 里。保留备作其它 SME kernel 用。）
// ============================================================================

#define SME_SMSTART_ZA() \
    __asm__ __volatile__("smstart" ::: "memory")

#define SME_SMSTOP() \
    __asm__ __volatile__("smstop" ::: "memory")

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
    // The input transform is a pure-SVE elementwise matrix product (no ZA /
    // FMOPA), so there is nothing for streaming mode to accelerate here. ACL's
    // sme_fp32_mla_6x6 is a single self-contained asm body it can safely wrap
    // in SMSTART/SMSTOP, but our transform_2d_sve lazily reallocs a
    // thread_local tmp buffer — running the allocator (which uses ADVSIMD)
    // inside streaming mode is illegal and corrupts the heap when a larger
    // shape triggers the growth. So keep it in normal mode (identical to the
    // SVE path).
    transform_2d_sve<TILE_SIZE, TILE_SIZE>(
        Bt_matrix, d, U, channels, channels
    );
}

// ============================================================================
// SME 输出变换（FMOPA + MOVA 内联汇编，助记符形式）
// ============================================================================
// 直接移植自 ACL 53.1.0 sme_fp32_mopa_4x4_3x3.cpp；历史上是 .inst 十六进制
// 编码，2026-09-08 逐条替换为等价汇编助记符（经 objdump 比对，800 条指令字
// 逐位一致）。用助记符 + zero {za0.s..} + smstart/smstop 写法需汇编器支持
// SME/SME2 语法（毕昇 clang ≥19 与 aarch64-linux-gnu-as 均可）。
//
// 核心思路（vec trick）：
//   f = A^T * M * A  →  vec(f) = (A^T ⊗ A^T) * vec(M)
//   (A^T ⊗ A^T) 是 16×36 矩阵，拆成 36 个外积秩-1 项，每项一条 FMOPA：
//   对 M 的每个 Winograd 元素 m（36 个，z16..z31 各持一个通道的向量），
//   按 Kronecker 系数的行/列模式（outer_terms × inner_terms）做 rank-1 累加。
//
// ZA 布局（SVE-512，fp32）——读懂本 kernel 的关键：
//   4 个 tile za0..za3 = 4 个 16 通道组；每次循环处理 ≤64 通道。
//   每个 tile 是 16 行 × 16 列：行 = 组内通道号，列 = 16 个 f 位置 (oi,oj)。
//   FMOPA 操作数角色：fmopa zaD.s, p5/m, p5/m, Zn.s, Zm.s 执行
//     zaD[row][col] += Zm[row] × Zn[col]
//   —— 第二个向量操作数(Zm)给出行（通道）侧，第一个(Zn)给出列（位置）侧。
//   例：bias 用 "fmopa zaD.s, p5/m, p5/m, z1.s, z0.s"（z1=1.0 广播 → 每行同值，
//   z0=bias → 沿列方向加）即把 bias 注入全部 16 个位置 × 16 通道。
//
// 读出（drain）用 MOVA 按列取回通道向量：mova zX.s, p5/m, zaNh.s[wR, #i]，
//   16 个 (wR∈{w12..w15}, #i∈0..3) 组合对应 16 个 f 位置——本 kernel 的写入
//   系数模式与该读出顺序是同一套自洽约定（ACL 原样），无需理解其物理列序；
//   只要写/读成对使用（本函数即如此），结果就正确。
//   x15/x14/x13/x12 的运行时值在内核中段设为 0/12/4/8。
//
// 参数映射（ACL → 本项目）：
//   inptr           = M          ([6][6][channels] Winograd 域数据)
//   matrix_stride   = channels   (相邻 tile 元素的步长)
//   bptr            = bias       ([channels] 或 nullptr)
//   output          = f          ([4][4][channels] 输出)
//   output_row_stride = 4 * channels
//   output_col_stride = channels
//   n_channels      = channels
//
// 数值注意：bias 经 FMOPA 注入 ZA（与 f_tile 累加同序），clamp 用 fmin(act_max)
// / fmax(act_min) —— 历史上这两条曾被抄反（全部输出坍缩到 act_max），已修复。

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
      "smstart\n"
      "ptrue p5.b\n"
      "ld1rw { z10.s }, p5/Z, [%x[params], %[ofs_amax]]\n"
      "ld1rw { z12.s }, p5/Z, [%x[params], %[ofs_amin]]\n"
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
      "zero {za0.s, za1.s, za2.s, za3.s}\n"
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
      "fmopa za0.s, p5/m, p5/m, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x24, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p2.b\n"
      "fmopa za1.s, p5/m, p5/m, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x23, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p1.b\n"
      "fmopa za2.s, p5/m, p5/m, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x22, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z1.s, z0.s\n"
      "2:\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "mov x15, #0\n"
      "mov x14, #0xc\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "whilelt p0.s, x25, %x[nchan]\n"
      "add x20, %x[output], %x[ocs], LSL #2\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "add x8, %x[output], %x[ors], LSL #2\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z9.s, z6.s[2]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z9.s, z6.s[3]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z9.s, z7.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z9.s, z7.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z8.s, z6.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z8.s, z6.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z8.s, z6.s[2]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z8.s, z6.s[3]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z8.s, z7.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z8.s, z7.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z15.s, z6.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z15.s, z6.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z15.s, z6.s[2]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z15.s, z6.s[3]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z15.s, z7.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z15.s, z7.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z4.s, z6.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z4.s, z6.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z4.s, z6.s[2]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z4.s, z6.s[3]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z4.s, z7.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z4.s, z7.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z3.s, z6.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z3.s, z6.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z3.s, z6.s[2]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z3.s, z6.s[3]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z3.s, z7.s[0]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z3.s, z7.s[1]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "ld1w { z30.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z2.s, z6.s[0]\n"
      "ld1w { z29.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "ld1w { z28.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "ld1w { z27.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "ld1w { z26.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z2.s, z6.s[1]\n"
      "ld1w { z25.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "ld1w { z24.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "ld1w { z23.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "ld1w { z22.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z2.s, z6.s[2]\n"
      "ld1w { z21.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "ld1w { z20.s }, p1/Z, [x21, x22, LSL #2]\n"
      "add x21, x21, %x[mstride], LSL #2\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "ld1w { z19.s }, p4/Z, [x21, x25, LSL #2]\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x21, x24, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z2.s, z6.s[3]\n"
      "ld1w { z17.s }, p2/Z, [x21, x23, LSL #2]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z31.s\n"
      "ld1w { z16.s }, p1/Z, [x21, x22, LSL #2]\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z30.s\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z29.s\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z28.s\n"
      "fmul z11.s, z2.s, z7.s[0]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z27.s\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z26.s\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z25.s\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z24.s\n"
      "fmul z5.s, z2.s, z7.s[1]\n"
      "fmopa za0.s, p5/m, p5/m, z11.s, z23.s\n"
      "fmopa za1.s, p5/m, p5/m, z11.s, z22.s\n"
      "fmopa za2.s, p5/m, p5/m, z11.s, z21.s\n"
      "fmopa za3.s, p5/m, p5/m, z11.s, z20.s\n"
      "fmul z11.s, z9.s, z6.s[0]\n"
      "fmopa za0.s, p5/m, p5/m, z5.s, z19.s\n"
      "fmopa za1.s, p5/m, p5/m, z5.s, z18.s\n"
      "fmopa za2.s, p5/m, p5/m, z5.s, z17.s\n"
      "fmopa za3.s, p5/m, p5/m, z5.s, z16.s\n"
      "fmul z5.s, z9.s, z6.s[1]\n"
      // ---- MOVA: 从 ZA tile 读出 4×4 输出 + fmin/fmax (ReLU) + st1w ----
      "mova z31.s, p5/m, za0h.s[w15, #0]\n"
      "mova z28.s, p5/m, za0h.s[w14, #0]\n"
      "fmin z31.s, p5/M, z31.s, z10.s\n"
      "mova z27.s, p5/m, za0h.s[w15, #1]\n"
      "fmin z28.s, p5/M, z28.s, z10.s\n"
      "mova z24.s, p5/m, za0h.s[w14, #1]\n"
      "fmin z27.s, p5/M, z27.s, z10.s\n"
      "mov x13, #0x4\n"
      "mov x12, #0x8\n"
      "mova z30.s, p5/m, za0h.s[w13, #0]\n"
      "fmin z24.s, p5/M, z24.s, z10.s\n"
      "mova z29.s, p5/m, za0h.s[w12, #0]\n"
      "fmax z31.s, p5/M, z31.s, z12.s\n"
      "fmin z30.s, p5/M, z30.s, z10.s\n"
      "mova z26.s, p5/m, za0h.s[w13, #1]\n"
      "fmin z29.s, p5/M, z29.s, z10.s\n"
      "fmax z28.s, p5/M, z28.s, z12.s\n"
      "mova z25.s, p5/m, za0h.s[w12, #1]\n"
      "fmax z27.s, p5/M, z27.s, z12.s\n"
      "fmin z26.s, p5/M, z26.s, z10.s\n"
      "mova z23.s, p5/m, za0h.s[w15, #2]\n"
      "fmin z25.s, p5/M, z25.s, z10.s\n"
      "fmax z24.s, p5/M, z24.s, z12.s\n"
      "mova z22.s, p5/m, za0h.s[w13, #2]\n"
      "fmax z30.s, p5/M, z30.s, z12.s\n"
      "fmin z23.s, p5/M, z23.s, z10.s\n"
      "mova z21.s, p5/m, za0h.s[w12, #2]\n"
      "fmax z29.s, p5/M, z29.s, z12.s\n"
      "fmin z22.s, p5/M, z22.s, z10.s\n"
      "mova z20.s, p5/m, za0h.s[w14, #2]\n"
      "fmax z26.s, p5/M, z26.s, z12.s\n"
      "fmin z21.s, p5/M, z21.s, z10.s\n"
      "mova z19.s, p5/m, za0h.s[w15, #3]\n"
      "fmax z25.s, p5/M, z25.s, z12.s\n"
      "fmin z20.s, p5/M, z20.s, z10.s\n"
      "mova z18.s, p5/m, za0h.s[w13, #3]\n"
      "fmax z23.s, p5/M, z23.s, z12.s\n"
      "fmin z19.s, p5/M, z19.s, z10.s\n"
      "mova z17.s, p5/m, za0h.s[w12, #3]\n"
      "fmax z22.s, p5/M, z22.s, z12.s\n"
      "fmin z18.s, p5/M, z18.s, z10.s\n"
      "mova z16.s, p5/m, za0h.s[w14, #3]\n"
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
      "mova z31.s, p5/m, za1h.s[w15, #0]\n"
      "mova z30.s, p5/m, za1h.s[w13, #0]\n"
      "fmin z31.s, p5/M, z31.s, z10.s\n"
      "mova z29.s, p5/m, za1h.s[w12, #0]\n"
      "fmin z30.s, p5/M, z30.s, z10.s\n"
      "mova z28.s, p5/m, za1h.s[w14, #0]\n"
      "fmin z29.s, p5/M, z29.s, z10.s\n"
      "mova z27.s, p5/m, za1h.s[w15, #1]\n"
      "fmin z28.s, p5/M, z28.s, z10.s\n"
      "mova z26.s, p5/m, za1h.s[w13, #1]\n"
      "fmax z31.s, p5/M, z31.s, z12.s\n"
      "fmin z27.s, p5/M, z27.s, z10.s\n"
      "mova z25.s, p5/m, za1h.s[w12, #1]\n"
      "fmax z30.s, p5/M, z30.s, z12.s\n"
      "fmin z26.s, p5/M, z26.s, z10.s\n"
      "mova z24.s, p5/m, za1h.s[w14, #1]\n"
      "fmax z29.s, p5/M, z29.s, z12.s\n"
      "fmin z25.s, p5/M, z25.s, z10.s\n"
      "mova z23.s, p5/m, za1h.s[w15, #2]\n"
      "fmax z28.s, p5/M, z28.s, z12.s\n"
      "fmin z24.s, p5/M, z24.s, z10.s\n"
      "mova z22.s, p5/m, za1h.s[w13, #2]\n"
      "fmax z27.s, p5/M, z27.s, z12.s\n"
      "fmin z23.s, p5/M, z23.s, z10.s\n"
      "mova z21.s, p5/m, za1h.s[w12, #2]\n"
      "fmax z26.s, p5/M, z26.s, z12.s\n"
      "fmin z22.s, p5/M, z22.s, z10.s\n"
      "add x20, %x[output], %x[ocs], LSL #2\n"
      "mova z20.s, p5/m, za1h.s[w14, #2]\n"
      "fmax z25.s, p5/M, z25.s, z12.s\n"
      "fmin z21.s, p5/M, z21.s, z10.s\n"
      "add x8, %x[output], %x[ors], LSL #2\n"
      "mova z19.s, p5/m, za1h.s[w15, #3]\n"
      "fmax z24.s, p5/M, z24.s, z12.s\n"
      "fmin z20.s, p5/M, z20.s, z10.s\n"
      "mova z18.s, p5/m, za1h.s[w13, #3]\n"
      "fmax z23.s, p5/M, z23.s, z12.s\n"
      "fmin z19.s, p5/M, z19.s, z10.s\n"
      "mova z17.s, p5/m, za1h.s[w12, #3]\n"
      "fmax z22.s, p5/M, z22.s, z12.s\n"
      "fmin z18.s, p5/M, z18.s, z10.s\n"
      "mova z16.s, p5/m, za1h.s[w14, #3]\n"
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
      "mova z31.s, p5/m, za2h.s[w15, #0]\n"
      "mova z30.s, p5/m, za2h.s[w13, #0]\n"
      "fmin z31.s, p5/M, z31.s, z10.s\n"
      "mova z29.s, p5/m, za2h.s[w12, #0]\n"
      "fmin z30.s, p5/M, z30.s, z10.s\n"
      "mova z28.s, p5/m, za2h.s[w14, #0]\n"
      "fmin z29.s, p5/M, z29.s, z10.s\n"
      "mova z27.s, p5/m, za2h.s[w15, #1]\n"
      "fmin z28.s, p5/M, z28.s, z10.s\n"
      "mova z26.s, p5/m, za2h.s[w13, #1]\n"
      "fmax z31.s, p5/M, z31.s, z12.s\n"
      "fmin z27.s, p5/M, z27.s, z10.s\n"
      "mova z25.s, p5/m, za2h.s[w12, #1]\n"
      "fmax z30.s, p5/M, z30.s, z12.s\n"
      "fmin z26.s, p5/M, z26.s, z10.s\n"
      "mova z24.s, p5/m, za2h.s[w14, #1]\n"
      "fmax z29.s, p5/M, z29.s, z12.s\n"
      "fmin z25.s, p5/M, z25.s, z10.s\n"
      "mova z23.s, p5/m, za2h.s[w15, #2]\n"
      "fmax z28.s, p5/M, z28.s, z12.s\n"
      "fmin z24.s, p5/M, z24.s, z10.s\n"
      "mova z22.s, p5/m, za2h.s[w13, #2]\n"
      "fmax z27.s, p5/M, z27.s, z12.s\n"
      "fmin z23.s, p5/M, z23.s, z10.s\n"
      "mova z21.s, p5/m, za2h.s[w12, #2]\n"
      "fmax z26.s, p5/M, z26.s, z12.s\n"
      "fmin z22.s, p5/M, z22.s, z10.s\n"
      "add x20, %x[output], %x[ocs], LSL #2\n"
      "mova z20.s, p5/m, za2h.s[w14, #2]\n"
      "fmax z25.s, p5/M, z25.s, z12.s\n"
      "fmin z21.s, p5/M, z21.s, z10.s\n"
      "add x8, %x[output], %x[ors], LSL #2\n"
      "mova z19.s, p5/m, za2h.s[w15, #3]\n"
      "fmax z24.s, p5/M, z24.s, z12.s\n"
      "fmin z20.s, p5/M, z20.s, z10.s\n"
      "mova z18.s, p5/m, za2h.s[w13, #3]\n"
      "fmax z23.s, p5/M, z23.s, z12.s\n"
      "fmin z19.s, p5/M, z19.s, z10.s\n"
      "mova z17.s, p5/m, za2h.s[w12, #3]\n"
      "fmax z22.s, p5/M, z22.s, z12.s\n"
      "fmin z18.s, p5/M, z18.s, z10.s\n"
      "mova z16.s, p5/m, za2h.s[w14, #3]\n"
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
      "mova z31.s, p5/m, za3h.s[w15, #0]\n"
      "mova z30.s, p5/m, za3h.s[w13, #0]\n"
      "fmin z31.s, p5/M, z31.s, z10.s\n"
      "mova z29.s, p5/m, za3h.s[w12, #0]\n"
      "fmin z30.s, p5/M, z30.s, z10.s\n"
      "mova z28.s, p5/m, za3h.s[w14, #0]\n"
      "fmin z29.s, p5/M, z29.s, z10.s\n"
      "mova z27.s, p5/m, za3h.s[w15, #1]\n"
      "fmin z28.s, p5/M, z28.s, z10.s\n"
      "mova z26.s, p5/m, za3h.s[w13, #1]\n"
      "fmax z31.s, p5/M, z31.s, z12.s\n"
      "fmin z27.s, p5/M, z27.s, z10.s\n"
      "mova z25.s, p5/m, za3h.s[w12, #1]\n"
      "fmax z30.s, p5/M, z30.s, z12.s\n"
      "fmin z26.s, p5/M, z26.s, z10.s\n"
      "mova z24.s, p5/m, za3h.s[w14, #1]\n"
      "fmax z29.s, p5/M, z29.s, z12.s\n"
      "fmin z25.s, p5/M, z25.s, z10.s\n"
      "mova z23.s, p5/m, za3h.s[w15, #2]\n"
      "fmax z28.s, p5/M, z28.s, z12.s\n"
      "fmin z24.s, p5/M, z24.s, z10.s\n"
      "mova z22.s, p5/m, za3h.s[w13, #2]\n"
      "fmax z27.s, p5/M, z27.s, z12.s\n"
      "fmin z23.s, p5/M, z23.s, z10.s\n"
      "mova z21.s, p5/m, za3h.s[w12, #2]\n"
      "fmax z26.s, p5/M, z26.s, z12.s\n"
      "fmin z22.s, p5/M, z22.s, z10.s\n"
      "mova z20.s, p5/m, za3h.s[w14, #2]\n"
      "fmax z25.s, p5/M, z25.s, z12.s\n"
      "fmin z21.s, p5/M, z21.s, z10.s\n"
      "add x20, %x[output], %x[ocs], LSL #2\n"
      "mova z19.s, p5/m, za3h.s[w15, #3]\n"
      "fmax z24.s, p5/M, z24.s, z12.s\n"
      "fmin z20.s, p5/M, z20.s, z10.s\n"
      "add x8, %x[output], %x[ors], LSL #2\n"
      "mova z18.s, p5/m, za3h.s[w13, #3]\n"
      "fmax z23.s, p5/M, z23.s, z12.s\n"
      "fmin z19.s, p5/M, z19.s, z10.s\n"
      "incw x25, ALL, MUL #4\n"
      "mova z17.s, p5/m, za3h.s[w12, #3]\n"
      "fmax z22.s, p5/M, z22.s, z12.s\n"
      "fmin z18.s, p5/M, z18.s, z10.s\n"
      "incw x24, ALL, MUL #4\n"
      "mova z16.s, p5/m, za3h.s[w14, #3]\n"
      "fmax z21.s, p5/M, z21.s, z12.s\n"
      "fmin z17.s, p5/M, z17.s, z10.s\n"
      "incw x23, ALL, MUL #4\n"
      "zero {za0.s, za1.s, za2.s, za3.s}\n"
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
      "fmopa za0.s, p5/m, p5/m, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x24, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p2.b\n"
      "fmopa za1.s, p5/m, p5/m, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x23, LSL #2]\n"
      "and p0.b, p5/Z, p8.b, p1.b\n"
      "fmopa za2.s, p5/m, p5/m, z1.s, z0.s\n"
      "ld1w { z0.s }, p0/Z, [%x[bptr], x22, LSL #2]\n"
      "fmopa za3.s, p5/m, p5/m, z1.s, z0.s\n"
      "b.any 2b\n"
      "3:\n"
      "smstop\n"
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
