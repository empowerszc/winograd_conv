#ifndef SWISH_SVE_H
#define SWISH_SVE_H

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

static inline svfloat32_t svexpa_f32_x(svbool_t pg, svuint32_t op) {
    svfloat32_t result;
    __asm__ __volatile__(
        "fexpa %0.s, %1/m, %2.s"
        : "=w" (result)
        : "Upa" (pg), "w" (op)
    );
    return result;
}

static inline svfloat32_t exp_sve_f32(svbool_t pg, svfloat32_t x) {
    x = svmin_f32_x(pg, x, svdup_f32(88.7228391f));
    x = svmax_f32_x(pg, x, svdup_f32(-103.972077f));

    svfloat32_t t = svmul_f32_x(pg, x, svdup_f32(1.44269502f));
    svfloat32_t n_f = svfrintm_f32_x(pg, t);
    svint32_t n_i = svcvt_s32_f32_x(pg, n_f);
    svfloat32_t r = svsub_f32_x(pg, t, n_f);

    svfloat32_t t0 = svadd_n_f32_x(pg, r, 1.0f);

    svuint32_t idx = svreinterpret_u32_f32(t0);
    idx = svlsr_n_u32_x(pg, idx, 17);

    svfloat32_t base = svexpa_f32_x(pg, idx);
    base = svscale_f32_x(pg, base, n_i);

    svuint32_t hi = svand_n_u32_x(pg, svreinterpret_u32_f32(t0), 0xFFFE0000u);
    svfloat32_t frac = svsub_f32_x(pg, t0, svreinterpret_f32_u32(hi));

    svfloat32_t poly = svmla_f32_x(pg, svdup_f32(0.2413862043f), frac,
                                   svdup_f32(0.6931473921f));
    poly = svmla_f32_x(pg, poly, frac, svdup_f32(1.0f));

    return svmul_f32_x(pg, base, poly);
}

static inline svfloat32_t sigmoid_sve_f32(svbool_t pg, svfloat32_t x) {
    svbool_t pos = svcmpgt_n_f32(pg, x, 0.0f);

    svuint32_t bits = svreinterpret_u32_f32(x);
    bits = svorr_n_u32_x(pg, bits, 0x80000000u);
    x = svreinterpret_f32_u32(bits);

    svfloat32_t e = exp_sve_f32(pg, x);

    svfloat32_t denom = svadd_n_f32_x(pg, e, 1.0f);
    svfloat32_t sigma = svdiv_f32_x(pg, e, denom);

    svfloat32_t inv = svsub_n_f32_x(pg, svdup_f32(1.0f), sigma);
    return svsel_f32(pos, inv, sigma);
}

static inline svfloat32_t swish_fwd_sve(svbool_t pg, svfloat32_t x, float alpha) {
    svfloat32_t x_orig = x;
    if (alpha != 1.0f)
        x = svmul_n_f32_x(pg, x, alpha);
    svfloat32_t s = sigmoid_sve_f32(pg, x);
    return svmul_f32_x(pg, x_orig, s);
}

static inline svfloat32_t silu_fwd_sve(svbool_t pg, svfloat32_t x) {
    svfloat32_t s = sigmoid_sve_f32(pg, x);
    return svmul_f32_x(pg, x, s);
}

static inline svfloat32_t swish_bwd_sve(svbool_t pg, svfloat32_t x, float alpha) {
    svfloat32_t R = (alpha != 1.0f) ? svmul_n_f32_x(pg, x, alpha) : x;
    svfloat32_t Q = sigmoid_sve_f32(pg, R);
    svfloat32_t T = svmls_f32_x(pg, R, R, Q);
    return svmla_f32_x(pg, Q, Q, T);
}

static inline void swish_fwd_buffer_sve(const float *src, float *dst,
                                        size_t n, float alpha) {
    for (size_t i = 0; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32((int64_t)i, (int64_t)n);
        svfloat32_t x = svld1_f32(pg, src + i);
        svst1_f32(pg, dst + i, swish_fwd_sve(pg, x, alpha));
    }
}

static inline void silu_fwd_buffer_sve(const float *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32((int64_t)i, (int64_t)n);
        svfloat32_t x = svld1_f32(pg, src + i);
        svst1_f32(pg, dst + i, silu_fwd_sve(pg, x));
    }
}

static inline void swish_bwd_buffer_sve(const float *src, const float *diff_dst,
                                        float *diff_src, size_t n, float alpha) {
    for (size_t i = 0; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32((int64_t)i, (int64_t)n);
        svfloat32_t x = svld1_f32(pg, src + i);
        svfloat32_t d = svld1_f32(pg, diff_dst + i);
        svfloat32_t deriv = swish_bwd_sve(pg, x, alpha);
        svst1_f32(pg, diff_src + i, svmul_f32_x(pg, deriv, d));
    }
}

#endif
