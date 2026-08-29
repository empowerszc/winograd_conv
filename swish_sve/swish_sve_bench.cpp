#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>

#include "swish_sve.h"

static inline float swish_ref(float x, float alpha) {
    float s = 1.0f / (1.0f + expf(-alpha * x));
    return x * s;
}

static inline float swish_bwd_ref(float x, float alpha) {
    float r = alpha * x;
    float q = 1.0f / (1.0f + expf(-r));
    return q * (1.0f + r * (1.0f - q));
}

static void test_correctness(int N, float alpha, float lo, float hi) {
    float *src = (float *)aligned_alloc(64, N * sizeof(float));
    float *dst_sve = (float *)aligned_alloc(64, N * sizeof(float));
    float *dst_ref = (float *)aligned_alloc(64, N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++)
        src[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);

    for (int i = 0; i < N; i++)
        dst_ref[i] = swish_ref(src[i], alpha);

    swish_fwd_buffer_sve(src, dst_sve, N, alpha);

    float max_err = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(dst_sve[i] - dst_ref[i]);
        float rel = err / (fabsf(dst_ref[i]) + 1e-30f);
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
    }

    printf("  fwd: max_abs_err=%.2e  max_rel_err=%.2e  (%d samples, alpha=%.4f, range=[%.1f,%.1f])\n",
           max_err, max_rel, N, alpha, lo, hi);

    for (int i = 0; i < N; i++)
        dst_ref[i] = swish_bwd_ref(src[i], alpha);

    float *diff_dst = (float *)aligned_alloc(64, N * sizeof(float));
    float *diff_src_sve = (float *)aligned_alloc(64, N * sizeof(float));
    float *diff_src_ref = (float *)aligned_alloc(64, N * sizeof(float));

    for (int i = 0; i < N; i++)
        diff_dst[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);

    for (int i = 0; i < N; i++)
        diff_src_ref[i] = diff_dst[i] * swish_bwd_ref(src[i], alpha);

    swish_bwd_buffer_sve(src, diff_dst, diff_src_sve, N, alpha);

    max_err = 0.0f; max_rel = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(diff_src_sve[i] - diff_src_ref[i]);
        float rel = err / (fabsf(diff_src_ref[i]) + 1e-30f);
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
    }
    printf("  bwd: max_abs_err=%.2e  max_rel_err=%.2e  (%d samples, alpha=%.4f)\n",
           max_err, max_rel, N, alpha);

    free(src); free(dst_sve); free(dst_ref);
    free(diff_dst); free(diff_src_sve); free(diff_src_ref);
}

static double bench_fwd(int N, int repeats, float alpha) {
    float *src = (float *)aligned_alloc(64, N * sizeof(float));
    float *dst = (float *)aligned_alloc(64, N * sizeof(float));

    for (int i = 0; i < N; i++)
        src[i] = (float)i * 0.001f - (N / 2) * 0.001f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < repeats; r++)
        swish_fwd_buffer_sve(src, dst, N, alpha);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double total_bytes = 2.0 * N * sizeof(float) * repeats;
    double gb_s = total_bytes / ns;
    double ns_elem = ns / (N * repeats);

    printf("  fwd: %d x %d elems | %.1f ns/elem | %.2f GB/s | alpha=%.4f\n",
           N, repeats, ns_elem, gb_s, alpha);

    free(src); free(dst);
    return gb_s;
}

static double bench_silu(int N, int repeats) {
    float *src = (float *)aligned_alloc(64, N * sizeof(float));
    float *dst = (float *)aligned_alloc(64, N * sizeof(float));

    for (int i = 0; i < N; i++)
        src[i] = (float)i * 0.001f - (N / 2) * 0.001f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < repeats; r++)
        silu_fwd_buffer_sve(src, dst, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double total_bytes = 2.0 * N * sizeof(float) * repeats;
    double gb_s = total_bytes / ns;
    double ns_elem = ns / (N * repeats);

    printf("  silu(alpha=1): %d x %d elems | %.1f ns/elem | %.2f GB/s\n",
           N, repeats, ns_elem, gb_s);

    free(src); free(dst);
    return gb_s;
}

static void bench_scalar_ref(int N, int repeats, float alpha) {
    float *src = (float *)aligned_alloc(64, N * sizeof(float));
    float *dst = (float *)aligned_alloc(64, N * sizeof(float));

    for (int i = 0; i < N; i++)
        src[i] = (float)i * 0.001f - (N / 2) * 0.001f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < repeats; r++)
        for (int i = 0; i < N; i++)
            dst[i] = swish_ref(src[i], alpha);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double total_bytes = 2.0 * N * sizeof(float) * repeats;
    double gb_s = total_bytes / ns;
    double ns_elem = ns / (N * repeats);

    printf("  scalar_ref: %d x %d elems | %.1f ns/elem | %.2f GB/s | alpha=%.4f\n",
           N, repeats, ns_elem, gb_s, alpha);

    free(src); free(dst);
}

int main(int argc, char **argv) {
    printf("=== Swish SVE Benchmark ===\n");
    printf("SVE vector length: %d bits (%d floats/vec)\n\n",
           svcntb() * 8, svcntw());

    int N = 1 << 20;
    int repeats = 100;

    printf("--- Correctness ---\n");
    test_correctness(N, 1.0f, -10.0f, 10.0f);
    test_correctness(N, 1.0f, -50.0f, 50.0f);
    test_correctness(N, 0.1f, -10.0f, 10.0f);
    test_correctness(N, 2.0f, -30.0f, 30.0f);
    test_correctness(1 << 16, 1.0f, -100.0f, 100.0f);
    printf("\n");

    printf("--- Performance (1M elements) ---\n");
    bench_scalar_ref(N, repeats, 1.0f);
    bench_silu(N, repeats);
    bench_fwd(N, repeats, 1.0f);
    bench_fwd(N, repeats, 0.1f);
    bench_fwd(N, repeats, 2.0f);

    printf("\n--- Performance (64K elements) ---\n");
    bench_silu(1 << 16, 1000);
    bench_fwd(1 << 16, 1000, 1.0f);

    printf("\n--- Performance (4K elements, L1-resident) ---\n");
    bench_silu(1 << 12, 10000);

    printf("\nDone.\n");
    return 0;
}
