// profile_case.cpp - Standalone profiling tool for a single Winograd case
//
// Usage:
//   ./profile_case [options]
//
// Options:
//   --n N          Batch size (default: 4)
//   --ic IC        Input channels (default: 192)
//   --ih IH        Input height (default: 40)
//   --iw IW        Input width (default: 40)
//   --oc OC        Output channels (default: 192)
//   --isa ISA      NEON / SVE / SME (default: auto-detect)
//   --layout L     NCHW / NHWC (default: NHWC)
//   --gemm G       openblas / naive / arm_gemm (default: compiled default)
//   --threads T    OpenMP threads (default: omp_get_max_threads())
//   --warmup W     Warmup iterations (default: 5)
//   --repeats R    Timed iterations (default: 20)
//   --timing       Enable fine-grained per-step timing
//   --verify       Verify against direct convolution
//   --perf         Print perf stat hints (perf command)
//   --help         Show help
//
// Examples:
//   # Profile Case 0 (4x192x40x40) with SVE, 16 threads, timing mode
//   ./profile_case --n 4 --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --timing
//
//   # Verify correctness
//   ./profile_case --ic 96 --ih 80 --iw 80 --oc 96 --verify
//
//   # Get perf command suggestion
//   ./profile_case --n 4 --ic 384 --ih 80 --iw 80 --oc 96 --perf
//
//   # Profile with perf (example)
//   perf stat -e cycles,instructions,cache-misses,cache-references,L1-dcache-load-misses \
//     ./profile_case --n 4 --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --warmup 3 --repeats 1
//
// Part of the winograd_conv project.

#include "../include/winograd_convolution.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace winograd_conv;

struct Args {
    int N = 4, IC = 192, IH = 40, IW = 40, OC = 192;
    ISALevel isa = ISALevel::NEON;
    Layout layout = Layout::NHWC;
    int threads = 1;
    int warmup = 5;
    int repeats = 20;
    bool timing = false;
    bool verify = false;
    bool perf = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    const char* isa_str = nullptr;
    const char* layout_str = nullptr;

    // Auto-detect ISA
    a.isa = detect_isa();

#ifdef ENABLE_OPENMP
    a.threads = omp_get_max_threads();
#else
    a.threads = 1;
#endif

    if (const char* env = std::getenv("WINOGRAD_ISA")) {
        a.isa = parse_isa(env);
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* { return (i+1 < argc) ? argv[++i] : ""; };

        if (arg == "--n") a.N = atoi(next());
        else if (arg == "--ic") a.IC = atoi(next());
        else if (arg == "--ih") a.IH = atoi(next());
        else if (arg == "--iw") a.IW = atoi(next());
        else if (arg == "--oc") a.OC = atoi(next());
        else if (arg == "--isa") isa_str = next();
        else if (arg == "--layout") layout_str = next();
        else if (arg == "--threads") a.threads = atoi(next());
        else if (arg == "--warmup") a.warmup = atoi(next());
        else if (arg == "--repeats") a.repeats = atoi(next());
        else if (arg == "--timing") a.timing = true;
        else if (arg == "--verify") a.verify = true;
        else if (arg == "--perf") a.perf = true;
        else if (arg == "--help") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Shape:\n");
            printf("  --n N          Batch size (default: 4)\n");
            printf("  --ic IC        Input channels (default: 192)\n");
            printf("  --ih IH        Input height (default: 40)\n");
            printf("  --iw IW        Input width (default: 40)\n");
            printf("  --oc OC        Output channels (default: 192)\n");
            printf("Execution:\n");
            printf("  --isa ISA      NEON / SVE / SME (default: auto-detect)\n");
            printf("  --layout L     NCHW / NHWC (default: NHWC)\n");
            printf("  --threads T    OpenMP threads (default: max)\n");
            printf("  --warmup W     Warmup iterations (default: 5)\n");
            printf("  --repeats R    Timed iterations (default: 20)\n");
            printf("Modes:\n");
            printf("  --timing       Fine-grained per-step timing\n");
            printf("  --verify       Verify against direct convolution\n");
            printf("  --perf         Print perf stat command suggestion\n");
            printf("\nPresets:\n");
            printf("  Case 0: --ic 192 --ih 40 --iw 40 --oc 192\n");
            printf("  Case 1: --ic 96  --ih 80 --iw 80 --oc 96\n");
            printf("  Case 2: --ic 48  --ih 160 --iw 160 --oc 48\n");
            printf("  Case 3: --ic 192 --ih 20 --iw 20 --oc 192\n");
            printf("  Case 4: --ic 384 --ih 80 --iw 80 --oc 96\n");
            printf("  Case 5: --ic 768 --ih 40 --iw 40 --oc 96\n");
            printf("  Case 6: --ic 768 --ih 20 --iw 20 --oc 96\n");
            printf("  Case 7: --ic 96  --ih 40 --iw 40 --oc 96\n");
            printf("  Case 8: --ic 96  --ih 20 --iw 20 --oc 96\n");
            exit(0);
        }
    }

    if (isa_str) {
        if (!strcmp(isa_str, "neon") || !strcmp(isa_str, "NEON")) a.isa = ISALevel::NEON;
        else if (!strcmp(isa_str, "sve") || !strcmp(isa_str, "SVE")) a.isa = ISALevel::SVE;
        else if (!strcmp(isa_str, "sme") || !strcmp(isa_str, "SME")) a.isa = ISALevel::SME;
    }
    if (layout_str) {
        if (!strcmp(layout_str, "nchw") || !strcmp(layout_str, "NCHW")) a.layout = Layout::NCHW;
        else if (!strcmp(layout_str, "nhwc") || !strcmp(layout_str, "NHWC")) a.layout = Layout::NHWC;
    }
    return a;
}

void print_perf_command(const Args& a) {
    printf("# Suggested perf commands:\n\n");

    const char* isa_str = isa_name(a.isa);
    printf("# Overall profiling (1 timed run after 5 warmup):\n");
    printf("perf stat -e cycles,instructions,cache-misses,cache-references,\\\n");
    printf("  L1-dcache-load-misses,LLC-load-misses,branch-misses,task-clock \\\n");
    printf("  ./profile_case --n %d --ic %d --ih %d --iw %d --oc %d --isa %s --threads %d --warmup 5 --repeats 1\n\n",
           a.N, a.IC, a.IH, a.IW, a.OC, isa_str, a.threads);

    printf("# Top-down analysis:\n");
    printf("perf stat -e topdown-fetch-bubbles,topdown-recovery-bubbles,\\\n");
    printf("  topdown-retiring,topdown-bad-spec,\\\n");
    printf("  int_ex_ret.active,int_ex_ret.far_branch,int_ex_ret间接br \\\n");
    printf("  ./profile_case --n %d --ic %d --ih %d --iw %d --oc %d --isa %s --threads %d --warmup 5 --repeats 1\n\n",
           a.N, a.IC, a.IH, a.IW, a.OC, isa_str, a.threads);

    printf("# Memory bandwidth:\n");
    printf("perf stat -e arm_spe/data_access,arm_spe/ll_cache_miss \\\n");
    printf("  ./profile_case --n %d --ic %d --ih %d --iw %d --oc %d --isa %s --threads %d --warmup 5 --repeats 1\n\n",
           a.N, a.IC, a.IH, a.IW, a.OC, isa_str, a.threads);

    printf("# Flame graph:\n");
    printf("perf record -g -F 999 -- ./profile_case --n %d --ic %d --ih %d --iw %d --oc %d --isa %s --threads %d --warmup 5 --repeats 10\n",
           a.N, a.IC, a.IH, a.IW, a.OC, isa_str, a.threads);
    printf("# Then: perf script | flamegraph.pl > flame.svg\n\n");

    printf("# NUMA profiling:\n");
    printf("numactl --interleave=all perf stat -e cache-misses,cache-references \\\n");
    printf("  ./profile_case --n %d --ic %d --ih %d --iw %d --oc %d --isa %s --threads %d --warmup 5 --repeats 1\n",
           a.N, a.IC, a.IH, a.IW, a.OC, isa_str, a.threads);
}

// Fine-grained timing (replicates pipeline step by step)
struct StepTiming {
    double weight_ms, tile_ext_ms, in_xform_ms, in_scat_ms;
    double gemm_ms, out_gath_ms, out_xform_ms, out_write_ms;
    double total_ms;
};

StepTiming run_with_timing(
    const float* src, const float* wei, const float* bias, float* dst,
    int N, int IC, int IH, int IW, int OC, int OH, int OW,
    ISALevel isa, Layout layout
) {
    const int TS = 6, OT = 4, NM = 36;
    const bool is_f44 = true;

    auto t0 = std::chrono::high_resolution_clock::now();
    // Weight transform
    int V_size = TS * TS * OC * IC;
    std::vector<float> V(V_size, 0.0f);
    for (int oc = 0; oc < OC; oc++) {
        std::vector<float> g(9 * IC);
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[(kh * 3 + kw) * IC + ic] = wei[((oc * IC + ic) * 3 + kh) * 3 + kw];
        std::vector<float> V_oc(TS * TS * IC);
        dispatch_weight_transform(g.data(), V_oc.data(), IC, is_f44, isa);
        for (int m = 0; m < TS * TS; m++)
            for (int ic = 0; ic < IC; ic++)
                V[m * OC * IC + oc * IC + ic] = V_oc[m * IC + ic];
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    int n_tile_rows = (OH + OT - 1) / OT;
    int n_tile_cols = (OW + OT - 1) / OT;
    int n_tiles = n_tile_rows * n_tile_cols;
    std::vector<float> U(NM * n_tiles * IC, 0.0f);
    std::vector<float> M_buf(NM * n_tiles * OC, 0.0f);
    std::vector<float> d_tile(TS * TS * IC, 0.0f);
    std::vector<float> U_tile(TS * TS * IC, 0.0f);
    std::vector<float> M_tile(TS * TS * OC, 0.0f);
    std::vector<float> f_tile(OT * OT * OC, 0.0f);

    double t_ext = 0, t_in = 0, t_scat = 0;
    double t_gath = 0, t_out = 0, t_write = 0;

    for (int n = 0; n < N; n++) {
        for (int tr = 0; tr < n_tile_rows; tr++) {
            for (int tc = 0; tc < n_tile_cols; tc++) {
                int tile_idx = tr * n_tile_cols + tc;

                auto sa = std::chrono::high_resolution_clock::now();
                std::fill(d_tile.begin(), d_tile.end(), 0.0f);
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ih = tr * OT - 1 + ti;
                        int iw = tc * OT - 1 + tj;
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            if (layout == Layout::NHWC) {
                                const float* sp = src + ((n * IH + ih) * IW + iw) * IC;
                                float* dp = d_tile.data() + (ti * TS + tj) * IC;
                                int ic = 0;
                                for (; ic + 4 <= IC; ic += 4)
                                    vst1q_f32(dp + ic, vld1q_f32(sp + ic));
                                for (; ic < IC; ic++) dp[ic] = sp[ic];
                            } else {
                                for (int ic = 0; ic < IC; ic++)
                                    d_tile[(ti * TS + tj) * IC + ic] =
                                        src[((n * IC + ic) * IH + ih) * IW + iw];
                            }
                        }
                    }
                }
                auto sb = std::chrono::high_resolution_clock::now();
                t_ext += std::chrono::duration<double, std::milli>(sb - sa).count();

                dispatch_input_transform(d_tile.data(), U_tile.data(), IC, is_f44, isa);
                auto sc = std::chrono::high_resolution_clock::now();
                t_in += std::chrono::duration<double, std::milli>(sc - sb).count();

                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ts_idx = ti * TS + tj;
                        float* dp = U.data() + (ts_idx * n_tiles + tile_idx) * IC;
                        const float* sp = U_tile.data() + (ti * TS + tj) * IC;
                        int ic = 0;
                        for (; ic + 4 <= IC; ic += 4)
                            vst1q_f32(dp + ic, vld1q_f32(sp + ic));
                        for (; ic < IC; ic++) dp[ic] = sp[ic];
                    }
                }
                auto sd = std::chrono::high_resolution_clock::now();
                t_scat += std::chrono::duration<double, std::milli>(sd - sc).count();
            }
        }
    }

    auto t_gemm_s = std::chrono::high_resolution_clock::now();
    for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
        winograd_gemm(U.data() + ts_idx * n_tiles * IC,
                      V.data() + ts_idx * OC * IC,
                      M_buf.data() + ts_idx * n_tiles * OC,
                      n_tiles, OC, IC);
    }
    auto t_gemm_e = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < N; n++) {
        for (int tr = 0; tr < n_tile_rows; tr++) {
            for (int tc = 0; tc < n_tile_cols; tc++) {
                int tile_idx = tr * n_tile_cols + tc;

                auto sa = std::chrono::high_resolution_clock::now();
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ts_idx = ti * TS + tj;
                        const float* sp = M_buf.data() + (ts_idx * n_tiles + tile_idx) * OC;
                        float* dp = M_tile.data() + (ti * TS + tj) * OC;
                        int oc = 0;
                        for (; oc + 4 <= OC; oc += 4)
                            vst1q_f32(dp + oc, vld1q_f32(sp + oc));
                        for (; oc < OC; oc++) dp[oc] = sp[oc];
                    }
                }
                auto sb = std::chrono::high_resolution_clock::now();
                t_gath += std::chrono::duration<double, std::milli>(sb - sa).count();

                dispatch_output_transform(M_tile.data(), f_tile.data(), OC,
                                          bias, -1e30f, 1e30f, is_f44, isa);
                auto sc = std::chrono::high_resolution_clock::now();
                t_out += std::chrono::duration<double, std::milli>(sc - sb).count();

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
                                for (; oc < OC; oc++) dp[oc] = sp[oc];
                            } else {
                                for (int oc = 0; oc < OC; oc++)
                                    dst[((n * OC + oc) * OH + oh) * OW + ow] =
                                        f_tile[(oi * OT + oj) * OC + oc];
                            }
                        }
                    }
                }
                auto sd = std::chrono::high_resolution_clock::now();
                t_write += std::chrono::duration<double, std::milli>(sd - sc).count();
            }
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    StepTiming st;
    st.weight_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    st.tile_ext_ms = t_ext;
    st.in_xform_ms = t_in;
    st.in_scat_ms = t_scat;
    st.gemm_ms = std::chrono::duration<double, std::milli>(t_gemm_e - t_gemm_s).count();
    st.out_gath_ms = t_gath;
    st.out_xform_ms = t_out;
    st.out_write_ms = t_write;
    st.total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
    return st;
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    set_isa_level(a.isa);

#ifdef ENABLE_OPENMP
    omp_set_num_threads(a.threads);
#endif

    int OH = a.IH, OW = a.IW;
    const char* layout_str = (a.layout == Layout::NHWC) ? "NHWC" : "NCHW";

    printf("=== Winograd Convolution Profiling ===\n");
    printf("Shape: N=%d IC=%d IH=%d IW=%d OC=%d OH=%d OW=%d\n", a.N, a.IC, a.IH, a.IW, a.OC, OH, OW);
    printf("ISA: %s | Layout: %s | Threads: %d | GEMM: ", isa_name(a.isa), layout_str, a.threads);
#ifdef USE_ARM_GEMM
    printf("arm_gemm\n");
#elif defined(USE_OPENBLAS)
    printf("OpenBLAS (single-thread)\n");
#else
    printf("naive\n");
#endif
    int n_tile_rows = (OH + 3) / 4;
    int n_tile_cols = (OW + 3) / 4;
    printf("Tiles: %d (%dx%d) | Winograd: F(4,4,3,3) | GEMMs: 36\n\n", n_tile_rows * n_tile_cols, n_tile_rows, n_tile_cols);

    if (a.perf) {
        print_perf_command(a);
        return 0;
    }

    // Allocate
    int src_size = a.N * a.IC * a.IH * a.IW;
    int wei_size = a.OC * a.IC * 9;
    int dst_size = a.N * a.OC * OH * OW;
    std::vector<float> src(src_size), wei(wei_size), bias(a.OC, 0.0f), dst(dst_size);

    // Fill with random data
    srand(42);
    for (auto& v : src) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : wei) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : bias) v = static_cast<float>(rand()) / RAND_MAX * 0.5f;

    // Verify
    if (a.verify) {
        printf("--- Correctness Verification ---\n");
        std::vector<float> ref_dst(dst_size, 0.0f);
        direct_convolution_3x3(src.data(), wei.data(), bias.data(), ref_dst.data(),
                               a.N, a.IC, a.IH, a.IW, a.OC, OH, OW, -1e30f, 1e30f);

        std::vector<float> wino_dst(dst_size, 0.0f);
        winograd_convolution_f44(src.data(), wei.data(), bias.data(), wino_dst.data(),
                                 a.N, a.IC, a.IH, a.IW, a.OC, OH, OW, -1e30f, 1e30f, a.layout);

        float max_err = 0;
        for (int i = 0; i < dst_size; i++)
            max_err = std::max(max_err, std::fabs(ref_dst[i] - wino_dst[i]));
        bool pass = (max_err < 1e-3f);
        printf("  Max error: %.6f  %s\n\n", max_err, pass ? "PASS" : "FAIL");
        if (!pass) return 1;
    }

    // Warmup
    for (int i = 0; i < a.warmup; i++) {
        winograd_convolution_f44(src.data(), wei.data(), bias.data(), dst.data(),
                                  a.N, a.IC, a.IH, a.IW, a.OC, OH, OW, -1e30f, 1e30f, a.layout);
    }

    // Benchmark (standard mode)
    if (!a.timing) {
        double best_ms = 1e30;
        for (int i = 0; i < a.repeats; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            winograd_convolution_f44(src.data(), wei.data(), bias.data(), dst.data(),
                                      a.N, a.IC, a.IH, a.IW, a.OC, OH, OW, -1e30f, 1e30f, a.layout);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best_ms = std::min(best_ms, ms);
        }
        double flops = 2.0 * a.N * a.OC * OH * OW * a.IC * 9.0;
        double gflops = flops / (best_ms * 1e-3) / 1e9;
        printf("--- Standard Mode ---\n");
        printf("  Best of %d: %.3f ms | %.2f GFLOPS\n", a.repeats, best_ms, gflops);
    }

    // Fine-grained timing
    if (a.timing) {
        printf("--- Fine-Grained Timing (serial, 1 thread) ---\n");
        printf("  Note: OpenMP parallelism disabled for step-by-step analysis\n");
#ifdef ENABLE_OPENMP
        int saved_threads = omp_get_max_threads();
        omp_set_num_threads(1);
#endif

        StepTiming best;
        best.total_ms = 1e30;
        for (int i = 0; i < a.repeats; i++) {
            auto st = run_with_timing(src.data(), wei.data(), bias.data(), dst.data(),
                                      a.N, a.IC, a.IH, a.IW, a.OC, OH, OW, a.isa, a.layout);
            if (st.total_ms < best.total_ms) best = st;
        }

#ifdef ENABLE_OPENMP
        omp_set_num_threads(saved_threads);
#endif

        printf("  %-16s %8.2f ms  (%4.1f%%)\n", "Weight transform", best.weight_ms, best.weight_ms / best.total_ms * 100);
        printf("  %-16s %8.2f ms  (%4.1f%%)\n", "Tile extract", best.tile_ext_ms, best.tile_ext_ms / best.total_ms * 100);
        printf("  %-16s %8.2f ms  (%4.1f%%)\n", "Input transform", best.in_xform_ms, best.in_xform_ms / best.total_ms * 100);
        printf("  %-16s %8.2f ms  (%4.1f%%)\n", "Scatter", best.in_scat_ms, best.in_scat_ms / best.total_ms * 100);
        printf("  %-16s %8.2f ms  (%4.1f%%)\n", "GEMM", best.gemm_ms, best.gemm_ms / best.total_ms * 100);
        printf("  %-16s %8.2f ms  (%4.1f%%)\n", "Gather", best.out_gath_ms, best.out_gath_ms / best.total_ms * 100);
        printf("  %-16s %8.2f ms  (%4.1f%%)\n", "Output transform", best.out_xform_ms, best.out_xform_ms / best.total_ms * 100);
        printf("  %-16s %8.2f ms  (%4.1f%%)\n", "Output writeback", best.out_write_ms, best.out_write_ms / best.total_ms * 100);
        printf("  %-16s %8.2f ms\n", "TOTAL", best.total_ms);

        double flops = 2.0 * a.N * a.OC * OH * OW * a.IC * 9.0;
        double gflops = flops / (best.total_ms * 1e-3) / 1e9;
        printf("  %-16s %8.2f GFLOPS\n", "GFLOPS (serial)", gflops);
    }

    printf("\n");
    return 0;
}
