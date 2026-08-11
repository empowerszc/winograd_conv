// test_ref_vs_nchw.cpp - Verify the NCHW wrapper matches the archived native-NCHW kernel
//
// The 2026-08-11 layout refactor re-implemented the NCHW path of
// winograd_convolution() as: NCHW -> NHWC convert -> shared NHWC compute core
// -> NCHW convert back. The pre-refactor native-NCHW kernel is archived in
// ref/winograd_conv_nchw_ref.cpp.
//
// Two modes:
//   (default)          Assert the two implementations agree BIT-EXACTLY on a
//                      fixed set of shapes. The conversion is a pure data move
//                      (no arithmetic), and both paths place the exact same
//                      values into the same d_tile/M_tile buffers, so bit-exact
//                      is expected.
//   --bench            Additionally time BOTH paths per case (warmup + best-of-
//                      repeats) and report wrapper-vs-native ms and ratio. This
//                      is the measurement that decides whether the conversion
//                      wrapper is a net win for NCHW inputs on the target
//                      (see PERFORMANCE_ANALYSIS.md §12): ratio < 1 means the
//                      conversion wrapper is faster than the native kernel.
//
// Usage:
//   ./test_ref_vs_nchw                       # bit-exact only
//   ./test_ref_vs_nchw --bench               # bit-exact + timing
//   ./test_ref_vs_nchw --bench --threads 16 --warmup 3 --repeats 10
//
// Part of the winograd_conv project.

#include "winograd_convolution.hpp"
#include "winograd_conv_nchw_ref.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace winograd_conv;

namespace {

struct Shape {
    int N, IC, IH, IW, OC;
};

const Shape kShapes[] = {
    {1, 8,  16, 16, 8},    // tiny, non-multiple of 16 channels
    {2, 16, 16, 16, 16},   // exact SVE lane block
    {1, 48, 160, 160, 48}, // bench Case 2 (many tiles)
    {4, 192, 40, 40, 192}, // bench Case 0
    {4, 96, 80, 80, 96},   // bench Case 1 (odd-ish)
    {2, 384, 20, 20, 96},  // bench Case 4 shape (IC large, not /16)
    {1, 192, 7, 9, 192},   // tiny odd spatial dims (edge tiles)
    {3, 33, 24, 24, 17},   // odd IC/OC
};

// Bit-exact correctness gate (both modes).
bool compare_case(const Shape& s, const WinogradConfig& cfg, const char* cfg_name) {
    const int OH = s.IH, OW = s.IW;
    const int HW = s.IH * s.IW, OHW = OH * OW;

    std::vector<float> src(s.N * s.IC * HW);
    std::vector<float> wei(s.OC * s.IC * 9);
    std::vector<float> bias(s.OC);
    for (auto& v : src) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : wei) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : bias) v = static_cast<float>(rand()) / RAND_MAX;

    std::vector<float> ref(s.N * s.OC * OHW, 0.0f);  // archived native NCHW
    std::vector<float> got(s.N * s.OC * OHW, 0.0f);  // new NCHW wrapper

    const float lo = -1e30f, hi = 1e30f;
    winograd_convolution_nchw_ref(src.data(), wei.data(), bias.data(), ref.data(),
                                  s.N, s.IC, s.IH, s.IW, s.OC, OH, OW, cfg, lo, hi);
    winograd_convolution(src.data(), wei.data(), bias.data(), got.data(),
                         s.N, s.IC, s.IH, s.IW, s.OC, OH, OW, cfg, lo, hi,
                         Layout::NCHW);

    int n_mismatch = 0;
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); i++) {
        float d = std::fabs(ref[i] - got[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (ref[i] != got[i]) n_mismatch++;   // must be bit-exact
    }
    if (n_mismatch > 0) {
        printf("  FAIL %s [%d,%d,%d,%d]/%d: %d/%zu elements differ, max|diff|=%g\n",
               cfg_name, s.N, s.IC, s.IH, s.IW, s.OC, n_mismatch, ref.size(), max_abs_diff);
        return false;
    }
    printf("  PASS %s [%d,%d,%d,%d]/%d: bit-exact, max|diff|=%g\n",
           cfg_name, s.N, s.IC, s.IH, s.IW, s.OC, max_abs_diff);
    return true;
}

// Best-of-repeats wall time for a callable, after warmup.
template <typename F>
double best_time_ms(F&& run, int warmup, int repeats) {
    for (int i = 0; i < warmup; i++) run();
    double best = 1e30;
    for (int i = 0; i < repeats; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int threads = 1, warmup = 2, repeats = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [--bench] [--threads T] [--warmup W] [--repeats R]\n", argv[0]);
            printf("  (no args)    bit-exact: NCHW wrapper vs archived native-NCHW kernel\n");
            printf("  --bench      also time both paths per case (wrapper vs native)\n");
            return 0;
        }
    }

#ifdef ENABLE_OPENMP
    omp_set_num_threads(threads);
#endif

    // Version banner: the pre--bench build has NO arg parsing and ignores
    // unknown flags, so it would silently run the same correctness test with
    // --bench passed. This line makes the two builds distinguishable on sight.
    printf("test_ref_vs_nchw --bench %s (threads=%d warmup=%d repeats=%d)\n\n",
           bench ? "ON" : "OFF", threads, warmup, repeats);

    srand(12345);

    // --- Correctness gate (always): bit-exact vs the archived native kernel ---
    int n_fail = 0, n_run = 0;
    for (const auto& s : kShapes) {
        n_run++;
        if (!compare_case(s, WinogradConfig::F44_33(), "F44")) n_fail++;
        n_run++;
        if (!compare_case(s, WinogradConfig::F22_33(), "F22")) n_fail++;
    }
    printf("%s: %d cases run, %d failed\n", (n_fail ? "FAILED" : "PASSED"), n_run, n_fail);
    if (n_fail) return 1;

    if (!bench) return 0;

    // --- Timing: wrapper vs archived native-NCHW kernel ---
    printf("\n=== Timing (best of %d after %d warmup, %d threads) ===\n",
           repeats, warmup, threads);
    printf("  ratio < 1 => conversion wrapper is FASTER than native NCHW\n");
    printf("%-18s %-4s %12s %12s %8s\n", "shape [N,IC,IH,IW,OC]", "cfg", "wrapper(ms)", "native(ms)", "ratio");

    double log_ratio_sum = 0.0;
    int n_ratio = 0;
    for (const auto& s : kShapes) {
        for (int c = 0; c < 2; c++) {
            const char* cfg_name = c ? "F22" : "F44";
            const WinogradConfig cfg = c ? WinogradConfig::F22_33() : WinogradConfig::F44_33();
            const int OH = s.IH, OW = s.IW;
            const int HW = s.IH * s.IW, OHW = OH * OW;

            std::vector<float> src(s.N * s.IC * HW);
            std::vector<float> wei(s.OC * s.IC * 9);
            std::vector<float> bias(s.OC);
            std::vector<float> w_dst(s.N * s.OC * OHW, 0.0f);
            std::vector<float> n_dst(s.N * s.OC * OHW, 0.0f);
            for (auto& v : src) v = static_cast<float>(rand()) / RAND_MAX;
            for (auto& v : wei) v = static_cast<float>(rand()) / RAND_MAX;
            for (auto& v : bias) v = static_cast<float>(rand()) / RAND_MAX;

            const float lo = -1e30f, hi = 1e30f;
            double w_ms = best_time_ms([&] {
                winograd_convolution(src.data(), wei.data(), bias.data(), w_dst.data(),
                                     s.N, s.IC, s.IH, s.IW, s.OC, OH, OW, cfg, lo, hi,
                                     Layout::NCHW);
            }, warmup, repeats);
            double n_ms = best_time_ms([&] {
                winograd_convolution_nchw_ref(src.data(), wei.data(), bias.data(), n_dst.data(),
                                              s.N, s.IC, s.IH, s.IW, s.OC, OH, OW, cfg, lo, hi);
            }, warmup, repeats);

            double ratio = n_ms > 0 ? w_ms / n_ms : 0.0;
            printf("%-18s %-4s %12.4f %12.4f %8.3f%s\n",
                   (std::to_string(s.N) + "," + std::to_string(s.IC) + "," +
                    std::to_string(s.IH) + "," + std::to_string(s.IW) + "," +
                    std::to_string(s.OC)).c_str(),
                   cfg_name, w_ms, n_ms, ratio,
                   ratio < 1.0 ? "  <--- wrapper faster" : "");
            log_ratio_sum += std::log(ratio);
            n_ratio++;
        }
    }
    if (n_ratio) {
        double geomean = std::exp(log_ratio_sum / n_ratio);
        printf("\nGeomean ratio (wrapper/native) over %d runs: %.3f\n", n_ratio, geomean);
        printf("  < 1.0 => conversion wrapper wins on this machine\n");
        printf("  > 1.0 => native NCHW kernel wins; wrapper should be reverted\n");
        printf("          (one-line change: Layout::NCHW -> winograd_convolution_nchw_ref)\n");
    }

    return 0;
}
