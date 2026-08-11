// test_ref_vs_nchw.cpp - Verify the NCHW wrapper matches the archived native-NCHW kernel
//
// The 2026-08-11 layout refactor re-implemented the NCHW path of
// winograd_convolution() as: NCHW -> NHWC convert -> shared NHWC compute core
// -> NCHW convert back. The pre-refactor native-NCHW kernel is archived in
// ref/winograd_conv_nchw_ref.cpp.
//
// The conversion is a pure data move (no arithmetic), and both paths place the
// exact same values into the same d_tile/M_tile buffers, so the two
// implementations must agree BIT-EXACTLY. This test asserts that.

#include "winograd_convolution.hpp"
#include "winograd_conv_nchw_ref.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

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

} // namespace

int main() {
    srand(12345);
    int n_fail = 0, n_run = 0;
    for (const auto& s : kShapes) {
        n_run++;
        if (!compare_case(s, WinogradConfig::F44_33(), "F44")) n_fail++;
        n_run++;
        if (!compare_case(s, WinogradConfig::F22_33(), "F22")) n_fail++;
    }
    printf("%s: %d cases run, %d failed\n", (n_fail ? "FAILED" : "PASSED"), n_run, n_fail);
    return n_fail ? 1 : 0;
}
