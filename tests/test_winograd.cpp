// test_winograd.cpp - Verification: Winograd vs direct convolution
//
// This program tests the Winograd convolution implementation by comparing
// its output against a reference direct convolution. It runs multiple
// test cases with random data and reports the maximum error.
//
// Usage:
//   ./test_winograd          # Run all tests
//   ./test_winograd --f22    # Only F(2,2,3,3)
//   ./test_winograd --f44    # Only F(4,4,3,3)
//   ./test_winograd --relu   # Test with ReLU activation
//
// Part of the winograd_conv project.

#include "../include/winograd_convolution.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>

using namespace winograd_conv;

// Fill a buffer with random floats in [0, 1)
void fill_random(std::vector<float>& buf) {
    for (auto& v : buf) {
        v = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// Compute max absolute difference between two arrays
float max_error(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = std::fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Run a single test case
bool run_test(int N, int IC, int IH, int IW, int OC,
              bool use_relu, bool use_f44) {
    int OH = IH;  // stride=1, pad=1
    int OW = IW;

    printf("  Test: N=%d IC=%d IH=%d IW=%d OC=%d OH=%d OW=%d %s %s\n",
           N, IC, IH, IW, OC, OH, OW,
           use_f44 ? "F(4,4,3,3)" : "F(2,2,3,3)",
           use_relu ? "+ReLU" : "");

    // Allocate and fill inputs
    int src_size = N * IC * IH * IW;
    int wei_size = OC * IC * 3 * 3;
    int dst_size = N * OC * OH * OW;
    std::vector<float> src(src_size), wei(wei_size);
    std::vector<float> bias(OC);
    fill_random(src);
    fill_random(wei);
    for (auto& b : bias) b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.5f;

    float act_min = use_relu ? 0.0f : -1e30f;
    float act_max = 1e30f;

    // Reference: direct convolution
    std::vector<float> ref_dst(dst_size, 0.0f);
    direct_convolution_3x3(src.data(), wei.data(), bias.data(), ref_dst.data(),
                           N, IC, IH, IW, OC, OH, OW, act_min, act_max);

    // Winograd convolution
    std::vector<float> wino_dst(dst_size, 0.0f);
    if (use_f44) {
        winograd_convolution_f44(src.data(), wei.data(), bias.data(), wino_dst.data(),
                                 N, IC, IH, IW, OC, OH, OW, act_min, act_max);
    } else {
        winograd_convolution_f22(src.data(), wei.data(), bias.data(), wino_dst.data(),
                                 N, IC, IH, IW, OC, OH, OW, act_min, act_max);
    }

    // Compare
    float err = max_error(ref_dst.data(), wino_dst.data(), dst_size);
    bool pass = (err < 1e-3f);  // tolerance for float32
    printf("    Max error: %.6f  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char** argv) {
    printf("=== Winograd Convolution Verification ===\n\n");

    bool test_f22 = true;
    bool test_f44 = true;
    bool test_relu = false;
    ISALevel isa = detect_isa();

    // Parse environment variable first
    if (const char* env = std::getenv("WINOGRAD_ISA")) {
        isa = parse_isa(env);
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--f22") { test_f44 = false; test_f22 = true; }
        else if (arg == "--f44") { test_f22 = false; test_f44 = true; }
        else if (arg == "--relu") { test_relu = true; }
        else if (arg == "--neon") { isa = ISALevel::NEON; }
        else if (arg == "--sve") { isa = ISALevel::SVE; }
        else if (arg == "--sme") { isa = ISALevel::SME; }
        else if (arg == "--help") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --f22    Test only F(2,2,3,3)\n");
            printf("  --f44    Test only F(4,4,3,3)\n");
            printf("  --relu   Test with ReLU activation\n");
            printf("  --neon   Force NEON transforms\n");
            printf("  --sve    Force SVE transforms\n");
            printf("  --sme    Force SME transforms\n");
            printf("  --help   Show this help\n");
            printf("Environment: WINOGRAD_ISA=neon|sve|sme\n");
            return 0;
        }
    }

    set_isa_level(isa);
    printf("Using ISA: %s (detected: %s)\n\n", isa_name(isa), isa_name(detect_isa()));

    // ---- Debug: check F(4,4,3,3) weight transform ----
    printf("--- F(4,4,3,3) Weight Transform Debug ---\n");
    {
        // g = [[1,0,0],[0,0,0],[0,0,0]] (1 channel)
        float g[9] = {1,0,0, 0,0,0, 0,0,0};
        float V[36];
        dispatch_weight_transform(g, V, 1, true, isa);

        // Expected: V[i][j] = G[i][0] * G[j][0] / 576
        // G[*][0] = [6, -4, -4, 1, 1, 0]
        float G0[6] = {6, -4, -4, 1, 1, 0};
        int errors = 0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                float expected = G0[i] * G0[j] / 576.0f;
                float got = V[i*6+j];
                if (fabs(got - expected) > 1e-6) {
                    printf("  V[%d][%d]: expected %.6f, got %.6f  ERROR\n", i, j, expected, got);
                    errors++;
                }
            }
        }
        if (errors == 0) printf("  Weight transform: PASS\n");
        else printf("  Weight transform: %d errors\n", errors);
    }

    // ---- Debug: check F(4,4,3,3) input transform ----
    printf("--- F(4,4,3,3) Input Transform Debug ---\n");
    {
        // d = identity (d[0][0]=1, rest=0), 1 channel
        float d[36] = {0};
        d[0] = 1.0f;  // d[0][0] = 1
        float U[36];
        dispatch_input_transform(d, U, 1, true, isa);

        // Expected: U[i][j] = B^T[i][0] * B^T[j][0]
        // (because only d[0][0] is non-zero)
        float Bt0[6] = {4, 0, -5, 0, 1, 0};  // column 0 of B^T = row 0 of B^T
        int errors = 0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                float expected = Bt0[i] * Bt0[j];  // B^T[i][0] * B^T[j][0]
                float got = U[i*6+j];
                if (fabs(got - expected) > 1e-4) {
                    printf("  U[%d][%d]: expected %.4f, got %.4f  ERROR\n", i, j, expected, got);
                    errors++;
                }
            }
        }
        if (errors == 0) printf("  Input transform: PASS\n");
        else printf("  Input transform: %d errors\n", errors);
    }

    // ---- Debug: check F(4,4,3,3) output transform ----
    printf("--- F(4,4,3,3) Output Transform Debug ---\n");
    {
        // M = identity (M[0][0]=1, rest=0), 1 channel, no bias
        float M[36] = {0};
        M[0] = 1.0f;  // M[0][0] = 1
        float f[16];
        dispatch_output_transform(M, f, 1, nullptr, -1e30f, 1e30f, true, isa);

        // Expected: f[i][j] = sum_k A^T[i][k] * M[k][l] * A^T[j][l]
        // With M[0][0]=1: f[i][j] = A^T[i][0] * A^T[j][0]
        // A^T[*][0] = [1, 0, 0, 0, 0, 0] (first column of A^T = first row of A^T)
        // Wait, A^T[i][0] is the (i,0) element of A^T = F44_A::val[i][0]
        float At0[4] = {1, 0, 0, 0};  // F44_A::val[*][0]
        int errors = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float expected = At0[i] * At0[j];
                float got = f[i*4+j];
                if (fabs(got - expected) > 1e-6) {
                    printf("  f[%d][%d]: expected %.6f, got %.6f  ERROR\n", i, j, expected, got);
                    errors++;
                }
            }
        }
        if (errors == 0) printf("  Output transform: PASS\n");
        else printf("  Output transform: %d errors\n", errors);
    }
    printf("\n");

    int total = 0, passed = 0;
    bool pass;

    printf("--- F(2,2,3,3) Tests ---\n");
    // Small test
    pass = run_test(1, 3, 4, 4, 3, test_relu, false); total++; passed += pass;
    pass = run_test(1, 4, 4, 4, 4, test_relu, false); total++; passed += pass;
    pass = run_test(1, 8, 8, 8, 8, test_relu, false); total++; passed += pass;
    pass = run_test(2, 16, 7, 7, 16, test_relu, false); total++; passed += pass;
    pass = run_test(1, 32, 14, 14, 32, test_relu, false); total++; passed += pass;
    pass = run_test(1, 64, 28, 28, 64, test_relu, false); total++; passed += pass;
    printf("\n");

    printf("--- F(4,4,3,3) Tests ---\n");
    pass = run_test(1, 3, 4, 4, 3, test_relu, true); total++; passed += pass;
    pass = run_test(1, 4, 8, 8, 4, test_relu, true); total++; passed += pass;
    pass = run_test(1, 8, 8, 8, 8, test_relu, true); total++; passed += pass;
    pass = run_test(2, 16, 8, 8, 16, test_relu, true); total++; passed += pass;
    pass = run_test(1, 32, 16, 16, 32, test_relu, true); total++; passed += pass;
    pass = run_test(1, 64, 28, 28, 64, test_relu, true); total++; passed += pass;
    printf("\n");

    printf("=== Summary: %d/%d tests passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
