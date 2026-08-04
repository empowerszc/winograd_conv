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

        // U[i][j] = B^T[i][0] * B^T[j][0] (column 0 of B^T)
        // B^T[*][0] = first element of each row = [4, 0, 0, 0, 0, 0]
        float Bt_col0[6] = {4, 0, 0, 0, 0, 0};
        int errors = 0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                float expected = Bt_col0[i] * Bt_col0[j];
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

    // ---- Debug: check F(4,4,3,3) full pipeline (1D-equivalent) ----
    printf("--- F(4,4,3,3) Full Pipeline Debug ---\n");
    {
        // d_tile[1][1] = 1 (simulates input[0][0]=1 with pad=1), g=[[1,0,0],...]
        // Expected: y[1][1] = 1, all else 0
        float d_tile[36] = {0};
        d_tile[1 * 6 + 1] = 1.0f;  // d[1][1] = 1
        float U_tile[36];
        dispatch_input_transform(d_tile, U_tile, 1, true, isa);

        float g[9] = {1,0,0, 0,0,0, 0,0,0};
        float V_oc[36];
        dispatch_weight_transform(g, V_oc, 1, true, isa);

        // M = U ⊙ V
        float M_tile[36];
        for (int m = 0; m < 36; m++)
            M_tile[m] = U_tile[m] * V_oc[m];

        // f = A^T * M * A
        float f_tile[16];
        dispatch_output_transform(M_tile, f_tile, 1, nullptr, -1e30f, 1e30f, true, isa);

        int errors = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float expected = (i == 1 && j == 1) ? 1.0f : 0.0f;
                float got = f_tile[i*4+j];
                if (fabs(got - expected) > 1e-3) {
                    printf("  f[%d][%d]: expected %.4f, got %.4f  ERROR\n", i, j, expected, got);
                    errors++;
                }
            }
        }
        if (errors == 0) printf("  Full pipeline (direct): PASS\n");
        else printf("  Full pipeline (direct): %d errors\n", errors);
    }

    // ---- Debug: check F(4,4,3,3) end-to-end via winograd_convolution ----
    printf("--- F(4,4,3,3) End-to-End Debug ---\n");
    {
        // IC=1, OC=1, IH=4, IW=4
        // Input: all zeros except input[0][0]=1
        // Weight: g[0][0][0][0]=1, rest=0 (identity)
        // Expected: y[1][1] = 1, all else 0 (with pad=1, stride=1)
        float src[4] = {1,0,0,0};   // [1][1][4][4] but only need 4 values for IC=1
        float wei[9] = {1,0,0, 0,0,0, 0,0,0};  // [1][1][3][3]
        float bias_arr[1] = {0};
        float dst[4] = {0,0,0,0};   // [1][1][4][4]

        winograd_convolution_f44(src, wei, bias_arr, dst,
                                  1, 1, 4, 4, 1, 4, 4,
                                  -1e30f, 1e30f);

        int errors = 0;
        for (int i = 0; i < 4; i++) {
            float expected = (i == 1) ? 1.0f : 0.0f;
            if (fabs(dst[i] - expected) > 1e-3) {
                printf("  dst[%d]: expected %.4f, got %.4f  ERROR\n", i, expected, dst[i]);
                errors++;
            }
        }
        if (errors == 0) printf("  End-to-end (IC=1): PASS\n");
        else printf("  End-to-end (IC=1): %d errors\n", errors);
    }

    // ---- Debug: check F(4,4,3,3) end-to-end with IC=3 ----
    printf("--- F(4,4,3,3) End-to-End IC=3 Debug ---\n");
    {
        // IC=3, OC=3, IH=4, IW=4
        // Input: all zeros except input[0][ic][0][0]=1 for all ic
        // Weight: wei[oc][ic][0][0]=1, rest=0 (each oc picks sum of all ic)
        // Expected: y[1][1][oc] = 3 (sum of 3 input channels), all else 0
        int IC=3, OC=3, IH=4, IW=4;
        std::vector<float> src(IC * IH * IW, 0.0f);
        for (int ic = 0; ic < IC; ic++)
            src[ic * IH * IW + 0 * IW + 0] = 1.0f;  // input[0][ic][0][0] = 1

        std::vector<float> wei(OC * IC * 9, 0.0f);
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                wei[(oc * IC + ic) * 9 + 0] = 1.0f;  // wei[oc][ic][0][0] = 1

        std::vector<float> bias(OC, 0.0f);
        std::vector<float> dst(OC * IH * IW, 0.0f);

        winograd_convolution_f44(src.data(), wei.data(), bias.data(), dst.data(),
                                  1, IC, IH, IW, OC, IH, IW,
                                  -1e30f, 1e30f);

        int errors = 0;
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < IH; oh++) {
                for (int ow = 0; ow < IW; ow++) {
                    float expected = (oh == 1 && ow == 1) ? (float)IC : 0.0f;
                    float got = dst[oc * IH * IW + oh * IW + ow];
                    if (fabs(got - expected) > 1e-2) {
                        if (errors < 10)  // limit output
                            printf("  dst[oc=%d][%d][%d]: expected %.4f, got %.4f  ERROR\n",
                                   oc, oh, ow, expected, got);
                        errors++;
                    }
                }
            }
        }
        if (errors == 0) printf("  End-to-end (IC=3): PASS\n");
        else printf("  End-to-end (IC=3): %d errors\n", errors);
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
