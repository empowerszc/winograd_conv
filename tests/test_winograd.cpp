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
        int IC=1, OC=1, IH=4, IW=4;
        std::vector<float> src(IC * IH * IW, 0.0f);
        src[0] = 1.0f;  // input[0][0][0][0] = 1
        std::vector<float> wei(OC * IC * 9, 0.0f);
        wei[0] = 1.0f;  // wei[0][0][0][0] = 1
        std::vector<float> bias(OC, 0.0f);
        std::vector<float> dst(OC * IH * IW, 0.0f);

        winograd_convolution_f44(src.data(), wei.data(), bias.data(), dst.data(),
                                  1, IC, IH, IW, OC, IH, IW,
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
        int IC=3, OC=3, IH=4, IW=4;
        std::vector<float> src(IC * IH * IW, 0.0f);
        for (int ic = 0; ic < IC; ic++)
            src[ic * IH * IW + 0 * IW + 0] = 1.0f;

        std::vector<float> wei(OC * IC * 9, 0.0f);
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                wei[(oc * IC + ic) * 9 + 0] = 1.0f;

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
                        if (errors < 10)
                            printf("  dst[oc=%d][%d][%d]: expected %.4f, got %.4f  ERROR\n",
                                   oc, oh, ow, expected, got);
                        errors++;
                    }
                }
            }
        }
        if (errors == 0) printf("  End-to-end (IC=3, identity kernel): PASS\n");
        else printf("  End-to-end (IC=3, identity kernel): %d errors\n", errors);
    }

    // ---- Debug: F(4,4,3,3) with IDENTITY kernel + RANDOM input ----
    printf("--- F(4,4,3,3) Identity Kernel + Random Input Debug ---\n");
    {
        int IC=3, OC=3, IH=4, IW=4;
        std::vector<float> src(IC * IH * IW);
        fill_random(src);

        std::vector<float> wei(OC * IC * 9, 0.0f);
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                wei[(oc * IC + ic) * 9 + 0] = 1.0f;  // identity kernel

        std::vector<float> bias(OC, 0.0f);
        std::vector<float> ref_dst(OC * IH * IW, 0.0f);
        std::vector<float> wino_dst(OC * IH * IW, 0.0f);

        direct_convolution_3x3(src.data(), wei.data(), bias.data(), ref_dst.data(),
                                1, IC, IH, IW, OC, IH, IW, -1e30f, 1e30f);
        winograd_convolution_f44(src.data(), wei.data(), bias.data(), wino_dst.data(),
                                  1, IC, IH, IW, OC, IH, IW, -1e30f, 1e30f);

        float err = max_error(ref_dst.data(), wino_dst.data(), OC * IH * IW);
        printf("  Max error: %.6f  %s\n", err, err < 1e-3 ? "PASS" : "FAIL");
    }

    // ---- Debug: F(4,4,3,3) with RANDOM kernel + delta input ----
    printf("--- F(4,4,3,3) Random Kernel + Delta Input Debug ---\n");
    {
        int IC=3, OC=3, IH=4, IW=4;
        std::vector<float> src(IC * IH * IW, 0.0f);
        for (int ic = 0; ic < IC; ic++)
            src[ic * IH * IW + 0 * IW + 0] = 1.0f;  // delta at [0][0]

        std::vector<float> wei(OC * IC * 9);
        fill_random(wei);

        std::vector<float> bias(OC, 0.0f);
        std::vector<float> ref_dst(OC * IH * IW, 0.0f);
        std::vector<float> wino_dst(OC * IH * IW, 0.0f);

        direct_convolution_3x3(src.data(), wei.data(), bias.data(), ref_dst.data(),
                                1, IC, IH, IW, OC, IH, IW, -1e30f, 1e30f);
        winograd_convolution_f44(src.data(), wei.data(), bias.data(), wino_dst.data(),
                                  1, IC, IH, IW, OC, IH, IW, -1e30f, 1e30f);

        float err = max_error(ref_dst.data(), wino_dst.data(), OC * IH * IW);
        printf("  Max error: %.6f  %s\n", err, err < 1e-3 ? "PASS" : "FAIL");

        // Dump first few output values to see where they differ
        for (int oc = 0; oc < OC && oc < 2; oc++) {
            for (int oh = 0; oh < 4; oh++) {
                int ow = 0;
                int idx = oc * IH * IW + oh * IW + ow;
                printf("  oc=%d oh=%d ow=%d: ref=%.6f wino=%.6f diff=%.6f\n",
                       oc, oh, ow, ref_dst[idx], wino_dst[idx],
                       fabs(ref_dst[idx] - wino_dst[idx]));
            }
        }

        // Also compare with F(2,2,3,3) using same data to check if the
        // direct convolution and Winograd agree for F(2,2) but not F(4,4)
        std::vector<float> ref22_dst(OC * IH * IW, 0.0f);
        std::vector<float> wino22_dst(OC * IH * IW, 0.0f);
        direct_convolution_3x3(src.data(), wei.data(), bias.data(), ref22_dst.data(),
                                1, IC, IH, IW, OC, IH, IW, -1e30f, 1e30f);
        winograd_convolution_f22(src.data(), wei.data(), bias.data(), wino22_dst.data(),
                                  1, IC, IH, IW, OC, IH, IW, -1e30f, 1e30f);
        float err22 = max_error(ref22_dst.data(), wino22_dst.data(), OC * IH * IW);
        printf("  F(2,2) same data: %.6f  %s\n", err22, err22 < 1e-3 ? "PASS" : "FAIL");
    }

    // ---- Debug: manual pipeline step-by-step for F(4,4,3,3) ----
    printf("--- F(4,4,3,3) Manual Pipeline Step-by-Step ---\n");
    {
        int IC=3, OC=3, IH=4, IW=4;
        int TS=6, OT=4, NM=36;
        std::vector<float> src(IC * IH * IW, 0.0f);
        for (int ic = 0; ic < IC; ic++)
            src[ic * IH * IW + 0 * IW + 0] = 1.0f;

        std::vector<float> wei(OC * IC * 9);
        fill_random(wei);
        std::vector<float> bias(OC, 0.0f);

        // Step 1: Weight transform (same as winograd_convolution)
        std::vector<float> V(NM * OC * IC, 0.0f);
        for (int oc = 0; oc < OC; oc++) {
            std::vector<float> g(9 * IC);
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        g[(kh * 3 + kw) * IC + ic] =
                            wei[((oc * IC + ic) * 3 + kh) * 3 + kw];
            std::vector<float> V_oc(NM * IC);
            dispatch_weight_transform(g.data(), V_oc.data(), IC, true, isa);
            for (int m = 0; m < NM; m++)
                for (int ic = 0; ic < IC; ic++)
                    V[m * OC * IC + oc * IC + ic] = V_oc[m * IC + ic];
        }

        // Step 2: Tile extraction + input transform
        int n_tiles = 1;
        int tile_idx = 0;
        std::vector<float> d_tile(TS * TS * IC, 0.0f);
        for (int ti = 0; ti < TS; ti++) {
            for (int tj = 0; tj < TS; tj++) {
                int ih = -1 + ti;  // tr=0
                int iw = -1 + tj;  // tc=0
                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                    for (int ic = 0; ic < IC; ic++)
                        d_tile[(ti * TS + tj) * IC + ic] =
                            src[((0 * IC + ic) * IH + ih) * IW + iw];
                }
            }
        }

        std::vector<float> U_tile(TS * TS * IC);
        dispatch_input_transform(d_tile.data(), U_tile.data(), IC, true, isa);

        // Step 3: Scatter U
        std::vector<float> U(NM * n_tiles * IC, 0.0f);
        for (int ti = 0; ti < TS; ti++) {
            for (int tj = 0; tj < TS; tj++) {
                int ts_idx = ti * TS + tj;
                for (int ic = 0; ic < IC; ic++)
                    U[(ts_idx * n_tiles + tile_idx) * IC + ic] =
                        U_tile[(ti * TS + tj) * IC + ic];
            }
        }

        // Step 4: GEMM
        std::vector<float> M_buf(NM * n_tiles * OC, 0.0f);
        for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
            const float* U_slice = U.data() + ts_idx * n_tiles * IC;
            const float* V_slice = V.data() + ts_idx * OC * IC;
            float* M_slice = M_buf.data() + ts_idx * n_tiles * OC;
            winograd_gemm(U_slice, V_slice, M_slice, n_tiles, OC, IC);
        }

        // Step 5: Gather M
        std::vector<float> M_tile(TS * TS * OC, 0.0f);
        for (int ti = 0; ti < TS; ti++) {
            for (int tj = 0; tj < TS; tj++) {
                int ts_idx = ti * TS + tj;
                for (int oc = 0; oc < OC; oc++)
                    M_tile[(ti * TS + tj) * OC + oc] =
                        M_buf[(ts_idx * n_tiles + tile_idx) * OC + oc];
            }
        }

        // Step 6: Output transform
        std::vector<float> f_tile(OT * OT * OC, 0.0f);
        dispatch_output_transform(M_tile.data(), f_tile.data(), OC,
                                  bias.data(), -1e30f, 1e30f, true, isa);

        // Step 7: Writeback
        std::vector<float> manual_dst(OC * IH * IW, 0.0f);
        for (int oi = 0; oi < OT; oi++) {
            for (int oj = 0; oj < OT; oj++) {
                int oh = oi, ow = oj;
                if (oh < IH && ow < IW) {
                    for (int oc = 0; oc < OC; oc++)
                        manual_dst[oc * IH * IW + oh * IW + ow] =
                            f_tile[(oi * OT + oj) * OC + oc];
                }
            }
        }

        // Compare manual pipeline with winograd_convolution_f44
        std::vector<float> wino_dst(OC * IH * IW, 0.0f);
        winograd_convolution_f44(src.data(), wei.data(), bias.data(), wino_dst.data(),
                                  1, IC, IH, IW, OC, IH, IW, -1e30f, 1e30f);

        // Compare manual pipeline with direct convolution
        std::vector<float> ref_dst(OC * IH * IW, 0.0f);
        direct_convolution_3x3(src.data(), wei.data(), bias.data(), ref_dst.data(),
                                1, IC, IH, IW, OC, IH, IW, -1e30f, 1e30f);

        float err_manual_vs_wino = max_error(manual_dst.data(), wino_dst.data(), OC * IH * IW);
        float err_manual_vs_ref = max_error(manual_dst.data(), ref_dst.data(), OC * IH * IW);
        float err_wino_vs_ref = max_error(wino_dst.data(), ref_dst.data(), OC * IH * IW);

        printf("  Manual vs Winograd: %.6f  %s\n", err_manual_vs_wino, err_manual_vs_wino < 1e-3 ? "PASS" : "FAIL");
        printf("  Manual vs Direct:   %.6f  %s\n", err_manual_vs_ref, err_manual_vs_ref < 1e-3 ? "PASS" : "FAIL");
        printf("  Winograd vs Direct: %.6f  %s\n", err_wino_vs_ref, err_wino_vs_ref < 1e-3 ? "PASS" : "FAIL");

        // Dump first few values
        for (int oc = 0; oc < OC && oc < 2; oc++) {
            for (int oh = 0; oh < 4; oh++) {
                int idx = oc * IH * IW + oh * IW + 0;
                printf("  oc=%d oh=%d: ref=%.6f manual=%.6f wino=%.6f\n",
                       oc, oh, ref_dst[idx], manual_dst[idx], wino_dst[idx]);
            }
        }
    }

    // ---- Debug: scalar reference weight transform comparison ----
    printf("--- F(4,4,3,3) Scalar vs NEON Weight Transform ---\n");
    {
        // Random weight, IC=3
        int IC = 3;
        float g[27];  // [3][3][3]
        for (int i = 0; i < 27; i++) g[i] = static_cast<float>(rand()) / RAND_MAX;

        // NEON dispatch
        float V_neon[36 * 3];  // [36][3]
        dispatch_weight_transform(g, V_neon, IC, true, ISALevel::NEON);

        // Scalar reference: V[i][j][c] = sum_k1 sum_k2 G[i][k1]*g[k1][k2][c]*G[j][k2] / 576
        float V_scalar[36 * 3];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int c = 0; c < IC; c++) {
                    float val = 0.0f;
                    for (int k1 = 0; k1 < 3; k1++) {
                        for (int k2 = 0; k2 < 3; k2++) {
                            val += F44_G::val[i][k1] * g[(k1*3+k2)*IC + c] * F44_G::val[j][k2];
                        }
                    }
                    V_scalar[(i*6+j)*IC + c] = val / 576.0f;
                }
            }
        }

        float err = max_error(V_neon, V_scalar, 36 * IC);
        printf("  Max error: %.6f  %s\n", err, err < 1e-5 ? "PASS" : "FAIL");
        if (err >= 1e-5) {
            for (int m = 0; m < 36; m++) {
                for (int c = 0; c < IC; c++) {
                    if (fabs(V_neon[m*IC+c] - V_scalar[m*IC+c]) > 1e-5) {
                        printf("  V[%d][%d]: neon=%.6f scalar=%.6f\n", m, c,
                               V_neon[m*IC+c], V_scalar[m*IC+c]);
                    }
                }
            }
        }
    }

    // ---- Debug: compute correct B^T from Winograd condition ----
    printf("--- F(4,4,3,3) Solve Correct B^T ---\n");
    {
        // Winograd 1D condition:
        // sum_j A^T[i][j] * B^T[j][a] * G[j][b] = 24 * delta(a, i+b)
        // For each column a, solve for B^T[0..5][a]
        // Using d=e_a, g=e_b: y[i] = delta(a-b, i)

        float At[4][6], Gm[6][3];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 6; j++) At[i][j] = F44_A::val[i][j];
        for (int i = 0; i < 6; i++) for (int j = 0; j < 3; j++) Gm[i][j] = F44_G::val[i][j];

        float Bt_correct[6][6] = {0};

        for (int a = 0; a < 6; a++) {
            // For each a, solve: sum_j (A^T[i][j] * G[j][b]) * B^T[j][a] = 24*delta(a, i+b)
            // This gives 12 equations (i=0..3, b=0..2) for 6 unknowns (j=0..5)
            // Use least-squares (normal equations): M^T * M * x = M^T * r

            double MtM[6][6] = {0};
            double Mtr[6] = {0};

            for (int i = 0; i < 4; i++) {
                for (int b = 0; b < 3; b++) {
                    // Equation: sum_j M[i*3+b][j] * x[j] = r[i*3+b]
                    // where M[i*3+b][j] = A^T[i][j] * G[j][b]
                    // and r[i*3+b] = 24 * delta(a, i+b)
                    double Mrow[6];
                    for (int j = 0; j < 6; j++)
                        Mrow[j] = At[i][j] * Gm[j][b];
                    double rval = (a == i + b) ? 24.0 : 0.0;

                    for (int j1 = 0; j1 < 6; j1++) {
                        Mtr[j1] += Mrow[j1] * rval;
                        for (int j2 = 0; j2 < 6; j2++)
                            MtM[j1][j2] += Mrow[j1] * Mrow[j2];
                    }
                }
            }

            // Solve MtM * x = Mtr using Gaussian elimination
            double aug[6][7];
            for (int r = 0; r < 6; r++) {
                for (int c = 0; c < 6; c++) aug[r][c] = MtM[r][c];
                aug[r][6] = Mtr[r];
            }
            // Forward elimination
            for (int p = 0; p < 6; p++) {
                // Find pivot
                int best = p;
                for (int r = p+1; r < 6; r++)
                    if (fabs(aug[r][p]) > fabs(aug[best][p])) best = r;
                if (best != p) for (int c = 0; c <= 6; c++) std::swap(aug[p][c], aug[best][c]);
                // Eliminate
                for (int r = p+1; r < 6; r++) {
                    double f = aug[r][p] / aug[p][p];
                    for (int c = p; c <= 6; c++) aug[r][c] -= f * aug[p][c];
                }
            }
            // Back substitution
            double x[6];
            for (int r = 5; r >= 0; r--) {
                x[r] = aug[r][6];
                for (int c = r+1; c < 6; c++) x[r] -= aug[r][c] * x[c];
                x[r] /= aug[r][r];
            }

            for (int j = 0; j < 6; j++)
                Bt_correct[j][a] = (float)x[j];
        }

        printf("  Correct B^T:\n");
        for (int i = 0; i < 6; i++) {
            printf("  {");
            for (int j = 0; j < 6; j++) printf("%s%.1f", j ? ", " : "", Bt_correct[i][j]);
            printf("}\n");
        }
        printf("  Current B^T:\n");
        for (int i = 0; i < 6; i++) {
            printf("  {");
            for (int j = 0; j < 6; j++) printf("%s%.1f", j ? ", " : "", F44_Bt::val[i][j]);
            printf("}\n");
        }

        // Verify: use correct B^T and check 1D Winograd
        float err = 0;
        for (int a = 0; a < 6; a++) {
            for (int b = 0; b < 3; b++) {
                for (int i = 0; i < 4; i++) {
                    float val = 0;
                    for (int j = 0; j < 6; j++)
                        val += At[i][j] * Bt_correct[j][a] * Gm[j][b];
                    float expected = (a == i+b) ? 24.0f : 0.0f;
                    err = std::max(err, (float)fabs(val - expected));
                }
            }
        }
        printf("  Verification error: %.6f  %s\n", err, err < 1e-3 ? "PASS" : "FAIL");
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
