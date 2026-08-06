// bench_winograd.cpp - Performance benchmark for Winograd convolution
//
// Reads shapes from a CSV file, filters for stride=1, group=1, 3x3 kernel,
// and benchmarks F(4,4,3,3) Winograd convolution across different thread counts.
//
// Usage:
//   ./bench_winograd [options] shapes.csv
//
// Options:
//   --neon / --sve / --sme   Select ISA
//   --warmup N               Warmup iterations (default 3)
//   --repeats N              Timed iterations (default 10)
//   --threads t1,t2,...      Thread counts to test (default: 1,8,16,32,38)
//   --output result.csv      Write results to CSV file
//   --timing                 Fine-grained per-step timing mode
//   --help                   Show help
//
// Input CSV format (comma-separated, first line is header):
//   mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count
//
// Part of the winograd_conv project.

#include "../include/winograd_convolution.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace winograd_conv;

struct ShapeInfo {
    int mb, ic, ih, iw, oc, kh, kw;
    int stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, grp, count;
    bool valid;
    std::string skip_reason;
};

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    for (char ch : line) {
        if (ch == ',') { fields.push_back(field); field.clear(); }
        else if (ch != '\r' && ch != '\n') field += ch;
    }
    if (!field.empty()) fields.push_back(field);
    return fields;
}

bool read_shapes(const std::string& filename, std::vector<ShapeInfo>& shapes) {
    std::ifstream fin(filename);
    if (!fin) {
        fprintf(stderr, "Cannot open %s\n", filename.c_str());
        return false;
    }
    std::string line;
    int line_no = 0;
    std::getline(fin, line); // header
    line_no++;
    while (std::getline(fin, line)) {
        line_no++;
        // Trim whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
            line.pop_back();
        if (line.empty()) continue;

        auto f = split_csv(line);
        if (f.size() < 15) {
            fprintf(stderr, "  Line %d: only %zu fields (need 15), skipped\n", line_no, f.size());
            continue;
        }

        ShapeInfo s;
        s.mb = atoi(f[0].c_str());
        s.ic = atoi(f[1].c_str());
        s.ih = atoi(f[2].c_str());
        s.iw = atoi(f[3].c_str());
        s.oc = atoi(f[4].c_str());
        s.kh = atoi(f[5].c_str());
        s.kw = atoi(f[6].c_str());
        s.stride_h = atoi(f[7].c_str());
        s.stride_w = atoi(f[8].c_str());
        s.pad_h = atoi(f[9].c_str());
        s.pad_w = atoi(f[10].c_str());
        s.dil_h = atoi(f[11].c_str());
        s.dil_w = atoi(f[12].c_str());
        s.grp = atoi(f[13].c_str());
        s.count = atoi(f[14].c_str());

        s.valid = true;
        s.skip_reason = "";
        if (s.stride_h != 1 || s.stride_w != 1) {
            s.valid = false;
            s.skip_reason = "stride!=1";
        } else if (s.grp != 1) {
            s.valid = false;
            s.skip_reason = "group!=1";
        } else if (s.kh != 3 || s.kw != 3) {
            s.valid = false;
            s.skip_reason = "not 3x3";
        } else if (s.pad_h != 1 || s.pad_w != 1) {
            s.valid = false;
            s.skip_reason = "pad!=1";
        }
        shapes.push_back(s);
    }
    return true;
}

// Fine-grained timing: manually run each step
struct StepTimings {
    double weight_transform_ms;
    double input_transform_ms;
    double gemm_ms;
    double output_transform_ms;
    double total_ms;
};

StepTimings run_with_timing(
    const float* src, const float* wei, const float* bias, float* dst,
    int N, int IC, int IH, int IW, int OC, int OH, int OW,
    ISALevel isa
) {
    const int TS = 6, OT = 4, NM = 36;
    const bool is_f44 = true;

    // Step 1: Weight transform
    auto t0 = std::chrono::high_resolution_clock::now();
    int V_size = TS * TS * OC * IC;
    std::vector<float> V(V_size, 0.0f);
    for (int oc = 0; oc < OC; oc++) {
        std::vector<float> g(9 * IC);
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[(kh * 3 + kw) * IC + ic] =
                        wei[((oc * IC + ic) * 3 + kh) * 3 + kw];
        std::vector<float> V_oc(TS * TS * IC);
        dispatch_weight_transform(g.data(), V_oc.data(), IC, is_f44, isa);
        for (int m = 0; m < TS * TS; m++)
            for (int ic = 0; ic < IC; ic++)
                V[m * OC * IC + oc * IC + ic] = V_oc[m * IC + ic];
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // Pre-allocate workspace
    int n_tile_rows = (OH + OT - 1) / OT;
    int n_tile_cols = (OW + OT - 1) / OT;
    int n_tiles = n_tile_rows * n_tile_cols;
    std::vector<float> U(NM * n_tiles * IC, 0.0f);
    std::vector<float> M_buf(NM * n_tiles * OC, 0.0f);
    std::vector<float> d_tile(TS * TS * IC, 0.0f);
    std::vector<float> U_tile(TS * TS * IC, 0.0f);
    std::vector<float> M_tile(TS * TS * OC, 0.0f);
    std::vector<float> f_tile(OT * OT * OC, 0.0f);

    // Step 2: Input transform + tile extraction
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < N; n++) {
        for (int tr = 0; tr < n_tile_rows; tr++) {
            for (int tc = 0; tc < n_tile_cols; tc++) {
                int tile_idx = tr * n_tile_cols + tc;
                std::fill(d_tile.begin(), d_tile.end(), 0.0f);
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ih = tr * OT - 1 + ti;
                        int iw = tc * OT - 1 + tj;
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            for (int ic = 0; ic < IC; ic++)
                                d_tile[(ti * TS + tj) * IC + ic] =
                                    src[((n * IC + ic) * IH + ih) * IW + iw];
                        }
                    }
                }
                dispatch_input_transform(d_tile.data(), U_tile.data(), IC, is_f44, isa);
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ts_idx = ti * TS + tj;
                        float* dst_ptr = U.data() + (ts_idx * n_tiles + tile_idx) * IC;
                        const float* src_ptr = U_tile.data() + (ti * TS + tj) * IC;
                        int ic = 0;
                        for (; ic + 4 <= IC; ic += 4)
                            vst1q_f32(dst_ptr + ic, vld1q_f32(src_ptr + ic));
                        for (; ic < IC; ic++)
                            dst_ptr[ic] = src_ptr[ic];
                    }
                }
            }
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    // Step 3: GEMM
    for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
        const float* U_slice = U.data() + ts_idx * n_tiles * IC;
        const float* V_slice = V.data() + ts_idx * OC * IC;
        float* M_slice = M_buf.data() + ts_idx * n_tiles * OC;
        winograd_gemm(U_slice, V_slice, M_slice, n_tiles, OC, IC);
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // Step 4: Output transform
    for (int n = 0; n < N; n++) {
        for (int tr = 0; tr < n_tile_rows; tr++) {
            for (int tc = 0; tc < n_tile_cols; tc++) {
                int tile_idx = tr * n_tile_cols + tc;
                for (int ti = 0; ti < TS; ti++) {
                    for (int tj = 0; tj < TS; tj++) {
                        int ts_idx = ti * TS + tj;
                        const float* src_ptr = M_buf.data() + (ts_idx * n_tiles + tile_idx) * OC;
                        float* dst_ptr = M_tile.data() + (ti * TS + tj) * OC;
                        int oc = 0;
                        for (; oc + 4 <= OC; oc += 4)
                            vst1q_f32(dst_ptr + oc, vld1q_f32(src_ptr + oc));
                        for (; oc < OC; oc++)
                            dst_ptr[oc] = src_ptr[oc];
                    }
                }
                dispatch_output_transform(M_tile.data(), f_tile.data(), OC,
                                          bias, -1e30f, 1e30f, is_f44, isa);
                for (int oi = 0; oi < OT; oi++) {
                    for (int oj = 0; oj < OT; oj++) {
                        int oh = tr * OT + oi;
                        int ow = tc * OT + oj;
                        if (oh < OH && ow < OW) {
                            for (int oc = 0; oc < OC; oc++)
                                dst[((n * OC + oc) * OH + oh) * OW + ow] =
                                    f_tile[(oi * OT + oj) * OC + oc];
                        }
                    }
                }
            }
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    StepTimings st;
    st.weight_transform_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    st.input_transform_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    st.gemm_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    st.output_transform_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    st.total_ms = std::chrono::duration<double, std::milli>(t5 - t0).count();
    return st;
}

int main(int argc, char** argv) {
    ISALevel isa = detect_isa();
    std::string input_file;
    std::string output_file;
    int warmup = 3;
    int repeats = 10;
    bool timing_mode = false;
    std::vector<int> thread_counts = {1, 8, 16, 32, 38};

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--neon") isa = ISALevel::NEON;
        else if (arg == "--sve") isa = ISALevel::SVE;
        else if (arg == "--sme") isa = ISALevel::SME;
        else if (arg == "--warmup" && i+1 < argc) { warmup = atoi(argv[++i]); }
        else if (arg == "--repeats" && i+1 < argc) { repeats = atoi(argv[++i]); }
        else if (arg == "--timing") { timing_mode = true; }
        else if (arg == "--threads" && i+1 < argc) {
            thread_counts.clear();
            std::string t = argv[++i];
            std::string num;
            for (char ch : t) {
                if (ch == ',') { if (!num.empty()) { thread_counts.push_back(atoi(num.c_str())); num.clear(); } }
                else num += ch;
            }
            if (!num.empty()) thread_counts.push_back(atoi(num.c_str()));
        }
        else if (arg == "--output" && i+1 < argc) { output_file = argv[++i]; }
        else if (arg == "--help") {
            printf("Usage: %s [options] shapes.csv\n", argv[0]);
            printf("Options:\n");
            printf("  --neon / --sve / --sme   Select ISA\n");
            printf("  --warmup N               Warmup iterations (default 3)\n");
            printf("  --repeats N              Timed iterations (default 10)\n");
            printf("  --threads t1,t2,...      Thread counts (default 1,8,16,32,38)\n");
            printf("  --output result.csv      Write results to CSV file\n");
            printf("  --timing                 Fine-grained per-step timing\n");
            return 0;
        } else {
            input_file = arg;
        }
    }

    if (const char* env = std::getenv("WINOGRAD_ISA")) {
        isa = parse_isa(env);
    }
    set_isa_level(isa);

    // OpenMP diagnostic
#ifdef ENABLE_OPENMP
    printf("OpenMP: ENABLE_OPENMP defined, omp.h included\n");
    printf("OpenMP: max threads = %d\n", omp_get_max_threads());
    omp_set_num_threads(16);
    printf("OpenMP: after set(16), max threads = %d\n", omp_get_max_threads());
    #pragma omp parallel
    {
        #pragma omp single
        printf("OpenMP: inside parallel region, actual threads = %d\n", omp_get_num_threads());
    }
#else
    printf("OpenMP: NOT compiled (ENABLE_OPENMP not defined)\n");
#endif

    if (input_file.empty()) {
        fprintf(stderr, "Usage: %s [options] shapes.csv\n", argv[0]);
        return 1;
    }

    std::vector<ShapeInfo> all_shapes;
    if (!read_shapes(input_file, all_shapes)) return 1;

    // Print parsing summary
    int n_valid = 0, n_skipped = 0;
    for (const auto& s : all_shapes) {
        if (s.valid) n_valid++;
        else { n_skipped++; }
    }
    printf("ISA: %s (detected: %s)\n", isa_name(isa), isa_name(detect_isa()));
    printf("Parsed %zu shapes from %s (%d valid, %d skipped)\n",
           all_shapes.size(), input_file.c_str(), n_valid, n_skipped);
    for (const auto& s : all_shapes) {
        if (!s.valid) {
            printf("  Skipped: (%d,%d,%d,%d) (%d,%d,%d,%d) stride=%d grp=%d — %s\n",
                   s.mb, s.ic, s.ih, s.iw, s.oc, s.kh, s.kw, 3,
                   s.stride_h, s.grp, s.skip_reason.c_str());
        }
    }
    printf("Warmup: %d, Repeats: %d, Timing: %s\n",
           warmup, repeats, timing_mode ? "ON" : "OFF");
    printf("Threads: ");
    for (size_t i = 0; i < thread_counts.size(); i++)
        printf("%s%d", i ? "," : "", thread_counts[i]);
    printf("\n\n");

    // Filter to valid shapes only
    std::vector<ShapeInfo> shapes;
    for (const auto& s : all_shapes)
        if (s.valid) shapes.push_back(s);

    // ---- Timing mode: fine-grained per-step breakdown ----
    if (timing_mode) {
        for (int nt : thread_counts) {
#ifdef ENABLE_OPENMP
            omp_set_num_threads(nt);
#endif
            printf("=== Timing mode: %d threads, ISA: %s ===\n", nt, isa_name(isa));
            printf("#  %-4s %-5s %-5s %-5s %-5s %-3s %-3s %-3s %-3s %-3s %-3s %-3s %-3s %-4s "
                   "%-12s %-12s %-12s %-12s %-12s %-12s\n",
                   "MB", "IC", "IH", "IW", "OC", "KH", "KW", "SH", "SW", "PH", "PW", "DH", "DW", "GRP",
                   "Weight(ms)", "Input(ms)", "GEMM(ms)", "Output(ms)", "Total(ms)", "GFLOPS");

            int row = 0;
            for (const auto& s : shapes) {
                int N = s.mb, IC = s.ic, IH = s.ih, IW = s.iw;
                int OC = s.oc, OH = IH, OW = IW;

                std::vector<float> src(N * IC * IH * IW), wei(OC * IC * 9), bias(OC, 0.0f), dst(N * OC * OH * OW);
                for (auto& v : src) v = static_cast<float>(rand()) / RAND_MAX;
                for (auto& v : wei) v = static_cast<float>(rand()) / RAND_MAX;

                // Warmup
                winograd_convolution_f44(src.data(), wei.data(), bias.data(), dst.data(),
                                          N, IC, IH, IW, OC, OH, OW, -1e30f, 1e30f);

                // Take best of several runs
                StepTimings best;
                best.total_ms = 1e30;
                for (int i = 0; i < repeats; i++) {
                    auto st = run_with_timing(src.data(), wei.data(), bias.data(), dst.data(),
                                              N, IC, IH, IW, OC, OH, OW, isa);
                    if (st.total_ms < best.total_ms) best = st;
                }

                double flops = 2.0 * N * OC * OH * OW * IC * 9.0;
                double gflops = flops / (best.total_ms * 1e-3) / 1e9;

                printf("%-3d %-4d %-5d %-5d %-5d %-5d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-4d "
                       "%10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n",
                       row, N, IC, IH, IW, OC, s.kh, s.kw, s.stride_h, s.stride_w,
                       s.pad_h, s.pad_w, s.dil_h, s.dil_w, s.grp, s.count,
                       best.weight_transform_ms, best.input_transform_ms,
                       best.gemm_ms, best.output_transform_ms, best.total_ms, gflops);
                row++;
            }
            printf("\n");
        }
    }

    // ---- Standard mode: multi-thread benchmark ----
    if (!timing_mode) {
        printf("#  %-4s %-5s %-5s %-5s %-5s %-3s %-3s %-3s %-3s %-3s %-3s %-3s %-3s %-4s",
               "MB", "IC", "IH", "IW", "OC", "KH", "KW", "SH", "SW", "PH", "PW", "DH", "DW", "GRP");
        for (int t : thread_counts)
            printf(" %s_t%d(ms) %s_t%d_GFLOPS", isa_name(isa), t, isa_name(isa), t);
        printf("\n");
    }

    int row = 0;
    std::vector<std::string> csv_rows;
    for (const auto& s : shapes) {
        int N = s.mb, IC = s.ic, IH = s.ih, IW = s.iw;
        int OC = s.oc, OH = IH, OW = IW;

        int src_size = N * IC * IH * IW;
        int wei_size = OC * IC * 9;
        int dst_size = N * OC * OH * OW;

        std::vector<float> src(src_size), wei(wei_size), bias(OC, 0.0f), dst(dst_size);
        for (auto& v : src) v = static_cast<float>(rand()) / RAND_MAX;
        for (auto& v : wei) v = static_cast<float>(rand()) / RAND_MAX;

        double flops = 2.0 * N * OC * OH * OW * IC * 9.0;

        if (!timing_mode) {
            printf("%-3d %-4d %-5d %-5d %-5d %-5d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-4d",
                   row, N, IC, IH, IW, OC, s.kh, s.kw, s.stride_h, s.stride_w,
                   s.pad_h, s.pad_w, s.dil_h, s.dil_w, s.grp, s.count);
        }

        std::string csv_row;
        csv_row = std::to_string(row) + "," + std::to_string(N) + "," +
                  std::to_string(IC) + "," + std::to_string(IH) + "," +
                  std::to_string(IW) + "," + std::to_string(OC) + "," +
                  std::to_string(s.kh) + "," + std::to_string(s.kw) + "," +
                  std::to_string(s.stride_h) + "," + std::to_string(s.stride_w) + "," +
                  std::to_string(s.pad_h) + "," + std::to_string(s.pad_w) + "," +
                  std::to_string(s.dil_h) + "," + std::to_string(s.dil_w) + "," +
                  std::to_string(s.grp) + "," + std::to_string(s.count) + "," +
                  isa_name(isa);

        for (int nt : thread_counts) {
#ifdef ENABLE_OPENMP
            omp_set_num_threads(nt);
#endif
            // Warmup
            for (int i = 0; i < warmup; i++) {
                winograd_convolution_f44(src.data(), wei.data(), bias.data(), dst.data(),
                                          N, IC, IH, IW, OC, OH, OW, -1e30f, 1e30f);
            }

            // Benchmark
            double best_ms = 1e30;
            for (int i = 0; i < repeats; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                winograd_convolution_f44(src.data(), wei.data(), bias.data(), dst.data(),
                                          N, IC, IH, IW, OC, OH, OW, -1e30f, 1e30f);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                best_ms = std::min(best_ms, ms);
            }

            double gflops = flops / (best_ms * 1e-3) / 1e9;
            if (!timing_mode) {
                printf(" %10.3f %10.2f", best_ms, gflops);
            }
            char buf[64];
            snprintf(buf, sizeof(buf), ",%.3f,%.2f", best_ms, gflops);
            csv_row += buf;
        }
        if (!timing_mode) printf("\n");
        csv_rows.push_back(csv_row);
        row++;
    }

    // Write results to CSV file
    if (!output_file.empty()) {
        std::ofstream fout(output_file);
        fout << "row,mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count,isa";
        for (int t : thread_counts)
            fout << ",t" << t << "_ms,t" << t << "_gflops";
        fout << "\n";
        for (const auto& r : csv_rows)
            fout << r << "\n";
        printf("\nResults written to %s\n", output_file.c_str());
    }

    return 0;
}
