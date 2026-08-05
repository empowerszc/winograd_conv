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
    std::getline(fin, line); // header
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto f = split_csv(line);
        if (f.size() < 15) continue;
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

        if (s.stride_h == 1 && s.stride_w == 1 && s.grp == 1 &&
            s.kh == 3 && s.kw == 3 && s.pad_h == 1 && s.pad_w == 1) {
            shapes.push_back(s);
        }
    }
    return true;
}

int main(int argc, char** argv) {
    ISALevel isa = detect_isa();
    std::string input_file;
    std::string output_file;
    int warmup = 3;
    int repeats = 10;
    std::vector<int> thread_counts = {1, 8, 16, 32, 38};

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--neon") isa = ISALevel::NEON;
        else if (arg == "--sve") isa = ISALevel::SVE;
        else if (arg == "--sme") isa = ISALevel::SME;
        else if (arg == "--warmup" && i+1 < argc) { warmup = atoi(argv[++i]); }
        else if (arg == "--repeats" && i+1 < argc) { repeats = atoi(argv[++i]); }
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
            return 0;
        } else {
            input_file = arg;
        }
    }

    if (const char* env = std::getenv("WINOGRAD_ISA")) {
        isa = parse_isa(env);
    }
    set_isa_level(isa);

    if (input_file.empty()) {
        fprintf(stderr, "Usage: %s [options] shapes.csv\n", argv[0]);
        return 1;
    }

    std::vector<ShapeInfo> shapes;
    if (!read_shapes(input_file, shapes)) return 1;

    printf("ISA: %s (detected: %s)\n", isa_name(isa), isa_name(detect_isa()));
    printf("Warmup: %d, Repeats: %d\n", warmup, repeats);
    printf("Threads: ");
    for (size_t i = 0; i < thread_counts.size(); i++)
        printf("%s%d", i ? "," : "", thread_counts[i]);
    printf("\n\n");

    int ncols = thread_counts.size() * 2;
    printf("#  %-4s %-5s %-5s %-5s %-5s %-3s %-3s %-3s %-3s %-3s %-3s %-3s %-3s %-4s",
           "MB", "IC", "IH", "IW", "OC", "KH", "KW", "SH", "SW", "PH", "PW", "DH", "DW", "GRP");
    for (int t : thread_counts)
        printf(" %s_t%d(ms) %s_t%d_GFLOPS", isa_name(isa), t, isa_name(isa), t);
    printf("\n");

    int row = 0;
    for (const auto& s : shapes) {
        int N = s.mb, IC = s.ic, IH = s.ih, IW = s.iw;
        int OC = s.oc, OH = IH, OW = IW;

        int src_size = N * IC * IH * IW;
        int wei_size = OC * IC * 3 * 3;
        int dst_size = N * OC * OH * OW;

        std::vector<float> src(src_size), wei(wei_size), bias(OC, 0.0f), dst(dst_size);
        for (auto& v : src) v = static_cast<float>(rand()) / RAND_MAX;
        for (auto& v : wei) v = static_cast<float>(rand()) / RAND_MAX;

        double flops = 2.0 * N * OC * OH * OW * IC * 9.0;

        printf("%-3d %-4d %-5d %-5d %-5d %-5d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-4d",
               row, N, IC, IH, IW, OC, s.kh, s.kw, s.stride_h, s.stride_w,
               s.pad_h, s.pad_w, s.dil_h, s.dil_w, s.grp, s.count);

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
            printf(" %10.3f %10.2f", best_ms, gflops);
        }
        printf("\n");
        row++;
    }

    // Write results to CSV file
    if (!output_file.empty()) {
        std::ofstream fout(output_file);
        fout << "row,mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count,isa";
        for (int t : thread_counts)
            fout << ",t" << t << "_ms,t" << t << "_gflops";
        fout << "\n";

        row = 0;
        for (const auto& s : shapes) {
            int N = s.mb, IC = s.ic, IH = s.ih, IW = s.iw;
            int OC = s.oc, OH = IH, OW = IW;

            std::vector<float> src(N * IC * IH * IW), wei(OC * IC * 9), bias(OC, 0.0f), dst(N * OC * OH * OW);
            for (auto& v : src) v = static_cast<float>(rand()) / RAND_MAX;
            for (auto& v : wei) v = static_cast<float>(rand()) / RAND_MAX;

            double flops = 2.0 * N * OC * OH * OW * IC * 9.0;

            fout << row << "," << N << "," << IC << "," << IH << "," << IW << ","
                 << OC << "," << s.kh << "," << s.kw << ","
                 << s.stride_h << "," << s.stride_w << ","
                 << s.pad_h << "," << s.pad_w << ","
                 << s.dil_h << "," << s.dil_w << ","
                 << s.grp << "," << s.count << "," << isa_name(isa);

            for (int nt : thread_counts) {
#ifdef ENABLE_OPENMP
                omp_set_num_threads(nt);
#endif
                for (int i = 0; i < warmup; i++) {
                    winograd_convolution_f44(src.data(), wei.data(), bias.data(), dst.data(),
                                              N, IC, IH, IW, OC, OH, OW, -1e30f, 1e30f);
                }
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
                fout << "," << std::fixed << std::setprecision(3) << best_ms
                     << "," << std::setprecision(2) << gflops;
            }
            fout << "\n";
            row++;
        }
        printf("\nResults written to %s\n", output_file.c_str());
    }

    return 0;
}
