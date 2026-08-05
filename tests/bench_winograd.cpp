// bench_winograd.cpp - Performance benchmark for Winograd convolution
//
// Reads shapes from stdin or a file, filters for stride=1, group=1,
// and benchmarks F(4,4,3,3) Winograd convolution.
//
// Usage:
//   ./bench_winograd shapes.csv           # read from file
//   cat shapes.csv | ./bench_winograd     # read from stdin
//   ./bench_winograd --neon shapes.csv    # force NEON
//   ./bench_winograd --sve shapes.csv     # force SVE
//   ./bench_winograd --sme shapes.csv      # force SME
//
// CSV format (tab-separated, first line is header):
//   Input Shape    Weight Shape    Stride    Pad    Dil    Grp    Count
//   (4, 192, 40, 40)    (192, 192, 3, 3)    [1, 1]    [1, 1]    [1, 1]    1    24
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

using namespace winograd_conv;

// Parse a tuple like "(4, 192, 40, 40)" into 4 ints
bool parse_tuple4(const std::string& s, int& a, int& b, int& c, int& d) {
    char open_paren, close_paren, comma1, comma2, comma3;
    std::istringstream iss(s);
    iss >> open_paren >> a >> comma1 >> b >> comma2 >> c >> comma3 >> d >> close_paren;
    return open_paren == '(' && close_paren == ')' &&
           comma1 == ',' && comma2 == ',' && comma3 == ',';
}

// Parse a bracket pair like "[1, 1]" into 2 ints
bool parse_bracket2(const std::string& s, int& a, int& b) {
    char open_b, close_b, comma;
    std::istringstream iss(s);
    iss >> open_b >> a >> comma >> b >> close_b;
    return open_b == '[' && close_b == ']' && comma == ',';
}

struct ShapeInfo {
    int N, IC, IH, IW;
    int OC, WIC, KH, KW;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dil_h, dil_w;
    int group;
    int count;
};

int main(int argc, char** argv) {
    ISALevel isa = detect_isa();
    std::string input_file;
    int warmup = 3;
    int repeats = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--neon") isa = ISALevel::NEON;
        else if (arg == "--sve") isa = ISALevel::SVE;
        else if (arg == "--sme") isa = ISALevel::SME;
        else if (arg == "--warmup" && i+1 < argc) { warmup = atoi(argv[++i]); }
        else if (arg == "--repeats" && i+1 < argc) { repeats = atoi(argv[++i]); }
        else if (arg == "--help") {
            printf("Usage: %s [options] [shapes_file]\n", argv[0]);
            printf("Options:\n");
            printf("  --neon / --sve / --sme  Select ISA\n");
            printf("  --warmup N              Warmup iterations (default 3)\n");
            printf("  --repeats N             Repeat iterations (default 10)\n");
            printf("  --help                  Show this help\n");
            return 0;
        } else {
            input_file = arg;
        }
    }

    if (const char* env = std::getenv("WINOGRAD_ISA")) {
        isa = parse_isa(env);
    }
    set_isa_level(isa);
    printf("ISA: %s (detected: %s)\n\n", isa_name(isa), isa_name(detect_isa()));

    // Read shapes
    FILE* fp = input_file.empty() ? stdin : fopen(input_file.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", input_file.c_str());
        return 1;
    }

    char line[512];
    // Skip header
    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "Empty input\n");
        return 1;
    }

    std::vector<ShapeInfo> shapes;
    while (fgets(line, sizeof(line), fp)) {
        // Parse: Input Shape, Weight Shape, Stride, Pad, Dil, Grp, Count
        // Tab-separated
        std::vector<std::string> fields;
        char* p = line;
        char* start = p;
        while (*p) {
            if (*p == '\t' || *p == '\n') {
                *p = '\0';
                fields.push_back(start);
                start = p + 1;
            }
            p++;
        }
        if (*start) fields.push_back(start);

        if (fields.size() < 7) continue;

        ShapeInfo s;
        if (!parse_tuple4(fields[0], s.N, s.IC, s.IH, s.IW)) continue;
        if (!parse_tuple4(fields[1], s.OC, s.WIC, s.KH, s.KW)) continue;
        if (!parse_bracket2(fields[2], s.stride_h, s.stride_w)) continue;
        if (!parse_bracket2(fields[3], s.pad_h, s.pad_w)) continue;
        if (!parse_bracket2(fields[4], s.dil_h, s.dil_w)) continue;
        s.group = atoi(fields[5].c_str());
        s.count = atoi(fields[6].c_str());

        // Filter: stride=1, group=1, 3x3 kernel
        if (s.stride_h == 1 && s.stride_w == 1 && s.group == 1 &&
            s.KH == 3 && s.KW == 3) {
            shapes.push_back(s);
        }
    }
    if (fp != stdin) fclose(fp);

    printf("%-45s %-20s %8s %12s %12s %10s\n",
           "Shape (N,IC,IH,IW)", "(OC,IC,3,3)", "Count", "Time(ms)", "GFLOPS", "ISA");
    printf("%s\n", std::string(120, '-').c_str());

    for (const auto& s : shapes) {
        int N = s.N, IC = s.IC, IH = s.IH, IW = s.IW;
        int OC = s.OC, OH = IH, OW = IW;

        // Allocate
        int src_size = N * IC * IH * IW;
        int wei_size = OC * IC * 3 * 3;
        int dst_size = N * OC * OH * OW;

        std::vector<float> src(src_size), wei(wei_size), bias(OC, 0.0f), dst(dst_size);

        // Fill with random data
        for (auto& v : src) v = static_cast<float>(rand()) / RAND_MAX;
        for (auto& v : wei) v = static_cast<float>(rand()) / RAND_MAX;

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

        // GFLOPS = 2 * N * OC * OH * OW * IC * 9 / time
        double flops = 2.0 * N * OC * OH * OW * IC * 9.0;
        double gflops = flops / (best_ms * 1e-3) / 1e9;

        char shape_str[64], wshape_str[32];
        snprintf(shape_str, sizeof(shape_str), "(%d, %d, %d, %d)", N, IC, IH, IW);
        snprintf(wshape_str, sizeof(wshape_str), "(%d, %d, 3, 3)", OC, IC);

        printf("%-45s %-20s %8d %10.2f %12.2f %10s\n",
               shape_str, wshape_str, s.count, best_ms, gflops, isa_name(isa));
    }

    return 0;
}
