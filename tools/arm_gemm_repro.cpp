// arm_gemm_repro.cpp -- minimal GEMM-stage repro/diagnosis tool.
//
// Hammers winograd_gemm() directly with random A/B and checks against a naive
// triple-loop reference. Zero transforms, zero OpenMP pipeline -- isolates
// "is the GEMM stage itself wrong" from everything else.
//
// Usage:
//   ./arm_gemm_repro [--threads T] [--iters I] [--bconst] [M N K ...]
//
//   --bconst  make every B row CONSTANT along K (V[oc][ic] = c_oc). In-pipeline
//             evidence says identity weights (= K-constant V) PASS while any
//             general W fails; this flag tests that hypothesis at the bare
//             GEMM level. Run once without (expect FAILs) and once with.
//
// Env knobs honored by the library driver:
//   WINO_GEMM_FILTER=...    kernel filter substring ("" = let arm_gemm choose)
//   WINO_GEMM_NAIVE=1       bypass arm_gemm entirely (control run)
#if !defined(ARM_GEMM_REPRO_STANDALONE)
#include "../include/winograd_convolution.hpp"   // declares winograd_gemm
#else
// Standalone syntax-check mode: declare just what we use so this file can be
// type-checked on any host without pulling in ISA-specific headers.
extern void winograd_gemm(const float *U, const float *V, float *M,
                          int n_tiles, int OC, int IC);
#endif

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <atomic>

static void naive_gemm(const float *U, const float *V, float *M,
                       int n_tiles, int OC, int IC)
{
    for (int t = 0; t < n_tiles; t++)
        for (int oc = 0; oc < OC; oc++)
        {
            float sum = 0.0f;
            for (int ic = 0; ic < IC; ic++)
                sum += U[t * IC + ic] * V[oc * IC + ic];
            M[t * OC + oc] = sum;
        }
}

struct Result { long fails = 0; double worst = 0; bool nonfinite = false; };

static bool g_bconst = false;

static Result check_case(int n_tiles, int OC, int IC, int iters)
{
    Result r;
    std::mt19937 rng(12345 + n_tiles * 7 + OC * 131 + IC);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    size_t usz = (size_t)n_tiles * IC;
    size_t vsz = (size_t)OC * IC;
    size_t msz = (size_t)n_tiles * OC;
    std::vector<float> U(usz), V(vsz), M(msz), R(msz);

    for (int it = 0; it < iters; it++)
    {
        for (auto &x : U) x = dist(rng);
        if (g_bconst)
        {
            // K-constant B: one random scalar per row, replicated across K.
            for (int oc = 0; oc < OC; oc++)
            {
                const float c = dist(rng);
                for (int ic = 0; ic < IC; ic++) V[(size_t)oc * IC + ic] = c;
            }
        }
        else
        {
            for (auto &x : V) x = dist(rng);
        }

        winograd_gemm(U.data(), V.data(), M.data(), n_tiles, OC, IC);
        naive_gemm(U.data(), V.data(), R.data(), n_tiles, OC, IC);

        // scale-free relative check; GEMM of ~N(0,1) rows has |ref| ~ sqrt(IC)
        double refnorm = 0.0;
        for (int i = 0; i < n_tiles * OC; i++)
        {
            if (!std::isfinite(M[i])) { r.nonfinite = true; }
            double d = std::fabs((double)M[i] - (double)R[i]);
            if (d > r.worst) r.worst = d;
            refnorm += (double)R[i] * (double)R[i];
        }
        refnorm = std::sqrt(refnorm / msz);
        if (r.worst > 5e-3 * std::max(1.0, refnorm)) r.fails++;
    }
    return r;
}

int main(int argc, char **argv)
{
    int threads = 1, iters = 4;
    std::vector<std::array<int,3>> cases;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bconst")) g_bconst = true;
        else {
            // positional triples: M N K
            if (i + 2 >= argc) { fprintf(stderr, "bad args\n"); return 2; }
            cases.push_back({atoi(argv[i]), atoi(argv[i+1]), atoi(argv[i+2])});
            i += 2;
        }
    }
    if (cases.empty())
    {
        // Default sweep: hits small/hybrid-window shapes and larger ones.
        const int defs[][3] = {
            {1,1,1}, {1,3,3}, {7,16,8}, {8,64,64},
            {25,192,192}, {100,96,96}, {49,512,256},
        };
        for (auto &c : defs) cases.push_back({c[0], c[1], c[2]});
    }

    printf("arm_gemm_repro: threads=%d iters=%d/case bconst=%d filter='%s'\n",
           threads, iters, g_bconst ? 1 : 0,
           getenv("WINO_GEMM_FILTER") ? getenv("WINO_GEMM_FILTER") : "sve_(default)");
    int bad_cases = 0;

    for (auto &c : cases)
    {
        const int nt = c[0], oc = c[1], ic = c[2];

        std::atomic<long> fails{0};
        std::atomic<int>  nnan{0};
        std::atomic<double> worst{0};

        // Each thread runs its own independent copy of the same case
        // (private buffers) -- concurrent calls through winograd_gemm.
        auto worker = [&](void) {
            Result r = check_case(nt, oc, ic, iters);
            fails += r.fails;
            if (r.nonfinite) nnan++;
            double w = worst.load();
            while (r.worst > w && !worst.compare_exchange_weak(w, r.worst)) {}
        };
        // For threads==1 stay sequential; for >1 re-run the same case
        // concurrently on separate data to expose race/static-state bugs.
        if (threads <= 1)
        {
            Result r = check_case(nt, oc, ic, iters);
            const char *failtag = r.fails || r.nonfinite ? "[FAIL]" : "[ok]";
            printf("case M=%d N=%d K=%d seq: worst=%.6g %s%s\n",
                   nt, oc, ic, r.worst, failtag,
                   r.nonfinite ? " [NONFINITE]" : "");
            if (r.fails || r.nonfinite) bad_cases++;
        }
        else
        {
            std::vector<std::thread> ts;
            for (int t = 0; t < threads; t++) ts.emplace_back(worker);
            for (auto &th : ts) th.join();
            const long nf = fails.load();
            const char *failtag = (nf || nnan.load()) ? "[FAIL]" : "[ok]";
            printf("case M=%d N=%d K=%d %dt: worst=%.6g fail_iters=%ld %s%s\n",
                   nt, oc, ic, threads, worst.load(), nf, failtag,
                   nnan.load() ? " [NONFINITE]" : "");
            if (nf || nnan.load()) bad_cases++;
        }
    }

    printf("=== %d/%zu failing cases ===\n", bad_cases, cases.size());
    return bad_cases ? 1 : 0;
}
