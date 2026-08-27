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
//   --dump   fingerprint mode: forget the sweep, run ONE iteration of
//            M=1 N=3 K=3 with fixed data + SENTINEL GUARD zones around A/B,
//            and print every input/output verbatim. Purpose: reconstruct off-
//            device exactly which linear combination arm_gemm summed (real B
//            rows vs guard bytes vs scaled mixes) -- a single dot product's
//            INTERNAL K-order cannot change its value, so any corruption must
//            show up as cross-row mixing or additive junk, both visible here.
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

// ---------------------------------------------------------------------------
// --dump: fixed tiny case, sentinel-guarded, verbatim printout.
// Guard zones are filled with an index-encoded pattern (1000+idx)*s so any
// out-of-bounds K read produces a recognizable linear signature.
// ---------------------------------------------------------------------------
static void dump_mode()
{
    const int M = 1, N = 3, IC = 3;
    constexpr int GU = 4;                       // guard floats per side
    // Layout: [GU guard][K real][GU guard] per row, rows contiguous. The GEMM
    // sees ld=IC pointing at the real region only; guards surround it in
    // memory so over/underruns by up to GU elements land on sentinels.
    float Ubuf[GU + M * IC + GU];
    float Vbuf[GU + N * IC + GU];
    float Mb[M * N], Rb[M * N];

    for (int i = 0; i < GU + M * IC + GU; i++)
        Ubuf[i] = (i < GU || i >= GU + M * IC) ? -(float)(1000 + i)
                                               : 0.f;
    for (int i = 0; i < GU + N * IC + GU; i++)
        Vbuf[i] = (i < GU || i >= GU + N * IC) ? (float)(2000 + i)
                                               : 0.f;

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    float *U = Ubuf + GU;
    float *V = Vbuf + GU;
    for (int i = 0; i < M * IC; i++) U[i] = dist(rng);
    for (int i = 0; i < N * IC; i++) V[i] = dist(rng);

    printf("== dump case M=%d N=%d K=%d ==\n", M, N, IC);

    winograd_gemm(U, V, Mb, M, N, IC);
    naive_gemm(U, V, Rb, M, N, IC);

    printf("A (U), %d row(s), ld=%d, leading guards:", M, IC);
    for (int i = 0; i < GU; i++) printf(" %.4g", Ubuf[i]);
    printf("\n");
    for (int t = 0; t < M; t++) {
        printf("U[%d]:", t);
        for (int ic = -GU; ic < IC + GU; ic++)
            printf(" %8.5f", (ic >= 0 && ic < IC) ? U[t * IC + ic]
                                                  : Ubuf[GU + t * IC + ic]);
        printf("\n");
    }
    printf("B (V), %d row(s), ld=%d, trailing guards: ", N, IC);
    for (int i = GU + N * IC; i < GU + N * IC + GU; i++)
        printf("%.4g ", Vbuf[i]);
    printf("\n");
    for (int r = 0; r < N; r++) {
        printf("V[%d]:", r);
        for (int ic = -GU; ic < IC + GU; ic++)
            printf(" %8.5f",
                   (ic >= 0 && ic < IC) ? V[r * IC + ic]
                                        : Vbuf[GU + r * IC + ic]);
        printf("(flat idx %d..%d)\n", GU + r * IC, GU + r * IC + IC - 1);
    }

    for (int t = 0; t < M; t++)
        for (int oc = 0; oc < N; oc++)
            printf("C[%d][%d]: got=%.8g want=%.8g diff=%.6g\n", t, oc,
                   Mb[t * N + oc], Rb[t * N + oc],
                   (double)Mb[t * N + oc] - (double)Rb[t * N + oc]);

    // Post-run guard integrity: nonzero entries mean execute() wrote out of
    // range somewhere around A/B/C regions of interest.
    printf("A guards after run:");
    for (int i = 0; i < GU + M * IC + GU; i++)
        if (i < GU || i >= GU + M * IC)
            printf(" [%d]=%.4g%s", i, Ubuf[i],
                   Ubuf[i] != -(float)(1000 + i) ? " <== TOUCHED" : "");
    printf("\n");
    printf("B guards after run:");
    for (int i = 0; i < GU + N * IC + GU; i++)
        if (i < GU || i >= GU + N * IC)
            printf(" [%d]=%.4g%s", i, Vbuf[i],
                   Vbuf[i] != (float)(2000 + i) ? " <== TOUCHED" : "");
    printf("\n");
}

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
        else if (!strcmp(argv[i], "--dump")) { dump_mode(); return 0; }
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
