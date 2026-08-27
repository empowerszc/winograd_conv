// winograd_conv.cpp - End-to-end Winograd convolution implementation
//
// This file implements the full Winograd convolution pipeline:
//   1. Weight transform (one-time)
//   2. Input tile extraction + input transform
//   3. Batched GEMM (simplified naive version)
//   4. Output transform + bias + ReLU
//
// It supports both F(2,2,3,3) and F(4,4,3,3) configurations.
//
// Part of the winograd_conv project.
// Based on ACL's Winograd implementation approach.

#include "winograd_convolution.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <arm_neon.h>

#ifdef USE_OPENBLAS
#include <cblas.h>
// Prevent OpenBLAS from spawning its own threads — conflicts with OpenMP
extern "C" void openblas_set_num_threads(int);
#endif

#ifdef USE_ARM_GEMM
// <string> must come first: arm_gemm.hpp uses std::string without including it.
#include <string>
#include <atomic>
#include <mutex>
#include <arm_gemm.hpp>
// arm_gemm usage:
//   Build: -DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/ComputeLibrary-23.11 or -53.1.0
//   The embedded arm_gemm copy is compiled standalone (see CMakeLists.txt):
//   we shadow arm_compute/core/CPP/CPPTypes.h with our own minimal header so
//   no arm_compute dependency is pulled in.
//   For Winograd: each GEMM is (n_tiles × IC) × (OC × IC)^T → (n_tiles × OC),
//   which is arm_gemm's native A(M×K)·B(N×K)^T layout with A=U, B=V.
//   Modern API: arm_gemm::gemm<fp32>(GemmArgs) factory + pretranspose_B_array.
#if defined(ARM_GEMM_NEW_API)
#include "gemm_implementation.hpp"      // get_compatible_kernels (debug print)
#endif
//
// Diagnostics env vars (all optional, defaults preserve plain behavior):
//   WINO_GEMM_NAIVE=1     skip arm_gemm, use the scalar triple loop (baseline)
//   WINO_GEMM_FILTER=f    override the kernel filter substring ("" = let
//                         arm_gemm choose freely, incl. NEON kernels)
//   WINO_GEMM_VERIFY_GEMM=1  after each execute() recompute with the naive
//                         loop and report max|diff| for the first 10 mismatches
//   WINO_GEMM_DEBUG=1     once per distinct (M,N,K): list all compatible
//                         kernels marking the selected one. NOTE: in ACL
//                         >=24.x a kernel WITHOUT a cycle estimator reports
//                         estimate=0 and find_implementation() treats 0 as
//                         "select immediately" -- so e.g. filter="sve_" +
//                         Nsize<12 deterministically picks
//                         sve_hybrid_fp32_mla_8x1VL regardless of fitness.
#endif

// Include ISA-specific transform implementations
#include "winograd_transforms.hpp"           // NEON (always available on AArch64)

#if defined(__ARM_FEATURE_SVE)
#include "winograd_transforms_sve.hpp"       // SVE
#endif

#if defined(__ARM_FEATURE_SME)
#include "winograd_transforms_sme.hpp"       // SME
#endif

namespace winograd_conv {

// ============================================================================
// Per-thread grow-on-demand scratch buffers
// ============================================================================
// Replaces per-call std::vector allocations. Each logical buffer owns a
// thread_local allocation that grows to the max size ever needed and is reused
// across calls, so repeated winograd_convolution() calls (benchmark / inference
// loops) stop churning the heap. malloc'd memory is never zeroed — every
// scratch is fully overwritten before being read by its caller. Mirrors the
// pattern already used inside transform_2d_*.
//
// IMPORTANT: each logical buffer needs its OWN Scratch. scratch_f32() grows a
// Scratch in place and returns its pointer, so handing one Scratch to two
// "different" buffers makes them alias the same memory (a real bug we hit when
// U/M_buf/V were all served from one thread_local Scratch).

namespace {

struct Scratch {
    float* ptr = nullptr;
    size_t cap = 0;
    ~Scratch() { free(ptr); }
};

inline float* scratch_f32(size_t n, Scratch& s) {
    if (s.cap < n) {
        free(s.ptr);
        s.ptr = static_cast<float*>(malloc(n * sizeof(float)));
        s.cap = n;
    }
    return s.ptr;
}

// Shared work buffers: the whole parallel region reads/writes them. Allocated
// by the master thread before the region starts; thread_local so they survive
// across calls (reuse without realloc).
thread_local Scratch sU, sM, sV;

// Per-thread tile buffers — each thread gets its own set, each with its own
// allocation.
thread_local Scratch sd_tile, sU_tile, sM_tile, sf_tile, sg_wt, sV_oc_wt;

// Layout-conversion temp buffers for the NCHW path (NCHW -> NHWC -> compute ->
// NHWC -> NCHW). Allocated by the master thread before the conversion regions.
thread_local Scratch sSrcNhwc, sDstNhwc;

} // anonymous namespace

// ============================================================================
// ISA-dispatched transform wrappers
// ============================================================================
// These functions select between NEON/SVE/SME based on runtime ISA level.
// The actual transform implementations are in the corresponding headers.

// Weight transform dispatcher
void dispatch_weight_transform(
    const float* g, float* V, int channels,
    bool is_f44, ISALevel isa
) {
    if (is_f44) {
        switch (isa) {
            case ISALevel::NEON:
                weight_transform_f44_neon(g, V, channels); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                weight_transform_f44_sve(g, V, channels); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                weight_transform_f44_sme(g, V, channels); break;
#endif
            default:
                weight_transform_f44_neon(g, V, channels); break;
        }
    } else {
        switch (isa) {
            case ISALevel::NEON:
                weight_transform_f22_neon(g, V, channels); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                weight_transform_f22_sve(g, V, channels); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                weight_transform_f22_sme(g, V, channels); break;
#endif
            default:
                weight_transform_f22_neon(g, V, channels); break;
        }
    }
}

// Input transform dispatcher
void dispatch_input_transform(
    const float* d, float* U, int channels,
    bool is_f44, ISALevel isa
) {
    if (is_f44) {
        switch (isa) {
            case ISALevel::NEON:
                input_transform_f44_neon(d, U, channels); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                input_transform_f44_sve(d, U, channels); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                input_transform_f44_sme(d, U, channels); break;
#endif
            default:
                input_transform_f44_neon(d, U, channels); break;
        }
    } else {
        switch (isa) {
            case ISALevel::NEON:
                input_transform_f22_neon(d, U, channels); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                input_transform_f22_sve(d, U, channels); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                input_transform_f22_sme(d, U, channels); break;
#endif
            default:
                input_transform_f22_neon(d, U, channels); break;
        }
    }
}

// Output transform dispatcher
void dispatch_output_transform(
    const float* M, float* f, int channels,
    const float* bias, float act_min, float act_max,
    bool is_f44, ISALevel isa
) {
    if (is_f44) {
        switch (isa) {
            case ISALevel::NEON:
                output_transform_f44_neon(M, f, channels, bias, act_min, act_max); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                output_transform_f44_sve(M, f, channels, bias, act_min, act_max); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                output_transform_f44_sme(M, f, channels, bias, act_min, act_max); break;
#endif
            default:
                output_transform_f44_neon(M, f, channels, bias, act_min, act_max); break;
        }
    } else {
        switch (isa) {
            case ISALevel::NEON:
                output_transform_f22_neon(M, f, channels, bias, act_min, act_max); break;
#if defined(__ARM_FEATURE_SVE)
            case ISALevel::SVE:
                output_transform_f22_sve(M, f, channels, bias, act_min, act_max); break;
#endif
#if defined(__ARM_FEATURE_SME)
            case ISALevel::SME:
                output_transform_f22_sme(M, f, channels, bias, act_min, act_max); break;
#endif
            default:
                output_transform_f22_neon(M, f, channels, bias, act_min, act_max); break;
        }
    }
}

// ============================================================================
// Batched GEMM for Winograd domain
// ============================================================================
// M[n][oc] = sum_ic U[n][ic] * V[oc][ic]
// This is M[n_tiles x OC] = U[n_tiles x IC] * V[OC x IC]^T
// Called n_multis times (16 for F(2,2,3,3), 36 for F(4,4,3,3)).

// Naive triple-loop GEMM — the portable fallback (also used by the arm_gemm
// driver if strategy selection ever fails).
static void winograd_gemm_naive(
    const float* U,   // [n_tiles][IC]
    const float* V,   // [OC][IC]
    float* M,         // [n_tiles][OC]
    int n_tiles,
    int OC,
    int IC
) {
    for (int t = 0; t < n_tiles; t++) {
        for (int oc = 0; oc < OC; oc++) {
            float sum = 0.0f;
            for (int ic = 0; ic < IC; ic++) {
                sum += U[t * IC + ic] * V[oc * IC + ic];
            }
            M[t * OC + oc] = sum;
        }
    }
}

#if defined(USE_ARM_GEMM)
// ----------------------------------------------------------------------------
// arm_gemm JIT driver (modern API, ACL 23.11 embedded copy).
//
// arm_gemm computes C(MxN) = A(MxK) * B(NxK)^T with A/B row-major, which maps
// straight onto our data: A=U (lda=IC), B=V (ldb=IC), C=M (ldc=OC).
//
// Hybrid/interleaved kernels require B to be pre-transposed before execute();
// the pretranspose + workspace buffers are cached per-thread so the steady
// state has no mallocs. The GEMM object itself is created per call (this is
// called from inside `#pragma omp parallel`, so a shared/cached object would
// race); maxthreads=1 matches the OpenBLAS-1-thread baseline.
//
// The GemmConfig filter forces SVE kernels — on Kunpeng 920F (SVE-512) that
// is the entire point of using arm_gemm. If a shape ever runs faster with a
// NEON kernel, change/remove the filter to let arm_gemm decide.
// ----------------------------------------------------------------------------
namespace arm_gemm_driver
{
// 64-byte aligned per-thread scratch buffer (grows, never shrinks).
struct Buf
{
    std::vector<uint8_t> storage;
    uint8_t             *p   = nullptr;
    size_t               cap = 0;

    uint8_t *ensure(size_t n)
    {
        if (n > cap)
        {
            storage.resize(n + 64);                               // room to align
            const uintptr_t base = reinterpret_cast<uintptr_t>(storage.data());
            const uintptr_t up   = (base + 63) & ~static_cast<uintptr_t>(63);
            p   = reinterpret_cast<uint8_t *>(up);
            cap = n;
        }
        return p;
    }
};

void run(const float *U, const float *V, float *M, int n_tiles, int OC, int IC)
{
    // Env knobs are read once per process (function-local static init is
    // thread-safe under OpenMP; the branch below is negligible per call).
    static const bool use_naive = [] {
        const char *e = getenv("WINO_GEMM_NAIVE");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    if (use_naive)
    {
        winograd_gemm_naive(U, V, M, n_tiles, OC, IC);
        return;
    }

    static arm_compute::CPUInfo &ci = arm_compute::CPUInfo::get();
    thread_local Buf pretrans;
    thread_local Buf work;

    // Force SVE fp32 kernels (the reason to use arm_gemm on SVE-512 hardware),
    // overridable for diagnosis ("" lets arm_gemm pick NEON ones too).
    static const char *filter_env = getenv("WINO_GEMM_FILTER"); // may be nullptr
    arm_gemm::GemmConfig cfg;
    cfg.filter = filter_env ? filter_env : "sve_";

#if defined(ARM_GEMM_NEW_API)
    // ACL 53.1.0: GemmArgs gained an 'accumulate' param before cfg. Without
    // the explicit false the cfg pointer would bind to the bool (accumulate
    // silently ON, C never zeroed) and cfg would stay nullptr (filter ignored).
    arm_gemm::GemmArgs args(&ci, n_tiles, OC, IC,
                            1 /*Ksections*/, 1 /*nbatches*/, 1 /*nmulti*/,
                            false /*indirect_input*/, {}, /*no activation*/
                            1 /*maxthreads*/,
                            false /*fixed_format*/, false /*fast_mode*/,
                            false /*accumulate*/, &cfg);
#else
    arm_gemm::GemmArgs args(&ci, n_tiles, OC, IC,
                            1 /*Ksections*/, 1 /*nbatches*/, 1 /*nmulti*/,
                            false /*indirect_input*/, {}, /*no activation*/
                            1 /*maxthreads*/,
                            false /*fixed_format*/, false /*fast_mode*/, &cfg);
#endif

#if defined(ARM_GEMM_NEW_API)
    // 53.1.0: gemm() no longer defaults Tret to Tlop -- pass it explicitly.
    auto gemm = arm_gemm::gemm<float, float, float>(args);
#else
    auto gemm = arm_gemm::gemm<float, float>(args);
#endif
    if (!gemm)
    {
        // Strategy selection failed (shouldn't happen) — fall back to naive.
        winograd_gemm_naive(U, V, M, n_tiles, OC, IC);
        return;
    }

    // Debug: once per distinct (M,N,K) shape, dump the compatible-kernel list
    // with the selected one marked. Exactly one thread wins the fetch_add per
    // shape; the old single "static bool" spammed the line once per OpenMP
    // thread. get_compatible_kernels() marks is_default via the same
    // find_implementation() the factory uses, so this shows the ACTUAL pick.
    static const bool debug = (getenv("WINO_GEMM_DEBUG") != nullptr);
    if (debug)
    {
        // Serialize the shape-key check + printing (debug-only path).
        static std::mutex   print_mu;
        static std::string  last_shape;
        std::lock_guard<std::mutex> lk(print_mu);
        // First line ever: build/toolchain banner (replaces asking the user
        // for flags.make / c++ --version when debugging remotely).
        static const bool banner = [] {
            int sve = 0, bits = 0, fp16 = 0, bf16 = 0, i8mm = 0, f32mm = 0;
#if defined(__ARM_FEATURE_SVE)
            sve = 1;
#endif
#ifdef __ARM_FEATURE_SVE_BITS
            bits = __ARM_FEATURE_SVE_BITS;
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            fp16 = 1;
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
            bf16 = 1;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
            i8mm = 1;
#endif
#if defined(__ARM_FEATURE_MATMUL_FP32)
            f32mm = 1;
#endif
            fprintf(stderr,
                    "[winograd_gemm] build: %s | SVE=%d SVE_BITS=%d "
                    "fp16=%d bf16=%d i8mm=%d f32mm=%d\n",
                    __VERSION__, sve, bits, fp16, bf16, i8mm, f32mm);
            return true;
        }();
        (void)banner;
        const std::string key = std::to_string(n_tiles) + "x" +
                                std::to_string(OC) + "x" + std::to_string(IC);
        if (key != last_shape)
        {
            last_shape = key;
#if defined(ARM_GEMM_NEW_API)
            // 53.1.0 only: get_compatible_kernels(args, Nothing()) doesn't
            // exist in the 23.11 layout.
            const auto kds = arm_gemm::get_compatible_kernels<float, float, float>(
                args, arm_gemm::Nothing());
            fprintf(stderr, "[winograd_gemm] filter='%s' shapes M=%d N=%d K=%d -- %zu kernels:\n",
                    cfg.filter.c_str(), n_tiles, OC, IC, kds.size());
            for (const auto &kd : kds)
                fprintf(stderr, "    %-42s est=%-10llu %s\n", kd.name.c_str(),
                        static_cast<unsigned long long>(kd.cycle_estimate),
                        kd.is_default ? "<== SELECTED" : "");
#else
            fprintf(stderr, "[winograd_gemm] filter='%s' shapes M=%d N=%d K=%d (23.11: no kernel list)\n",
                    cfg.filter.c_str(), n_tiles, OC, IC);
#endif
        }
    }

    // Pre-transpose B (=V, OCxIC row-major) into the cached per-thread buffer.
    const size_t bsz = gemm->get_B_pretransposed_array_size();
#if defined(ARM_GEMM_NEW_API)
    // 53.1.0: pretranspose_B_array takes a trailing 'transposed' flag. B is
    // row-major NxK here (V, ld=IC) -- the non-transposed case, which the SVE
    // interleaved/hybrid kernels require (std_transforms_sve asserts it).
    gemm->pretranspose_B_array(pretrans.ensure(bsz), V, IC, 0, false);
#else
    gemm->pretranspose_B_array(pretrans.ensure(bsz), V, IC, 0);
#endif
    gemm->set_pretransposed_B_data(pretrans.p);

    gemm->set_arrays(U, IC, 0, 0,    // A: MxK, ld=IC
                     V, IC, 0,       // B: NxK, ld=IC (ignored; pretransposed)
                     M, OC, 0, 0,    // C: MxN, ld=OC
                     nullptr, 0);    // no bias

    const size_t wsz = gemm->get_working_size();
    if (wsz != 0)
        gemm->set_working_space(work.ensure(wsz));

    const arm_gemm::ndrange_t win = gemm->get_window_size();
    // execute() takes an ndcoord_t (position + size per dim); a full-range
    // coord with all dims at their window sizes computes the whole GEMM.
    arm_gemm::ndcoord_t coord{{0, win.get_size(0)},
                              {0, win.get_size(1)},
                              {0, win.get_size(2)},
                              {0, win.get_size(3)},
                              {0, win.get_size(4)},
                              {0, win.get_size(5)}};
    gemm->execute(coord, arm_gemm::ndcoord_t{}, 0);

    // Optional GEMM-stage self-check: recompute with the naive loop and
    // report max|arm_gemm - naive| for the first few mismatching calls.
    // Pure diagnostics (WINO_GEMM_VERIFY_GEMM=1) -- doubles GEMM work.
    static const bool verify = [] {
        const char *e = getenv("WINO_GEMM_VERIFY_GEMM");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    if (verify)
    {
        std::vector<float> ref(static_cast<size_t>(n_tiles) * OC);
        winograd_gemm_naive(U, V, ref.data(), n_tiles, OC, IC);

        float maxerr = 0.0f;
        int   bad_i  = -1;
        bool  nonfinite = false;
        for (int i = 0; i < n_tiles * OC; i++)
        {
            if (!std::isfinite(M[i])) { nonfinite = true; bad_i = i; break; }
            const float err = std::fabs(M[i] - ref[i]);
            if (err > maxerr) { maxerr = err; bad_i = i; }
        }
        if (nonfinite ||
            maxerr > 1e-3f * std::max(1.0f, std::fabs(ref[bad_i < 0 ? 0 : bad_i])))
        {
            static std::atomic<int> reported{0};
            if (reported.fetch_add(1) < 10)
                fprintf(stderr,
                        "[winograd_gemm VERIFY] MISMATCH M=%d N=%d K=%d: "
                        "max|diff|=%.6g at flat %d (got=%.6g want=%.6g)\n",
                        n_tiles, OC, IC, maxerr, bad_i,
                        M[bad_i], ref[bad_i]);
        }
    }
}
} // namespace arm_gemm_driver
#endif // USE_ARM_GEMM

void winograd_gemm(
    const float* U,   // [n_tiles][IC]
    const float* V,    // [OC][IC]
    float* M,           // [n_tiles][OC]
    int n_tiles,
    int OC,
    int IC
) {
#if defined(USE_ARM_GEMM)
    // arm_gemm JIT (modern API). Requires:
    //   -DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/ComputeLibrary-23.11/ComputeLibrary-23.11
    arm_gemm_driver::run(U, V, M, n_tiles, OC, IC);
#elif defined(USE_OPENBLAS)
    // OpenBLAS: cblas_sgemm
    // Requires: -DUSE_OPENBLAS=ON
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_tiles, OC, IC,
                1.0f, U, IC, V, IC,
                0.0f, M, OC);
#else
    // Naive fallback: triple loop (no external dependency)
    winograd_gemm_naive(U, V, M, n_tiles, OC, IC);
#endif
}

// ============================================================================
// Direct (reference) 3x3 convolution with stride=1, pad=1
// ============================================================================
// Standard implementation for verification.
// Input format: NCHW (batch, channels, height, width)

void direct_convolution_3x3(
    const float* src,   // [N][IC][IH][IW]
    const float* wei,   // [OC][IC][3][3]
    const float* bias,  // [OC] or nullptr
    float* dst,         // [N][OC][OH][OW]
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    float act_min, float act_max
) {
    // Assume stride=1, pad=1, dilation=1, so OH=IH, OW=IW
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float sum = bias ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int ih = oh - 1 + kh;  // pad=1: ih = oh + kh - 1
                                int iw = ow - 1 + kw;

                                // Bounds check with zero padding
                                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                    float s = src[((n * IC + ic) * IH + ih) * IW + iw];
                                    float w = wei[((oc * IC + ic) * 3 + kh) * 3 + kw];
                                    sum += s * w;
                                }
                                // else: padding contributes 0
                            }
                        }
                    }

                    // Activation (clamp)
                    if (sum > act_max) sum = act_max;
                    if (sum < act_min) sum = act_min;

                    dst[((n * OC + oc) * OH + oh) * OW + ow] = sum;
                }
            }
        }
    }
}

// ============================================================================
// Winograd convolution
// ============================================================================

// Internal NHWC-only core of the Winograd convolution. The public
// winograd_convolution() either calls this directly (NHWC input) or converts
// NCHW -> NHWC first and converts the result back. Keeping the compute kernel
// layout-free means there is exactly one hot path to maintain and tune.
static void winograd_convolution_nhwc_core(
    const float* src,   // NHWC: [N][IH][IW][IC]
    const float* wei,   // [OC][IC][3][3]
    const float* bias,  // [OC] or nullptr
    float* dst,         // NHWC: [N][OH][OW][OC]
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    const WinogradConfig& config,
    float act_min, float act_max
) {
#ifdef USE_OPENBLAS
    // OpenBLAS must use 1 thread — our OpenMP parallelism is on tiles, not GEMM
    openblas_set_num_threads(1);
#endif

    const int TS = config.input_tile_rows;  // tile size = m + r - 1
    const int OT = config.output_tile_rows;  // output tile = m
    const int NM = config.n_multis;          // number of GEMMs = TS^2
    const bool is_f44 = (TS == 6);

    // Get runtime ISA level (can be set via set_isa_level() or env var)
    ISALevel isa = isa_level();

    // Check for environment variable override
    if (const char* env = std::getenv("WINOGRAD_ISA")) {
        isa = parse_isa(env);
    }

    int n_tile_rows = config.n_tile_rows(OH);
    int n_tile_cols = config.n_tile_cols(OW);
    int n_tiles = n_tile_rows * n_tile_cols;

    // ---- Step 1+2: Single OpenMP region ----
    // Weight transform → per-batch loop (input → GEMM → output)
    // Per-batch: 1 weight barrier + 2*N phase barriers = 1+2N total
    // (Flattening all batches saves barriers but increases memory N× → cache thrashing for large IC)
    int U_size = NM * n_tiles * IC;
    int M_size = NM * n_tiles * OC;
    int V_size = TS * TS * OC * IC;
    // U/M_buf/V are fully overwritten before being read (Phase 1 scatter, GEMM
    // beta=0, weight transform), so allocate uninitialized and reuse the
    // per-thread buffer across calls instead of value-initializing + realloc'ing.
    float* U = scratch_f32(U_size, sU);
    float* M_buf = scratch_f32(M_size, sM);
    float* V = scratch_f32(V_size, sV);

    #pragma omp parallel
    {
        // Per-thread tile buffers, also reuse across calls (see scratch_f32).
        // Each has its own Scratch so buffers never alias each other.
        float* d_tile = scratch_f32(TS * TS * IC, sd_tile);
        float* U_tile = scratch_f32(TS * TS * IC, sU_tile);
        float* M_tile = scratch_f32(TS * TS * OC, sM_tile);
        float* f_tile = scratch_f32(OT * OT * OC, sf_tile);
        float* g_wt = scratch_f32(9 * IC, sg_wt);
        float* V_oc_wt = scratch_f32(TS * TS * IC, sV_oc_wt);

        // ---- Step 1: Weight transform (parallelized over OC) ----
        #pragma omp for schedule(dynamic, 4)
        for (int oc = 0; oc < OC; oc++) {
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++)
                        g_wt[(kh * 3 + kw) * IC + ic] =
                            wei[((oc * IC + ic) * 3 + kh) * 3 + kw];
            dispatch_weight_transform(g_wt, V_oc_wt, IC, is_f44, isa);
            for (int m = 0; m < TS * TS; m++)
                for (int ic = 0; ic < IC; ic++)
                    V[m * OC * IC + oc * IC + ic] = V_oc_wt[m * IC + ic];
        }

        // ---- Step 2: Per-batch loop ----
        for (int n = 0; n < N; n++) {
            // Phase 1: Input transform
            #pragma omp for collapse(2) schedule(dynamic, 2)
            for (int tr = 0; tr < n_tile_rows; tr++) {
                for (int tc = 0; tc < n_tile_cols; tc++) {
                    int tile_idx = tr * n_tile_cols + tc;

                    bool is_edge = (tr == 0 || tr == n_tile_rows - 1 ||
                                    tc == 0 || tc == n_tile_cols - 1);
                    if (is_edge) {
                        memset(d_tile, 0, TS * TS * IC * sizeof(float));
                    }

                    // Clip the input-tile rows/cols to the actual input extent.
                    // A tile spans ih in [tr*OT-1, tr*OT-1+TS); rows outside
                    // [0, IH) stay zero (from the edge memset above). The old
                    // "TS-1 for the last tile" rule only held for even sizes —
                    // odd IH/IW (e.g. IH=7, last tile row reads ih=7) read one
                    // past the last valid row, i.e. the next channel's row 0.
                    int ih_begin = tr * OT - 1;
                    int ti_start = (ih_begin < 0) ? -ih_begin : 0;
                    int ti_end   = (ih_begin + TS > IH) ? (IH - ih_begin) : TS;
                    if (ti_end < ti_start) ti_end = ti_start;
                    int iw_begin = tc * OT - 1;
                    int tj_start = (iw_begin < 0) ? -iw_begin : 0;
                    int tj_end   = (iw_begin + TS > IW) ? (IW - iw_begin) : TS;
                    if (tj_end < tj_start) tj_end = tj_start;

                    for (int ti = ti_start; ti < ti_end; ti++) {
                        int ih = tr * OT - 1 + ti;
                        for (int tj = tj_start; tj < tj_end; tj++) {
                            int iw = tc * OT - 1 + tj;
                            const float* sp = src + ((n * IH + ih) * IW + iw) * IC;
                            copy_f32(sp, d_tile + (ti * TS + tj) * IC, IC);
                        }
                    }

                    dispatch_input_transform(d_tile, U_tile, IC, is_f44, isa);

                    for (int ti = 0; ti < TS; ti++) {
                        for (int tj = 0; tj < TS; tj++) {
                            int ts_idx = ti * TS + tj;
                            copy_f32(U_tile + (ti * TS + tj) * IC,
                                     U + (ts_idx * n_tiles + tile_idx) * IC, IC);
                        }
                    }
                }
            }

            // Phase 2: GEMM
            #pragma omp for schedule(dynamic)
            for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
                const float* U_slice = U + ts_idx * n_tiles * IC;
                const float* V_slice = V + ts_idx * OC * IC;
                float* M_slice = M_buf + ts_idx * n_tiles * OC;
                winograd_gemm(U_slice, V_slice, M_slice, n_tiles, OC, IC);
            }

            // Phase 3: Output transform
            #pragma omp for collapse(2) schedule(dynamic, 2) nowait
            for (int tr = 0; tr < n_tile_rows; tr++) {
                for (int tc = 0; tc < n_tile_cols; tc++) {
                    int tile_idx = tr * n_tile_cols + tc;

                    for (int ti = 0; ti < TS; ti++) {
                        for (int tj = 0; tj < TS; tj++) {
                            int ts_idx = ti * TS + tj;
                            copy_f32(M_buf + (ts_idx * n_tiles + tile_idx) * OC,
                                     M_tile + (ti * TS + tj) * OC, OC);
                        }
                    }

                    dispatch_output_transform(M_tile, f_tile, OC,
                                              bias, act_min, act_max, is_f44, isa);

                    for (int oi = 0; oi < OT; oi++) {
                        for (int oj = 0; oj < OT; oj++) {
                            int oh = tr * OT + oi;
                            int ow = tc * OT + oj;
                            if (oh < OH && ow < OW) {
                                copy_f32(f_tile + (oi * OT + oj) * OC,
                                         dst + ((n * OH + oh) * OW + ow) * OC, OC);
                            }
                        }
                    }
                }
            }
            // implicit barrier at end of Phase 3 for (syncs before next batch)
        } // end for each batch
    } // end single parallel region
}

// ============================================================================
// NCHW <-> NHWC layout conversion
// ============================================================================
// The NCHW path of winograd_convolution() converts the input to NHWC, runs the
// (single, shared) NHWC compute core, then converts the output back. These two
// helpers are cache-blocked transposes: they read channel-major and write
// pixel-major (or vice versa) in 16-wide blocks, so both sides are contiguous
// runs and each element is read and written exactly once. 16 matches the
// SVE-512 lane count; copy_f32() handles the rest.
//
// NOTE (2026-08-11): whether this wrapper beats the archived native-NCHW path
// is a measured question, not a theoretical one — see PERFORMANCE_ANALYSIS.md
// §12. These are a correct first cut; if target measurements show the
// conversion dominates, the block transpose can be upgraded to an SVE
// register transpose (fewer L1 round-trips through the stack buffer).

namespace {

// NCHW [N][IC][IH][IW] -> NHWC [N][IH][IW][IC]
void nchw_to_nhwc(const float* src, float* dst, int N, int IC, int IH, int IW) {
    const int HW = IH * IW;
    constexpr int CH = 16;  // channel/pixel block width
    // collapse(3) over (n, cb, h): collapse(2) parallelizes only over
    // N * ceil(IC/16) chunks — for N=1 small-IC inputs that's 1-3 chunks, so
    // 13+ of 16 threads sit idle (measured: wrapper lost 1.13x-1.38x there).
    // Adding h exposes N * ceil(IC/16) * IH chunks, matching the native
    // kernel's tile-level parallelism. ch is recomputed per (n,cb,h); it must
    // live after the h loop header or the collapse loops are not perfectly
    // nested (OpenMP requires perfect nesting for collapse).
    #pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int cb = 0; cb < IC; cb += CH) {
            for (int h = 0; h < IH; h++) {
                int ch = std::min(cb + CH, IC) - cb;
                for (int wb = 0; wb < IW; wb += CH) {
                    int wn = std::min(CH, IW - wb);
                    // Read channel-major block [ch][wn], write pixel-major [wn][ch]
                    float tmp[CH][CH];
                    for (int k = 0; k < ch; k++) {
                        const float* sp = src + n * IC * HW + (cb + k) * HW + h * IW + wb;
                        for (int j = 0; j < wn; j++) tmp[j][k] = sp[j];
                    }
                    for (int j = 0; j < wn; j++)
                        copy_f32(tmp[j], dst + (n * HW + h * IW + wb + j) * IC + cb, ch);
                }
            }
        }
    }
}

// NHWC [N][IH][IW][IC] -> NCHW [N][IC][IH][IW]
void nhwc_to_nchw(const float* src, float* dst, int N, int IC, int IH, int IW) {
    const int HW = IH * IW;
    constexpr int CH = 16;
    // Same collapse(3) reasoning as nchw_to_nhwc.
    #pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int cb = 0; cb < IC; cb += CH) {
            for (int h = 0; h < IH; h++) {
                int ch = std::min(cb + CH, IC) - cb;
                for (int wb = 0; wb < IW; wb += CH) {
                    int wn = std::min(CH, IW - wb);
                    float tmp[CH][CH];
                    for (int j = 0; j < wn; j++)
                        copy_f32(src + (n * HW + h * IW + wb + j) * IC + cb, tmp[j], ch);
                    for (int k = 0; k < ch; k++) {
                        float* dp = dst + n * IC * HW + (cb + k) * HW + h * IW + wb;
                        for (int j = 0; j < wn; j++) dp[j] = tmp[j][k];
                    }
                }
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Public Winograd convolution
// ============================================================================
// Layout::NHWC  -> the NHWC core runs directly (the fast path).
// Layout::NCHW  -> convert to NHWC, run the NHWC core, convert the output
//                  back to NCHW. The native NCHW kernel this replaced is
//                  archived (buildable reference) in
//                  ref/winograd_conv_nchw_ref.cpp.

void winograd_convolution(
    const float* src,   // NCHW: [N][IC][IH][IW]  or  NHWC: [N][IH][IW][IC]
    const float* wei,   // [OC][IC][3][3]
    const float* bias,  // [OC] or nullptr
    float* dst,         // NCHW: [N][OC][OH][OW]  or  NHWC: [N][OH][OW][OC]
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    const WinogradConfig& config,
    float act_min, float act_max,
    Layout layout
) {
    if (layout != Layout::NHWC) {
        // NCHW: one in/out transpose around the shared NHWC core.
        const int HW = IH * IW;
        const int OHW = OH * OW;
        float* src_nhwc = scratch_f32(static_cast<size_t>(N) * IC * HW, sSrcNhwc);
        float* dst_nhwc = scratch_f32(static_cast<size_t>(N) * OC * OHW, sDstNhwc);
        nchw_to_nhwc(src, src_nhwc, N, IC, IH, IW);
        winograd_convolution_nhwc_core(src_nhwc, wei, bias, dst_nhwc,
                                       N, IC, IH, IW, OC, OH, OW,
                                       config, act_min, act_max);
        nhwc_to_nchw(dst_nhwc, dst, N, OC, OH, OW);
        return;
    }
    winograd_convolution_nhwc_core(src, wei, bias, dst,
                                   N, IC, IH, IW, OC, OH, OW,
                                   config, act_min, act_max);
}

} // namespace winograd_conv
