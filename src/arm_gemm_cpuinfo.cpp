/*
 * CPUInfo implementation for the embedded arm_gemm (ACL 23.11 copy).
 *
 * Target: Huawei Kunpeng 920F -- aarch64, SVE-512, no SVE2/BF16/SME.
 * Tuned defaults for that machine; the SVE probe is done via Linux HWCAPs so
 * the same build also behaves on other aarch64 Linux boxes (e.g. Neoverse).
 *
 * On non-aarch64 builds this still compiles (all features off, GENERIC model)
 * so the driver can be syntax-checked on x86.
 */
#include "arm_compute/core/CPP/CPPTypes.h"

#include <cstdlib>
#include <cstring>

/* Linux arm64 HWCAP bit definitions (stable kernel ABI). Defined here
 * unconditionally so the feature checks also compile on non-aarch64 hosts;
 * the actual probing below is aarch64/Linux-only. */
#ifndef HWCAP_FPHP
#define HWCAP_FPHP (1 << 9)
#endif
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1 << 10)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP_FHM
#define HWCAP_FHM (1 << 23)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#ifndef HWCAP2_SVEI8MM
#define HWCAP2_SVEI8MM (1 << 8)
#endif
#ifndef HWCAP2_SVEF32MM
#define HWCAP2_SVEF32MM (1 << 9)
#endif
#ifndef HWCAP2_SVEBF16
#define HWCAP2_SVEBF16 (1 << 11)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 12)
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1 << 13)
#endif

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <unistd.h>
#endif /* __aarch64__ && __linux__ */

namespace arm_compute
{
namespace
{
inline unsigned long hwcap(void)
{
#if defined(__aarch64__) && defined(__linux__)
    return getauxval(AT_HWCAP);
#else
    return 0;
#endif
}

inline unsigned long hwcap2(void)
{
#if defined(__aarch64__) && defined(__linux__)
    return getauxval(AT_HWCAP2);
#else
    return 0;
#endif
}

inline unsigned int kb_from_env(const char *name, unsigned int def_kb)
{
    const char *v = getenv(name);
    if (v == nullptr)
        return def_kb;
    return static_cast<unsigned int>(std::atoi(v));
}
} // namespace

/* ------------------------------------------------------------------ */
/* ISA feature probes.                                                */
/* ------------------------------------------------------------------ */
bool CPUInfo::has_sve() const
{
    return (hwcap() & HWCAP_SVE) != 0;
}
bool CPUInfo::has_sve2() const
{
    return (hwcap2() & HWCAP2_SVE2) != 0;
}
bool CPUInfo::has_bf16() const
{
    return (hwcap2() & HWCAP2_BF16) != 0;
}
bool CPUInfo::has_svebf16() const
{
    return (hwcap2() & HWCAP2_SVEBF16) != 0;
}
bool CPUInfo::has_fp16() const
{
    return (hwcap() & (HWCAP_FPHP | HWCAP_ASIMDHP)) != 0;
}
bool CPUInfo::has_dotprod() const
{
    return (hwcap() & HWCAP_ASIMDDP) != 0;
}
bool CPUInfo::has_i8mm() const
{
    return (hwcap2() & HWCAP2_I8MM) != 0;
}
bool CPUInfo::has_svei8mm() const
{
    return (hwcap2() & HWCAP2_SVEI8MM) != 0;
}
bool CPUInfo::has_svef32mm() const
{
    return (hwcap2() & HWCAP2_SVEF32MM) != 0;
}
bool CPUInfo::has_sme() const
{
    return false; /* SME unsupported on 920F and on GCC < 12 toolchains */
}
bool CPUInfo::has_sme2() const
{
    return false;
}
bool CPUInfo::has_fhm() const
{
    /* FEAT_FHM (fp16fml). Present on A76-class cores. */
    return (hwcap() & HWCAP_FHM) != 0;
}
bool CPUInfo::has_sme_b16f32() const
{
    return false; /* SME outer-product features: no SME on 920F */
}
bool CPUInfo::has_sme_f16f32() const
{
    return false;
}
bool CPUInfo::has_sme_f32f32() const
{
    return false;
}
bool CPUInfo::has_sme_i8i32() const
{
    return false;
}

/* ------------------------------------------------------------------ */
/* Model / sizes.                                                     */
/* ------------------------------------------------------------------ */
CPUModel CPUInfo::get_cpu_model(unsigned int) const
{
    /* Kunpeng 920F cores are Cortex-A76 class (Taishan V110). The perf
     * parameters for A76 in arm_gemm's fp32 kernels use the default path;
     * with the SVE filter forced in the driver the model is not used for
     * strategy selection. */
    return CPUModel::A76;
}
CPUModel CPUInfo::get_cpu_model() const
{
    return get_cpu_model(0);
}

unsigned int CPUInfo::get_L1_cache_size() const
{
    /* 920F: 64 KiB L1D per core. Override via WINO_GEMM_L1_KB. */
    return kb_from_env("WINO_GEMM_L1_KB", 64) * 1024u;
}
unsigned int CPUInfo::get_L2_cache_size() const
{
    /* 920F: 768 KiB L2 per core (there is no L3). Override via WINO_GEMM_L2_KB. */
    return kb_from_env("WINO_GEMM_L2_KB", 768) * 1024u;
}

unsigned int CPUInfo::get_cpu_num() const
{
#if defined(__aarch64__) && defined(__linux__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<unsigned int>(n) : 1u;
#else
    return 1u;
#endif
}

CPUInfo &CPUInfo::get()
{
    static CPUInfo instance;
    return instance;
}
} // namespace arm_compute
