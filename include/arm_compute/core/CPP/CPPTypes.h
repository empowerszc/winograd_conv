/*
 * Minimal self-contained replacement for ACL's arm_compute/core/CPP/CPPTypes.h.
 *
 * The embedded copy of arm_gemm (ComputeLibrary 23.11, src/cpu/kernels/assembly/)
 * only reaches arm_compute through a single glue header, arm_gemm_local.hpp:
 *
 *     #include "arm_compute/core/CPP/CPPTypes.h"
 *     using CPUModel = arm_compute::CPUModel;
 *     using CPUInfo  = arm_compute::CPUInfo;
 *
 * Supplying this header (plus src/arm_gemm_cpuinfo.cpp) lets arm_gemm compile
 * fully standalone -- no arm_compute dependency, no ACL build. Only the CPU
 * model enum and the CPUInfo members that arm_gemm's fp32 SVE strategies call
 * are declared here; the union of `_ci->` calls across the arm_gemm sources is:
 *
 *   get_cpu_model(), get_L1_cache_size(), get_L2_cache_size(),
 *   has_fp16(), has_bf16(), has_svebf16(), has_dotprod(), has_svef32mm(),
 *   has_i8mm(), has_svei8mm(), has_sve(), has_sve2(), has_sme(), has_sme2()
 *
 * The enum values and the CPUModel name list mirror ACL 23.11 exactly so the
 * strategy kernel headers (which switch on CPUModel::A76 etc.) compile as-is.
 */
#pragma once

#include <cstddef>

namespace arm_compute
{
#define ARM_COMPUTE_CPU_MODEL_LIST \
    X(GENERIC)                     \
    X(GENERIC_FP16)                \
    X(GENERIC_FP16_DOT)            \
    X(A53)                         \
    X(A55r0)                       \
    X(A55r1)                       \
    X(A35)                         \
    X(A73)                         \
    X(A76)                         \
    X(A510)                        \
    X(X1)                          \
    X(V1)                          \
    X(A64FX)                       \
    X(N1)

/** CPU models types (mirrors arm_compute/core/CPP/CPPTypes.h). */
enum class CPUModel
{
#define X(model) model,
    ARM_COMPUTE_CPU_MODEL_LIST
#undef X
};

#undef ARM_COMPUTE_CPU_MODEL_LIST

/** Minimal CPUInfo. Implementation lives in src/arm_gemm_cpuinfo.cpp. */
class CPUInfo final
{
public:
    static CPUInfo &get();

    bool has_fp16() const;
    bool has_bf16() const;
    bool has_svebf16() const;
    bool has_dotprod() const;
    bool has_svef32mm() const;
    bool has_i8mm() const;
    bool has_svei8mm() const;
    bool has_sve() const;
    bool has_sve2() const;
    bool has_sme() const;
    bool has_sme2() const;
    CPUModel get_cpu_model(unsigned int cpuid) const;
    CPUModel get_cpu_model() const;
    unsigned int get_L1_cache_size() const;
    unsigned int get_L2_cache_size() const;
    unsigned int get_cpu_num() const;
};
} // namespace arm_compute
