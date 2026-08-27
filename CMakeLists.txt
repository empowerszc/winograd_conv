cmake_minimum_required(VERSION 3.14)
project(winograd_conv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ISA selection options
option(ENABLE_SVE "Enable SVE (-march=armv8.6-a+sve2)" OFF)
option(ENABLE_SME "Enable SME+SVE (-march=armv9-a+sme)" OFF)
option(ENABLE_OPENMP "Enable OpenMP threading" OFF)
option(USE_OPENBLAS "Use OpenBLAS for GEMM (cblas_sgemm)" OFF)
option(USE_ARM_GEMM "Use arm_gemm for GEMM (requires arm_gemm library)" OFF)
set(OPENBLAS_ROOT "" CACHE PATH "Path to OpenBLAS installation")
set(ARM_GEMM_ROOT "" CACHE PATH "Path to arm_gemm library (e.g. ComputeLibrary/src/core/NEON/kernels/convolution/common/gemm")

# Base optimization
add_compile_options(-O3 -g)

# ISA flags
if(ENABLE_SME)
    add_compile_options(-march=armv9-a+sme)
    message(STATUS "=== SME+SVE enabled (-march=armv9-a+sme) ===")
elseif(ENABLE_SVE)
    add_compile_options(-march=armv8.6-a+sve2)
    message(STATUS "=== SVE enabled (-march=armv8.6-a+sve2) ===")
else()
    message(STATUS "=== NEON only (default) ===")
endif()

# OpenMP support (use modern imported target)
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS} | ${OpenMP_CXX_LIBRARIES}")
endif()

# OpenBLAS support
if(USE_OPENBLAS AND NOT USE_ARM_GEMM)
    find_path(OPENBLAS_INCLUDE_DIR cblas.h
              PATHS ${OPENBLAS_ROOT}/include /usr/include /usr/local/include
                    /usr/include/aarch64-linux-gnu)
    find_library(OPENBLAS_LIBRARIES NAMES openblas
                 PATHS ${OPENBLAS_ROOT}/lib /usr/lib /usr/local/lib
                       /usr/lib/aarch64-linux-gnu)
    if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARIES)
        add_compile_definitions(USE_OPENBLAS)
        include_directories(${OPENBLAS_INCLUDE_DIR})
        link_libraries(${OPENBLAS_LIBRARIES})
        message(STATUS "OpenBLAS found: ${OPENBLAS_LIBRARIES}")
    else()
        message(FATAL_ERROR "OpenBLAS not found. Install openblas-dev or set -DOPENBLAS_ROOT=/path")
    endif()
endif()

# arm_gemm support (compiles the fp32-SVE subset of the embedded arm_gemm copy
# from an ACL 23.11 source tree, fully standalone -- no arm_compute dependency;
# our include/arm_compute/core/CPP/CPPTypes.h shim provides the CPUInfo types)
if(USE_ARM_GEMM)
    if(NOT ENABLE_SVE)
        message(FATAL_ERROR "USE_ARM_GEMM requires -DENABLE_SVE=ON (SVE kernels on AArch64)")
    endif()
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        message(FATAL_ERROR "USE_ARM_GEMM requires an AArch64 target (SVE). Current: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    # arm_gemm.hpp moved to src/cpu/kernels/assembly/arm_gemm/ in ACL 53.1.0
    # (was directly in src/cpu/kernels/assembly/ in 23.11).
    find_path(ARM_GEMM_GUARD arm_gemm.hpp
              PATHS ${ARM_GEMM_ROOT}/src/cpu/kernels/assembly/arm_gemm
                    ${ARM_GEMM_ROOT}/src/cpu/kernels/assembly
                    ${ARM_GEMM_ROOT}/src/core/NEON/kernels/arm_gemm
              NO_DEFAULT_PATH)
    if(NOT ARM_GEMM_GUARD)
        message(FATAL_ERROR "arm_gemm.hpp not found under -DARM_GEMM_ROOT=${ARM_GEMM_ROOT}")
    endif()
    add_compile_definitions(USE_ARM_GEMM)

    set(AG ${ARM_GEMM_ROOT})
    set(AG_ARM_GEMM ${AG}/src/core/NEON/kernels/arm_gemm)
    # Exactly the sources gemm_fp32.cpp's fp32 list needs with only
    # ARM_COMPUTE_ENABLE_SVE set (no BF16/SME2/FIXED_FORMAT/SVEF32MM).
    set(ARM_GEMM_SRCS
        ${AG_ARM_GEMM}/gemm_fp32.cpp
        ${AG_ARM_GEMM}/interleave_indirect.cpp
        ${AG_ARM_GEMM}/interleave_indirect-sve.cpp
        ${AG_ARM_GEMM}/mergeresults.cpp
        ${AG_ARM_GEMM}/mergeresults-sve.cpp
        ${AG_ARM_GEMM}/misc.cpp
        ${AG_ARM_GEMM}/transform.cpp
        ${AG_ARM_GEMM}/transform-sve.cpp
        # a64 (NEON) fp32 kernels referenced by the gemm_fp32 list
        ${AG_ARM_GEMM}/kernels/a64_sgemm_8x6/generic.cpp
        ${AG_ARM_GEMM}/kernels/a64_sgemm_8x12/generic.cpp
        ${AG_ARM_GEMM}/kernels/a64_smallK_hybrid_fp32_mla_6x4/generic.cpp
        ${AG_ARM_GEMM}/kernels/a64_smallK_hybrid_fp32_mla_8x4/generic.cpp
        ${AG_ARM_GEMM}/kernels/a64_hybrid_fp32_mla_4x24/generic.cpp
        ${AG_ARM_GEMM}/kernels/a64_hybrid_fp32_mla_6x16/generic.cpp
        ${AG_ARM_GEMM}/kernels/a64_hybrid_fp32_mla_8x4/generic.cpp
        # SVE fp32 kernels -- the point: SVE-512 on Kunpeng 920F
        ${AG_ARM_GEMM}/kernels/sve_hybrid_fp32_mla_6x4VL/generic.cpp
        ${AG_ARM_GEMM}/kernels/sve_hybrid_fp32_mla_8x1VL/generic.cpp
        ${AG_ARM_GEMM}/kernels/sve_interleaved_fp32_mla_8x3VL/generic.cpp
        # our CPUInfo shim implementation
        ${CMAKE_CURRENT_SOURCE_DIR}/src/arm_gemm_cpuinfo.cpp
    )
    # ACL >= 24.x merged misc-sve.cpp into misc.cpp (no -sve suffix file);
    # 23.11 still has both.
    if(EXISTS ${AG_ARM_GEMM}/misc-sve.cpp)
        list(APPEND ARM_GEMM_SRCS ${AG_ARM_GEMM}/misc-sve.cpp)
    endif()
    add_library(arm_gemm STATIC ${ARM_GEMM_SRCS})
    # The SVE kernel asm is large; O2 keeps the build sane (O3 gains nothing).
    # ACL >= 24.x arm_gemm headers use std::string without including <string>
    # (arm_gemm.hpp: KernelDescription.name / GemmConfig.filter; arm_common/
    # internal/utils.hpp: get_type_name). Force-include it so the embedded
    # sources compile standalone on the 920F.
    target_compile_options(arm_gemm PRIVATE -O2 -include string)
    # Put our shim first so it shadows any arm_compute headers on the machine.
    target_include_directories(arm_gemm PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${AG}
        ${AG}/src/cpu/kernels/assembly
        ${AG_ARM_GEMM}
    )
    target_compile_definitions(arm_gemm PUBLIC ARM_COMPUTE_ENABLE_SVE)
    # ACL 53.1.0: arm_gemm.hpp lives in src/cpu/kernels/assembly/arm_gemm/ and
    # GemmArgs gained an 'accumulate' param before cfg while pretranspose_B_array
    # gained a trailing 'transposed' bool. Tell the driver (winograd_conv.cpp)
    # to use the new call signatures when this layout is detected.
    if(EXISTS ${AG}/src/cpu/kernels/assembly/arm_gemm/arm_gemm.hpp)
        target_compile_definitions(arm_gemm PUBLIC ARM_GEMM_NEW_API=1)
        message(STATUS "arm_gemm: new-API layout detected (ACL >= 24.x, e.g. 53.1.0)")
    else()
        message(STATUS "arm_gemm: legacy-API layout detected (ACL 23.11)")
    endif()
    message(STATUS "arm_gemm fp32-SVE subset compiled from ${AG}")
endif()

# Include directories
include_directories(include)

# Library
add_library(winograd_conv STATIC src/winograd_conv.cpp)

# arm_gemm (built above) is required by winograd_conv when USE_ARM_GEMM is on.
if(USE_ARM_GEMM)
    target_link_libraries(winograd_conv PUBLIC arm_gemm)
endif()

# Apply OpenMP to library target (propagates to all consumers)
if(ENABLE_OPENMP)
    target_compile_options(winograd_conv PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(winograd_conv PUBLIC ${OpenMP_CXX_LIBRARIES})
    target_compile_definitions(winograd_conv PUBLIC ENABLE_OPENMP)
endif()

# Test
add_executable(test_winograd tests/test_winograd.cpp)
target_link_libraries(test_winograd winograd_conv)

# Benchmark
add_executable(bench_winograd tests/bench_winograd.cpp)
target_link_libraries(bench_winograd winograd_conv)

# Profiling (single case, perf-friendly)
add_executable(profile_case tests/profile_case.cpp)
target_link_libraries(profile_case winograd_conv)

# Archived native-NCHW reference kernel (pre-2026-08-11 layout refactor).
# Kept as a separate library so it can be linked as a correctness/performance
# baseline against the current NHWC-based NCHW wrapper. Not part of the main
# library on purpose.
add_library(winograd_nchw_ref STATIC ref/winograd_conv_nchw_ref.cpp)
target_include_directories(winograd_nchw_ref PUBLIC include ref)
if(ENABLE_OPENMP)
    target_compile_options(winograd_nchw_ref PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(winograd_nchw_ref PUBLIC ${OpenMP_CXX_LIBRARIES})
    target_compile_definitions(winograd_nchw_ref PUBLIC ENABLE_OPENMP)
endif()

# Compare the NCHW wrapper vs the archived native-NCHW kernel (must be bit-exact)
add_executable(test_ref_vs_nchw tests/test_ref_vs_nchw.cpp)
target_link_libraries(test_ref_vs_nchw winograd_nchw_ref winograd_conv)

# Swish SVE benchmark. swish_sve/ is user work that is not in git yet, so only
# build it when the sources are actually present in this checkout (otherwise the
# 920F clone without swish_sve/ would fail to configure).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/swish_sve/swish_sve_bench.cpp)
    add_executable(swish_sve_bench swish_sve/swish_sve_bench.cpp)
    target_include_directories(swish_sve_bench PRIVATE swish_sve)
    if(ENABLE_OPENMP)
        target_compile_options(swish_sve_bench PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(swish_sve_bench PRIVATE ${OpenMP_CXX_LIBRARIES})
    endif()
endif()

# Install
set(INSTALL_TARGETS winograd_conv test_winograd bench_winograd profile_case)
if(TARGET swish_sve_bench)
    list(APPEND INSTALL_TARGETS swish_sve_bench)
endif()
install(TARGETS ${INSTALL_TARGETS}
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/swish_sve)
    install(DIRECTORY swish_sve/ DESTINATION include/swish_sve)
endif()
