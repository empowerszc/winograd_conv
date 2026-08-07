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

# arm_gemm support (requires ACL source tree)
if(USE_ARM_GEMM)
    find_path(ARM_GEMM_INCLUDE_DIR arm_gemm.hpp
              PATHS ${ARM_GEMM_ROOT} ${ARM_GEMM_ROOT}/include)
    if(ARM_GEMM_INCLUDE_DIR)
        add_compile_definitions(USE_ARM_GEMM)
        include_directories(${ARM_GEMM_INCLUDE_DIR})
        message(STATUS "arm_gemm found: ${ARM_GEMM_INCLUDE_DIR}")
        # arm_gemm is header-only with JIT, needs to be compiled as source
        # The user needs to add arm_gemm.cpp to the build or compile it as a library
        # See: https://github.com/ARM-software/ComputeLibrary/tree/main/src/core/NEON/kernels/convolution/common/gemm
    else()
        message(FATAL_ERROR "arm_gemm.hpp not found. Set -DARM_GEMM_ROOT=/path/to/gemm")
    endif()
endif()

# Include directories
include_directories(include)

# Library
add_library(winograd_conv STATIC src/winograd_conv.cpp)

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

# Install
install(TARGETS winograd_conv test_winograd bench_winograd
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
