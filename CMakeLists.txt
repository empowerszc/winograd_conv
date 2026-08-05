cmake_minimum_required(VERSION 3.14)
project(winograd_conv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ISA selection options
option(ENABLE_SVE "Enable SVE (-march=armv8.6-a+sve2)" OFF)
option(ENABLE_SME "Enable SME+SVE (-march=armv9-a+sme)" OFF)
option(ENABLE_OPENMP "Enable OpenMP threading" OFF)
option(USE_OPENBLAS "Use OpenBLAS for GEMM (cblas_sgemm)" OFF)
set(OPENBLAS_ROOT "" CACHE PATH "Path to OpenBLAS installation (e.g. /usr/local/openblas)")

# Base optimization
add_compile_options(-O3 -g)

# ISA flags — applied directly, no platform detection needed.
# If compiler doesn't support the -march, it will error (expected behavior).
if(ENABLE_SME)
    add_compile_options(-march=armv9-a+sme)
    message(STATUS "=== SME+SVE enabled (-march=armv9-a+sme) ===")
elseif(ENABLE_SVE)
    add_compile_options(-march=armv8.6-a+sve2)
    message(STATUS "=== SVE enabled (-march=armv8.6-a+sve2) ===")
else()
    message(STATUS "=== NEON only (default) ===")
endif()

# OpenMP support
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_compile_options(${OpenMP_CXX_FLAGS})
    add_compile_definitions(ENABLE_OPENMP)
    link_libraries(${OpenMP_CXX_LIBRARIES})
    message(STATUS "OpenMP enabled")
endif()

# OpenBLAS support for GEMM
if(USE_OPENBLAS)
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
        message(FATAL_ERROR "OpenBLAS not found. Install openblas-dev or set -DOPENBLAS_ROOT=/path/to/openblas")
    endif()
endif()

# Include directories
include_directories(include)

# Library
add_library(winograd_conv STATIC src/winograd_conv.cpp)

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
