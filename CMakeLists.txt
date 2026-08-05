cmake_minimum_required(VERSION 3.14)
project(winograd_conv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ISA selection options
option(ENABLE_SVE "Enable SVE intrinsics (requires SVE-capable CPU)" OFF)
option(ENABLE_SME "Enable SME intrinsics (requires SME-capable CPU)" OFF)
option(ENABLE_OPENMP "Enable OpenMP threading" OFF)

# Detect AArch64 platform
set(IS_AARCH64 OFF)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(IS_AARCH64 ON)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(IS_AARCH64 ON)
endif()

if(IS_AARCH64)
    add_compile_options(-O3 -g)

    if(ENABLE_SME)
        # SME requires Armv9 + SME (implies SVE in streaming mode)
        add_compile_options(-march=armv9-a+sme)
        message(STATUS "SME+SVE enabled (-march=armv9-a+sme)")
    elseif(ENABLE_SVE)
        add_compile_options(-march=armv8.6-a+sve2)
        message(STATUS "SVE enabled (-march=armv8.6-a+sve2)")
    else()
        message(STATUS "Using NEON only (default)")
    endif()
else()
    message(WARNING "Not AArch64 platform; ISA options ignored")
endif()

# OpenMP support
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_libraries(${OpenMP_CXX_LIBRARIES})
    message(STATUS "OpenMP enabled")
endif()

# Include directories
include_directories(include)

# Library target
add_library(winograd_conv STATIC
    src/winograd_conv.cpp
)

# Test executable
add_executable(test_winograd tests/test_winograd.cpp)
target_link_libraries(test_winograd winograd_conv)

# Benchmark executable
add_executable(bench_winograd tests/bench_winograd.cpp)
target_link_libraries(bench_winograd winograd_conv)

# Install
install(TARGETS winograd_conv test_winograd bench_winograd
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
