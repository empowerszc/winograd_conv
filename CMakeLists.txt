cmake_minimum_required(VERSION 3.14)
project(winograd_conv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ISA selection options
# At least NEON is always available on AArch64.
# SVE and SME can be enabled independently or together.
option(ENABLE_SVE "Enable SVE intrinsics (requires SVE-capable CPU)" OFF)
option(ENABLE_SME "Enable SME intrinsics (requires SME-capable CPU)" OFF)
option(ENABLE_OPENMP "Enable OpenMP threading" OFF)

if(AARCH64)
    # Base: NEON always on
    add_compile_options(-O2 -g)
    
    if(ENABLE_SVE)
        # SVE: add -march with sve feature
        add_compile_options(-march=armv8.6-a+sve2)
        add_compile_definitions(ARM_COMPUTE_ENABLE_SVE __ARM_FEATURE_SVE=1)
        message(STATUS "SVE intrinsics enabled")
    endif()
    
    if(ENABLE_SME)
        # SME: requires Armv9 + SME
        if(NOT ENABLE_SVE)
            # SME implies SVE (SME needs streaming SVE)
            add_compile_options(-march=armv9-a+sme)
            add_compile_definitions(ARM_COMPUTE_ENABLE_SVE __ARM_FEATURE_SVE=1)
        else()
            add_compile_options(-march=armv9-a+sme)
        endif()
        add_compile_definitions(ARM_COMPUTE_ENABLE_SME __ARM_FEATURE_SME=1)
        message(STATUS "SME intrinsics enabled (includes SVE)")
    endif()
    
    if(NOT ENABLE_SVE AND NOT ENABLE_SME)
        message(STATUS "Using NEON only (default)")
    endif()
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

# Library target (header-only for transforms, compiled for convolution)
add_library(winograd_conv STATIC
    src/winograd_conv.cpp
)

# Test executable
add_executable(test_winograd tests/test_winograd.cpp)
target_link_libraries(test_winograd winograd_conv)

# Install
install(TARGETS winograd_conv test_winograd
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
