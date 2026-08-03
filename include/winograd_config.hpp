// winograd_config.hpp - Winograd convolution configuration
//
// This file defines the configuration for Winograd convolution.
// It selects between F(2,2,3,3) and F(4,4,3,3) based on compile-time options.
//
// Part of the winograd_conv project: a standalone reimplementation of
// ACL's Winograd convolution for AArch64 (NEON/SVE/SME).

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace winograd_conv {

// Winograd tile configuration
// F(m, r): output tile = m x m, kernel = r x r
// Input tile = (m + r - 1) x (m + r - 1)
// Number of GEMMs = (m + r - 1)^2

struct WinogradConfig {
    int output_tile_rows;  // m: output tile height
    int output_tile_cols;  // m: output tile width
    int kernel_rows;       // r: kernel height
    int kernel_cols;       // r: kernel width
    int input_tile_rows;   // m + r - 1
    int input_tile_cols;   // m + r - 1
    int n_multis;          // (m + r - 1)^2 = number of GEMMs
    bool has_normalization; // true for F(4,3) which needs /576
    float normalization;   // 1.0f / 576.0f for F(4,3), 1.0f for F(2,3)

    // Compute derived values
    static WinogradConfig F22_33() {
        // F(2,2,3,3): output 2x2, kernel 3x3, input 4x4, 16 GEMMs
        return {2, 2, 3, 3, 4, 4, 16, false, 1.0f};
    }

    static WinogradConfig F44_33() {
        // F(4,4,3,3): output 4x4, kernel 3x3, input 6x6, 36 GEMMs
        return {4, 4, 3, 3, 6, 6, 36, true, 1.0f / 576.0f};
    }

    int tile_stride_rows() const { return output_tile_rows; }
    int tile_stride_cols() const { return output_tile_cols; }

    int n_tile_rows(int oh) const {
        return (oh + output_tile_rows - 1) / output_tile_rows;
    }
    int n_tile_cols(int ow) const {
        return (ow + output_tile_cols - 1) / output_tile_cols;
    }
};

// ISA detection for selecting transform implementations
enum class ISALevel {
    NEON,   // Always available on AArch64 (128-bit, 4 floats)
    SVE,    // SVE (scalable, up to 512-bit = 16 floats)
    SME,    // SME (matrix tile + FMOPA)
};

// Get the best available ISA (compile-time detection)
inline ISALevel detect_isa() {
#if defined(__ARM_FEATURE_SME)
    return ISALevel::SME;
#elif defined(__ARM_FEATURE_SVE)
    return ISALevel::SVE;
#else
    return ISALevel::NEON;
#endif
}

// Runtime ISA selection (can be overridden by user)
// Set via set_isa_level() or environment variable WINOGRAD_ISA
inline ISALevel& isa_level() {
    static ISALevel level = detect_isa();
    return level;
}

inline void set_isa_level(ISALevel level) {
    isa_level() = level;
}

// Parse ISA level from string
inline ISALevel parse_isa(const char* str) {
    if (str) {
        if (strcmp(str, "neon") == 0 || strcmp(str, "NEON") == 0)
            return ISALevel::NEON;
        if (strcmp(str, "sve") == 0 || strcmp(str, "SVE") == 0)
            return ISALevel::SVE;
        if (strcmp(str, "sme") == 0 || strcmp(str, "SME") == 0)
            return ISALevel::SME;
    }
    return detect_isa();
}

// Get ISA name string
inline const char* isa_name(ISALevel level) {
    switch (level) {
        case ISALevel::NEON: return "NEON";
        case ISALevel::SVE:  return "SVE";
        case ISALevel::SME:  return "SME";
    }
    return "UNKNOWN";
}

// Vector length in floats
inline int vec_length() {
#if defined(__ARM_FEATURE_SVE)
    return 16;  // SVE-512: 16 floats (simplified; actual is runtime)
#else
    return 4;   // NEON: 4 floats
#endif
}

} // namespace winograd_conv
