// winograd_conv_nchw_ref.hpp - Declares the archived native-NCHW kernel
//
// See winograd_conv_nchw_ref.cpp for the full explanation. This kernel is a
// frozen snapshot of the pre-2026-08-11 NCHW path and is kept only as a
// buildable reference for correctness/performance comparison against the new
// NHWC-based wrapper in src/winograd_conv.cpp.

#pragma once

#include "winograd_config.hpp"

namespace winograd_conv {

// Archived native-NCHW Winograd convolution (always operates on NCHW layout).
// Semantics are identical to the public winograd_convolution(..., NCHW).
void winograd_convolution_nchw_ref(
    const float* src,   // [N][IC][IH][IW]
    const float* wei,   // [OC][IC][3][3]
    const float* bias,  // [OC] or nullptr
    float* dst,         // [N][OC][OH][OW]
    int N, int IC, int IH, int IW,
    int OC, int OH, int OW,
    const WinogradConfig& config,
    float act_min = -1e30f, float act_max = 1e30f
);

} // namespace winograd_conv
