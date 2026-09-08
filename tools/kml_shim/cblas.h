// cblas.h — KML kblas compatibility shim.
//
// Huawei KML ships a cblas-compatible BLAS (same cblas_sgemm signature and
// CblasRowMajor/CblasNoTrans enums as OpenBLAS). The winograd code includes
// <cblas.h> under USE_OPENBLAS; on a KML-only machine this shim redirects that
// include to <kblas.h> so the existing OpenBLAS code path builds unchanged.
//
// Selected by the USE_KML CMake option (see CMakeLists.txt), which also adds
// tools/kml_shim/kml_shim.cpp to provide the OpenBLAS-only symbols.
#pragma once
#include <kblas.h>
