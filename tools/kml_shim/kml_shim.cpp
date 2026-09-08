// kml_shim.cpp — OpenBLAS-only symbols for the KML cblas backend.
//
// The winograd code calls openblas_set_num_threads(1) once per convolution to
// keep the GEMM single-threaded (parallelism lives in the outer OpenMP tile
// loop). USE_KML links KML's *nolocking* (inherently single-threaded) kblas
// build, so this call is a no-op there. If you ever point USE_KML at KML's
// multi-threaded "multi" build, replace this with a real KML thread cap.
extern "C" void openblas_set_num_threads(int) {}
