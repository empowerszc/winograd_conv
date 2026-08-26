# arm_gemm JIT 集成构建指南（920F）

Winograd GEMM 改用 arm_gemm（ACL 23.11 内嵌版）fp32-SVE 子集，替代 OpenBLAS 1 线程。
arm_gemm 是 ARM 手写的 SVE/NEON 高性能 GEMM（每个内核 ~2000 行汇编），其 SVE 混合内核
正是为 920F（SVE-512, A76 类核心）这类芯片调优的。

## 改了什么

| 文件 | 作用 |
|---|---|
| `src/winograd_conv.cpp` | `winograd_gemm` 的 `USE_ARM_GEMM` 分支重写为现代 arm_gemm API（`GemmArgs` + `gemm()` 工厂 + `pretranspose_B_array` + `execute`）。每次调用单线程（`maxthreads=1`），与 OpenBLAS 基线同构。`GemmConfig.filter="sve_"` 强制 SVE 内核。 |
| `include/arm_compute/core/CPP/CPPTypes.h` | **自包含 CPUInfo shim**。arm_gemm 只通过 `arm_gemm_local.hpp` 依赖 `arm_compute::CPUModel/CPUInfo`；用这个最小头文件 + `src/arm_gemm_cpuinfo.cpp` 替代，**不再需要编译整个 ACL**。 |
| `src/arm_gemm_cpuinfo.cpp` | CPUInfo 实现：getauxval HWCAP 探测 SVE 等特性，L1/L2 缓存按 920F（64K/768K 字节）返回，模型报 A76。 |
| `CMakeLists.txt` | `USE_ARM_GEMM` 块重写：AArch64 + SVE 守卫、从 `ARM_GEMM_ROOT` 编译 fp32-SVE 子集（19 个 ACL 源 + 我们的 cpuinfo）、include 顺序让 shim 先命中。 |
| `tools/build_arm_gemm.md` | 本文件。 |

## 920F 上构建

```bash
# 1) ACL 23.11 源码在 920F 上的路径（本仓库只引用它，不提交它）
git clone --depth 1 --branch v23.11 https://github.com/ARM-software/ComputeLibrary.git
# 或从本地已有目录指定

# 2) 配置 + 编译
cd winograd_conv
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SVE=ON \
  -DENABLE_OPENMP=ON \
  -DUSE_ARM_GEMM=ON \
  -DARM_GEMM_ROOT=/path/to/ComputeLibrary-23.11/ComputeLibrary-23.11
make -j$(nproc)

# 3) 确认 SVE 内核被选中（一次打印）
WINO_GEMM_DEBUG=1 ./test_winograd
#   期望输出形如:
#   [winograd_gemm] arm_gemm selected: sve_hybrid_fp32_mla_6x4VL (M=.. N=.. K=..)
#   （也可能选 sve_interleaved_fp32_mla_8x3VL / sve_hybrid_fp32_mla_8x1VL，都是 SVE）

# 4) 正确性门（必须全 PASS）
./test_winograd --verify        # 若支持该参数；否则按 tests/README 的验证方式
./bench_winograd --verify ...

# 5) 性能 A/B（arm_gemm vs OpenBLAS）
#    a) 无 USE_ARM_GEMM 的 OpenBLAS 基线：-DUSE_OPENBLAS=ON
#    b) arm_gemm：-DUSE_ARM_GEMM=ON
#    两边同样形状、同样 --threads/--repeats，对比端到端 ms。
```

## 预期与注意事项

- **单线程 GEMM/次调用**：`maxthreads=1` 与 OpenBLAS 基线（`openblas_set_num_threads(1)`）同构，
  9/16 线程并行仍来自外层 OpenMP 对 36 个 multi-tile 的并行。
- **V 预转置是每调用一次**（与 OpenBLAS 每调用 pack B 同代价）。下一步优化：
  把 36 个 `V_slice` 的预转置缓存起来（V 在整次卷积内不变），消除该开销。
- **强制 SVE**：`cfg.filter="sve_"`。若某个 shape 实测 NEON 内核更快，改回默认（删 filter）让
  arm_gemm 自己选。
- **CPUInfo 可调**：`WINO_GEMM_L1_KB` / `WINO_GEMM_L2_KB` 覆盖缓存大小假设。
- **首次在 920F 编译可能有个别汇编/宏问题**：本机是 x86 无法编译 AArch64 SVE，驱动已按
  ACL 23.11 头文件逐行核对并在 x86 上做过 API 类型检查（`build/arm_gemm_driver_check.cpp`）。
  若有编译错误，贴出来我修。

## 为什么不用「手写 SVE GEMM」替代 arm_gemm

arm_gemm 的 SVE 混合内核本身就是 ARM 手写的高性能 SVE GEMM（汇编级调优），
我们手写一份大概率前期更慢、且无法复用它的多 shape 选择逻辑。真正的优化点是：
把 V 的预转置从「每次调用」变为「每次卷积一次」——这已列入优化计划（cache V）。
若实测后 arm_gemm 的 per-call 开销仍显著，再考虑把预转置并入权重变换阶段或手写。
（详见 why_faster §10 优化路径。）
