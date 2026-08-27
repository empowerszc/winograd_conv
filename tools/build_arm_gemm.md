# arm_gemm JIT 集成构建指南（920F）

Winograd GEMM 改用 arm_gemm（ACL 内嵌版，23.11 / 53.1.0 均可）fp32-SVE 子集，替代 OpenBLAS 1 线程。
arm_gemm 是 ARM 手写的 SVE/NEON 高性能 GEMM（每个内核 ~2000 行汇编），其 SVE 混合内核
正是为 920F（SVE-512, A76 类核心）这类芯片调优的。

## 版本兼容（23.11 vs 53.1.0）

CMake 自动探测两种布局，无需手改：

| | 23.11 | 53.1.0 |
|---|---|---|
| arm_gemm.hpp | `src/cpu/kernels/assembly/` | `src/cpu/kernels/assembly/arm_gemm/` |
| misc-sve.cpp | 存在 | **已并入 misc.cpp**（不再单独编译） |
| `GemmArgs` | 13 参数 | **多了 `accumulate`**（cfg 前） |
| `pretranspose_B_array` | 4 参数 | **多了尾部 `transposed` bool**（无默认值） |
| `gemm<T,T>()` | `Tret=Tlop` 默认 | **必须显式 `gemm<T,T,T>()`** |
| 头文件 `<string>` | 传递包含偶然覆盖 | **arm_gemm 头文件自己缺 `#include <string>`**（见下） |
| `get_gemm_method<T,T>()` | 有定义 | **只声明无定义（dead API）**，驱动调试打印改用 `cfg.filter` |
| fp32 内核源码 | 全在 `generic.cpp` | **拆成 per-variant 文件**（`a53/a55/a55r1/x1`、`_a64fx`），CMake 按 `if(EXISTS)` 追加 |

检测到新布局时 CMake 定义 `ARM_GEMM_NEW_API`，驱动自动用新签名
（见 `src/winograd_conv.cpp` arm_gemm_driver 的 `#if defined(ARM_GEMM_NEW_API)`）。
若在 53.1.0 上误用了旧签名：`&cfg` 会错位绑定到 `accumulate`（变成每次都累加、
filter 失效），pretranspose 4 参调用直接编译失败。x86 上已用 53.1.0 头文件做过 API 类型检查。

> **53.1.0 的 `<string>` 缺失**：`arm_gemm.hpp`（`KernelDescription.name`、
> `GemmConfig.filter`）和 `arm_common/internal/utils.hpp`（`get_type_name`）都用
> `std::string`，但整个头文件链都不 `#include <string>`，只经 `<memory>`→`iosfwd`
> 拿到前向声明 → 几十个「incomplete type」级联报错。CMake 用
> `target_compile_options(arm_gemm PRIVATE -include string)` 强制预包含修复，
> 不动 ACL 源码。（23.11 靠传递包含侥幸通过。）

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
# 1) ACL 源码在 920F 上的路径（本仓库只引用它，不提交它）。23.11 或 53.1.0 均可：
git clone --depth 1 --branch v23.11 https://github.com/ARM-software/ComputeLibrary.git
# 或已有 53.1.0：/workspace/.../ComputeLibrary-53.1.0

# 2) 先 git pull 拿最新 master（含 swish_sve 存在性守卫 + 53.1.0 适配）
cd winograd_conv && git pull

# 3) 配置 + 编译
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SVE=ON \
  -DENABLE_OPENMP=ON \
  -DUSE_ARM_GEMM=ON \
  -DARM_GEMM_ROOT=/workspace/.../ComputeLibrary-53.1.0   # 或 23.11 路径
make -j$(nproc)

# 4) 确认 SVE 内核被选中（一次打印）
WINO_GEMM_DEBUG=1 ./test_winograd
#   期望输出形如:
#   [winograd_gemm] arm_gemm selected: sve_hybrid_fp32_mla_6x4VL (M=.. N=.. K=..)
#   （也可能选 sve_interleaved_fp32_mla_8x3VL / sve_hybrid_fp32_mla_8x1VL，都是 SVE）

# 5) 正确性门（必须全 PASS）
./test_winograd --verify        # 若支持该参数；否则按 tests/README 的验证方式
./bench_winograd --verify ...

# 6) 性能 A/B（arm_gemm vs OpenBLAS）
#    a) 无 USE_ARM_GEMM 的 OpenBLAS 基线：-DUSE_OPENBLAS=ON
#    b) arm_gemm：-DUSE_ARM_GEMM=ON
#    两边同样形状、同样 --threads/--repeats，对比端到端 ms。
```

## 数值错误排查（arm_gemm 结果不对时）

驱动内置诊断开关（默认全关，不改运行行为），2026-08 为排查「接入 53.1.0 后
test_winograd 大面积 FAIL（误差随 IC 增长、跨次运行不稳定）」而加：

| 开关 | 作用 |
|---|---|
| `WINO_GEMM_NAIVE=1` | 完全绕过 arm_gemm，走标量三重循环（对照组：其余管线应 22/22 PASS） |
| `WINO_GEMM_FILTER=<子串>` | 覆盖内核过滤串；置空 `""` 让 arm_gemm 自由选（含 NEON 内核），可逐个试 `sve_hybrid_fp32_mla_6x4VL` 等具体内核名 |
| `WINO_GEMM_VERIFY_GEMM=1` | 每次 execute 后用标量循环复算并比对，前 10 处 mismatch 打到 stderr（慢一倍，仅诊断用） |
| `WINO_GEMM_DEBUG=1` | 打印 filter/M/N/K（每进程一次，已修 OpenMP 多线程重复打印竞态） |

另配独立最小复现工具（不进 git，`build/arm_gemm_repro.cpp`）：直接反复调用
`winograd_gemm()` 与标量参考比对，不带任何变换/OpenMP 流水线，用于隔离
「GEMM 层自身算错 / 并发状态污染」。把它临时加进 CMake 编一个
`arm_gemm_repro` 目标，或按 test_winograd 同款链接参数手工编。

920F 上定位顺序（依次贴回输出）：

```bash
# 1) 对照组：关 arm_gemm —— 若仍 FAIL 则问题根本不在 GEMM 层
WINO_GEMM_NAIVE=1 ./test_winograd --sve --nhwc

# 2) GEMM 自检：定位是 GEMM 段算错还是外围段错、误差量级
WINO_GEMM_VERIFY_GEMM=1 ./test_winograd --f44 --sve

# 3) 解除 SVE 强制：NEON 内核对 vs SVE 内核错 ⇒ 指向 SVE 内核子集编译
WINO_GEMM_FILTER= ./test_winograd --f44 --sve

# 4) 最小 GEMM 复现（无变换）：单线程 & 多线程各自哪些 shape 错
./arm_gemm_repro                        # 默认 shape 扫描, 单线程
./arm_gemm_repro --threads 16           # 同上, 16 路并发(暴露静态/共享态污染)

# 5) 若 #3 下 NEON 全对而 SVE 全错：降级验证内核源文件编译选项
#    确认 arm_gemm 目标拿到了 -march 含 sve 且 ARM_COMPUTE_ENABLE_SVE：
grep -m1 "flags" CMakeFiles/arm_gemm.dir/flags.make
grep -m1 "ARM_COMPUTE_ENABLE_SVE" CMakeFiles/arm_gemm.dir/flags.make

# 6) 终极对照：换 ACL 23.11 重编同一份驱动（CMake 兼容两版）
#    23.11 对 / 53.1.0 错 ⇒ 问题锁定在 53.1.0 库子集的适配，非驱动逻辑。
```

已知现象记录（53.1.0, 2026-08-27）：变换单项测试与 identity-kernel 端到端均
PASS，随机权重即错且误差随 IC 近似线性增长；同二进制多次运行失败模式不同
（一次恒等/随机双双通过后手动管线错误 0.99，另一次 Manual=GEMM 正确而整管
线乱数 5 万+）⇒ 高度怀疑非确定性行为（并发/静态状态/内核选择变化），而非
确定性参数绑定错误——驱动的全部 API 调用已逐一对照 ACL 自己的
`CpuGemmAssemblyDispatch.cpp` 用法核实。

## 预期与注意事项

- **单线程 GEMM/次调用**：`maxthreads=1` 与 OpenBLAS 基线（`openblas_set_num_threads(1)`）同构，
  9/16 线程并行仍来自外层 OpenMP 对 36 个 multi-tile 的并行。
- **V 预转置是每调用一次**（与 OpenBLAS 每调用 pack B 同代价）。下一步优化：
  把 36 个 `V_slice` 的预转置缓存起来（V 在整次卷积内不变），消除该开销。
- **强制 SVE**：`cfg.filter="sve_"`。若某个 shape 实测 NEON 内核更快，改回默认（删 filter）让
  arm_gemm 自己选。
- **CPUInfo 可调**：`WINO_GEMM_L1_KB` / `WINO_GEMM_L2_KB` 覆盖缓存大小假设。
- **首次在 920F 编译可能有个别汇编/宏问题**：本机是 x86 无法编译 AArch64 SVE，驱动已按
  ACL 23.11 / 53.1.0 头文件逐行核对，并分别在两套布局（旧签名 / `ARM_GEMM_NEW_API`）
  下做过 x86 上的 API 类型检查（`build/arm_gemm_driver_check.cpp`，用
  `-DARM_GEMM_NEW_API` 切换新签名）。若有编译错误，贴出来我修。

## 为什么不用「手写 SVE GEMM」替代 arm_gemm

arm_gemm 的 SVE 混合内核本身就是 ARM 手写的高性能 SVE GEMM（汇编级调优），
我们手写一份大概率前期更慢、且无法复用它的多 shape 选择逻辑。真正的优化点是：
把 V 的预转置从「每次调用」变为「每次卷积一次」——这已列入优化计划（cache V）。
若实测后 arm_gemm 的 per-call 开销仍显著，再考虑把预转置并入权重变换阶段或手写。
（详见 why_faster §10 优化路径。）

arm_gemm 的 SVE 混合内核本身就是 ARM 手写的高性能 SVE GEMM（汇编级调优），
我们手写一份大概率前期更慢、且无法复用它的多 shape 选择逻辑。真正的优化点是：
把 V 的预转置从「每次调用」变为「每次卷积一次」——这已列入优化计划（cache V）。
若实测后 arm_gemm 的 per-call 开销仍显著，再考虑把预转置并入权重变换阶段或手写。
（详见 why_faster §10 优化路径。）
