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

另配独立最小复现工具 `tools/arm_gemm_repro.cpp`（有 CMake 目标 `arm_gemm_repro`，
与 test_winograd 链同一库）：直接反复调用 `winograd_gemm()` 与标量参考比对，
不带任何变换/OpenMP 流水线，用于隔离「GEMM 层自身算错 / 并发状态污染」。
支持 `--threads T --iters I --bconst [M N K ...]`；`--bconst` 把 B 每行沿 K
复制成常数（模拟 identity 权重的 V 结构）。

920F 上定位顺序（2026-08-27 实测进度：①NAIVE 对照已全对 ⇒ 管线其余部分无问题；
②VERIFY_GEMM 显示连 M=1 N=3 K=3 的极小 GEMM 都错且逐位可复现 ⇒ 问题锁定在
arm_gemm 执行层；③跨引擎二分全部同错（6x4VL / 8x1VL / ""）⇒ 引擎共性层；
④test_winograd 里所有 PASS 的用例权重都是 identity（其 V 沿 K 为常数），
权重一般随机即错 ⇒ 主嫌疑是 B 面（pretranspose/K 配对），下一步跑 repro 工具的
--bconst 两步对照验证。注意：ACL>=24.x 里没有 cycle_estimator 的内核
estimate=0，find_implementation() 见 0 即「立即选中」——filter="sve_" 时 N<12
的 GEMM 一律落到 sve_hybrid_fp32_mla_8x1VL，与适不适配无关）：

```bash
# 0) 内核选择可视化（每个不同 M,N,K 打一次候选表, 标出实际选中者）
WINO_GEMM_DEBUG=1 ./test_winograd --f44 --sve 2>&1 | grep -A8 "shapes M="

# 1) 按引擎二分（不用重编, filter 直接指定具体内核名）：
for f in sve_hybrid_fp32_mla_6x4VL sve_interleaved_fp32_mla_8x3VL \
         sve_hybrid_fp32_mla_8x1VL ""; do
  echo "== filter='$f'"; WINO_GEMM_FILTER="$f" ./test_winograd --f44 2>/dev/null | tail -2
done
#    判读: 某 engine 下 PASS⇒只有它被坏配置坑; 全 SVE 坏而 ""(自动, 可能选 NEON)好
#          ⇒ SVE 子集共性错误; 仅 8x1VL 坏⇒无估计器短路选中它的问题(N<12 时)。
#
# 1.5) 最小 GEMM 复现（已证实各引擎同坏后跑这个, 不带卷积管线）：
#   ./arm_gemm_repro --iters 3            # 随机 A/B, 预期多个 case FAIL
#   ./arm_gemm_repro --iters 3 --bconst   # B 沿 K 为常数, 若全 ok ⇒ 锁定 B 面
#   两步对照判读: --bconst 全过而随机全坏 ⇒ pretranspose/K 维配对问题;
#   两者同坏 ⇒ 与 B 结构无关（回到缓冲/workspace 假设）。
#
# 2) 若确认与「哪个引擎」无关（各引擎同坏）：换 ACL 23.11 重编同一份驱动 A/B，
#    23.11 对 / 53.1.0 错 ⇒ 问题锁定在 53.1.0 库子集适配；
#    并检查实际编译开关：
grep -m2 "CXX_FLAGS\|DEFINES" CMakeFiles/arm_gemm.dir/flags.make
c++ --version; grep -m1 "COMPILER" CMakeCache.txt
```

判读要点：测试输出里 stderr 的 VERIFY 行先于 stdout 出现属正常缓冲现象；
VERBOSE 内核表的 `est=0 ... <== SELECTED` 组合即短路选中的直接证据。

## 结构优化：K 主序 V + nmulti 批量调用（2026-08-27 第二步改造）

> **⚠️ 第三步修正（2026-08-28）**：本节「Phase 1 权重散写改为 K 主序」**已回退**。
> kt 散写的每条 store 跨 OC×4B，16 条 store 才凑一条 64B 缓存行，在本机（无 L3）
> 上产生 ~16x 写放大——同作业实测 1,2048,7²,512 从打包路径的 10~12ms 涨到
> 62~65ms，与 2.4GB 脏行流量模型吻合；对照组 OpenBLAS 后端（行主序跨步读，
> 同样 16x 但在读侧）同形状 59.8ms，互相印证。**现行架构**：Phase 1 行主序连续
> 散写（`V[(m*OC+oc)*IC+ic]`，满行写）+ 驱动按 nmulti 一次性把全部面板打包成 Bt
> （跨步读限制在单个 V 面板 = OC*IC*4B 内，缓存可驻）+ 一次批量 pretranspose/execute。
> 第二步的 nmulti 批量调用收益保留。`winograd_gemm_batched_kt`/`winograd_gemm_kt`
> 入口保留（给已持有 K 主序数据的调用方），管线改走新的 `winograd_gemm_batched`
> （行主序）。旧版驱动源码快照在 `tools/old_src/winograd_conv.cpp`（c48761e，
> 供 diag_ab.sh E5 做同作业新旧对照）。

### 背景

全量 A/B 显示 arm_gemm 在中高 IC 输 OpenBLAS 1.2~2.6x，嫌疑之一是 Phase 2 的
每 ts_idx 固定开销：36 次 GEMM 对象构造 + 选核 + B 预转置/打包。第二步把两者一并消掉：

### 契约变化（对外 API 保持兼容）

| 入口 | V 布局 | 行为 |
|---|---|---|
| `winograd_gemm` | `[OC][IC]` 行主序 | **契约不变**（repro/test 仍走这条），arm_gemm 分支内部照旧做 Bt 暂存打包 |
| `winograd_gemm_kt` | `[IC][OC]` K 主序 | 管线实际产出的布局；arm_gemm 直接零拷贝绑定，OpenBLAS 换成 NN 公式 |
| `winograd_gemm_batched_kt` | 同上 ×nmulti | 把连续多个 ts_idx 切片折成**一次** arm_gemm 调用（`GemmArgs.nmulti`），对象构造/选核/B 预转置全部摊薄 |

### 内部改动

- **Phase 1 权重散写改为 K 主序**：`V[(m*IC+ic)*OC+oc] = V_oc_wt[m*IC+ic]`
  （panel m 占据 `[m*IC*OC, (m+1)*IC*OC)`，内层 [ic][oc]）。整卷一次的开销，
  换取 GEMM 阶段零 B 暂存。
- **Phase 2（USE_ARM_GEMM 分支）静态切分 + 批量**：不再 `omp for dynamic`
  逐 ts_idx 单发；按 `b = tid*NM/nth` 近均分给每个线程一段连续切片，线程内
  **一次** `winograd_gemm_batched_kt(..., cnt)` 完成（等 FLOP 切片，dynamic 无收益）。
  显式 `#pragma omp barrier` 补上原 omp-for 的隐式栅栏（Phase 3 是 nowait）。
- 驱动 `run()` 增加 `v_kmajor / nmulti` 参数：kt 时跳过 Bt 暂存直接绑
  （`B_ld=OC`）；`GemmArgs.nmulti=cnt`，`set_arrays` 传 multi 步长
  （A=n_tiles*IC，B=OC*IC，C=n_tiles*OC）；`pretranspose_B_array` 第 4 参
  （它自己的 B_multi_stride，与 set_arrays 无关）必须传 OC*IC——传 0 会把
  panel 0 重打包进所有 multi 槽。
- 已核实 ACL 库语义支持该用法：hybrid_indirect 的
  `get_B_pretransposed_array_size()` 含 `*nmulti`、`pretranspose_B_array_part`
  按 `B + multi*B_multi_stride` 循环覆盖全部面板、execute 按 multi 步长遍历
  A/B/C。
- 非 arm 后端（OpenBLAS/naive）与非 OpenMP 构建的旧循环原样保留。
- x86 类型检查三方通过：53.1.0 新签名（`ARM_GEMM_NEW_API`）、23.11 旧布局、
  无外部后端的 plain 构建（`build/x86_stubs/arm_neon.h` 为本仓库自带的
  仅声明桩，只含工程实际用到的 NEON 符号，供 -fsyntax-only 用）。

### 性能验证

编译后先跑正确性门：

```bash
WINO_GEMM_VERIFY_GEMM=1 ./test_winograd   # 或 tests/README 的验证方式
```

再对比 A/B 数据；引擎轮换定位脚本见 `tools/perf_engine_sweep.md`
（注意：本改造后固定开销已摊薄，与历史数字比较时时间基数会整体下移）。

## 根因与修复（2026-08-27 定位）

**现象**：53.1.0 下所有 SVE 引擎全错；identity 权重（V 沿 K 常数）反而 PASS；
NAIVE 对照全对。`--bconst` 裸 GEMM 对照同坏 ⇒ 排除 K 排列类。

**定位手段**：`./arm_gemm_repro --dump`（M=1 N=3 K=3，哨兵保护区）。三个输出方程
被精确解出：

    got[t][n] = Σ_k U[t][k] * V_flat[k*N + n]     （B 被按列向量消费）

**根因**：现代 arm_gemm（≥24.x / 53.1.0）对 B 的布局约定是「调用方已按 K 主序
存放」，即元素 (n,k) 位于 `ptr[k*ldb + n]`（等价于传入 Bᵀ）。此前驱动传的是行主序
N×K 的 V（ld=K），引擎把它的**列**当成 B 行用了 ⇒ U·V[:,n] 而非 U·V[n,:]。
ACL 自家流水线在进 arm_gemm 前有 Interleave/Transpose 内核预打包，掩盖了该约定。

**修复**：驱动内把每个 `V_slice` 打包成 Bt（K×N，ld=OC）后再交给
pretranspose_B_array / set_arrays（thread_local 缓冲，无锁安全）。
环境开关 `WINO_GEMM_BTRANS=0` 可回退旧行为做 A/B。
后续优化：把该转置上提到权重变换阶段（整次卷积一次），消除每 ts_idx 一次的打包。

**工具链事实**：920F 构建用毕昇 Clang 17.0.6（BiSheng Enterprise 4.2.0.2.B002），
SVE 可变长（SVE_BITS=0），fp16/bf16/i8mm 开、f32mm 关。WINO_GEMM_DEBUG 首行会打印
这些信息（banner），无需再查 flags.make。

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
