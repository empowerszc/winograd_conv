# Winograd 卷积复刻项目

> 基于 ACL（Arm Compute Library）的 Winograd 卷积实现思路，用 C++ + NEON/SVE intrinsics + SME 内联汇编复刻的独立可运行项目。
>
> 支持 F(2,2,3,3) 和 F(4,4,3,3) 两种 Winograd 配置，包含权重/输入/输出三步变换、GEMM（OpenBLAS/arm_gemm/naive 三选一）、端到端卷积、NHWC 计算核心 + NCHW 转换包装、OpenMP 多线程、以及与直接卷积的正确性验证。
> 
> 在鲲鹏 920F（Armv9 + SVE-512 + SME, 16 NUMA × 38 核）上，**8/9 测试 case 的 16 线程性能超越 oneDNN**（快 1.68~3.54x，A1/A2/A3 落地后复测）。
>
> 逐 case 正确性经 **fp64 直接卷积参考**验证：fp32 舍入误差相对 ~2-6e-6（随 IC 增长的是绝对误差，相对误差恒定），符合 fp32 精度预期。

## 目录结构

```
winograd_conv/
├── CMakeLists.txt                         ← 构建配置
├── AGENTS.md                              ← AI 代理指南
├── README.md                              ← 本文件
├── PERFORMANCE_ANALYSIS.md                ← 性能分析与优化历程（含新思路 A-F）
├── include/
│   ├── winograd_config.hpp                ← 配置 + ISA 检测/选择 + Layout 枚举
│   ├── winograd_matrices.hpp              ← G/B^T/A^T 变换矩阵
│   ├── winograd_transforms.hpp            ← NEON intrinsics 变换
│   ├── winograd_transforms_sve.hpp        ← SVE intrinsics 变换
│   ├── winograd_transforms_sme.hpp        ← SME FMOPA 内联汇编变换
│   └── winograd_convolution.hpp           ← 端到端接口 + dispatch 声明
├── src/
│   └── winograd_conv.cpp                  ← 端到端实现 + GEMM + 直接卷积 + dispatch（NHWC 计算核心）
├── ref/
│   ├── winograd_conv_nchw_ref.hpp         ← 存档原生 NCHW 内核声明（2026-08-11 布局重构前）
│   └── winograd_conv_nchw_ref.cpp         ← 冻结快照，可 build 成库做对照基准
├── tests/
│   ├── test_winograd.cpp                  ← 正确性验证 + 变换调试
│   ├── test_ref_vs_nchw.cpp               ← NCHW 包装 vs 原生 ref 必须 bit-exact
│   ├── bench_winograd.cpp                 ← 性能基准（读 CSV，测 GFLOPS）
│   └── profile_case.cpp                   ← 单 case profiling（perf 友好，含 --timing）
└── docs/
    ├── algorithm.md                       ← 算法详解（数学、布局、缓冲、并行、ISA 内核、精度）
    ├── why_faster_than_acl_23.11.md       ← 为什么比 ACL 23.11 快（独立分析，含例子）
    └── acl_reference/                     ← ACL 参考文档（从 oneDNN 源码树复制）
        ├── README.md                      ← 文档索引与使用场景
        ├── acl_wino_neon_intrinsics_annotated.md
        ├── acl_wino_neon_asm_annotated.md
        ├── acl_wino_sve_asm_annotated.md
        ├── acl_wino_sme_asm_annotated.md
        ├── acl_wino_implementation_details.md
        ├── acl_wino_transform_kernels_explained.md
        ├── acl_23.11_wino_transform_kernels_explained.md
        └── acl_23.11_vs_53.1.0_wino_analysis.md
```

## 快速开始

### 编译（AArch64 环境）

```bash
# 重要：必须清除旧缓存，否则 CMake 会用旧的 CMakeLists.txt 配置
rm -rf build && mkdir build && cd build

# 默认：NEON only
cmake .. -DCMAKE_BUILD_TYPE=Release

# 启用 SVE
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SVE=ON

# 启用 SME（包含 SVE）
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SME=ON

# 启用 OpenBLAS 高性能 GEMM
cmake .. -DUSE_OPENBLAS=ON

# 或用 arm_gemm JIT GEMM（与 oneDNN 相同，需 ACL 源码树）
cmake .. -DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/gemm

# 全部启用
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON -DUSE_OPENBLAS=ON

make -j
```

> **注意**：修改 CMakeLists.txt 后必须清除缓存（`rm -rf build`），否则旧的编译选项不会更新。

### 运行验证

```bash
# 自动检测 ISA
./test_winograd

# 强制使用 SME（需要 -DENABLE_SME=ON 编译）
./test_winograd --sme

# 测试 NHWC 布局（12 NCHW + 10 NHWC = 22 tests）
./test_winograd --nhwc

# NCHW 包装 vs 存档原生 NCHW 参考（必须 bit-exact，8 shape × F44+F22）
./test_ref_vs_nchw
# 加 --bench 同测两条路径耗时，判定转换包装 vs 原生 NCHW 是否净赚（ratio<1=包装快）
./test_ref_vs_nchw --bench --threads 16 --warmup 3 --repeats 10

# 或通过环境变量
WINOGRAD_ISA=sme ./test_winograd
```

### 运行性能测试

```bash
# 准备 CSV 文件（逗号分隔，第一行表头）
# mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count
# 自动过滤 stride=1, group=1, 3x3 kernel, pad=1 的行

# 标准模式（多线程对比）
./bench_winograd --sve --nhwc --threads 1,8,16,32,38 shapes.csv

# 细粒度计时模式（各阶段时间占比）
./bench_winograd --timing --threads 16 --sve --nhwc shapes.csv

# 输出结果到 CSV
./bench_winograd --sve --nhwc --threads 32 --output result.csv shapes.csv

# 逐 case 正确性验证（与 fp64 直接卷积对比，相对容差 1e-4，任一 FAIL 则中止）
./bench_winograd --sve --nhwc --verify shapes.csv

# NUMA 优化
numactl --interleave=all ./bench_winograd --sve --nhwc --threads 32 shapes.csv
```

### 单 case profiling（perf 友好）

```bash
# 查看帮助（含 9 个 case preset）
./profile_case --help

# 标准模式：测某个 case 的性能
./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16

# 细粒度计时：8 个子步骤 + 占比
./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --timing

# 正确性验证
./profile_case --ic 96 --ih 80 --iw 80 --oc 96 --verify

# 获取 perf 命令建议（cache/topdown/SPE/NUMA/flamegraph 5 种）
./profile_case --ic 384 --ih 80 --iw 80 --oc 96 --perf

# 实际 perf profiling
perf stat -e cycles,instructions,cache-misses,cache-references \
  ./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --repeats 1

# Flame graph
perf record -g -- ./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --repeats 10
perf script | flamegraph.pl > flame.svg
```

输出示例：
```
=== Winograd Convolution Profiling ===
Shape: N=4 IC=192 IH=40 IW=40 OC=192 OH=40 OW=40
ISA: SVE | Layout: NHWC | Threads: 16 | GEMM: OpenBLAS (single-thread)
Tiles: 100 (10x10) | Winograd: F(4,4,3,3) | GEMMs: 36

--- Standard Mode ---
  Best of 20: 7.201 ms | 583.86 GFLOPS

--- Fine-Grained Timing (serial, 1 thread) ---
  Weight transform      1.96 ms  (12.4%)
  Tile extract          1.08 ms  ( 6.8%)
  Input transform       2.78 ms  (17.6%)
  ...
```

## 算法概述

### Winograd 卷积三步流程

```
直接卷积: src ──[9 次乘法/像素]──→ dst

Winograd: src ──[① 输入变换]──→ U ──[② GEMM]──→ M ──[③ 输出变换]──→ dst
                  B^T·d·B         (少量乘法)       A^T·M·A (+bias+ReLU)
```

### 两种配置

| | F(2,2,3,3) | F(4,4,3,3) |
|---|---|---|
| 输出 tile | 2×2 | 4×4 |
| 输入 tile | 4×4 | 6×6 |
| GEMM 数 | 16 | 36 |
| 每像素等效乘法 | 4 (原来 9) | 2.25 |
| 变换乘法 | 0 (全加减) | ~90 |

### 变换矩阵

**F(2,2,3,3)**:
```
G   = [1,0,0; 0.5,0.5,0.5; 0.5,-0.5,0.5; 0,0,1]       (4×3), norm=1
B^T = [1,0,-1,0; 0,1,1,0; 0,-1,1,0; 0,1,0,-1]          (4×4)
A^T = [1,1,1,0; 0,1,-1,-1]                              (2×4)
```

**F(4,4,3,3)**:
```
G   = [6,0,0; -4,-4,-4; -4,4,-4; 1,2,4; 1,-2,4; 0,0,24]  (6×3), norm=1/576
B^T = [4,0,-5,0,1,0; 0,-4,-4,1,1,0; 0,4,-4,-1,1,0; 0,-2,-1,2,1,0; 0,2,-1,-2,1,0; 0,4,0,-5,0,1]  (6×6)
A^T = [1,1,1,1,1,0; 0,1,-1,2,-2,0; 0,1,1,4,4,0; 0,1,-1,8,-8,1]  (4×6)
```

> 矩阵系数经 Winograd 多项式条件数值验证（误差 < 1e-6）。F44_B^T 的正确值通过求解器从 A^T 和 G 反解得到。

## 文件说明

### `winograd_config.hpp`
- `WinogradConfig` 结构体：定义 tile 大小、kernel 大小、GEMM 数量
- `F22_33()` / `F44_33()`：创建 F(2,2,3,3) / F(4,4,3,3) 配置
- `detect_isa()`：编译时检测 NEON/SVE/SME
- `set_isa_level()` / `parse_isa()`：运行时 ISA 选择

### `winograd_matrices.hpp`
- 定义 F(2,2,3,3) 和 F(4,4,3,3) 的 G、B^T、A^T 矩阵
- F(4,4,3,3) 的 G 用整数系数（最大 24），归一化因子 = 1/576 = 1/24²

### `winograd_transforms.hpp` (NEON)
- `transform_1d_neon<>()`：通用 1D 矩阵-向量变换，4 通道并行 + 3 段 tail 降级
- `transform_2d_neon<>()`：2D 变换 = 两次 1D 变换（列变换 + 行变换）
- `weight_transform_neon<>()`：权重变换 V = G·g·G^T/norm
- `input_transform_neon<>()`：输入变换 U = B^T·d·B
- `output_transform_neon<>()`：输出变换 f = A^T·M·A + bias + ReLU clamp

### `winograd_transforms_sve.hpp` (SVE)
- 与 NEON 版完全对应的 SVE intrinsics 实现
- `svld1_f32` + `svwhilelt_b32` 谓词化加载，自动处理 tail（无需降级）
- `svmla_n_f32_x` 谓词化乘加，VL 通道并行（SVE-512 时 16 float/指令）

### `winograd_transforms_sme.hpp` (SME)
- **输入变换**：SVE 指令 + `SMSTART ZA`/`SMSTOP` 包裹（流式模式 SVE）
- **输出变换**：直接移植 ACL `sme_fp32_mopa_4x4_3x3.cpp` 的内联汇编
  - 157 条 FMOPA 指令（`.inst` 编码）
  - 69 条 MOVA 指令读出 ZA tile
  - Kronecker 积 + 外积累加算法
- **权重变换**：复用 NEON（权重变换非热路径，ACL 也只有 NEON 版）

### `winograd_conv.cpp`
- `dispatch_weight/input/output_transform()`：根据 ISA 选择 NEON/SVE/SME 实现
- `winograd_gemm()`：3 种 GEMM 内核（编译期选择：arm_gemm > OpenBLAS > naive）
- `direct_convolution_3x3()`：参考直接卷积（用于验证正确性）
- `winograd_convolution()`：端到端 Winograd 卷积
  - 权重变换 + 输入变换 + GEMM + 输出变换全部在单个 `#pragma omp parallel` 区域内
  - **计算核心只认 NHWC**（`winograd_convolution_nhwc_core`）；`Layout::NCHW` 是包装：`nchw_to_nhwc` 转换 → 计算 → `nhwc_to_nchw` 转回（转换方案未在目标机实测，见 §12）
  - 原生 NCHW 内核存档在 `ref/`（`winograd_convolution_nchw_ref`），`test_ref_vs_nchw` 断言包装与它 bit-exact

### `test_winograd.cpp`
- F(4,4,3,3) 变换级调试（权重/输入/输出/全流水线 + B^T 矩阵求解器）
- 12 个 NCHW 测试 + 10 个 NHWC 测试（`--nhwc`），对比直接卷积，容差 1e-3

## ISA 调度机制

项目支持三种 ISA，可运行时切换：

| ISA | 编译选项 | 运行选项 | 环境变量 | 说明 |
|-----|---------|---------|---------|------|
| NEON | (默认) | `--neon` | `WINOGRAD_ISA=neon` | 128-bit，4 float/指令 |
| SVE | `-DENABLE_SVE=ON` | `--sve` | `WINOGRAD_ISA=sve` | 可伸缩，16 float/指令（SVE-512） |
| SME | `-DENABLE_SME=ON` | `--sme` | `WINOGRAD_ISA=sme` | 矩阵 tile + FMOPA |

### 三种 ISA 的实现差异

| 方面 | NEON | SVE | SME |
|------|------|-----|-----|
| 文件 | `winograd_transforms.hpp` | `winograd_transforms_sve.hpp` | `winograd_transforms_sme.hpp` |
| 向量宽度 | 128 bit (4 float) | VL (16 float, SVE-512) | VL (16 float, 流式模式) |
| 加载/存储 | `vld1q_f32` | `svld1_f32` + 谓词 | 同 SVE + SMSTART/SMSTOP |
| 乘加 | `vmlaq_n_f32` | `svmla_n_f32_x` | FMOPA（输出变换） |
| tail 处理 | 3 段降级 (4→2→1) | 谓词掩码（无降级） | 同 SVE |
| 输出变换算法 | 两次 1D 矩阵乘 | 同 NEON | Kronecker 积 + FMOPA |
| 通道并行度 | 4 | 16 | 64（4×VL） |

### 运行时 ISA 选择

```cpp
// 方式 1：编译时自动检测
ISALevel isa = detect_isa();  // NEON < SVE < SME

// 方式 2：运行时设置
set_isa_level(ISALevel::SVE);  // 强制用 SVE

// 方式 3：环境变量
// export WINOGRAD_ISA=sve
// 自动被 winograd_convolution() 读取

// 方式 4：命令行
// ./test_winograd --sve
```

## 与 ACL 实现的对比

| 方面 | 本项目 | ACL |
|------|--------|-----|
| 变换实现 | NEON/SVE intrinsics + SME .inst | NEON/SVE 汇编 + SME 汇编 |
| GEMM | 3 种可选：arm_gemm JIT > OpenBLAS > naive | arm_gemm 库（自动选 SVE/SME 内核） |
| 数据布局 | NHWC 计算核心；NCHW 经转换包装（原生 NCHW 存档于 `ref/`） | NHWC + Winograd 专用布局 |
| 多线程 | OpenMP（权重+输入+GEMM+输出 全并行，合并区域） | NEScheduler |
| 正确性 | ✅ 22/22 验证通过（NCHW + NHWC）；bench `--verify` 用 fp64 参考 + 相对容差 | ✅ |
| 性能 | **8/9 case 超越 oneDNN**（A1/A2/A3 后快 1.68~3.54x），仅 Case 2 慢 1.13x | 高度优化 |

### 性能分析

使用 `--timing` 模式可查看各阶段时间占比：
```bash
./bench_winograd --timing --threads 16 --sve --nhwc shapes.csv
```

细粒度计时将输入/输出各拆为 3 个子步骤：

```
输入侧: TileExt(NCHW/NHWC提取) → InXform(变换) → InScat(NEON拷贝)
输出侧: OutGath(NEON拷贝) → OutXform(变换+ReLU) → OutWrite(NCHW/NHWC写回)
```

实测数据（4×192×40×40, NHWC, SVE, 1 线程, timing 模式）：

| 阶段 | 时间(ms) | 占比 | 瓶颈 |
|------|---------|------|------|
| 权重变换 | 1.96 | 12% | 已并行（#pragma omp for over OC） |
| 输入变换 | 2.78 | 18% | NEON/SVE 计算 |
| 输出变换 | 2.02 | 13% | NEON/SVE 计算 |
| GEMM | 1.22 | 8% | OpenBLAS |
| NEON gather/scatter | 2.59 | 17% | 向量化拷贝 |
| NHWC tile 提取 | 1.08 | 7% | NEON 连续读 |
| NHWC 写回 | 0.38 | 2% | NEON 连续写 |

**主要瓶颈**（针对慢 case 0/1/2）：OpenMP barrier 开销（多 tile + 小 IC 时占比大）、GEMM 内核质量（OpenBLAS 对小 K 矩阵不如 arm_gemm JIT）、变换实现（泛型模板 vs ACL 手写展开）。

**已实施的优化**：
1. NHWC 布局（tile 提取 3.8x 加速，写回 7.4x）
2. GEMM 并行（`#pragma omp for` over 36 Winograd domain elements）
3. 合并 OpenMP 区域（权重+输入+GEMM+输出 全在 1 个 `#pragma omp parallel`）
4. `schedule(dynamic, 2)` 减少 tile 调度开销
5. `thread_local static float*` + `malloc` 替代 `std::vector::resize`
6. 内部 tile 跳过 `memset` 清零 + 预计算有效行列范围
7. 输出变换 `nowait` 消除多余 barrier
8. OpenBLAS `openblas_set_num_threads(1)` 避免 GEMM 线程冲突
9. 权重变换并行（`#pragma omp for` over OC，合并进主 region）
10. arm_gemm JIT GEMM 内核可选（与 oneDNN 相同，`-DUSE_ARM_GEMM=ON`）
11. SVE 化内存拷贝（`copy_f32()`：tile 提取/scatter/gather/写回，SVE-512 16 float/指令，NEON 构建回退 4 float）
12. 消除 U/M_buf/V 无用清零 + per-thread 缓冲跨调用复用（`scratch_f32()` thread_local 增长式）

**优化历程**（Case 0, t32）：
```
① NCHW 基线:        14.0ms (3.89x vs oneDNN)
③ +GEMM并行:         9.1ms (2.53x)
④ +合并OMP区域:      6.7ms (1.86x)
⑦ +权重并行(合并):    7.0ms (1.94x, Case 3-8 改善明显)
```

**最终结果**（2026-08-10 A1/A2/A3 后复测, 16 线程, 9 case）：
- **8/9 case 超越 oneDNN**（快 1.68x~3.54x）；仅 Case 2 慢 1.13x（从慢 2.56x 追到 11.5%）
- t16 几何平均较旧版最优加速 ~1.67x；Case 0/1/2（拷贝/缓冲受限）加速 2.3~3.7x
- Case 4/5（大 IC，GEMM 受限）仅 ~1.1x —— 下一目标：`-DUSE_ARM_GEMM` JIT 内核
- 高线程扩展 6.2~26.6x（t1→t38），小 tile 数 case 16 线程后内存带宽受限平台化

### 数值精度说明（2026-08-11）

`--verify` 对比 **fp64 直接卷积参考**（非 fp32 参考），判据为**相对容差** `err < 1e-4 × max|ref| + 1e-5`。原因：

- F(4,4) Winograd 的 fp32 舍入误差**随 IC 线性增长**（GEMM 按 IC 串行累加，被输出变换 A 矩阵最高 ±8 的系数放大），但输出幅值同样随 IC 线性增长，所以**相对误差恒定在 ~2-6e-6**（fp32 正常水平）。
- 早期用固定绝对容差 1e-3 导致大 IC case（IC=384/768）误报 FAIL——那是容差指标问题，不是实现 bug。
- fp64 参考让 `--verify` 测的是 Winograd 相对精确数学的**真实误差**，而非两个 fp32 实现的差值。

详见 `PERFORMANCE_ANALYSIS.md` 第 9 节。

## 扩展指南

### GEMM 内核切换

3 种 GEMM 内核编译期选择（互斥）：
- `USE_ARM_GEMM`：ACL JIT SVE 内核（与 oneDNN 相同），`-DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=...`
- `USE_OPENBLAS`：`cblas_sgemm`，`-DUSE_OPENBLAS=ON`（当前使用）
- 默认：naive 三重循环

### 下一步优化方向

详见 `AGENTS.md` 的"下一步优化方向"和 `PERFORMANCE_ANALYSIS.md` 的"新增优化思路"。

主要方向（针对慢 case 0/1/2）：
1. **Tile 分块处理**：8 tile/组，数据放 L2（768KB）（注意：分块会放大 V 的重复读取，只在 V 小时有效，详见 AGENTS.md）
2. **变换函数专用化**：F(4,4) B^T 有 50% 零元素可跳过
3. **arm_gemm 替换 OpenBLAS**：JIT 针对小矩阵优化

### 添加新配置（如 F(6,6,3,3)）

1. 在 `winograd_matrices.hpp` 添加矩阵并用 Winograd 多项式条件验证
2. 在 `winograd_config.hpp` 添加 `WinogradConfig::F66_33()`
3. 在各变换文件添加便捷包装
4. 在 `winograd_conv.cpp` dispatch 中添加分支

## 许可证

Apache 2.0（与 oneDNN/ACL 一致）
