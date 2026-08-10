# AGENTS.md

> 本文件为 AI 代理（如 opencode、Claude、Copilot 等）在此代码库中工作时提供指导。

## 项目概述

Winograd 卷积复刻项目：基于 ACL（Arm Compute Library）的算法思路，用 C++17 + AArch64 NEON/SVE intrinsics + SME 内联汇编，独立实现的 Winograd 快速卷积。

**目标**：教学与研究用途，帮助理解 ACL Winograd 实现的算法原理和 ISA 优化策略。

**目标平台**：AArch64（华为鲲鹏 920F / Armv9 + SVE-512 + SME）。
L1=32KB/核, L2=768KB/核, **无 L3**, 16 NUMA × 38 cores = 608 cores。

## 架构

### 数据流

```
输入特征图 [N][IC][IH][IW]
    │
    ├── 权重变换 (一次性): wei[OC][IC][3][3] → V[36][OC*IC] = G·g·G^T/576
    │   调用 dispatch_weight_transform() → NEON/SVE/SME
    │
    ├── 输入变换 (每 tile): d_tile[6][6][IC] → U_tile[6][6][IC] = B^T·d·B
    │   调用 dispatch_input_transform() → NEON/SVE/SME
    │
    ├── GEMM (每 Winograd 元素): M[ts][tile][OC] = Σ U[ts][tile][IC]·V[ts][OC][IC]
    │   调用 winograd_gemm() (naive, 可替换为 OpenBLAS cblas_sgemm)
    │
    └── 输出变换 (每 tile): M_tile[6][6][OC] → f_tile[4][4][OC] = A^T·M·A + bias + ReLU
        调用 dispatch_output_transform() → NEON/SVE/SME
```

### 文件职责

| 文件 | 职责 | 关键类型/函数 |
|------|------|--------------|
| `winograd_config.hpp` | 配置 + ISA 检测 | `WinogradConfig`, `ISALevel`, `detect_isa()` |
| `winograd_matrices.hpp` | 变换矩阵常量 | `F22_G/Bt/A`, `F44_G/Bt/A` |
| `winograd_transforms.hpp` | NEON 变换 | `transform_2d_neon<>`, `weight/input/output_transform_neon<>` |
| `winograd_transforms_sve.hpp` | SVE 变换 | `transform_2d_sve<>`, `weight/input/output_transform_sve<>` |
| `winograd_transforms_sme.hpp` | SME 变换 | `input_transform_sme<>`, `sme_output_transform_f44()` |
| `winograd_convolution.hpp` | 端到端接口 | `winograd_convolution()`, `dispatch_*()` |
| `winograd_conv.cpp` | 端到端实现 | `winograd_gemm()`, `direct_convolution_3x3()` |
| `test_winograd.cpp` | 正确性验证 | `run_test()`, 变换级调试, B^T 矩阵求解器 |
| `bench_winograd.cpp` | 性能基准 | CSV 解析, GFLOPS 测量, 细粒度计时, 多线程对比 |
| `profile_case.cpp` | 单 case profiling | `--timing` 8 子步骤, `--verify`, `--perf` 命令生成, 9 case preset |

### ISA 调度

三种 ISA（NEON < SVE < SME）通过运行时 dispatch 选择：

```cpp
ISALevel isa = isa_level();  // 可被 --neon/--sve/--sme 或 WINOGRAD_ISA 环境变量覆盖
dispatch_weight_transform(g, V, IC, is_f44, isa);
dispatch_input_transform(d, U, IC, is_f44, isa);
dispatch_output_transform(M, f, OC, bias, act_min, act_max, is_f44, isa);
```

## 构建

```bash
# 重要：必须清除旧缓存
rm -rf build && mkdir build && cd build

# NEON only (默认)
cmake .. -DCMAKE_BUILD_TYPE=Release

# SVE
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SVE=ON

# SME (包含 SVE)
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SME=ON

# OpenBLAS GEMM（推荐）
cmake .. -DUSE_OPENBLAS=ON

# arm_gemm GEMM（与 oneDNN 相同的 JIT 内核，需 ACL 源码树）
cmake .. -DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/gemm

# 全部启用（OpenBLAS 版）
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON -DUSE_OPENBLAS=ON

# 全部启用（arm_gemm 版）
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON -DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=...

make -j && ./test_winograd
```

**关键**：
- `__ARM_FEATURE_SVE`/`__ARM_FEATURE_SME` 是编译器内部宏，由 `-march` 触发，不能手动 `#define`
- 修改 CMakeLists.txt 后必须 `rm -rf build` 清缓存
- GEMM 内核优先级：arm_gemm > OpenBLAS > naive（互斥，arm_gemm 优先）
- OpenBLAS 需要 `cblas.h` + `libopenblas`：`apt install libopenblas-dev`
- arm_gemm 是 ACL 的 JIT GEMM 库（header-only + JIT），路径示例：`ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/common/gemm`

## 测试

```bash
# 正确性验证
./test_winograd              # 全部测试
./test_winograd --sme        # 强制 SME（需要 -DENABLE_SME=ON 编译）
./test_winograd --nhwc       # 测试 NHWC 布局（12 NCHW + 10 NHWC = 22 tests）
./test_winograd --f44        # 只测 F(4,4,3,3)
./test_winograd --relu       # 测 ReLU

# 性能基准（读 CSV，自动过滤 stride=1 group=1 3x3）
./bench_winograd --sve --nhwc --threads 1,8,16,32,38 shapes.csv
./bench_winograd --timing --threads 16 --sve --nhwc shapes.csv  # 细粒度计时
cat shapes.csv | ./bench_winograd --neon

# 单 case profiling（perf 友好）
./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16         # 标准模式
./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --timing  # 8 子步骤计时
./profile_case --ic 96 --ih 80 --iw 80 --oc 96 --verify                          # 正确性验证
./profile_case --ic 384 --ih 80 --iw 80 --oc 96 --perf                           # 生成 perf 命令
./profile_case --help                                                           # 9 case preset 一览

# perf profiling
perf stat -e cycles,instructions,cache-misses,cache-references \
  ./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --repeats 1
perf record -g -- ./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --repeats 10
perf script | flamegraph.pl > flame.svg

# NUMA 优化
numactl --interleave=all env OMP_PROC_BIND=spread OMP_PLACES=cores \
  ./bench_winograd --sve --nhwc --threads 32 shapes.csv
```

测试包含：
1. **变换级调试**：权重/输入/输出/全流水线，用已知输入验证
2. **端到端测试**：12 个用例（小图到大图），对比直接卷积，容差 1e-3

## 代码约定

- **语言**：C++17
- **命名空间**：`winograd_conv`
- **模板参数**：`TILE_SIZE`（4 或 6）、`OUTPUT_TILE`（2 或 4）、`OUT_SIZE`/`IN_SIZE`
- **数据布局**：NCHW（输入/输出），`[tile][channels]`（Winograd 域）
- **ISA 守卫**：`#if defined(__ARM_FEATURE_SVE)` / `#if defined(__ARM_FEATURE_SME)`
- **SME 汇编**：使用 `.inst` 编码（与 ACL 一致），不依赖编译器 SME 助记符支持
- **无注释**：代码中不加注释（除非用户明确要求）

## 关键设计决策

1. **统一矩阵定义**：所有 ISA（NEON/SVE/SME）共用 `winograd_matrices.hpp` 中的矩阵常量。修复矩阵只需改一处，三种 ISA 路径同时修复

2. **泛型 2D 变换**：`transform_2d_neon<OUT, IN>` 用同一矩阵做列变换和行变换，因为 Winograd 的 B 和 A 满足 `output = M * input * M^T`

2. **三层降级（NEON）**：NEON Q 寄存器固定 128-bit，无法用谓词处理 tail，写三份代码（4→2→1 通道）

3. **谓词化（SVE）**：`whilelt` 谓词自动处理 tail，一份代码替代三层降级

4. **Kronecker 积（SME）**：输出变换用 `(A^T ⊗ A^T) · vec(M)` 替代两次 1D 矩阵乘，用 FMOPA 外积累加实现

5. **权重变换用 NEON**：权重变换在 prepare 阶段做一次，非热路径，ACL 也只有 NEON 版

6. **bias + ReLU 在变换内部**：输出变换函数内部添加 bias 并 clamp，避免端到端函数重复添加

7. **矩阵验证方法**：用 Winograd 多项式条件 `sum_j A^T[i][j]·B^T[j][a]·G[j][b] = C·delta(a, i+b)` 数值验证矩阵正确性。当矩阵来源不可靠时，可用求解器从 A^T 和 G 反解出正确的 B^T

8. **OpenMP 并行策略**：单个 `#pragma omp parallel` 区域包含全部阶段：
   - 权重变换：`#pragma omp for schedule(dynamic, 4)` over OC
   - barrier（V 必须完成才能进入 batch 循环）
   - 每个 batch：Phase 1（输入变换）`#pragma omp for collapse(2) schedule(dynamic, 2)` → barrier → Phase 2（GEMM）`#pragma omp for schedule(dynamic)` → barrier → Phase 3（输出变换）`#pragma omp for collapse(2) schedule(dynamic, 2) nowait`
   - 权重 + 所有 batch 合并在 1 个 parallel region，仅 1 次 fork/join
   - 6 个 per-thread 缓冲区（d_tile, U_tile, M_tile, f_tile, g_wt, V_oc_wt）在入口一次性分配、全程复用
   - OpenBLAS 单线程 `openblas_set_num_threads(1)` 避免冲突

   **教训**：权重变换最初放在独立 `#pragma omp parallel` 中，导致 Case 0/1 变慢（额外 fork/join ~1-2ms 抵消权重并行收益）。修复方法是合并进主 parallel region。Case 4/5（大 IC）则受益明显（-19%~-30%）。

9. **GEMM 内核切换**：编译期选择，优先级 arm_gemm > OpenBLAS > naive
   - `USE_ARM_GEMM`：ACL JIT SVE 内核（与 oneDNN 相同），`-DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=...`
   - `USE_OPENBLAS`：`cblas_sgemm`，`-DUSE_OPENBLAS=ON`
   - 默认：naive 三重循环

10. **NHWC 布局优化**：`Layout` 枚举选择 NCHW 或 NHWC。NHWC 下 tile 提取用 `vld1q_f32` 连续加载（3.8x 快于 NCHW 标量）

11. **transform_2d 临时缓冲区**：`thread_local static float*` + `malloc`/`free`，仅容量不足时重新分配

12. **tile 提取优化**：内部 tile 跳过 `memset` 清零，预计算有效行列范围消除 `if` 分支

## 已知限制

1. **GEMM 内核**：3 种可选（arm_gemm JIT > OpenBLAS > naive）。默认 naive，生产环境推荐 arm_gemm
2. **NCHW/NHWC 双布局**：默认 NCHW，`--nhwc` 切换 NHWC。NHWC tile 提取 3.8x 快于 NCHW
3. **OpenMP 并行**：权重变换 + 输入/输出变换 + GEMM 均已并行。Phase 1-3 合并 1 区域，2 barrier（输入→GEMM, GEMM→输出），输出 `nowait`。权重变换独立并行区域
4. **多线程扩展性受限**：8 线程后加速停滞，剩余瓶颈：OpenMP barrier 开销、GEMM 内核质量（OpenBLAS vs arm_gemm JIT）、NUMA 远程访问（920F 16 NUMA）
5. **SME 仅 F(4,4,3,3) 输出变换**：F(2,2,3,3) 输出变换回退到 SVE
6. **`--timing` 模式是串行的**：用于分析各阶段时间占比，不代表并行后的实际性能

### 性能对比（NHWC, SVE, 16 线程，完整 9 case）

| Case | Shape | 本项目 t16(ms) | oneDNN t16(ms) | 结果 |
|------|-------|--------------|---------------|------|
| 0 | 4,192,40,40 | 7.20 | 3.55 | 慢 2.03x |
| 1 | 4,96,80,80 | 7.54 | 4.22 | 慢 1.79x |
| 2 | 4,48,160,160 | 10.39 | 4.06 | 慢 2.56x |
| 3 | 4,192,20,20 | 1.17 | 2.83 | **快 59%** ✓ |
| 4 | 4,384,80,80 | 6.80 | 13.78 | **快 51%** ✓ |
| 5 | 4,768,40,40 | 4.43 | 9.09 | **快 51%** ✓ |
| 6 | 4,768,20,20 | 2.30 | 5.58 | **快 59%** ✓ |
| 7 | 4,96,40,40 | 1.08 | 1.90 | **快 43%** ✓ |
| 8 | 4,96,20,20 | 0.50 | 1.15 | **快 56%** ✓ |

**6/9 case 超越 oneDNN**（快 43-59%）。慢的 case 是多 tile + 小 IC（barrier 开销占比大）。

### t32 对比

| Case | 本项目 t32(ms) | oneDNN t32(估) | 结果 |
|------|--------------|--------------|------|
| 0 | 7.0 | 3.6 | 慢 1.94x |
| 1 | 6.9 | ~3.5 | 慢 ~2.0x |
| 2 | 8.7 | ~3.0 | 慢 ~2.9x |
| 3 | 1.0 | ~2.5 | **快 ~2.5x** ✓ |
| 4 | 4.9 | ~12 | **快 ~2.4x** ✓ |
| 5 | 3.6 | ~8 | **快 ~2.2x** ✓ |
| 6 | 2.0 | ~5 | **快 ~2.5x** ✓ |
| 7 | 1.1 | ~1.5 | **快 ~1.4x** ✓ |
| 8 | 0.5 | ~0.8 | **快 ~1.6x** ✓ |

Case 3/4/5 已大幅超越 oneDNN（大 IC 时变换计算量大，OpenMP 并行效率高）。Case 0/1/2 仍慢（小 IC，barrier 开销占比大）。详见 `PERFORMANCE_ANALYSIS.md`

## 历史修复记录

| 日期 | 问题 | 根因 |
|------|------|------|
| 2026-08 | F44 全部失败 | F44_Bt 矩阵 5 个元素错误，用数值求解器从 A^T 和 G 反解出正确 B^T |
| 2026-08 | F22 全部失败 | F22_A 矩阵 [1][1] 符号错误（-1 应为 +1） |
| 2026-08 | F44 OOB 读取 | weight_transform_neon 冗余外层 k 循环导致 vld1q 越界读取（UB） |
| 2026-08 | segfault | transform_2d_neon CHANNELS 模板参数未传，默认 0，零大小数组 |
| 2026-08 | SME bias 翻倍 | 输出变换内部已加 bias，端到端函数又加一次 |
| 2026-08 | 权重变换标量 | dispatch_weight_transform 已定义但未调用 |
| 2026-08 | SVE 编译报错 | `svptest_first` 需要 2 参数；`static inline` 函数在模板中触发两阶段查找，改用宏 |
| 2026-08 | OpenMP 不生效 | CMake `PRIVATE`→`PUBLIC`；`omp_set_num_threads` 原为桩函数 |
| 2026-08 | double free | OpenBLAS 内部线程与 OpenMP 冲突，`openblas_set_num_threads(1)` |
| 2026-08 | OpenMP 无加速 | 每 tile 在循环内 `std::vector` 分配导致堆锁竞争 |
| 2026-08 | 权重并行后变慢 | 独立 `#pragma omp parallel` 增加额外 fork/join，合并进主 region 修复 |

## 性能优化记录

### 优化历程（Case 0: 4×192×40×40, NHWC, SVE）

| 阶段 | t1(ms) | t8(ms) | t16(ms) | t32(ms) | vs oneDNN t32 |
|------|--------|--------|---------|---------|-------------|
| ① NCHW 基线 | 22.5 | 14.6 | 14.2 | 14.0 | 3.89x |
| ② +NHWC 布局 | 16.3 | ~14 | ~14 | ~14 | ~3.89x |
| ③ +GEMM并行+schedule(2)+raw malloc | 21.6 | 10.3 | 9.5 | 9.1 | 2.53x |
| ④ +合并OpenMP区域 | 18.9 | 7.9 | 7.0 | 6.7 | 1.86x |
| ⑤ +优化B+C(跳过fill+nowait) | — | — | — | — | — |
| ⑥ +权重变换并行（独立region） | 20.2 | 8.5 | 7.4 | 7.2 | 2.00x ↑变慢！ |
| ⑦ +合并权重到主region | 19.9 | 8.2 | 7.3 | 7.0 | 1.94x |
| ⑧ +展平N个batch(9→3 barriers) | 20.2 | 7.3 | 6.4 | 6.0 | 1.67x Case0好, Case4+76%↓ |
| ⑨ 回退到⑦(per-batch) | 20.6 | 8.1 | 7.2 | 7.0 | 1.94x (最终版) |
| oneDNN 参考 | 18.0 | 4.9 | 4.0 | 3.6 | 1.0x |

**关键教训**：
- ⑥ 权重变换放独立 `#pragma omp parallel`，Case 0 变慢（t32: 6.7→7.2），Case 4/5 改善（-19~-30%）。原因：额外 fork/join ~1-2ms 抵消小 OC 的权重并行收益
- ⑦ 合并进主 region 修复，Case 0/1 恢复，Case 3/4/5 保持改善
- ⑧ 展平 N 个 batch 减少 barrier（9→3），但 U/M_buf 内存增加 N 倍。Case 0（U=11MB）改善 -14%，Case 4（U=88MB）严重劣化 +76%（cache thrashing，920F 无 L3、L2 仅 768KB）
- ⑨ 回退到⑦（per-batch），最终最优方案。Case 0/1/2 慢于 oneDNN 但 Case 3-8 超越 oneDNN

### 最终性能对比（完整 9 case, NHWC, SVE, 16 线程）

| Case | Shape (N,IC,IH,IW) | (OC,IC) | Tiles | t16(ms) | oneDNN t16(ms) | 结果 |
|------|---------------------|---------|-------|---------|---------------|------|
| 0 | 4,192,40,40 | 192,192 | 100 | 7.20 | 3.55 | 慢 2.03x |
| 1 | 4,96,80,80 | 96,96 | 400 | 7.54 | 4.22 | 慢 1.79x |
| 2 | 4,48,160,160 | 48,48 | 1600 | 10.39 | 4.06 | 慢 2.56x |
| 3 | 4,192,20,20 | 192,192 | 25 | 1.17 | 2.83 | **快 59%** ✓ |
| 4 | 4,384,80,80 | 96,384 | 400 | 6.80 | 13.78 | **快 51%** ✓ |
| 5 | 4,768,40,40 | 96,768 | 100 | 4.43 | 9.09 | **快 51%** ✓ |
| 6 | 4,768,20,20 | 96,768 | 25 | 2.30 | 5.58 | **快 59%** ✓ |
| 7 | 4,96,40,40 | 96,96 | 100 | 1.08 | 1.90 | **快 43%** ✓ |
| 8 | 4,96,20,20 | 96,96 | 25 | 0.50 | 1.15 | **快 56%** ✓ |

**6/9 case 超越 oneDNN**（快 43-59%）

### 最终多线程扩展性（完整 9 case, 最终版⑦）

| Case | Shape | Tiles | t1 | t8 | t16 | t32 | t38 | t8加速 | t32加速 |
|------|-------|-------|-----|-----|-----|-----|-----|--------|--------|
| 0 | 4,192,40,40 | 100 | 20.6 | 8.1 | 7.2 | 7.0 | 6.8 | 2.54x | 2.94x |
| 1 | 4,96,80,80 | 400 | 28.6 | 9.0 | 7.5 | 6.9 | 6.7 | 3.18x | 4.14x |
| 2 | 4,48,160,160 | 1600 | 67.0 | 13.8 | 10.4 | 8.7 | 8.0 | 4.85x | 7.70x |
| 3 | 4,192,20,20 | 25 | 8.4 | 1.7 | 1.2 | 1.0 | 1.1 | 4.94x | 8.40x |
| 4 | 4,384,80,80 | 400 | 53.5 | 11.1 | 6.8 | 4.9 | 4.1 | 4.82x | 10.92x |
| 5 | 4,768,40,40 | 100 | 31.2 | 6.6 | 4.4 | 3.6 | 3.5 | 4.73x | 8.67x |
| 6 | 4,768,20,20 | 25 | 17.7 | 3.2 | 2.3 | 2.0 | 1.8 | 5.48x | 8.85x |
| 7 | 4,96,40,40 | 100 | 9.0 | 1.6 | 1.1 | 1.1 | 1.1 | 5.63x | 8.18x |
| 8 | 4,96,20,20 | 25 | 2.9 | 0.7 | 0.5 | 0.5 | 0.5 | 4.38x | 5.79x |

### 细粒度计时数据（NHWC, SVE, 1 线程, timing 模式）

注意：timing 模式是串行的（不含 `#pragma omp`），OpenBLAS 可能用默认多线程

| 阶段 | Case 0 (192×192) | Case 4 (384×96) | Case 5 (768×96) |
|------|-----------------|----------------|-----------------|
| 权重变换 | 1.96ms (12%) | 1.05ms (2%) | 1.96ms (8%) |
| Tile 提取(NHWC) | 1.08ms (7%) | 9.14ms (20%) | 4.58ms (20%) |
| 输入变换 | 2.78ms (18%) | 20.83ms (45%) | 9.81ms (42%) |
| Scatter | 0.77ms (5%) | 4.97ms (11%) | 2.32ms (10%) |
| GEMM | 1.22ms (8%) | 1.62ms (3%) | 2.16ms (9%) |
| Gather | 1.82ms (12%) | 2.71ms (6%) | 0.43ms (2%) |
| 输出变换 | 2.02ms (13%) | 4.70ms (10%) | 1.17ms (5%) |
| 输出写回 | 0.38ms (2%) | 1.06ms (2%) | 0.28ms (1%) |

### 胜负模式分析

**赢的条件**（6/9 case, 快 43-59%）：
- IC ≥ 384（GEMM 矩阵大，OpenBLAS 高效，权重并行收益大）
- 或 tiles ≤ 100 且 IC ≥ 96（barrier 开销占比小）

**输的条件**（3/9 case, 慢 1.79-2.56x）：
- IC ≤ 192 且 tiles ≥ 100（GEMM K 维小 + barrier 开销占比大）
- Case 2 最差（IC=48, tiles=1600）：GEMM K=48 极小 + 8 个 barrier

### 下一步优化方向（按预期收益排序）

详见 `PERFORMANCE_ANALYSIS.md` 和 `docs/acl_reference/`。

**针对慢 case（0/1/2）的优化**：

1. **arm_gemm 替换 OpenBLAS**（预期 Case 0/1/2 各 -1~2ms）
   - OpenBLAS 对小 K（48-192）矩阵效率不如 arm_gemm JIT
   - arm_gemm 针对具体矩阵大小自动生成最优 SVE 指令序列
   - 使用：`-DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/gemm`
   - 参考：`docs/acl_reference/acl_wino_implementation_details.md` 中 arm_gemm 选择机制

2. **变换用 SVE intrinsics 优化**（预期 -0.5~1ms）
   - 当前 `transform_2d` 用泛型模板循环，ACL 用手写 SVE intrinsics 完全展开
   - 参考 `docs/acl_reference/acl_wino_sve_asm_annotated.md`：ACL 的 SVE 实现 361 行 vs 我们 ~100 行
   - 可以在 SVE 路径中针对 F(4,4,3,3) 写专用展开（类似 ACL 的 `sve_fp32_6x6.cpp`）

3. **合并 scatter/gather 到变换**（预期 -0.5ms）
   - 当前变换先写 U_tile，再 NEON 拷贝到 U。可改为变换直接写 U
   - 需要修改 `transform_2d` 支持非连续输出 stride

4. **权重变换手写公式**（预期 -0.5ms）
   - 当前用 `weight_transform_neon<6>` 模板循环，ACL 用直接展开 G 矩阵行
   - 参考 `docs/acl_reference/acl_wino_neon_intrinsics_annotated.md`

5. **启发式展平**（对 Case 0 有效，对 Case 4 无害）
   - 当 `NM * n_tiles * IC * 4 < 16MB` 时展平 N 个 batch（9→3 barriers）
   - 当内存大时保持 per-batch（避免 cache thrashing）

**对已超越 oneDNN 的 case（3-8）**：当前已优，无需进一步优化

## 扩展指南

### GEMM 内核切换

已有 3 种 GEMM 内核（编译期选择，互斥）：

```cpp
#if defined(USE_ARM_GEMM)      // ACL JIT SVE 内核（与 oneDNN 相同）
    arm_gemm::GemmHybrid<...> gemm(...); gemm.matmul(...);
#elif defined(USE_OPENBLAS)     // OpenBLAS cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ...);
#else                           // naive 三重循环
    for (...) ...
#endif
```

CMake 选项：
- `-DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/gemm`（优先级最高，与 oneDNN 相同内核）
- `-DUSE_OPENBLAS=ON`（当前使用，对小 K 矩阵不如 arm_gemm）
- 默认 naive（开发/验证用）

arm_gemm 路径示例：`ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/common/gemm`

### 下一步优化方向（按预期收益排序）

详见 `PERFORMANCE_ANALYSIS.md` 和 `docs/acl_reference/`。

**针对慢 case（0/1/2, 慢 1.79-2.56x）的优化**：

1. **arm_gemm 替换 OpenBLAS**（预期 Case 0/1/2 各 -1~2ms）
   - OpenBLAS 对小 K（48-192）矩阵效率不如 arm_gemm JIT
   - arm_gemm 针对具体矩阵大小自动生成最优 SVE 指令序列
   - 使用：`-DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/gemm`
   - 参考：`docs/acl_reference/acl_wino_implementation_details.md` 中 arm_gemm 选择机制

2. **变换用 SVE intrinsics 优化**（预期 -0.5~1ms）
   - 当前 `transform_2d` 用泛型模板循环，ACL 用手写 SVE intrinsics 完全展开
   - 参考 `docs/acl_reference/acl_wino_sve_asm_annotated.md`：ACL 的 SVE 实现 361 行 vs 我们 ~100 行
   - 可以在 SVE 路径中针对 F(4,4,3,3) 写专用展开（类似 ACL 的 `sve_fp32_6x6.cpp`）

3. **合并 scatter/gather 到变换**（预期 -0.5ms）
   - 当前变换先写 U_tile，再 NEON 拷贝到 U。可改为变换直接写 U
   - 需要修改 `transform_2d` 支持非连续输出 stride

4. **权重变换手写公式**（预期 -0.5ms）
   - 当前用 `weight_transform_neon<6>` 模板循环，ACL 用直接展开 G 矩阵行
   - 参考 `docs/acl_reference/acl_wino_neon_intrinsics_annotated.md`

5. **启发式展平**（对 Case 0 有效，对 Case 4 无害）
   - 当 `NM * n_tiles * IC * 4 < 16MB` 时展平 N 个 batch（9→3 barriers）
   - 当内存大时保持 per-batch（避免 cache thrashing）

**对已超越 oneDNN 的 case（3-8, 快 43-59%）**：当前已优，无需进一步优化

### 新增优化思路（基于 920F 硬件特性 + 性能数据）

以下是基于 920F 无 L3 cache、L2 仅 768KB/核的硬件特性，以及对 3 个慢 case 的分析，提出的新优化方向：

#### A. Tile 分块处理（cache 友好）

**问题**：920F 无 L3，L2 仅 768KB/核。当前一次性处理所有 tile，U（2.7-11MB）远超 L2，每个 tile 的输入变换和 scatter 都直接走主存。

**方案**：将 tile 分为 K 个一组（如 K=8），每组数据量 8×6×6×192×4=221KB 能放 L2。在组内完成 input→GEMM→output，减少主存访问。

```cpp
const int CHUNK = 8;
for (int chunk_start = 0; chunk_start < n_tiles; chunk_start += CHUNK) {
    int chunk_end = min(chunk_start + CHUNK, n_tiles);
    // 1. 输入变换：chunk_start..chunk_end
    // 2. GEMM：36 个 ts × (chunk_end-chunk_start) tiles
    // 3. 输出变换：chunk_start..chunk_end
}
```

**预期**：减少主存访问次数，改善 Case 2（1600 tiles, IC=48）的 cache 局部性。

#### B. GEMM 批量合并（减少调用次数）

**问题**：当前 36 次 `cblas_sgemm` 调用，每次 (n_tiles × IC × OC)。36 次调用有函数调用开销和 OpenMP 调度开销。

**方案**：将多个 ts_idx 的 GEMM 合并为一次大 GEMM。将 U[ts0..ts3][tile][ic] 重排为 U_batch[4*n_tiles][ic]，V[ts0..ts3][oc][ic] 重排为 V_batch[4*OC][ic]，一次 GEMM 得到 M_batch[4*n_tiles][4*OC]。但输出布局复杂，需要额外的重排。

更简单的方案：将 U 和 V 沿 K 维拼接（不改变 M/N），利用 cblas_sgemm 的 beta 参数做累加：

```cpp
// 不需要改布局，用 beta=1.0 累加多次 GEMM 结果到同一 M
for (int ts = 0; ts < 36; ts++) {
    cblas_sgemm(..., 1.0f, U_ts, V_ts, ts == 0 ? 0.0f : 1.0f, M, ...);
}
// 但这样不并行，不如当前的 #pragma omp for
```

此方案对大 IC case（GEMM 已经高效）帮助不大，但对小 IC case 可能减少调度开销。

#### C. SVE 替代 NEON 做 tile 提取/scatter/gather

**问题**：当前 tile 提取、scatter、gather 用 NEON（128-bit, 4 float/指令）。920F 支持 SVE-512（16 float/指令）。

**方案**：在 SVE 编译路径下，将 NEON `vld1q_f32`/`vst1q_f32` 替换为 SVE `svld1_f32`/`svst1_f32`。tile 提取从 4 float/指令→16 float/指令，4× 加速。

```cpp
// 当前 NEON（4 float）
for (; ic + 4 <= IC; ic += 4)
    vst1q_f32(dp + ic, vld1q_f32(sp + ic));

// SVE 路径（16 float，谓词自动处理 tail）
if (isa >= ISALevel::SVE) {
    svbool_t pg = svwhilelt_b32(ic, IC);
    do {
        svst1_f32(pg, dp + ic, svld1_f32(pg, sp + ic));
        ic += svcntw();
        pg = svwhilelt_b32(ic, IC);
    } while (svptest_first(svptrue_b32(), pg));
}
```

**预期**：tile 提取从 1.10ms → ~0.28ms（Case 0），scatter/gather 也有类似加速。对 Case 2（1600 tiles）收益最大。

#### D. 双缓冲（batch 间重叠）

**问题**：当前 batch 间串行：batch N 的输出完成后才开始 batch N+1 的输入。

**方案**：分配双份 U/M_buf，batch N 的输出变换与 batch N+1 的输入变换重叠：

```
batch 0: input → GEMM → output
batch 1:              input → GEMM → output  (与 batch 0 output 并行)
```

用 OpenMP task 实现：

```cpp
#pragma omp taskgroup
{
    #pragma omp task
    { process_batch(0, U_A, M_A); }
    #pragma omp task
    { process_batch(1, U_B, M_B); } // 与 batch 0 的 output 并行
}
```

**代价**：内存翻倍（2×U + 2×M），但对 Case 0（U=2.7MB）可接受。

#### E. 权重变换直接写 V（消除 V_oc 中间缓冲）

**问题**：当前权重变换先写 V_oc（per-OC），再拷贝到 V。多一次内存拷贝。

**方案**：修改 `weight_transform_neon` 支持输出 stride，直接写 V[m*OC*IC + oc*IC + ic]。但 V 的 stride 与 V_oc 不同，需要参数化 transform 函数。

#### F. 变换函数专用化（F(4,4,3,3) 硬编码）

**问题**：当前 `transform_1d_neon<6,6>` 是泛型模板，循环内有 `if (matrix[o][k] == 0.0f)` 分支判断。ACL 为 F(4,4,3,3) 写了专用代码，完全展开，无分支。

**方案**：为 F(4,4,3,3) 的输入变换（B^T 矩阵有大量零元素）和输出变换（A^T 矩阵有零元素）写专用函数，跳过零系数的行/列：

```cpp
// F(4,4) 专用输入变换，B^T[0] = [4,0,-5,0,1,0]
// 只需处理 k=0,2,4（跳过 k=1,3,5 的零系数）
void input_transform_f44_specialized(...) {
    // 列变换：6 列，每列只需 3 个非零行 × 3 个非零系数 = 9 FMA（vs 泛型 6×6=36）
}
```

**预期**：变换计算量减少 ~50%（F(4,4) 的 B^T 有 50% 零元素）。

### 运行方式

```bash
# NUMA 交织（让内存均匀分布在所有 NUMA 节点）
numactl --interleave=all ./bench_winograd --sve --nhwc --threads 32 shapes.csv

# 单 NUMA 绑定（38 核，减少跨 NUMA 访问）
numactl --cpunodebind=0 --membind=0 ./bench_winograd --sve --nhwc --threads 38 shapes.csv

# 线程绑定（防止线程迁移跨 NUMA）
OMP_PROC_BIND=spread OMP_PLACES=cores ./bench_winograd --sve --nhwc --threads 32 shapes.csv
```

### 添加新配置（如 F(6,6,3,3)）

1. 在 `winograd_matrices.hpp` 添加 `F66_G`、`F66_Bt`、`F66_A`
2. 在 `winograd_config.hpp` 添加 `WinogradConfig::F66_33()`
3. 在 `winograd_transforms.hpp` 添加 `weight/input/output_transform_f66_neon` 便捷包装
4. 在 `winograd_conv.cpp` dispatch 中添加 `is_f66` 分支

## 相关文档

- **ACL 参考文档**：`docs/acl_reference/` — 从 oneDNN 源码树复制的 ACL Winograd 实现分析文档（8 个文件）。用于指导后续优化（SVE/SME 汇编变换、arm_gemm GEMM 内核、权重变换手写公式等）
- **性能分析**：`PERFORMANCE_ANALYSIS.md` — 完整优化历程（9 阶段）、9 case 对比数据、细粒度计时、差距分析、新优化思路（A-F）
