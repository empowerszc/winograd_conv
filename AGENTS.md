# AGENTS.md

> 本文件为 AI 代理（如 opencode、Claude、Copilot 等）在此代码库中工作时提供指导。

## 项目概述

Winograd 卷积复刻项目：基于 ACL（Arm Compute Library）的算法思路，用 C++17 + AArch64 NEON/SVE intrinsics + SME 内联汇编，独立实现的 Winograd 快速卷积。

**目标**：教学与研究用途，帮助理解 ACL Winograd 实现的算法原理和 ISA 优化策略。

**目标平台**：AArch64（华为鲲鹏 920F / Armv9 + SVE-512 + SME）。

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
| `test_winograd.cpp` | 正确性验证 | `run_test()`, 变换级调试 |
| `bench_winograd.cpp` | 性能基准 | CSV 解析, GFLOPS 测量 |

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
./test_winograd --f44        # 只测 F(4,4,3,3)
./test_winograd --relu       # 测 ReLU

# 性能基准（读 CSV，自动过滤 stride=1 group=1 3x3）
./bench_winograd --sme shapes.csv
./bench_winograd --sve --warmup 5 --repeats 20 shapes.csv
cat shapes.csv | ./bench_winograd --neon
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

### 性能对比（NHWC, SVE, 16 线程，全部 case）

| Case | Shape | 本项目(ms) | oneDNN(ms) | 差距 |
|------|-------|-----------|-----------|------|
| 0 | 4,192,40,40 | 7.0 | 3.55 | 1.97x |
| 1 | 4,96,80,80 | 5.9 | 4.22 | 1.40x |
| 2 | 4,48,160,160 | 10.6 | 4.06 | 2.61x |
| 3 | 4,192,20,20 | 2.3 | 2.83 | **0.81x ✓更快** |
| 4 | 4,384,80,80 | 8.2 | 13.78 | **0.60x ✓快40%** |
| 5 | 4,768,40,40 | 6.3 | 9.09 | **0.69x ✓快31%** |

Case 3/4/5 已快于 oneDNN（大 IC），Case 0/1/2 仍慢（小 IC，barrier 开销占比大）。详见 `PERFORMANCE_ANALYSIS.md`

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
| ④ +合并OpenMP区域 | 18.9 | 7.9 | 7.0 | 6.7 | **1.86x** |
| ⑤ +优化B+C(跳过fill+nowait) | — | — | — | — | — |
| ⑥ +权重变换并行（独立region） | 20.2 | 8.5 | 7.4 | 7.2 | 2.00x ↑变慢！ |
| ⑦ +合并权重到主region | 待测 | 待测 | 待测 | 待测 | — |
| oneDNN 参考 | 18.0 | 4.9 | 4.0 | 3.6 | 1.0x |

**关键教训**：
- ⑥ 权重变换放独立 `#pragma omp parallel`，Case 0 变慢（t32: 6.7→7.2），Case 4/5 改善（-19~-30%）。原因：额外 fork/join ~1-2ms 抵消小 OC 的权重并行收益
- ⑦ 合并进主 region 修复，预期 Case 0 恢复到 ⑥之前的 6.7ms 或更好，同时保留 Case 4/5 的改善

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

### 多线程扩展性（合并 OpenMP 区域后，标准模式）

| Case | Shape | Tiles | t1 | t8 | t8加速 | t32 | t32加速 |
|------|-------|-------|-----|-----|--------|-----|--------|
| 0 | 4,192,40,40 | 100 | 18.9 | 7.9 | 2.39x | 6.7 | 2.82x |
| 1 | 4,96,80,80 | 400 | 25.3 | 7.4 | 3.42x | 5.3 | 4.78x |
| 2 | 4,48,160,160 | 1600 | 69.6 | 14.3 | 4.87x | 8.9 | 7.82x |
| 3 | 4,192,20,20 | 25 | 8.6 | 2.9 | 2.96x | 2.3 | 3.74x |
| 4 | 4,384,80,80 | 400 | 54.7 | 12.5 | 4.38x | 6.3 | 8.68x |
| 5 | 4,768,40,40 | 100 | 32.3 | 8.4 | 3.84x | 5.6 | 5.77x |

### 权重变换独立并行后的多线程数据（⑥，有回退）

| Case | Shape | t1 | t8 | t16 | t32 | t38 | vs ④ t32 |
|------|-------|-----|-----|-----|-----|-----|---------|
| 0 | 4,192,40,40 | 20.2 | 8.5 | 7.4 | 7.2 | 7.1 | ↑ +0.5ms |
| 1 | 4,96,80,80 | 26.4 | 8.4 | 6.9 | 6.3 | 6.0 | ↑ +1.0ms |
| 2 | 4,48,160,160 | 67.6 | 13.9 | 10.3 | 8.7 | 8.0 | ↓ -0.2ms |
| 3 | 4,192,20,20 | 8.4 | 1.7 | 1.2 | 1.0 | 1.1 | ↓ -1.3ms |
| 4 | 4,384,80,80 | 53.9 | 11.3 | 7.0 | 5.1 | 4.3 | ↓ -1.2ms |
| 5 | 4,768,40,40 | 31.1 | 7.0 | 4.8 | 4.0 | 3.6 | ↓ -1.6ms |

Case 3/4/5（大 IC 或少 tile）改善，Case 0/1（中 IC 多 tile）变慢

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
- `-DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/gemm`（优先级最高）
- `-DUSE_OPENBLAS=ON`（推荐，已验证可用）
- 默认 naive（开发/验证用）

### 下一步优化方向（按预期收益排序）

详见 `PERFORMANCE_ANALYSIS.md` 和 `OPTIMIZATION_ANALYSIS.md`。

1. **重构数据布局消除 barrier**（预期 -0.8ms）：U[tile][ts][ic] 替代 U[ts][tile][ic]，每 tile 独立 pipeline，0 barrier
2. **arm_gemm 替换 OpenBLAS**（预期 -0.5ms）：JIT 针对小矩阵优化，需 ACL 源码树
3. **变换汇编化**（预期 -0.3ms）：参考 `docs/acl_reference/acl_wino_sve_asm_annotated.md`
4. **权重变换手写公式**（预期 -0.5ms）：参考 `docs/acl_reference/acl_wino_neon_intrinsics_annotated.md`

### 添加新配置（如 F(6,6,3,3)）

1. 在 `winograd_matrices.hpp` 添加 `F66_G`、`F66_Bt`、`F66_A`
2. 在 `winograd_config.hpp` 添加 `WinogradConfig::F66_33()`
3. 在 `winograd_transforms.hpp` 添加 `weight/input/output_transform_f66_neon` 便捷包装
4. 在 `winograd_conv.cpp` dispatch 中添加 `is_f66` 分支

## 相关文档

- **ACL 参考文档**：`docs/acl_reference/` — 从 oneDNN 源码树复制的 ACL Winograd 实现分析文档（8 个文件）。用于指导后续优化（SVE/SME 汇编变换、arm_gemm GEMM 内核、权重变换手写公式等）。当前与 ACL SVE 版本仍有 1.86x 性能差距
- **性能分析**：`PERFORMANCE_ANALYSIS.md` — 优化历程、差距分解、下一步建议
- **优化方案**：`OPTIMIZATION_ANALYSIS.md` — 5 个优化提案的详细展开
