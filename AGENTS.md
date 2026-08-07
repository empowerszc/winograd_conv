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

# 全部启用
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON -DUSE_OPENBLAS=ON

make -j && ./test_winograd
```

**关键**：
- `__ARM_FEATURE_SVE`/`__ARM_FEATURE_SME` 是编译器内部宏，由 `-march` 触发，不能手动 `#define`
- 修改 CMakeLists.txt 后必须 `rm -rf build` 清缓存
- OpenBLAS 需要 `cblas.h` 头文件和 `libopenblas` 库。如系统未安装：`apt install libopenblas-dev`

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

8. **OpenMP 并行策略**：tile 循环用 `#pragma omp parallel { 预分配; #pragma omp for schedule(dynamic,2) }` 模式，per-thread 缓冲区在 parallel 区域内分配一次、复用。GEMM 循环用 `#pragma omp parallel for schedule(dynamic)` 并行 36 个 GEMM。OpenBLAS 设为单线程 `openblas_set_num_threads(1)` 避免 GEMM 内部线程与 OpenMP 线程冲突

9. **NHWC 布局优化**：`Layout` 枚举选择 NCHW 或 NHWC。NHWC 下通道连续，tile 提取用 `vld1q_f32` NEON 批量加载（4 float/指令 vs NCHW 标量逐元素）。输出写回同理。tile 提取从 4.20ms → 1.10ms（3.8x），写回从 2.95ms → 0.40ms（7.4x）

10. **transform_2d 临时缓冲区**：用 `thread_local static float*` + `malloc`/`free`，避免 `std::vector::resize()` 的函数调用和容量检查开销。仅在容量不足时才重新分配

## 已知限制

1. **GEMM 用 OpenBLAS**：启用 `-DUSE_OPENBLAS=ON`。OpenBLAS 单线程（`openblas_set_num_threads(1)`），GEMM 循环用 `#pragma omp parallel for` 并行。与 oneDNN 的 arm_gemm JIT 相比仍有 2-3x GEMM 性能差距
2. **NCHW/NHWC 双布局**：默认 NCHW（`Layout::NCHW`），可用 `--nhwc` 切换 NHWC。NHWC 的 tile 提取/写回用 NEON 批量加载（3.8x/7.4x 快于 NCHW 标量）
3. **OpenMP 并行**：tile 循环 + GEMM 循环均并行。每 batch 有 3 个并行区域（输入→GEMM→输出），有 fork/join 开销
4. **多线程扩展性受限**：8 线程后加速停滞，原因：GEMM 内核质量、OpenMP fork/join 开销（12 次/batch）、NUMA 远程访问（920F 16 NUMA）
5. **SME 仅 F(4,4,3,3) 输出变换**：F(2,2,3,3) 输出变换回退到 SVE（ACL 也如此）
6. **`--timing` 模式是串行的**：`run_with_timing()` 手动重现管道各步骤，不使用 OpenMP。用于分析各阶段时间占比

### 性能对比（Case 0: 4×192×40×40, NHWC, SVE）

| 线程 | 本项目(ms) | oneDNN(ms) | 差距 |
|------|-----------|-----------|------|
| 1 | 21.6 | 18.0 | 1.20x |
| 8 | 10.3 | 4.9 | 2.10x |
| 16 | 9.5 | 4.0 | 2.38x |
| 32 | 9.1 | 3.6 | 2.53x |

差距来源：GEMM 内核（OpenBLAS vs arm_gemm JIT）、OpenMP fork/join 开销、NUMA 效应。详见 `PERFORMANCE_ANALYSIS.md`

## 历史修复记录

| 日期 | 问题 | 根因 |
|------|------|------|
| 2026-08 | F44 全部失败 | F44_Bt 矩阵 5 个元素错误（B^T[3][2], B^T[4][2], B^T[5][1,2,3]），违反 Winograd 多项式条件。用数值求解器从 A^T 和 G 反解出正确 B^T |
| 2026-08 | F22 全部失败 | F22_A 矩阵 [1][1] 符号错误（-1 应为 +1） |
| 2026-08 | F44 OOB 读取 | weight_transform_neon 冗余外层 k 循环导致 vld1q 越界读取（UB） |
| 2026-08 | segfault | transform_2d_neon CHANNELS 模板参数未传，默认 0，零大小数组 |
| 2026-08 | SME bias 翻倍 | 输出变换内部已加 bias，端到端函数又加一次 |
| 2026-08 | 权重变换标量 | dispatch_weight_transform 已定义但未调用 |

## 扩展指南

### 替换 GEMM 为 OpenBLAS

```cpp
// 在 winograd_conv.cpp 中替换 winograd_gemm() 函数体
#include <cblas.h>
void winograd_gemm(const float* U, const float* V, float* M,
                   int n_tiles, int OC, int IC) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_tiles, OC, IC, 1.0f, U, IC, V, IC, 0.0f, M, OC);
}
```

### 添加 prepare/execute 分离

```cpp
struct WinogradContext {
    std::vector<float> V;
    WinogradConfig config;
    ISALevel isa;
    std::vector<float> U_workspace, M_workspace;
};
WinogradContext prepare(const float* weights, int OC, int IC, ...);
void execute(WinogradContext& ctx, const float* input, float* output, ...);
```

### 添加新配置（如 F(6,6,3,3)）

1. 在 `winograd_matrices.hpp` 添加 `F66_G`、`F66_Bt`、`F66_A`
2. 在 `winograd_config.hpp` 添加 `WinogradConfig::F66_33()`
3. 在 `winograd_transforms.hpp` 添加 `weight/input/output_transform_f66_neon` 便捷包装
4. 在 `winograd_conv.cpp` dispatch 中添加 `is_f66` 分支

## 相关文档

- ACL 源码：`D:\300Code\ComputeLibrary-53.1.0\src\core\NEON\kernels\convolution\winograd\`
- ACL 注释文档：`D:\300Code\oneDNN-3.12.1\...\src\cpu\aarch64\acl_wino_*_annotated.md`（4 个文件）
- 项目分析：`D:\300Code\oneDNN-3.12.1\...\src\cpu\aarch64\winograd_conv_project_analysis.md`
