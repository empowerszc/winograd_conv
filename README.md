# Winograd 卷积复刻项目

> 基于 ACL（Arm Compute Library）的 Winograd 卷积实现思路，用 C++ + NEON/SVE intrinsics + SME 内联汇编复刻的独立可运行项目。
>
> 支持 F(2,2,3,3) 和 F(4,4,3,3) 两种 Winograd 配置，包含权重/输入/输出三步变换、简化 GEMM、端到端卷积、以及与直接卷积的正确性验证。

## 目录结构

```
winograd_conv/
├── CMakeLists.txt                         ← 构建配置
├── AGENTS.md                              ← AI 代理指南
├── README.md                              ← 本文件
├── PERFORMANCE_ANALYSIS.md                ← 性能分析与优化历程
├── OPTIMIZATION_ANALYSIS.md               ← 优化方案详细展开
├── include/
│   ├── winograd_config.hpp                ← 配置 + ISA 检测/选择 + Layout 枚举
│   ├── winograd_matrices.hpp              ← G/B^T/A^T 变换矩阵
│   ├── winograd_transforms.hpp            ← NEON intrinsics 变换
│   ├── winograd_transforms_sve.hpp        ← SVE intrinsics 变换
│   ├── winograd_transforms_sme.hpp        ← SME FMOPA 内联汇编变换
│   └── winograd_convolution.hpp           ← 端到端接口 + dispatch 声明
├── src/
│   └── winograd_conv.cpp                  ← 端到端实现 + GEMM + 直接卷积 + dispatch
├── tests/
│   ├── test_winograd.cpp                  ← 正确性验证 + 变换调试
│   └── bench_winograd.cpp                 ← 性能基准（读 CSV，测 GFLOPS）
└── docs/
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

# 或通过环境变量
WINOGRAD_ISA=sme ./test_winograd
```

### 运行性能测试

```bash
# 准备 shapes 文件（tab 分隔，第一行是表头）
# 格式：Input Shape  Weight Shape  Stride  Pad  Dil  Grp  Count
# 自动过滤 stride=1, group=1, 3x3 kernel 的行

# 用 SME 测试
./bench_winograd --sme shapes.csv

# 用 SVE 测试
./bench_winograd --sve shapes.csv

# 用 NEON 测试
./bench_winograd --neon shapes.csv

# 从 stdin 读取
cat shapes.csv | ./bench_winograd --sme

# 调整 warmup/repeats
./bench_winograd --sme --warmup 5 --repeats 20 shapes.csv
```

输出示例：
```
ISA: SME (detected: SME)

Shape (N,IC,IH,IW)              (OC,IC,3,3)          Count   Time(ms)      GFLOPS        ISA
----------------------------------------------------------------------------------------------------
(4, 192, 40, 40)                (192, 192, 3, 3)        24     XX.XX       XX.XX        SME
(4, 96, 80, 80)                 (96, 96, 3, 3)          17     XX.XX       XX.XX        SME
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
- `winograd_gemm()`：naive 三重循环 GEMM（可替换为 OpenBLAS `cblas_sgemm`）
- `direct_convolution_3x3()`：参考直接卷积（用于验证正确性）
- `winograd_convolution()`：端到端 Winograd 卷积
  - 步骤 1：权重变换（通过 dispatch，一次性）
  - 步骤 2a：输入 tile 提取 + 输入变换（通过 dispatch）
  - 步骤 2b：NM 个 batched GEMM
  - 步骤 2c：输出变换（通过 dispatch，含 bias + ReLU）+ 写回

### `test_winograd.cpp`
- F(4,4,3,3) 变换级调试（权重/输入/输出/全流水线）
- 12 个测试用例（小图到大图，F22 和 F44 各 6 个）
- 对比 Winograd 与直接卷积，容差 1e-3

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
| 数据布局 | NCHW + NHWC（`--nhwc` 切换） | NHWC + Winograd 专用布局 |
| 多线程 | OpenMP（权重+输入+GEMM+输出 全并行，合并区域） | NEScheduler |
| 正确性 | ✅ 22/22 验证通过（NCHW + NHWC） | ✅ |
| 性能 | t32=6.7ms vs oneDNN 3.6ms（1.86x，权重并行后预期 ~1.1x） | 高度优化 |

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

实测数据（4×192×40×40, NHWC, SVE, 1 线程）：

| 阶段 | 时间(ms) | 占比 | 瓶颈 |
|------|---------|------|------|
| 权重变换 | 3.18 | 19% | 串行，模板 intrinsics |
| 输入变换 | 2.79 | 17% | NEON/SVE 计算 |
| 输出变换 | 1.94 | 12% | NEON/SVE 计算 |
| GEMM | 1.99 | 12% | OpenBLAS |
| NEON gather/scatter | 2.25 | 14% | 向量化拷贝 |
| NHWC tile 提取 | 1.10 | 7% | NEON 连续读 |
| NHWC 写回 | 0.40 | 2% | NEON 连续写 |

**主要瓶颈**：权重变换（串行，占 t32 的 47%）、GEMM 内核质量（OpenBLAS vs arm_gemm JIT）、OpenMP barrier 开销。

**已实施的优化**：
1. NHWC 布局（tile 提取 3.8x 加速）
2. 权重变换并行（`#pragma omp for` over OC，占 t32 的 47%→~3%）
3. GEMM 并行（36 个 GEMM 分布到所有线程）
4. 合并 OpenMP 区域（3→1 个 parallel region，fork/join 减 3x）
5. `schedule(dynamic, 2)` 减少 tile 调度开销
6. `thread_local static float*` + `malloc` 替代 `std::vector::resize`
7. 内部 tile 跳过 `memset` 清零 + 预计算有效行列范围
8. 输出变换 `nowait` 消除多余 barrier
9. OpenBLAS 单线程避免 GEMM 线程冲突
10. arm_gemm JIT GEMM 内核可选（与 oneDNN 相同）

**优化历程**（Case 0, t32）：
```
NCHW 基线:  14.0ms → +NHWC: → +GEMM并行: 9.1ms → +合并OMP+优化B/C: 6.7ms → +权重并行: 待测
3.89x → 2.53x → 1.86x → (预期 ~1.1x)
```

## 扩展指南

### 替换 GEMM

将 `winograd_gemm()` 替换为 OpenBLAS：

```cpp
#include <cblas.h>
void winograd_gemm(const float* U, const float* V, float* M,
                   int n_tiles, int OC, int IC) {
    // M[n_tiles][OC] = U[n_tiles][IC] × V[OC][IC]^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_tiles, OC, IC, 1.0f, U, IC, V, IC, 0.0f, M, OC);
}
```

### 添加 prepare/execute 分离

```cpp
struct WinogradContext {
    std::vector<float> V;  // 预变换的权重
    WinogradConfig config;
    ISALevel isa;
    std::vector<float> U_workspace, M_workspace;  // 预分配
};
WinogradContext prepare(const float* weights, ...);
void execute(WinogradContext& ctx, const float* input, float* output, ...);
```

## 许可证

Apache 2.0（与 oneDNN/ACL 一致）
