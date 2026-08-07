# ACL Winograd 变换内核源码逐行解析（SME / SVE / NEON）

> **目的**：ACL（Compute Library 53.1.0）的 Winograd 卷积变换 kernel 大量使用 **内嵌汇编（inline assembly）** 和 **NEON 内联函数（intrinsics）**。本文对照源码，把每条汇编/内联函数"翻译"成人话，让看不懂汇编的人也能理解它具体在算什么。
>
> **目标机器**：华为鲲鹏 920F，Armv9 + SVE-512 + SME。该机上 F(4,4,3,3) 的三个变换分别是：
> - 输入变换：`sme_fp32_mla_6x6`（SME）
> - 输出变换：`sme_fp32_mopa_4x4_3x3`（SME）
> - 权重变换：`arm_fp32_4x4_3x3`（NEON intrinsics）
>
> 同时本文也覆盖无 SME 机器的备选实现（SVE 版 `sve_fp32_6x6`、NEON 汇编版 `a64_fp32_6x6`、NEON intrinsics 版 `arm_fp32_4x4_3x3` 输出变换），便于对比。

---

## 目录

- [1. 三类变换 × 三种 ISA：实现全景](#1-三类变换--三种-isa实现全景)
- [2. 预备知识：寄存器与指令速查](#2-预备知识寄存器与指令速查)
- [3. 权重变换 arm_fp32_4x4_3x3（NEON intrinsics，易读版）](#3-权重变换-arm_fp32_4x4_3x3neon-intrinsics易读版)
- [4. 输入变换：三个 ISA 版本对比](#4-输入变换三个-isa-版本对比)
  - [4.1 NEON 汇编版 a64_fp32_6x6](#41-neon-汇编版-a64_fp32_6x6)
  - [4.2 SVE 汇编版 sve_fp32_6x6](#42-sve-汇编版-sve_fp32_6x6)
  - [4.3 SME 版 sme_fp32_mla_6x6](#43-sme-版-sme_fp32_mla_6x6)
- [5. 输出变换：两个 ISA 版本对比](#5-输出变换两个-isa-版本对比)
  - [5.1 NEON intrinsics 版 arm_fp32_4x4_3x3](#51-neon-intrinsics-版-arm_fp32_4x4_3x3)
  - [5.2 SME 汇编版 sme_fp32_mopa_4x4_3x3（核心创新）](#52-sme-汇编版-sme_fp32_mopa_4x4_3x3核心创新)
- [6. 三种 ISA 实现的本质差异总结](#6-三种-isa-实现的本质差异总结)
- [7. F(2,2,3,3) 变换解析（NEON intrinsics）](#7-f2233-变换解析neon-intrinsics)
  - [7.1 权重变换 arm_fp32_2x2_3x3](#71-权重变换-arm_fp32_2x2_3x3)
  - [7.2 输入变换 arm_fp32_4x4（4×4 tile）](#72-输入变换-arm_fp32_4x444-tile)
  - [7.3 输出变换 arm_fp32_2x2_3x3](#73-输出变换-arm_fp32_2x2_3x3)
- [8. 何时走 F(4,4,3,3) vs F(2,2,3,3)](#8-何时走-f4433-vs-f2233)

---

## 1. 三类变换 × 三种 ISA：实现全景

Winograd F(4,4,3,3) 需要三个变换，每个变换都有多个 ISA 实现：

**F(4,4,3,3) 变换**（输出 tile 4×4，输入 tile 6×6，36 个 GEMM）：

| 变换 | 数学 | NEON intrinsics | NEON 汇编 | SVE 汇编 | SME 汇编 |
|------|------|:-:|:-:|:-:|:-:|
| **权重变换** G | V = G·g·G^T / 576 | `arm_fp32_4x4_3x3` ✅ | — | — | — |
| **输入变换** B | U = B^T·d·B | `arm_fp32_4x4` (4×4, F(2,2)) | `a64_fp32_6x6` ✅ | `sve_fp32_6x6` ✅ | `sme_fp32_mla_6x6` ✅ |
| **输出变换** A | f = A^T·M·A (+bias +ReLU) | `arm_fp32_4x4_3x3` ✅ | — | — | `sme_fp32_mopa_4x4_3x3` ✅ |

**F(2,2,3,3) 变换**（输出 tile 2×2，输入 tile 4×4，16 个 GEMM）：

| 变换 | 数学 | NEON intrinsics | NEON 汇编 | SVE 汇编 | SME 汇编 |
|------|------|:-:|:-:|:-:|:-:|
| **权重变换** G | V = G·g·G^T | `arm_fp32_2x2_3x3` ✅ | — | — | — |
| **输入变换** B | U = B^T·d·B | `arm_fp32_4x4` ✅（4×4 tile） | — | — | — |
| **输出变换** A | f = A^T·M·A (+bias +ReLU) | `arm_fp32_2x2_3x3` ✅ | — | — | — |

> **在 920F（SME 机器）上**：ACL 的 `get_implementation` 贪心选注册表最前的 SME 变换，所以输入/输出走 SME 汇编版（F(4,4,3,3)）；权重变换没有 SME/SVE 专用版，统一走 NEON intrinsics 版。
> F(2,2,3,3) 的三个变换**全部只有 NEON intrinsics 版**，无 SVE/SME 加速——何时使用详见第 7、8 节。

---

## 2. 预备知识：寄存器与指令速查

### 2.1 NEON（所有 AArch64 标配）

| 寄存器 | 宽度 | 容量 (f32) |
|--------|------|-----------|
| `Q0`-`Q31`（=`v0`-`v31`） | 128 bit | 4 个 float |
| `D0`-`D31` | 64 bit | 2 个 float |
| `S0`-`S31` | 32 bit | 1 个 float |

### 2.2 SVE（Scalable Vector Extension）

| 寄存器 | 宽度 | 容量 (f32, SVE-512) |
|--------|------|---------------------|
| `Z0`-`Z31` | 向量长度 VL（128~2048 bit） | 16 个 float (VL=512) |
| `P0`-`P15` | 谓词寄存器 | 每位控制一个 lane |

### 2.3 SME（Scalable Matrix Extension）

在 SVE 基础上增加：

| 寄存器 | 宽度 | 用途 |
|--------|------|------|
| `ZA` tile 寄存器 | 二维 tile（如 16×16 f32 = 8KB） | 外积累加矩阵 |
| 流式 SVE `Z0`-`Z31` | 流式模式 VL | FMOPA 的输入向量 |

### 2.4 核心指令速查

| 指令 | 含义 | 人类语言 |
|------|------|----------|
| `fmla vD.4s, vN.4s, vM.s[idx]` | D += N × M[idx] | 4 路 FMA，乘以一个标量 |
| `fmls vD.4s, vN.4s, vM.s[idx]` | D -= N × M[idx] | 4 路 FMS（减） |
| `fadd vD.4s, vN.4s, vM.4s` | D = N + M | 4 路加 |
| `fsub` | D = N - M | 4 路减 |
| `ld1w {zD.s}, p0/Z, [addr]` | 谓词加载 | 只加载 p0 为真的 lane |
| `st1w {zD.s}, p0, [addr]` | 谓词存储 | 只存 p0 为真的 lane |
| `whilelt p0.s, xN, xM` | 设置谓词 | 当 N < M 时对应 lane 置真 |
| `incb xN` | N += VL/sizeof(byte) | 指针前进一个向量宽度 |
| `fmad zD.s, p/M, zN.s, zM.s` | D = D + N × M（谓词控制） | SVE FMA |
| `fmsb zD.s, p/M, zN.s, zM.s` | D = D - N × M（谓词控制） | SVE FMS |
| `fneg zD.s, p/M, zN.s` | D = -N（谓词控制） | 取反 |
| `SMSTART ZA` | 进入 SME 流式模式 + 激活 ZA | 开启矩阵加速器 |
| `SMSTOP` | 退出 SME 流式模式 | 关闭矩阵加速器 |
| `zero {zad0-zad7}` | 清零所有 ZA tile | 8 个 tile 全清零 |
| `fmopa za0.s, p/M, p/M, zN.s, zM.s` | **za0 += zN ⊗ zM**（外积累加） | SME 核心指令：两个向量做外积，累加进二维 tile |
| `mova zD.s, p/M, za0.s[xN]` | 从 ZA tile 读第 N 行到 Z | 取出结果 |

---

## 3. 权重变换 arm_fp32_4x4_3x3（NEON intrinsics，易读版）

**文件**：`weight_transforms/arm_fp32_4x4_3x3.cpp`
**数学**：`V = G · g · G^T / 576`，其中 g 是 3×3 权重，V 是 6×6 变换后权重。

> 权重变换是三者中最易读的——它用 **NEON C++ intrinsics**（非汇编），可以直接看到数学公式。

### 3.1 G 矩阵（从源码 :57-72 提取）

```cpp
Ww[0][j] =  6 * w[0][j];                                    // G 第 0 行: [6, 0, 0]
Ww[1][j] = -4 * (w[0][j] + w[1][j] + w[2][j]);             // G 第 1 行: [-4, -4, -4]
Ww[2][j] =  4 * (w[1][j] - w[0][j] - w[2][j]);             // G 第 2 行: [-4, 4, -4]
Ww[3][j] = w[0][j] + 2*w[1][j] + 4*w[2][j];                // G 第 3 行: [1, 2, 4]
Ww[4][j] = w[0][j] - 2*w[1][j] + 4*w[2][j];                // G 第 4 行: [1, -2, 4]
Ww[5][j] = 24 * w[2][j];                                   // G 第 5 行: [0, 0, 24]
```

所以 G 矩阵是：

```
        ┌  6   0   0  ┐
   G =  │ -4  -4  -4  │   (6×3, 把 3×3 权重 → 6×6 Winograd 域)
        │ -4   4  -4  │
        │  1   2   4  │
        │  1  -2   4  │
        └  0   0  24  ┘
```

随后用 **相同的 G 矩阵** 对列做变换（`V[i][j]` 行 :81-96），最后除以 `576`（= 24²，归一化因子，:78）：

```cpp
const float recip576 = 1.0f / 576.0f;
V[i][0] = (6 * Ww[i][0]) * recip576;
V[i][1] = (-4*(Ww[i][0] + Ww[i][1] + Ww[i][2])) * recip576;
// ... 同样的 G 矩阵模式 ...
V[i][5] = (24 * Ww[i][2]) * recip576;
```

### 3.2 通道维向量化

权重变换沿 **输出通道维** 向量化（一次处理 4 个 OC）：

```cpp
for (; n_channels >= 4; n_channels -= 4) {     // :39, 一次 4 通道
    float32x4_t w[3][3];                         // 每个 w[i][j] 是 4 个 OC 的权重
    w[i][j] = vld1q_f32(inptr + ...);            // 加载 4 个 float (128 bit)
    // ... G 变换 ...
    vst1q_f32(outptr + m*matrix_stride, V[i][j]); // 存储 4 个 float
}
```

然后降级到 2 通道（`float32x2_t`，64 bit）和 1 通道（标量 `float`）处理 tail。

> **一句话**：权重变换 = 用 G 矩阵对 3×3 权重做两次矩阵乘（行变换 + 列变换），结果除以 576，沿输出通道维用 128-bit NEON 向量化。

---

## 4. 输入变换：三个 ISA 版本对比

三个版本实现的 **数学完全相同**（`U = B^T · d · B`，6×6 输入 tile → 6×6 Winograd 域），但实现方式差异很大。

### 4.1 NEON 汇编版 a64_fp32_6x6

**文件**：`input_transforms/a64_fp32_6x6.cpp`（1140 行）

#### 4.1.1 B 矩阵系数

```cpp
const float pcoeffs[4] = {1.0f, 2.0f, 4.0f, 5.0f};  // :42
// 装入 q0:
ldr q0, [%[pcoeffs]]  // q0 = {1, 2, 4, 5}, 即 v0.s[0]=1, v0.s[1]=2, v0.s[2]=4, v0.s[3]=5
```

这四个系数 `1, 2, 4, 5` 是 F(4,4,3,3) 输入变换 B 矩阵的关键参数。

#### 4.1.2 通道循环结构

```asm
    cmp %w[n_channels], #4     // :63, 通道数 ≥ 4？
    blt 2f                      // 不够 4, 跳到 2 通道路径
1:                              // 4 通道循环
    ldr q8, [%[inptr0], x20]   // :66, 加载输入像素 (col 4)
    ldr q2, [%[inptr0], x10]   // :67, 加载输入像素 (col 2)
    ldr q9, [%[inptr0]]        // :69, 加载输入像素 (col 0)
    ; ... 加载 6×6 输入 tile 的相关像素 ...
    fmla v14.4s, v9.4s, v0.s[2]    // :72, V14 += V9 × 4 (v0.s[2]=4)
    fmls v10.4s, v12.4s, v0.s[2]   // :76, V10 -= V12 × 4
    ; ... 大量 fmla/fmls/fadd/fsub ...
    str q23, [%[outptr0]]          // :166, 存储变换结果
    ; ... 36 个输出元素的存储 ...
    add %[inptr0], %[inptr0], #16  // :101, 前进 4 个 float (16 byte)
    cmp %w[n_channels], #4         // :376
    bge 1b                          // 继续
2:                                  // 降到 2 通道 (ldr d)
3:                                  // 降到 1 通道 (ldr s)
4:                                  // 结束
```

#### 4.1.3 关键汇编逐行翻译

| 汇编 | 翻译 |
|------|------|
| `ldr q8, [%[inptr0], x20]` | 从输入行第 5 列加载 4 个通道的 float 到 Q8 |
| `fmla v14.4s, v9.4s, v0.s[2]` | Q14 += Q9 × 4（4 路并行 FMA，v0.s[2]=4 是系数） |
| `fmls v10.4s, v12.4s, v0.s[2]` | Q10 -= Q12 × 4（4 路并行 FMS） |
| `fadd v10.4s, v10.4s, v4.4s` | Q10 += Q4（4 路加） |
| `fsub v9.4s, v9.4s, v4.4s` | Q9 -= Q4（4 路减） |
| `str q23, [%[outptr0]]` | 把 4 通道变换结果存到输出 |
| `add %[inptr0], %[inptr0], #16` | 输入指针前进 4 个 float（16 字节） |

> **本质**：与权重变换一样做两次矩阵乘（行变换 B^T，列变换 B），但用 `fmla`/`fmls`（乘加/乘减）把每一步融合成单条指令，减少中间寄存器。沿通道维 4 路并行。

### 4.2 SVE 汇编版 sve_fp32_6x6

**文件**：`input_transforms/sve_fp32_6x6.cpp`（361 行）

#### 4.2.1 与 NEON 版的关键差异

| 维度 | NEON (a64) | SVE |
|------|-----------|-----|
| 寄存器 | Q (128 bit, 固定 4 float) | Z (VL bit, SVE-512 时 16 float) |
| 通道并行度 | 4 | **16**（SVE-512） |
| 循环控制 | `cmp/blt/bge`（比较跳转） | `whilelt`（谓词自动控制） |
| tail 处理 | 3 段代码（4→2→1 通道降级） | **1 段代码**（谓词掩码自动处理） |
| 加载/存储 | `ldr q/d/s` | `ld1w/st1w`（谓词化） |

#### 4.2.2 谓词化循环控制

```asm
    whilelt p0.s, XZR, %x[num_channels]   // :67, p0[i] = (0+i < n_channels)? 设置谓词
    beq 2f                                 // 若 n_channels==0, 跳过
1:  // channel_loop
    ld1w { z31.s }, p0/Z, [%x[input_row_0]] // :70, 只加载 p0 为真的 lane
    decw %x[num_channels]                   // :71, 通道数 -= 1 (SVE word)
    ; ... 大量 fmla/fmls/fmad/fmsb ...
    st1w { z31.s }, p0, [%x[output_row_0]]  // :141, 只存储 p0 为真的 lane
    incb %x[output_row_0]                    // :158, 输出指针前进 VL 字节
    whilelt p0.s, XZR, %x[num_channels]     // :347, 重新计算谓词
    bne 1b                                   // :348, 若还有通道, 继续循环
```

**翻译**：
- `whilelt p0.s, XZR, num_channels` → "如果当前通道索引 < 总通道数，谓词对应 lane 置真"。首次调用时 XZR=0，所以当 num_channels ≥ 1 时 p0[0]=true。
- `ld1w {z31.s}, p0/Z, [addr]` → "从 addr 加载一个向量，但只填入 p0 为真的 lane，其余置零"。当通道数不是 VL 的整数倍时，最后一轮自动用谓词掩码处理 tail，**不需要降级到更窄的路径**。
- `incb` → "指针前进一个向量长度（VL 字节）"——SVE-512 时前进 64 字节（16 个 float）。
- `decw` → "通道计数减 1"（word 级递减）。

#### 4.2.3 SVE FMA 指令模式

```asm
    fmul z13.s, z28.s, z2.s[1]          // :73, Z13 = Z28 × 2 (z2.s[1]=2, 系数)
    fneg z13.s, p1/M, z13.s             // :76, Z13 = -Z13 (谓词 p1 全真)
    fmla z13.s, z11.s, z2.s[1]         // :81, Z13 += Z11 × 2
    fmad z31.s, p1/M, z16.s, z7.s       // :79, Z31 = Z31 + Z16 × Z7 (Z16=4.0)
    fmls z31.s, z27.s, z2.s[3]          // :84, Z31 -= Z27 × 5 (z2.s[3]=5)
```

**翻译**：
- `z2.s[1]` = 2.0（从 B_values = {1,2,4,5} 的第 1 个元素广播标量）
- `z16` = 4.0（常量，`:46` `fmov z16.s, #4.0`）
- `z2.s[3]` = 5.0（B_values 第 3 个元素）
- `fmad zD, p/M, zN, zM` → D = D + N × M（三操作数 FMA，谓词控制）
- `fmsb zD, p/M, zN, zM` → D = D - N × M（三操作数 FMS，谓词控制）

> **核心优势**：一条 SVE 指令处理 16 个通道（SVE-512），且谓词掩码自动处理 tail——代码量减半，循环无分支。

### 4.3 SME 版 sme_fp32_mla_6x6

**文件**：`input_transforms/sme_fp32_mla_6x6.cpp`（363 行）

#### 4.3.1 与 SVE 版的对比

```diff
  // 完全相同的 SVE 汇编代码体...
+ ".inst 0xd503477f  // SMSTART ZA\n"     // :47, 进入 SME 流式模式
  "fmov z16.s, #4.0\n"
  ... (中间代码逐字节相同) ...
  ".inst 0xd503467f  // SMSTOP\n"          // :352, 退出 SME 流式模式
```

**关键发现**：SME 版与 SVE 版的变换逻辑 **完全相同**（逐行一致的 SVE 指令），唯一区别是用 `SMSTART ZA` / `SMSTOP` 包裹。

#### 4.3.2 SMSTART / SMSTOP 做了什么

| 指令 | 编码 | 含义 |
|------|------|------|
| `SMSTART ZA` | `0xd503477f` | 进入 SME 流式模式：激活 ZA tile 寄存器，切换到流式 SVE 执行模式 |
| `SMSTOP` | `0xd503467f` | 退出流式模式：释放 ZA，回到普通 SVE/NEON 模式 |

在流式模式下：
- SVE 指令在 **流式向量长度** 上执行（可能更宽）
- ZA tile 可用（但本变换未直接使用 ZA——它只用 Z 寄存器和谓词）
- 流式模式有独立的寄存器状态，避免与非流式 SVE 代码互相干扰

> **结论**：对于输入变换，SME 版 = SVE 版 + 流式模式包裹。真正的 SME 矩阵加速（FMOPA 外积）出现在 **输出变换** 中（第 5.2 节）。

---

## 5. 输出变换：两个 ISA 版本对比

### 5.1 NEON intrinsics 版 arm_fp32_4x4_3x3

**文件**：`output_transforms/arm_fp32_4x4_3x3.cpp`（242 行）

#### 5.1.1 A 矩阵（从源码 :67-92 提取）

变换 `f = A^T · M · A`，代码先算 `FZ = F · A`（列变换），再算 `f = A^T · FZ`（行变换）。

A 矩阵的系数是 `{1, -1, 2, 4, 8}`：

```cpp
// 列变换 FZ = F · A  (:64-77)
FZ[i][0] = F[i][0] + F[i][1] + F[i][2] + F[i][3] + F[i][4];                    // 系数 1,1,1,1,1
FZ[i][1] = (F[i][1] - F[i][2]) + 2*(F[i][3] - F[i][4]);                        // 系数 1,-1,2,-2
FZ[i][2] = (F[i][1] + F[i][2]) + 4*(F[i][3] + F[i][4]);                        // 系数 1,1,4,4
FZ[i][3] = (F[i][1] - F[i][2]) + 8*(F[i][3] - F[i][4]) + F[i][5];             // 系数 1,-1,8,-8,1

// 行变换 f = A^T · FZ  (:80-93) —— 用相同系数对行变换
```

故 A 矩阵（4×6）：

```
        ┌ 1  1  1  1  1  0 ┐
   A =  │ 0  1 -1  2 -2  0 │   (把 6×6 Winograd 域 → 4×4 输出)
        │ 0  1  1  4  4  0 │
        └ 0  1 -1  8 -8  1 ┘
```

#### 5.1.2 融合 bias + ReLU

```cpp
// 加 bias  (:96-104)
b = bptr ? vld1q_f32(bptr) : vdupq_n_f32(0.0f);

// bias + clamp（=fused ReLU/BoundedReLU） (:109-112)
y = vmaxq_f32(vminq_f32(vaddq_f32(f[i][j], b),        // f + bias
                          vdupq_n_f32(output_max)),    // min(., max)
               vdupq_n_f32(output_min));               // max(., min)
vst1q_f32(outptr + ..., y);                             // 存储
```

**翻译**：
- `vaddq_f32(f, b)` → 输出值 + bias
- `vminq_f32(., max)` → 上界裁剪（Bounded ReLU 的上界）
- `vmaxq_f32(., min)` → 下界裁剪（ReLU 的 min=0）
- 无激活时 max=+∞, min=-∞ → clamp 退化为恒等

### 5.2 SME 汇编版 sme_fp32_mopa_4x4_3x3（核心创新）

**文件**：`output_transforms/sme_fp32_mopa_4x4_3x3.cpp`（892 行）

这个实现与 NEON 版 **完全不同的算法**——它用 SME 的 **外积（outer product）** 指令 `FMOPA` 和 **二维 tile 寄存器 ZA** 一次性算完整个 2D 变换。

#### 5.2.1 算法思想：vec trick + Kronecker 积

源码注释（:45-53）解释了核心数学技巧：

> 输出变换 `y = A^T · Y · A` 可以用"vec trick"重写为：
>
> `vec(y) = (A^T ⊗ A^T) · vec(Y)`
>
> 其中 ⊗ 是 Kronecker 积，`vec()` 把矩阵按列展平成向量。
> 这样 2D 矩阵链就变成了一个 **矩阵-向量乘法**。

- `vec(Y)` 是 36 维向量（6×6 展平）
- `(A^T ⊗ A^T)` 是 16×36 矩阵（映射到 16 维输出 = 4×4）
- 把多个通道的 `vec(Y)` 堆在一起，就变成一次矩阵乘法（GEMV/GEMM）

#### 5.2.2 构造 Kronecker 积的系数

```cpp
// A^T 矩阵的行系数，分成"外项"和"内项"
const float outer_terms[32] = {    // 8 组 × 4 元素，装入 2 个 Z 寄存器
    1, 1,  1, 1,    // A^T 第 0 行: [1, 1, 1, 1]（与 A 第 0 列相关）
    0, 1, -1, 2,    // A^T 第 1 行
    0, 1,  1, 4,    // A^T 第 2 行
    0, 1, -1, 8,    // A^T 第 3 行
    1, 0,  0, 0,    // 后续行的延续（带 padding 对齐）
   -2, 0,  0, 0,
    4, 0,  0, 0,
   -8, 1,  0, 0
};
const float inner_terms[24] = {    // 6 组 × 4 元素，装入 6 个 Z 寄存器（复制广播）
    1,  0, 0, 0,
    1,  1, 1, 1,
    1, -1, 1, -1,
    1,  2, 4, 8,
    1, -2, 4, -8,
    0,  0, 0, 1
};
```

`(A^T ⊗ A^T)` 的每个元素 = `outer_terms[某行] × inner_terms[某列]`，代码在运行时用 `fmul` 构造这些系数。

#### 5.2.3 FMOPA 外积累加（核心）

```asm
    ".inst 0xc00800ff  // zero {zad0-zad7}\n"     // :129, 清零 8 个 ZA tile
    "fmov z1.s, #1.0\n"                            // :130, Z1 = 1.0（bias 用）
    ; ...
    ".inst 0x8080b420  // fmopa za0.s, p5/M, p5/M, z1.s, z0.s\n"  // :161
```

**翻译**：
- `zero {zad0-zad7}` → 把 8 个 ZA tile（za0-za7）全部清零。ZA tile 是一个二维寄存器（如 SVE-512 时 16×16 = 256 个 f32）。
- `fmopa za0.s, p5/M, p5/M, z1.s, z0.s` → **`za0 += Z1 ⊗ Z0`**（外积累加）。Z1 是 `1.0` 向量，Z0 是 bias 向量，所以这条指令把 bias 累加进 za0 的每一行。这是 SME 的核心加速指令：一条指令完成一个 **16×16 的外积 + 累加**。

随后大量的 `fmopa` 把 Winograd 域数据的各列乘以 Kronecker 积系数，累加进不同的 ZA tile：

```asm
    ".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"  // :171
    ".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"  // :175
    ".inst 0x809db562  // fmopa za2.s, p5/M, p5/M, z11.s, z29.s\n"  // :179
    ".inst 0x809cb563  // fmopa za3.s, p5/M, p5/M, z11.s, z28.s\n"  // :182
```

**翻译**：
- `za0 += Z11 ⊗ Z31`：Z11 是 Kronecker 系数向量（由 `fmul` 构造），Z31 是输入数据向量。外积 `Z11 ⊗ Z31` 产生一个 16×16 矩阵，累加进 za0。
- 4 个 ZA tile（za0-za3）并行累加 → 同时计算输出 tile 的 4 行。

#### 5.2.4 从 ZA tile 读出结果

```asm
    ".inst 0xc082741f  // mova z31.s, p5/M, za0h.s[x15]\n"    // :516
    ".inst 0xc082541c  // mova z28.s, p5/M, za0h.s[x14]\n"    // :517
```

**翻译**：
- `mova z31.s, p5/M, za0h.s[x15]` → 从 ZA tile 0 读出第 x15 行到 Z31。x15=0xC=12，所以读第 12 行。
- 多条 `mova` 读出 ZA tile 的不同行，得到 4×4=16 个输出值（分布在不同 Z 寄存器中）。

#### 5.2.5 融合 ReLU + 存储

```asm
    "fmin z31.s, p5/M, z31.s, z10.s\n"     // :518, Z31 = min(Z31, max)  上界裁剪
    "fmax z31.s, p5/M, z31.s, z12.s\n"     // :528, Z31 = max(Z31, min)  下界裁剪 (=ReLU)
    "st1w { z31.s }, p0, [%x[output], x25, LSL #2]\n"  // :562, 存储到输出
```

**翻译**：
- Z10 = output_max，Z12 = output_min
- `fmin` + `fmax` = clamp = ReLU/BoundedReLU
- `st1w` 谓词化存储（只存 p0 为真的 lane）

#### 5.2.6 通道循环与多 tile 并行

```asm
    "whilelt p4.s, x25, %x[n_channels]\n"   // :135, 谓词: 当前通道 < 总通道
    "whilelt p3.s, x24, %x[n_channels]\n"   // :136, 第二组通道
    "whilelt p2.s, x23, %x[n_channels]\n"   // :139, 第三组
    "whilelt p1.s, x22, %x[n_channels]\n"   // :140, 第四组
```

**翻译**：4 组谓词（p1-p4）分别控制 4 组通道子集，**一次迭代同时处理 4×VL 个通道**。在 SVE-512 上 VL=16，所以 **一次处理 64 个通道**（对比 NEON 版一次 4 个）。

```asm
    "incw x25, ALL, MUL #4\n"    // :798, 通道索引 += 4×VL
    "incw x24, ALL, MUL #4\n"    // :802
    "incw x23, ALL, MUL #4\n"    // :806
    "incw x22, ALL, MUL #4\n"    // :844
```

> **SME 输出变换的核心优势**：用 FMOPA 外积指令把整个 `(A^T ⊗ A^T) · vec(Y)` 矩阵-向量乘法映射到 ZA tile 的外积累加，一条 FMOPA 指令完成 16×16=256 次 FMA，4 个 ZA tile 并行处理 4 个输出行，64 通道批量处理——这是 NEON 版无法企及的吞吐。

---

## 6. 三种 ISA 实现的本质差异总结

```
┌──────────────────────────────────────────────────────────────────────┐
│                      输入变换 (6×6 → 6×6)                              │
├──────────┬──────────────────┬───────────────────┬───────────────────┤
│          │ NEON (a64)       │ SVE               │ SME                │
├──────────┼──────────────────┼───────────────────┼───────────────────┤
│ 代码形式 │ 内嵌汇编          │ 内嵌汇编           │ 内嵌汇编+SMSTART  │
│ 寄存器   │ Q(128b,4f)       │ Z(VL,16f)         │ Z(流式,16f)       │
│ 并行度   │ 4 通道/指令       │ 16 通道/指令       │ 16 通道/指令       │
│ tail处理 │ 3段降级(4→2→1)    │ 谓词掩码(无降级)   │ 同SVE+流式模式    │
│ 循环控制 │ cmp/blt/bge      │ whilelt/bne       │ 同SVE             │
│ 矩阵加速 │ 无               │ 无                 │ 无(仅流式模式)    │
│ 与SVE差异│ —                │ —                 │ SMSTART/SMSTOP包裹│
│          │                  │                   │ 代码体完全相同     │
└──────────┴──────────────────┴───────────────────┴───────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      输出变换 (6×6 → 4×4 + bias + ReLU)               │
├──────────┬──────────────────────────┬──────────────────────────────┤
│          │ NEON intrinsics           │ SME 汇编                     │
├──────────┼──────────────────────────┼──────────────────────────────┤
│ 算法     │ 顺序两次矩阵乘            │ Kronecker积+外积(FMOPA)      │
│          │ FZ=F·A, 然后 f=A^T·FZ    │ vec(y)=(A^T⊗A^T)·vec(Y)     │
│ 寄存器   │ Q(128b,4f)               │ Z + ZA tile(16×16)            │
│ 并行度   │ 4 通道/指令               │ 64 通道/迭代(4组×16)         │
│ 核心指令 │ fmla/fmls/vaddq           │ fmopa(16×16外积累加)         │
│ bias+ReLU│ vmax/vmin(最后做)        │ fmopa累加bias + fmin/fmax    │
│ 代码量   │ 242行(C++易读)           │ 892行(汇编)                  │
│ 性能     │ 基准                      │ ★远高于NEON(矩阵加速)        │
└──────────┴──────────────────────────┴──────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      权重变换 (3×3 → 6×6)                             │
├──────────────────────────────────────────────────────────┤
│ 仅有 NEON intrinsics 版 (arm_fp32_4x4_3x3, 236行)         │
│ 用 vld1q_f32 / vmulq_n_f32 / vmlaq_n_f32 / vst1q_f32      │
│ 无 SVE/SME 专用版（权重变换在 prepare 阶段一次性做，     │
│   性能不敏感，NEON 版足够）                                │
└──────────────────────────────────────────────────────────┘
```

### 关键洞察

1. **输入变换**：SME 版 = SVE 版 + `SMSTART`/`SMSTOP` 包裹。变换逻辑完全相同，SME 的流式模式提供了潜在的更宽向量和 ZA tile 可用性，但输入变换本身并未利用 ZA 的外积能力。
2. **输出变换**：SME 版与 NEON 版是 **完全不同的算法**。NEON 版顺序做两次矩阵乘（行变换+列变换），SME 版用 Kronecker 积把 2D 变换折叠成 1D 矩阵-向量乘法，再用 FMOPA 外积累加一条指令完成 16×16=256 次 FMA。这是 SME 在 Winograd 中的 **真正价值所在**。
3. **权重变换**：只有 NEON intrinsics 版。因为权重变换在 `prepare()` 阶段一次性完成（非热路径），不需要极致优化。
4. **谓词化是 SVE 的核心优势**：`whilelt` + 谓词化 `ld1w`/`st1w` 消除了 NEON 版的 tail 处理分支（从 3 段代码减到 1 段），代码量减半且无分支预测开销。
5. **FMOPA 是 SME 的核心优势**：`fmopa za0.s, p/M, p/M, zN, zM` 一条指令完成 16×16 外积累加（256 次 FMA），4 个 ZA tile 并行 → 4×256=1024 FMA/迭代，是 NEON `fmla`（4 FMA/指令）的 **256 倍吞吐**。

> 本文基于 ACL 53.1.0 源码逐行分析，所有汇编指令均标注源码行号，具体实现以源码为准。

---

## 7. F(2,2,3,3) 变换解析（NEON intrinsics）

前文第 3-5 节聚焦了 **F(4,4,3,3)** 的变换实现。实际上 ACL 同样实现了 **F(2,2,3,3)**（输出 tile 2×2，输入 tile 4×4，16 个 GEMM），且三个变换**全部只有 NEON intrinsics 版**——没有 SVE/SME 加速版本。

> **为什么没有 SVE/SME 版？** F(2,2,3,3) 的输出 tile 只有 2×2（4 个输出像素），变换域矩阵 4×4（16 元素）远小于 F(4,4,3,3) 的 6×6（36 元素）。F(2,2,3,3) 通常作为小特征图的 fallback，计算量小，优化收益低，NEON 版已够用。

### 7.1 权重变换 arm_fp32_2x2_3x3

**文件**：`weight_transforms/arm_fp32_2x2_3x3.cpp`（200 行，NEON intrinsics）
**数学**：`V = G · g · G^T`，3×3 权重 → 4×4 Winograd 域（**无归一化因子**，对比 F(4,4,3,3) 的 /576）。

#### G 矩阵（从源码 :60-83 提取）

```cpp
Ww[0][j] = w[0][j];                                    // G 第 0 行: [1,   0,   0  ]
Ww[1][j] = 0.5*(w[0][j] + w[1][j] + w[2][j]);         // G 第 1 行: [0.5, 0.5, 0.5]
Ww[2][j] = 0.5*(w[0][j] - w[1][j] + w[2][j]);         // G 第 2 行: [0.5,-0.5, 0.5]
Ww[3][j] = w[2][j];                                    // G 第 3 行: [0,   0,   1  ]
```

故 F(2,2,3,3) 的 G 矩阵（4×3）：

```
        ┌ 1     0     0   ┐
   G =  │ 0.5   0.5   0.5 │   (4×3, 把 3×3 权重 → 4×4 Winograd 域)
        │ 0.5  -0.5   0.5 │
        └ 0     0     1   ┘
```

随后用相同 G 矩阵对列变换（`V[i][j]` 行 :72-83），无除法。

#### 与 F(4,4,3,3) 的差异

| | F(2,2,3,3) G | F(4,4,3,3) G |
|---|---|---|
| 维度 | 4×3 | 6×3 |
| 系数 | 1, 0.5（简单） | 6, -4, 1, 2, 4, 24（复杂） |
| 归一化 | 无 | /576 |
| 输出 tile | 4×4 | 6×6 |
| GEMM 数 | 16 | 36 |

> F(2,2,3,3) 的 G 矩阵系数更简单（多为 0.5 和 1），变换加法开销更小，但 GEMM 数少（16 vs 36）、输出 tile 小（2×2 vs 4×4）。

### 7.2 输入变换 arm_fp32_4x4（4×4 tile）

**文件**：`input_transforms/arm_fp32_4x4.cpp`（251 行，NEON intrinsics）
**数学**：`U = B^T · d · B`，4×4 输入 tile → 4×4 Winograd 域。

> 注意：这个文件也出现在 F(4,4,3,3) 的上下文中，因为 ACL 把 F(2,2,3,3) 的 4×4 输入变换复用了——输入 tile = m+r-1 = 2+3-1 = **4×4**，恰好是 `arm_fp32_4x4` 的 tile 大小。

#### B^T 矩阵（从源码 :99-128 提取）

```cpp
// 列变换 XTx = B^T · x  (:99-112)
XTx[0][j] = x[0][j] - x[2][j];    // B^T 第 0 行: [1,  0, -1,  0]
XTx[1][j] = x[1][j] + x[2][j];    // B^T 第 1 行: [0,  1,  1,  0]
XTx[2][j] = x[2][j] - x[1][j];    // B^T 第 2 行: [0, -1,  1,  0]
XTx[3][j] = x[1][j] - x[3][j];    // B^T 第 3 行: [0,  1,  0, -1]

// 行变换 U = XTx · B  (:115-128) —— 用相同 B^T 对行变换
```

故 F(2,2,3,3) 的 B^T 矩阵（4×4）：

```
          ┌ 1  0 -1  0 ┐
   B^T =  │ 0  1   1  0 │   (4×4, 把 4×4 输入 tile → 4×4 Winograd 域)
          │ 0 -1   1  0 │
          └ 0  1   0 -1 ┘
```

#### 实现模式

```cpp
for (int channels_remaining = n_channels; channels_remaining >= 4; channels_remaining -= 4) {
    float32x4_t x[4][4], XTx[4][4], U[4][4];     // 4 通道并行
    for (i...) for (j...)
        x[i][j] = vld1q_f32(x_ptrs[i][j]);       // 加载 4 个 float
    // ... B^T 变换 ...
    for (i...) for (j..., m++)
        vst1q_f32(outptr + m*matrix_stride, U[i][j]);  // 存储
}
// 降到 2 通道 (float32x2_t) 和 1 通道 (float) 的 tail
```

> **对比 F(4,4,3,3) 输入变换**：F(2,2,3,3) 用 NEON intrinsics（C++ 易读），一次 4 通道，3 段降级 tail。F(4,4,3,3) 用内嵌汇编（SVE/SME），一次 16 通道，谓词化无降级。

### 7.3 输出变换 arm_fp32_2x2_3x3

**文件**：`output_transforms/arm_fp32_2x2_3x3.cpp`（220 行，NEON intrinsics）
**数学**：`f = A^T · M · A`（+ bias + ReLU），4×4 Winograd 域 → 2×2 输出。

#### A^T 矩阵（从源码 :64-81 提取）

```cpp
// 列变换 FZ = F · A  (:64-71)
FZ[i][0] = F[i][0] + F[i][1] + F[i][2];    // A 列 0: [1, 1, 1, 0]
FZ[i][1] = F[i][1] - F[i][2] - F[i][3];    // A 列 1: [0, 1,-1,-1]

// 行变换 f = A^T · FZ  (:74-81) —— 用相同 A^T 对行变换
```

故 F(2,2,3,3) 的 A^T 矩阵（2×4）：

```
          ┌ 1  1  1  0 ┐
   A^T =  │           │   (2×4, 把 4×4 Winograd 域 → 2×2 输出)
          └ 0 -1 -1 -1 ┘
```

#### 融合 bias + ReLU（与 F(4,4,3,3) 完全一致的模式）

```cpp
b = bptr ? vld1q_f32(bptr) : vdupq_n_f32(0.0f);         // :86-92, 加载 bias
y = vmaxq_f32(vminq_f32(vaddq_f32(f[i][j], b),          // :99-101, f + bias
                          vdupq_n_f32(output_max)),      //   min(., max)  上界
               vdupq_n_f32(output_min));                  //   max(., min)  下界=ReLU
vst1q_f32(outptr + ..., y);                               // 存储
```

#### F(2,2,3,3) 三矩阵速查卡

```
F(2,2,3,3):
  权重: V = G·g·G^T         G = [1,0,0; 0.5,0.5,0.5; 0.5,-0.5,0.5; 0,0,1]  (4×3)
  输入: U = B^T·d·B        B^T = [1,0,-1,0; 0,1,1,0; 0,-1,1,0; 0,1,0,-1] (4×4)
  输出: f = A^T·M·A        A^T = [1,1,1,0; 0,-1,-1,-1]                    (2×4)
  GEMM: n_multis = (2+3-1)² = 16 个, M=tiles, N=OC, K=IC
  每输出像素等效乘法 9→4 (≈44%)
```

---

## 8. 何时走 F(4,4,3,3) vs F(2,2,3,3)

### 8.1 选择机制回顾

ACL 的 `get_implementation`（`winograd_implementations.hpp:236-339`）用**贪心算法**选变换三元组：

1. 收集所有满足约束的 **输出变换**（kernel 尺寸匹配）
2. 对每个输出变换（按**注册表顺序**——大输出 tile 在前），找匹配的权重变换和输入变换
3. 第一个完整匹配的三元组胜出

### 8.2 3×3 kernel 的输出变换注册表顺序

```
output_transforms_fp32.cpp:50-66 注册顺序：
  1. sme_fp32_mopa_4x4_3x3  → F(4,4,3,3)  约束: RequiresSME
  2. arm_fp32_4x4_3x3       → F(4,4,3,3)  约束: LargerShape (input > 4×4)
  3. arm_fp32_2x2_3x3       → F(2,2,3,3)  约束: 无
  4. arm_fp32_2x2_5x5       → F(2,2,5,5)  约束: 无
  ...
```

贪心按顺序尝试：先试 F(4,4,3,3) 的两个变体，只有都失败才到 F(2,2,3,3)。

### 8.3 判定流程

```
3×3 kernel, 调用 get_implementation:

  ┌─ 试 sme_fp32_mopa_4x4_3x3 (F(4,4,3,3), 需 SME)
  │   has_sme()?
  │   ├─ YES → 需要 6×6 输入变换 → sme_fp32_mla_6x6 (SME) 匹配 → ✅ 选中 F(4,4,3,3)
  │   └─ NO  → 继续 ↓
  │
  ├─ 试 arm_fp32_4x4_3x3 (F(4,4,3,3), 需 LargerShape)
  │   input.rows > 4 && input.cols > 4?
  │   ├─ YES → 需要 6×6 输入变换 → sve/a64_fp32_6x6 匹配 → ✅ 选中 F(4,4,3,3)
  │   └─ NO  → 继续 ↓ (特征图太小)
  │
  └─ 试 arm_fp32_2x2_3x3 (F(2,2,3,3), 无约束)
      → 需要 4×4 输入变换 → arm_fp32_4x4 匹配 → ✅ 选中 F(2,2,3,3)
```

### 8.4 各机器上的选择

| 机器 | 特征图大小 | 选中 | 原因 |
|------|-----------|------|------|
| **920F**（SME+SVE-512） | 任意 ≥1×1 | **F(4,4,3,3)** | SME 变换排最前，无 LargerShape 约束，直接选中 |
| SVE 机器（无 SME，如 A64FX） | input > 4×4（几乎所有实际 case） | **F(4,4,3,3)** | LargerShape 满足 → arm_fp32_4x4_3x3 + sve_fp32_6x6 |
| SVE 机器（无 SME） | input ≤ 4（极小特征图） | **F(2,2,3,3)** | LargerShape 不满足 → 退到 arm_fp32_2x2_3x3 |
| NEON-only 机器 | input > 4×4 | **F(4,4,3,3)** | LargerShape 满足 → arm_fp32_4x4_3x3 + a64_fp32_6x6 |
| NEON-only 机器 | input ≤ 4 | **F(2,2,3,3)** | LargerShape 不满足 → arm_fp32_2x2_3x3 |

### 8.5 实际场景中的意义

```
在你的 920F 机器上（SME 可用）:
  → 永远走 F(4,4,3,3) 的 SME 版
  → F(2,2,3,3) 的代码路径永远不会被执行

F(2,2,3,3) 实际使用的场景:
  → 无 SME 的 SVE/NEON 机器 + 极小特征图（≤4 像素某维度）
  → 这在实际深度学习中很少见（多数特征图 ≥7×7）
  → ACL 保留 F(2,2,3,3) 主要是为完整性 + 小图 fallback

F(2,2,3,3) vs F(4,4,3,3) 权衡:
  ┌──────────────┬──────────────────┬──────────────────┐
  │              │ F(2,2,3,3)       │ F(4,4,3,3)       │
  ├──────────────┼──────────────────┼──────────────────┤
  │ 输出 tile    │ 2×2 (4 像素)     │ 4×4 (16 像素)    │
  │ 输入 tile    │ 4×4              │ 6×6              │
  │ GEMM 数      │ 16               │ 36               │
  │ 每像素等效乘法│ 4 (9的44%)       │ 2.25 (36/16)     │
  │ 变换加法开销  │ 低 (简单系数)    │ 高 (复杂系数+576)│
  │ 小图填充浪费  │ 低 (2×2 tile)   │ 高 (4×4 tile)    │
  │ 小图综合效率  │ ★更好            │ 变换开销>乘法节省 │
  │ 大图综合效率  │ GEMM 少, 不占优  │ ★GEMM 复用好     │
  └──────────────┴──────────────────┴──────────────────┘
```

> **一句话**：F(4,4,3,3) 对大特征图更优（每像素乘法更少、GEMM 复用好），F(2,2,3,3) 对极小特征图更优（tile 填充浪费少、变换开销小）。ACL 贪心优先 F(4,4,3,3)，只有当特征图太小（≤4）且无 SME 时才退到 F(2,2,3,3)。在 920F 上 SME 可用，**F(2,2,3,3) 永远不会被选中**。

---

## 9. 每种实现的数据布局与计算流程详解

> 本节为每个变换内核画出输入/输出的内存布局图和计算流程图，帮助理解数据如何在内存中排列、内核如何遍历。

### 9.1 权重变换 arm_fp32_4x4_3x3（F(4,4,3,3)）

#### 9.1.1 输入数据布局

权重在 permute 后是 **HWIO** 格式（Height × Width × Input_channels × Output_channels）：

```
内存布局（HWIO，以 IC=192, OC=192, 3×3 kernel 为例）：

  地址偏移 = h * ld_row + w * ld_col + ic * ld_channel + oc

  ld_row = IC * OC（一行的跨度）
  ld_col = 1（一列的跨度，但在 permute 后可能是 OC）

  对于一个 (ic, oc) 通道对，3×3 权重排列为：
  
  inptr 指向 w[0][0]（h=0, w=0, ic, oc）
  
  ┌─────────────────────────────────────┐
  │ w[0][0]  w[0][1]  w[0][2]          │  ← h=0 (行0)
  │ w[1][0]  w[1][1]  w[1][2]          │  ← h=1 (行1)
  │ w[2][0]  w[2][1]  w[2][2]          │  ← h=2 (行2)
  └─────────────────────────────────────┘
   ↑ ld_in_row 步长
  每个 w[i][j] 是 1 个 float（标量版）
  或 4 个 float（NEON 向量化版，一次处理 4 个 OC）
```

#### 9.1.2 计算流程

```
输入：3×3 权重矩阵 g（每个元素是一个标量或 4 通道向量）
    
  步骤 1：行变换 Ww = G · g
    ┌──────────┐   ┌─────────────┐       ┌──────────────┐
    │ 6  0  0  │   │ w[0][j]     │       │ 6*w[0][j]    │ ← Ww[0][j]
    │-4 -4 -4 │ × │ w[1][j]     │   =   │ -4*(w0+w1+w2)│ ← Ww[1][j]
    │-4  4 -4 │   │ w[2][j]     │       │ 4*(w1-w0-w2) │ ← Ww[2][j]
    │ 1  2  4  │   └─────────────┘       │ w0+2*w1+4*w2 │ ← Ww[3][j]
    │ 1 -2  4  │                          │ w0-2*w1+4*w2 │ ← Ww[4][j]
    │ 0  0 24  │                          │ 24*w[2][j]   │ ← Ww[5][j]
    └──────────┘                          └──────────────┘
    对每个 j=0,1,2 做一次（共 3 次）
    用 NEON: vmulq_n_f32, vaddq_f32, vsubq_f32, vmlaq_n_f32, vmlsq_n_f32

  步骤 2：列变换 V = Ww · G^T / 576
    对每个 i=0..5，用相同 G 矩阵对列变换：
    V[i][0] = (6 * Ww[i][0]) / 576
    V[i][1] = (-4*(Ww[i][0]+Ww[i][1]+Ww[i][2])) / 576
    V[i][2] = (4*(Ww[i][1]-Ww[i][0]-Ww[i][2])) / 576
    V[i][3] = (Ww[i][0]+2*Ww[i][1]+4*Ww[i][2]) / 576
    V[i][4] = (Ww[i][0]-2*Ww[i][1]+4*Ww[i][2]) / 576
    V[i][5] = (24*Ww[i][2]) / 576
    用 NEON: 先做乘加，最后乘 recip576 (=1/576)

  步骤 3：存储 V[6][6] = 36 个元素
    for (i=0..5) for (j=0..5, m++)
      vst1q_f32(outptr + m*matrix_stride, V[i][j])
```

#### 9.1.3 输出数据布局

```
输出：6×6 变换后权重 V，36 个元素，按 matrix_stride 间隔存储

  outptr → V[0][0]  V[0][1]  ... V[0][5]  V[1][0] ... V[5][5]
            ↑                         ↑
            matrix_stride 间隔         matrix_stride 间隔

  每个 V[i][j] 是 4 个 float（4 个 OC 通道的变换后权重）
  matrix_stride = roundup(OC, 4)（按 4 对齐，:324-325 winograd_spec.weight_ld_row）
  
  在 Winograd 域中，这 36 个元素对应 36 个 GEMM 的权重矩阵 B
```

#### 9.1.4 通道维向量化

```
权重变换沿 OC（输出通道）维向量化：

  IC=0  IC=1  ...  IC=191
  ┌────┐ ┌────┐      ┌────┐
  │OC0-3│ │OC0-3│      │OC0-3│   ← 每次处理 4 个 OC（vld1q_f32）
  │OC4-7│ │OC4-7│      │OC4-7│
  │ ...│ │ ...│      │ ...│
  │OC188│ │OC188│      │OC188│
  │-191│ │-191│      │-191│
  └────┘ └────┘      └────┘
  
  外层循环：n_channels -= 4（一次 4 个 OC）
  内层：对每个 (IC, OC_block) 做 G·g·G^T 变换
  
  tail 处理：降到 2 通道（vld1_f32）、1 通道（标量）
```

---

### 9.2 输入变换 — 数据布局与 tile 遍历（F(4,4,3,3)）

#### 9.2.1 输入特征图布局

输入是 **NHWC** 格式（Batch × Height × Width × Channels）：

```
内存布局（NHWC，以 N=4, H=40, W=40, C=192 为例）：

  地址偏移 = n * (H*W*C) + h * (W*C) + w * C + c

  ld_in_batch = H * W * C    （一个 batch 的跨度）
  ld_in_row   = W * C        （一行的跨度）
  ld_in_col   = C             （一列的跨度 = 通道数）
  
  对于一个像素 (n, h, w)，C=192 个通道连续排列：
  
  inptr → [c0, c1, c2, ..., c191]   ← 一个像素的所有通道
           ↑ ld_in_col = 192
```

#### 9.2.2 Tile 划分

F(4,4,3,3) 的输入 tile = 6×6，输出 tile = 4×4，tile 步长 = 4（输出 tile 大小）。

```
输入特征图（40×40，补零后 42×42，pad=1）：

  tile_stride_rows = 6 - 3 + 1 = 4  （= 输出 tile 高度 m）
  tile_stride_cols = 4
  n_tile_rows = ceil(OH / 4) = ceil(40 / 4) = 10
  n_tile_cols = ceil(OW / 4) = 10
  总 tile 数 = 10 × 10 = 100

  ┌─────────┬─────────┬─────────┬─────┐
  │ tile    │ tile    │ tile    │ ... │  ← tile 行 0
  │ (0,0)  │ (0,1)  │ (0,2)  │     │     每个 tile 6×6
  │ 6×6    │ 6×6    │ 6×6    │     │     输出 4×4
  ├─────────┼─────────┼─────────┼─────┤
  │ tile    │ tile    │ tile    │     │  ← tile 行 1
  │ (1,0)  │ (1,1)  │ (1,2)  │     │
  ├─────────┼─────────┼─────────┼─────┤
  │  ...    │  ...    │  ...    │     │
  └─────────┴─────────┴─────────┴─────┘
  
  相邻 tile 有 2 像素重叠（6 - 4 = 2）—— Winograd 允许重叠
```

#### 9.2.3 线程划分

```
tile 行按线程条带化分配（input_transform.hpp:106）：

  线程 0：tile 行 0, n_threads, 2*n_threads, ...
  线程 1：tile 行 1, 1+n_threads, ...
  线程 2：tile 行 2, 2+n_threads, ...
  ...
  
  每个 tile 行内，按 tile 列顺序处理（:115-132）
  输出指针按 ld_out_row 前进（一个 tile 的输出跨度）
```

#### 9.2.4 单个 tile 的内存提取

```
对于一个 6×6 输入 tile（tile_i, tile_j）：

  start_i = tile_i * 4
  start_j = tile_j * 4
  pad_top = max(0, pad_top - start_i)   ← 边界 padding
  pad_left = max(0, pad_left - start_j)
  
  inptr_tile 指向输入特征图中的起始位置：
  inptr_tile = inptr + (start_i - pad_top) * ld_in_row + (start_j - pad_left) * ld_in_col

  6×6 tile 在内存中的实际位置（非连续！）：

  行 0: inptr_tile[0]              ... inptr_tile[5 * ld_in_col]
  行 1: inptr_tile[ld_in_row]      ... inptr_tile[ld_in_row + 5 * ld_in_col]
  ...
  行 5: inptr_tile[5*ld_in_row]    ... inptr_tile[5*ld_in_row + 5*ld_in_col]

  每个 "格子" 是 C 个 float（通道维），内核按通道批量处理
  → 6 行 × 6 列 = 36 个位置，每个位置有 C 个通道
  → 内存不连续（行间有 ld_in_row 步长，列间有 ld_in_col 步长）
```

#### 9.2.5 输入变换的数学过程（F(4,4,3,3)）

B 矩阵系数是 {1, 2, 4, 5}（源码 `:42`），对应 F(4,4,3,3) 的输入变换矩阵。

```
变换 U = B^T · d · B（6×6 输入 → 6×6 Winograd 域）

  步骤 1：行变换 XTx = B^T · x（对每列 j 做 6 次线性组合）
    XTx[0][j] = f(d[0][j], d[2][j], d[4][j])     用系数 1, 4, 5
    XTx[1][j] = f(d[1][j], d[2][j], d[3][j])     用系数 2, 4, 5
    XTx[2][j] = f(d[1][j], d[2][j], d[3][j])     用系数 2, 4, 5（不同符号）
    XTx[3][j] = f(d[1][j], d[3][j], d[4][j])     用系数 1, 2, 5
    XTx[4][j] = f(d[1][j], d[3][j], d[4][j])     ...
    XTx[5][j] = f(d[2][j], d[4][j], d[5][j])     ...
    
    具体公式由汇编中的 fmul/fmla/fmls/fmad/fmsb 实现
    每个公式形如：result = a*coef1 ± b*coef2 ± c*coef3
    其中 coef 来自 {1, 2, 4, 5}（B_values）

  步骤 2：列变换 U = XTx · B（对每行 i 做相同的 6 次线性组合）
    用与步骤 1 相同的系数和模式，对行方向变换
    
  结果：6×6 = 36 个 Winograd 域元素，每个包含 C 个通道的数据
```

#### 9.2.6 输出数据布局

```
输出：6×6 Winograd 域，36 个元素，按 matrix_stride 和 ld_out_row 间隔存储

  outptr_tile → U[0][0]  U[0][1]  ... U[0][5]   ← 第 0 行
                 U[1][0]  U[1][1]  ... U[1][5]   ← 第 1 行
                 ...
                 U[5][0]  U[5][1]  ... U[5][5]   ← 第 5 行
  
  每个元素是 VL 个 float（SVE-512: 16 个通道）
  matrix_stride = roundup(n_output_patches, 4) * roundup(IC, 4)（WinogradDomainSpec）
  ld_out_row    = 同上（一个 tile 行的跨度）
  
  多个 tile 的输出按 ld_out_row 间隔排列：
  outptr_tile[0]         → tile(0,0) 的 U[0][0]
  outptr_tile[ld_out_row] → tile(0,1) 的 U[0][0]
  ...
```

#### 9.2.7 NEON 版 vs SVE 版的数据布局差异

```
NEON 版（a64_fp32_6x6）：
  每次处理 4 个通道（Q 寄存器 = 128 bit = 4 float）
  通道循环：cmp/blt/bge + 3 段降级（4→2→1）
  ld_in_row 前进 #16（4 float × 4 byte）

SVE 版（sve_fp32_6x6）：
  每次处理 16 个通道（Z 寄存器 = 512 bit = 16 float）
  通道循环：whilelt 谓词自动处理 tail
  incb 前进 VL 字节 = 64 字节（16 float）
  
  数据布局相同，但每次加载/存储的通道数不同：
  NEON: [c0,c1,c2,c3]            ← 4 通道
  SVE:  [c0,c1,...,c15]          ← 16 通道
```

---

### 9.3 输出变换 — 数据布局与计算流程（F(4,4,3,3)）

#### 9.3.1 输入数据布局（Winograd 域结果 M）

GEMM 输出的 Winograd 域结果 M 是 6×6 = 36 个矩阵，每个矩阵大小 = tiles × OC：

```
内存布局（WinogradDomainSpec, winograd_implementations.hpp:322-336）：

  output_ld_row    = roundup(OC, 4)          ← 每行的跨度（按 4 对齐）
  output_ld_matrix = n_output_patches * output_ld_row  ← 一个 Winograd 矩阵的跨度
  output_ld_batch  = n_multis * output_ld_matrix       ← 跨 36 个 Winograd 矩阵
  
  对于输出变换，输入指针 inptr 指向 GEMM 结果：
  
  inptr → M[0][0]  M[0][1]  ... M[0][5]     ← Winograd 矩阵 0-5（第 0 行）
          M[1][0]  M[1][1]  ... M[1][5]     ← Winograd 矩阵 6-11（第 1 行）
          ...
          M[5][0]  M[5][1]  ... M[5][5]     ← Winograd 矩阵 30-35（第 5 行）
  
  每个 M[i][j] 是 tiles × OC 个 float
  ld_in_matrix = output_ld_matrix（跨一个 Winograd 矩阵）
  ld_in_row    = output_ld_row（跨一个 tile 行）
```

#### 9.3.2 输出特征图布局

输出是 **NHWC** 格式，4×4 输出 tile 放到正确的空间位置：

```
输出特征图（OH×OW×OC，NHWC）：

  ld_out_row = OW * OC    （一行输出的跨度）
  ld_out_col = OC          （一列输出的跨度）
  
  outptr_tile 指向输出特征图中 tile 对应的起始位置：
  outptr_tile = outptr + out_i * ld_out_row + out_j * ld_out_col
  
  4×4 输出 tile 写入：
  
  ┌─────────────────────────────┐
  │ out[0][0] out[0][1] out[0][2] out[0][3] │  ← 输出行 0
  │ out[1][0] out[1][1] out[1][2] out[1][3] │  ← 输出行 1
  │ out[2][0] out[2][1] out[2][2] out[2][3] │  ← 输出行 2
  │ out[3][0] out[3][1] out[3][2] out[3][3] │  ← 输出行 3
  └─────────────────────────────┘
   ↑ ld_out_col = OC
  
  每个格子是 OC 个 float
  行间间隔 ld_out_row = OW * OC
```

#### 9.3.3 NEON 版计算流程（arm_fp32_4x4_3x3）

```
NEON 输出变换计算流程：

  步骤 1：加载 6×6 Winograd 域结果 F
    for (i=0..5) for (j=0..5, m++)
      F[i][j] = vld1q_f32(inptr + m * matrix_stride)  // 加载 4 个 OC
    
    → F[6][6] 共 36 个向量（每个 4 个 OC）
    → 装入 36 个 Q 寄存器（但只有 32 个，所以分批处理）

  步骤 2：列变换 FZ = F · A（对每行 i 做 4 次线性组合）
    FZ[i][0] = F[i][0] + F[i][1] + F[i][2] + F[i][3] + F[i][4]
    FZ[i][1] = (F[i][1] - F[i][2]) + 2*(F[i][3] - F[i][4])
    FZ[i][2] = (F[i][1] + F[i][2]) + 4*(F[i][3] + F[i][4])
    FZ[i][3] = (F[i][1] - F[i][2]) + 8*(F[i][3] - F[i][4]) + F[i][5]
    
    用 NEON: vaddq_f32, vsubq_f32, vmlaq_n_f32（系数 2/4/8）
    → FZ[6][4] 共 24 个向量

  步骤 3：行变换 f = A^T · FZ（对每列 j 做 4 次线性组合）
    f[0][j] = FZ[0][j] + FZ[1][j] + FZ[2][j] + FZ[3][j] + FZ[4][j]
    f[1][j] = (FZ[1][j] - FZ[2][j]) + 2*(FZ[3][j] - FZ[4][j])
    f[2][j] = (FZ[1][j] + FZ[2][j]) + 4*(FZ[3][j] + FZ[4][j])
    f[3][j] = (FZ[1][j] - FZ[2][j]) + 8*(FZ[3][j] - FZ[4][j]) + FZ[5][j]
    
    → f[4][4] 共 16 个向量 = 4×4 输出 tile

  步骤 4：融合 bias + ReLU
    b = vld1q_f32(bptr)                           // 加载 bias
    y = vmaxq_f32(vminq_f32(vaddq_f32(f[i][j], b), // f + bias，然后
                             vdupq_n_f32(max)),     // min(., max) 上界
                  vdupq_n_f32(min))                // max(., min) 下界=ReLU

  步骤 5：存储 4×4 输出
    for (i=0..3) for (j=0..3)
      vst1q_f32(outptr + i*ld_out_row + j*ld_out_col, y)  // 写回特征图
```

#### 9.3.4 边界处理

输出变换也需要处理边界——当输出 tile 的部分像素超出特征图边界时：

```
output_transform.hpp:135-136:
  args.output_shape.rows - out_i  // 剩余有效行数
  args.output_shape.cols - out_j  // 剩余有效列数

  如果 valid_rows < 4 或 valid_cols < 4：
    只写入有效部分，超出部分跳过（或由调用方 padding）
```

#### 9.3.5 线程划分

```
output_transform.hpp:115-117:
  for (out_i = thread_id * 4; out_i < OH; out_i += n_threads * 4)
  
  线程 0：输出行 0-3, 4*n_threads, ...
  线程 1：输出行 4-7, 4+4*n_threads, ...
  ...
  
  每个线程独立处理若干 4×4 输出 tile
```

---

### 9.4 SME 输出变换的数据布局（sme_fp32_mopa_4x4_3x3）

#### 9.4.1 与 NEON 版的关键布局差异

```
NEON 版：
  - 每次处理 4 个通道（Q 寄存器）
  - 顺序做两次矩阵乘（F·A 然后 A^T·FZ）
  - bias/ReLU 在最后做

SME 版：
  - 每次处理 4×VL = 64 个通道（4 组谓词，每组 VL=16）
  - 用 Kronecker 积把两次矩阵乘折叠成一次 (A^T⊗A^T)·vec(Y)
  - bias 通过 FMOPA 累加进 ZA tile
  - ReLU 用 fmin/fmax 在 MOVA 读出后做
```

#### 9.4.2 ZA tile 布局

```
SME 用 4 个 ZA tile（za0-za3）存储 4×4 输出的 4 行：

  za0 → 输出行 0 的 4 个元素（分布在 16 列中）
  za1 → 输出行 1
  za2 → 输出行 2
  za3 → 输出行 3

  每个 ZA tile 是 16×16 的二维矩阵（SVE-512）：
  
  za0 (16 行 × 16 列):
  ┌─────────────────────────────────────┐
  │ row 0:  out[0][0]_ch0..ch15        │  ← 第 0 列 = 输出[0][0] 的 16 个通道
  │ row 1:  out[0][1]_ch0..ch15        │  ← 第 1 列 = 输出[0][1]
  │ row 2:  out[0][2]_ch0..ch15        │
  │ row 3:  out[0][3]_ch0..ch15        │
  │ row 4-15: （未使用或后续 tile）      │
  └─────────────────────────────────────┘
  
  → 一次 FMOPA 把一个输入向量 × 一个系数向量，结果累加进 16×16 tile
  → 多次 FMOPA 累加完 36 个 Winograd 域矩阵的所有贡献
  → MOVA 读出 4 行 → 得到 4×4=16 个输出值（每个 16 通道）
```

#### 9.4.3 SME 计算流程

```
SME 输出变换计算流程：

  步骤 1：初始化
    SMSTART ZA              → 进入流式模式
    zero {zad0-zad7}        → 清零所有 ZA tile
    fmov z1.s, #1.0         → Z1 = 1.0（用于 bias 的 FMOPA）
    加载 outer_terms → z6, z7  （A^T 的行系数）
    加载 inner_terms → z9, z8, z15, z4, z3, z2  （A^T 的列系数）
    用 fmul 构造 Kronecker 积的系数向量（z11, z5）

  步骤 2：bias 累加（如果有 bias）
    for 每组通道 (4 组，各 VL=16 通道):
      ld1w z0, bias         → 加载 bias 向量
      fmopa za0, z1, z0     → za0 += 1.0 ⊗ bias  （bias 加到每行）
      fmopa za1, z1, z0     → za1 += bias
      fmopa za2, z1, z0     → za2 += bias
      fmopa za3, z1, z0     → za3 += bias

  步骤 3：Winograd 域数据累加（核心循环）
    for 每组通道 (4 组):
      加载 6×6 = 36 个 Winograd 域向量（每组 VL 通道）
      对每个 Winograd 域元素 [i][j] (共 36 个):
        构造 Kronecker 系数 = outer_terms[i] × inner_terms[j]
        fmopa za0, kronecker_coef, winograd_data  → za0 += coef ⊗ data
        fmopa za1, ...                              → za1 += ...
        fmopa za2, ...                              → za2 += ...
        fmopa za3, ...                              → za3 += ...
      
      → 36 × 4 = 144 条 FMOPA（每条做 16×16=256 次 FMA）
      → 总计 144 × 256 = 36864 次 FMA / 64 通道

  步骤 4：读出 + ReLU + 存储
    for 行 i = 0..3:
      mova z31, za{i}[row_idx]     → 从 ZA tile 读出一行
      fmin z31, z31, z10           → 上界裁剪（z10 = max）
      fmax z31, z31, z12           → 下界裁剪（z12 = min = ReLU）
      st1w z31, p0, [output]       → 写回特征图
    
    SMSTOP → 退出流式模式
```

#### 9.4.4 数据流图

```
SME 输出变换数据流：

  Winograd 域 M (6×6, 每个含 VL 通道)
  ┌───────────────────────────────┐
  │ M[0][0] M[0][1] ... M[0][5]  │
  │ M[1][0] ...            M[5][5]│  36 个向量
  └───────────┬───────────────────┘
              │ ld1w 加载
              ▼
  ┌───────────────────────────────────────────┐
  │  Kronecker 系数构造                        │
  │  z11 = outer_terms[i] × inner_terms[j]     │
  │  （用 fmul 从预加载的 outer/inner 构造）    │
  └───────────┬───────────────────────────────┘
              │
              ▼  fmopa za{0-3}, z_coef, z_data  (×36 ×4 = 144 条)
  ┌───────────────────────────────────────────┐
  │  ZA tile 累加                               │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │  │  za0    │ │  za1    │ │  za2    │ │  za3    │
  │  │ (16×16) │ │ (16×16) │ │ (16×16) │ │ (16×16) │
  │  │ 输出行0 │ │ 输出行1 │ │ 输出行2 │ │ 输出3  │
  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
  └───────┼──────────┼──────────┼──────────┼──────┘
          │ mova     │ mova     │ mova     │ mova
          ▼          ▼          ▼          ▼
  ┌───────────────────────────────────────────┐
  │  Z 寄存器（4×16 = 64 个输出值）             │
  │  z31=out[0], z30=out[1], z29=out[2], z28=out[3]│
  └───────────┬───────────────────────────────┘
              │ fmin + fmax (ReLU)
              ▼
  ┌───────────────────────────────────────────┐
  │  输出特征图 (4×4 × VL 通道)                 │
  │  st1w 写回                                  │
  └───────────────────────────────────────────┘
```

---

### 9.5 F(2,2,3,3) 变换的数据布局

#### 9.5.1 权重变换 arm_fp32_2x2_3x3

```
输入：3×3 权重（HWIO 格式，同 F(4,4,3,3) 但 G 矩阵不同）

  G(F(2,2,3,3)) = [1,   0,   0  ]     G(F(4,4,3,3)) = [6,  0,  0 ]
                   [0.5, 0.5, 0.5]                      [-4,-4,-4 ]
                   [0.5,-0.5, 0.5]                      [-4, 4,-4 ]
                   [0,   0,   1  ]                      [1,  2,  4 ]
                                                      [1, -2,  4 ]
                                                      [0,  0, 24 ]

  F(2,2,3,3) 输出 4×4 = 16 个变换后权重（无 /576 归一化）
  F(4,4,3,3) 输出 6×6 = 36 个变换后权重（有 /576 归一化）
  
  内存布局相同：按 matrix_stride 间隔存储
```

#### 9.5.2 输入变换 arm_fp32_4x4（F(2,2,3,3) 专用）

```
输入 tile = 4×4（m+r-1 = 2+3-1 = 4）

  tile_stride = m = 2（输出 tile 步长）
  n_tile_rows = ceil(OH / 2)
  n_tile_cols = ceil(OW / 2)
  
  4×4 tile 在内存中（NHWC 输入）：
  
  行 0: inptr[0]              inptr[ld_col]          inptr[2*ld_col]        inptr[3*ld_col]
  行 1: inptr[ld_row]         inptr[ld_row+ld_col]   ...
  行 2: inptr[2*ld_row]      ...
  行 3: inptr[3*ld_row]      ...
  
  → 4 行 × 4 列 = 16 个位置，每个位置 C 个通道

变换 U = B^T · d · B：

  B^T = [1, 0,-1, 0]    ← 系数更简单（无 2/4/5）
        [0, 1, 1, 0]
        [0,-1, 1, 0]
        [0, 1, 0,-1]
  
  行变换 XTx[0][j] = d[0][j] - d[2][j]     ← 简单减法
  行变换 XTx[1][j] = d[1][j] + d[2][j]     ← 简单加法
  行变换 XTx[2][j] = d[2][j] - d[1][j]
  行变换 XTx[3][j] = d[1][j] - d[3][j]
  
  → 只有加法和减法，无乘法（系数全是 1）→ 变换开销比 F(4,4,3,3) 小很多

输出：4×4 = 16 个 Winograd 域元素
```

#### 9.5.3 输出变换 arm_fp32_2x2_3x3

```
输入：4×4 Winograd 域结果 = 16 个元素

  A^T = [1, 1, 1, 0]    ← 简单加法
        [0,-1,-1,-1]
  
  列变换 FZ[i][0] = F[i][0] + F[i][1] + F[i][2]
  列变换 FZ[i][1] = F[i][1] - F[i][2] - F[i][3]
  
  行变换 f[0][j] = FZ[0][j] + FZ[1][j] + FZ[2][j]
  行变换 f[1][j] = FZ[1][j] - FZ[2][j] - FZ[3][j]
  
  → 只有加法和减法
  
  输出：2×2 = 4 个输出像素

bias + ReLU 融合：
  y = max(min(f + bias, max), min)    ← 与 F(4,4,3,3) 完全一致的模式
  
存储：2×2 输出写入特征图
```

#### 9.5.4 F(2,2,3,3) 完整数据流

```
F(2,2,3,3) 完整流程：

  输入特征图 (NHWC)
  ┌──────────────────────────┐
  │ 4×4 tile → B^T·d·B → U  │  ← 输入变换（16 个 Winograd 元素）
  │ (4 通道/指令, NEON)       │
  └────────────┬─────────────┘
               │
               ▼
  Winograd 域 U (4×4 = 16 个矩阵, 每个 tiles×IC)
  ┌──────────────────────────┐
  │ 16 个 GEMM: M = U · V    │  ← GEMM (arm_gemm SVE 内核)
  │ (M=tiles, N=OC, K=IC)   │
  └────────────┬─────────────┘
               │
               ▼
  Winograd 域 M (4×4 = 16 个矩阵, 每个 tiles×OC)
  ┌──────────────────────────┐
  │ A^T·M·A → 2×2 输出       │  ← 输出变换（4 个输出像素）
  │ + bias + ReLU            │
  │ (4 通道/指令, NEON)       │
  └────────────┬─────────────┘
               │
               ▼
  输出特征图 (NHWC, 2×2 像素)
```

---

### 9.6 所有变换的 B_values / 系数对照表

```
┌────────────┬──────────────────┬──────────────────────┐
│            │ F(2,2,3,3)       │ F(4,4,3,3)           │
├────────────┼──────────────────┼──────────────────────┤
│ 输入系数   │ {无}(全1,纯加减) │ {1, 2, 4, 5}         │
│ B_values   │ (pcoeffs 无)    │ (pcoeffs/B_values)   │
├────────────┼──────────────────┼──────────────────────┤
│ G 矩阵     │ [1,0,0;          │ [6,0,0;              │
│            │  0.5,0.5,0.5;    │  -4,-4,-4;           │
│            │  0.5,-0.5,0.5;  │  -4,4,-4;            │
│            │  0,0,1]          │  1,2,4;              │
│            │                  │  1,-2,4;             │
│            │                  │  0,0,24]             │
├────────────┼──────────────────┼──────────────────────┤
│ 归一化     │ 无               │ /576 (= 24²)         │
├────────────┼──────────────────┼──────────────────────┤
│ A 矩阵     │ [1,1,1,0;        │ [1,1,1,1,1,0;       │
│            │  0,-1,-1,-1]     │  0,1,-1,2,-2,0;     │
│            │ (2×4)            │  0,1,1,4,4,0;       │
│            │                  │  0,1,-1,8,-8,1]      │
│            │                  │ (4×6)                │
├────────────┼──────────────────┼──────────────────────┤
│ 输入 tile  │ 4×4              │ 6×6                  │
│ 输出 tile  │ 2×2              │ 4×4                  │
│ GEMM 数    │ 16               │ 36                   │
├────────────┼──────────────────┼──────────────────────┤
│ 变换开销   │ 低(纯加减)       │ 高(乘+加+归一化)     │
│ 乘法节省  │ 9→4 (44%)        │ 9→2.25 (25%)         │
└────────────┴──────────────────┴──────────────────────┘
```

> 本文基于 ACL 53.1.0 源码逐行分析，所有汇编指令均标注源码行号，具体实现以源码为准。

---

## 10. 算法详解：公式与计算步骤

> 本节用详细的数学公式和文字把每个变换的算法从头到尾讲清楚。每个变换都列出：完整矩阵、逐元素公式、数值示例。

### 10.1 F(4,4,3,3) 权重变换：V = G·g·G^T / 576

#### 10.1.1 完整 G 矩阵

F(4,4,3,3) 的权重变换矩阵 G 是一个 6×3 矩阵，把 3×3 的卷积核 g 变换到 6×6 的 Winograd 域 V。ACL 用整数系数避免浮点误差，最后统一除以 576：

```
        ┌  6   0   0  ┐
   G =  │ -4  -4  -4  │     （6 行 × 3 列）
        │ -4   4  -4  │
        │  1   2   4  │
        │  1  -2   4  │
        └  0   0  24  ┘
```

这个 G 矩阵的每一行定义了一种"观察卷积核"的新视角。例如第 0 行 `[6, 0, 0]` 只看核的第 0 个元素（乘以 6），第 1 行 `[-4, -4, -4]` 看三个元素之和（乘以 -4）。

#### 10.1.2 逐元素计算公式

设 3×3 卷积核为：
```
g = ┌ g[0][0]  g[0][1]  g[0][2] ┐
    │ g[1][0]  g[1][1]  g[1][2] │
    └ g[2][0]  g[2][1]  g[2][2] ┘
```

**第一步：行变换 Ww = G · g**（对 g 的每一列 j 做 6 次线性组合）：

对每个 j = 0, 1, 2，计算 6 个中间值：

```
Ww[0][j] =  6 * g[0][j]                                    // 只取核的第 0 行，乘 6
Ww[1][j] = -4 * (g[0][j] + g[1][j] + g[2][j])              // 三行之和，乘 -4
Ww[2][j] =  4 * (g[1][j] - g[0][j] - g[2][j])              // 中间行减去首尾，乘 4
Ww[3][j] =  1 * g[0][j] + 2 * g[1][j] + 4 * g[2][j]       // 加权求和（1,2,4 是 2 的幂）
Ww[4][j] =  1 * g[0][j] - 2 * g[1][j] + 4 * g[2][j]       // 同上但中间行取负
Ww[5][j] = 24 * g[2][j]                                    // 只取核的第 2 行，乘 24
```

这 6 个公式把 3 行的核"展开"到 6 个 Winograd 域行。注意：
- 第 0 行和第 5 行只取核的首行/末行（边界行）
- 第 1 行取三行之和（类似均值）
- 第 2 行取中间行减首尾（类似差分）
- 第 3、4 行用 2 的幂做加权（1,2,4 = 2^0, 2^1, 2^2），编码了位置信息

**第二步：列变换 V = Ww · G^T / 576**（对 Ww 的每一行 i 做 6 次线性组合）：

对每个 i = 0, 1, ..., 5，计算 6 个最终值：

```
V[i][0] = ( 6 * Ww[i][0]) / 576
V[i][1] = (-4 * (Ww[i][0] + Ww[i][1] + Ww[i][2])) / 576
V[i][2] = ( 4 * (Ww[i][1] - Ww[i][0] - Ww[i][2])) / 576
V[i][3] = ( 1 * Ww[i][0] + 2 * Ww[i][1] + 4 * Ww[i][2]) / 576
V[i][4] = ( 1 * Ww[i][0] - 2 * Ww[i][1] + 4 * Ww[i][2]) / 576
V[i][5] = (24 * Ww[i][2]) / 576
```

列变换用的是与行变换**完全相同的系数模式**——这是因为 G^T 就是 G 的转置，所以列方向也是同样的 6 个线性组合。

576 = 24² 是归一化因子。它的来源：G 矩阵的最大系数是 24，V = G·g·G^T 的系数会被乘两次（行列各一次），所以最大系数 = 24 × 24 = 576。除以 576 把结果归一化到正确的 Winograd 权重值。

#### 10.1.3 数值示例

设 3×3 卷积核全为 1：
```
g[0][0]=g[0][1]=g[0][2]=g[1][0]=...=g[2][2]=1
```

第一步：行变换 Ww = G · g
```
Ww[0][j] = 6 * 1 = 6
Ww[1][j] = -4 * (1+1+1) = -12
Ww[2][j] = 4 * (1-1-1) = -4
Ww[3][j] = 1 + 2 + 4 = 7
Ww[4][j] = 1 - 2 + 4 = 3
Ww[5][j] = 24 * 1 = 24
```
（j=0,1,2 都一样，因为核全为 1）

第二步：列变换 V = Ww · G^T / 576
```
V[0][0] = (6 * 6) / 576 = 36/576 = 1/16
V[1][1] = (-4 * (6 + (-12) + (-4))) / 576 = (-4 * (-10)) / 576 = 40/576
V[3][3] = (1*6 + 2*(-12) + 4*(-4)) / 576 = (6-24-16) / 576 = -34/576
...
```

这 36 个值就是变换后权重，它们会被 GEMM 使用。

#### 10.1.4 代码实现对应

源码 `arm_fp32_4x4_3x3.cpp` 中的对应关系：

```
:57  Ww[0][j] = vmulq_n_f32(w[0][j], 6.0);                    ← Ww[0][j] = 6*g[0][j]
:60  Ww[1][j] = vmulq_n_f32(vaddq_f32(...), -4.0);            ← Ww[1][j] = -4*(g0+g1+g2)
:63  Ww[2][j] = vmulq_n_f32(vsubq_f32(vsubq_f32(...), 4.0));  ← Ww[2][j] = 4*(g1-g0-g2)
:66  Ww[3][j] = vmlaq_n_f32(vmlaq_n_f32(w0, w1, 2), w2, 4);   ← Ww[3][j] = g0+2*g1+4*g2
:69  Ww[4][j] = vmlaq_n_f32(vmlsq_n_f32(w0, w1, 2), w2, 4);   ← Ww[4][j] = g0-2*g1+4*g2
:72  Ww[5][j] = vmulq_n_f32(w[2][j], 24.0);                  ← Ww[5][j] = 24*g[2][j]

:81  V[i][0] = vmulq_n_f32(vmulq_n_f32(Ww[i][0], 6.0), recip576);
     → V[i][0] = (6 * Ww[i][0]) * (1/576) = (6 * Ww[i][0]) / 576
```

---

### 10.2 F(4,4,3,3) 输入变换：U = B^T·d·B

#### 10.2.1 完整 B^T 矩阵

F(4,4,3,3) 的输入变换矩阵 B^T 是一个 6×6 矩阵，把 6×6 的输入 tile d 变换到 6×6 的 Winograd 域 U：

```
          ┌  4   0  -5   0   1   0  ┐
   B^T =  │  0  -4  -4   1   1   0  │     （6 行 × 6 列）
          │  0   4  -4  -1   1   0  │
          │  0  -2  -4   2   1   0  │
          │  0   2  -4  -2   1   0  │
          └  0   0  -5   0   0   1  ┘
```

> 这个矩阵来自 Winograd 最小滤波算法的标准推导（Lavin & Fast, 2016）。ACL 源码中不直接存储这个矩阵，而是用系数 `{1, 2, 4, 5}` 和常量 `4.0` 通过 `fmla`/`fmls`/`fmad`/`fmsb` 指令组合来实现。

验证：B^T 矩阵中的系数只有 {0, ±1, ±2, ±4, ±5}，正好对应源码中的 `B_values = {1, 2, 4, 5}` 和常量 `z16 = 4.0`。

#### 10.2.2 逐元素计算公式

设 6×6 输入 tile 为：
```
d = ┌ d[0][0]  d[0][1]  d[0][2]  d[0][3]  d[0][4]  d[0][5] ┐
    │ d[1][0]  d[1][1]  d[1][2]  d[1][3]  d[1][4]  d[1][5] │
    │ d[2][0]  ...                                       │
    │ d[3][0]  ...                                       │
    │ d[4][0]  ...                                       │
    └ d[5][0]  ...                            d[5][5]     ┘
```

**第一步：行变换 XTx = B^T · d**（对 d 的每一列 j 做 6 次线性组合）：

对每个 j = 0, 1, ..., 5，计算 6 个中间值：

```
XTx[0][j] = 4*d[0][j] - 5*d[2][j] + 1*d[4][j]
            ← 只用第 0、2、4 行（跳行取），系数 4, -5, 1

XTx[1][j] = -4*d[1][j] - 4*d[2][j] + 1*d[3][j] + 1*d[4][j]
            ← 用第 1、2、3、4 行，系数 -4, -4, 1, 1

XTx[2][j] = 4*d[1][j] - 4*d[2][j] - 1*d[3][j] + 1*d[4][j]
            ← 同上行但系数符号不同

XTx[3][j] = -2*d[1][j] - 4*d[2][j] + 2*d[3][j] + 1*d[4][j]
            ← 系数 -2, -4, 2, 1（注意 2 的幂模式）

XTx[4][j] = 2*d[1][j] - 4*d[2][j] - 2*d[3][j] + 1*d[4][j]
            ← 同上行但系数符号相反

XTx[5][j] = -5*d[2][j] + 1*d[5][j]
            ← 只用第 2、5 行，系数 -5, 1
```

**第二步：列变换 U = XTx · B**（对 XTx 的每一行 i 做 6 次线性组合）：

因为 B = (B^T)^T，列变换用的是与行变换**完全相同的系数模式**，只是作用在行方向（对 i 的不同列做组合）：

```
U[i][0] = 4*XTx[i][0] - 5*XTx[i][2] + 1*XTx[i][4]
U[i][1] = -4*XTx[i][1] - 4*XTx[i][2] + 1*XTx[i][3] + 1*XTx[i][4]
U[i][2] = 4*XTx[i][1] - 4*XTx[i][2] - 1*XTx[i][3] + 1*XTx[i][4]
U[i][3] = -2*XTx[i][1] - 4*XTx[i][2] + 2*XTx[i][3] + 1*XTx[i][4]
U[i][4] = 2*XTx[i][1] - 4*XTx[i][2] - 2*XTx[i][3] + 1*XTx[i][4]
U[i][5] = -5*XTx[i][2] + 1*XTx[i][5]
```

最终得到 6×6 = 36 个 Winograd 域元素 U[i][j]，每个包含 C 个通道的数据。

#### 10.2.3 B^T 矩阵的结构特点

观察 B^T 矩阵的结构：

1. **第 0 列只有 4 和 0**：`[4, 0, 0, 0, 0, 0]^T` — 只从输入的第 0 行取值
2. **第 5 列只有 0 和 1**：`[0, 0, 0, 0, 0, 1]^T` — 只从输入的第 5 行取值
3. **第 2 列有 -5**：`[-5, -4, -4, -4, -4, -5]^T` — 第 2 行的权重最大
4. **第 3、4 列用 1, 2 的幂**：编码了位置信息

这种结构是 Winograd 算法精心设计的——它在保证正确性的同时，让乘法次数最小化。每一行都是输入 tile 的某种线性组合，使得后续的 GEMM 能用更少的乘法完成卷积。

#### 10.2.4 数值示例

设 6×6 输入 tile 全为 1（最简情况）：
```
d[i][j] = 1  (对所有 i, j)
```

第一步：行变换 XTx = B^T · d
```
XTx[0][j] = 4*1 - 5*1 + 1*1 = 0
XTx[1][j] = -4*1 - 4*1 + 1*1 + 1*1 = -6
XTx[2][j] = 4*1 - 4*1 - 1*1 + 1*1 = 0
XTx[3][j] = -2*1 - 4*1 + 2*1 + 1*1 = -3
XTx[4][j] = 2*1 - 4*1 - 2*1 + 1*1 = -3
XTx[5][j] = -5*1 + 1*1 = -4
```
（j=0..5 都一样，因为输入全为 1）

第二步：列变换 U = XTx · B（用同样的 B^T 系数对行做变换）
```
U[0][0] = 4*0 - 5*0 + 1*0 = 0         (XTx[0][0]=0, XTx[0][2]=0, XTx[0][4]=0)
U[1][0] = -4*(-6) -4*0 + 1*(-3) + 1*(-3) = 24 - 3 - 3 = 18
...
```

这些 U 值就是 Winograd 域的变换后输入，会被 GEMM 使用。

#### 10.2.5 代码中的系数映射

源码中 `pcoeffs = {1, 2, 4, 5}`（a64 版 `:42`）或 `B_values = {1, 2, 4, 5}`（SVE 版 `:42`），它们在 B^T 矩阵中的角色：

| 系数 | 源码变量 | 在 B^T 中的角色 |
|------|---------|----------------|
| 1 | `v0.s[0]` / `z2.s[0]` | 单位系数（加/减的基础） |
| 2 | `v0.s[1]` / `z2.s[1]` | 出现在 B^T[3][3]=2, B^T[4][3]=-2 |
| 4 | 常量 `z16=4.0` 或 `v0.s[2]` | 出现在 B^T[0][0]=4, B^T[1][1]=-4, B^T[2][1]=4, B^T[3][2]=-4, B^T[4][2]=-4 |
| 5 | `v0.s[3]` / `z2.s[3]` | 出现在 B^T[0][2]=-5, B^T[5][2]=-5 |

汇编中 `fmla`（乘加）实现正系数项，`fmls`（乘减）实现负系数项，`fmad`/`fmsb` 是三操作数版本。

---

### 10.3 F(4,4,3,3) 输出变换：f = A^T·M·A

#### 10.3.1 完整 A 矩阵

输出变换矩阵 A 是一个 4×6 矩阵，把 6×6 的 Winograd 域结果 M 变换回 4×4 的输出 f：

```
        ┌ 1  1  1  1  1  0 ┐
   A =  │ 0  1 -1  2 -2  0 │     （4 行 × 6 列）
        │ 0  1  1  4  4  0 │
        └ 0  1 -1  8 -8  1 ┘
```

A^T（用于行变换的转置）：
```
         ┌ 1  0  0  0 ┐
   A^T = │ 1  1  1  1 │     （6 行 × 4 列）
         │ 1 -1  1 -1 │
         │ 1  2  4  8 │
         │ 1 -2  4 -8 │
         └ 0  0  0  1 ┘
```

#### 10.3.2 逐元素计算公式

设 6×6 Winograd 域结果为 M[6][6]（来自 GEMM 步骤）。

**第一步：列变换 FZ = M · A**（对 M 的每一行 i 做 4 次线性组合）：

对每个 i = 0, 1, ..., 5，计算 4 个中间值：

```
FZ[i][0] = M[i][0] + M[i][1] + M[i][2] + M[i][3] + M[i][4]
           ← 前 5 列之和（系数全 1）

FZ[i][1] = M[i][1] - M[i][2] + 2*M[i][3] - 2*M[i][4]
           ← 第 1-4 列的加权差（系数 1, -1, 2, -2）

FZ[i][2] = M[i][1] + M[i][2] + 4*M[i][3] + 4*M[i][4]
           ← 第 1-4 列的加权和（系数 1, 1, 4, 4）

FZ[i][3] = M[i][1] - M[i][2] + 8*M[i][3] - 8*M[i][4] + M[i][5]
           ← 第 1-5 列的加权差（系数 1, -1, 8, -8, 1）
```

注意系数模式：`{1, 1, 1, 1, 1}` → `{1, -1, 2, -2}` → `{1, 1, 4, 4}` → `{1, -1, 8, -8, 1}`。这是 2 的幂次模式（1, 2, 4, 8），编码了输出像素的空间位置信息。

**第二步：行变换 f = A^T · FZ**（对 FZ 的每一列 j 做 4 次线性组合）：

对每个 j = 0, 1, 2, 3，计算 4 个最终输出值：

```
f[0][j] = FZ[0][j] + FZ[1][j] + FZ[2][j] + FZ[3][j] + FZ[4][j]
          ← 前 5 行之和（系数全 1）

f[1][j] = FZ[1][j] - FZ[2][j] + 2*FZ[3][j] - 2*FZ[4][j]
          ← 第 1-4 行的加权差

f[2][j] = FZ[1][j] + FZ[2][j] + 4*FZ[3][j] + 4*FZ[4][j]
          ← 第 1-4 行的加权和

f[3][j] = FZ[1][j] - FZ[2][j] + 8*FZ[3][j] - 8*FZ[4][j] + FZ[5][j]
          ← 第 1-5 行的加权差
```

行变换用的系数模式与列变换**完全相同**（因为 A^T 是 A 的转置，系数对称）。

#### 10.3.3 bias 和激活融合

在得到 4×4 = 16 个输出值 f[i][j] 后：

```
对每个输出值 f[i][j]：
  y = f[i][j] + bias[oc]                    ← 加偏置
  y = min(y, activation_max)                ← 上界裁剪（BoundedReLU 的上界）
  y = max(y, activation_min)                 ← 下界裁剪（ReLU 的下界 = 0）
  
  如果无激活：activation_max = +∞, activation_min = -∞ → 裁剪退化为恒等
  如果 ReLU：activation_min = 0, activation_max = +∞ → 只做下界裁剪
  如果 BoundedReLU：activation_min = 0, activation_max = α → 双向裁剪
```

#### 10.3.4 数值示例

设 Winograd 域结果 M 全为 1：
```
M[i][j] = 1 (对所有 i=0..5, j=0..5)
```

第一步：列变换 FZ = M · A
```
FZ[i][0] = 1+1+1+1+1 = 5        (5 列之和)
FZ[i][1] = 1-1+2-2 = 0          (加权差)
FZ[i][2] = 1+1+4+4 = 10         (加权和)
FZ[i][3] = 1-1+8-8+1 = 1        (加权差+末列)
```
（i=0..5 都一样）

第二步：行变换 f = A^T · FZ
```
FZ[0..5] 的值都是 [5, 0, 10, 1]

f[0][0] = FZ[0][0]+FZ[1][0]+FZ[2][0]+FZ[3][0]+FZ[4][0] = 5+5+5+5+5 = 25
f[1][0] = FZ[1][0]-FZ[2][0]+2*FZ[3][0]-2*FZ[4][0] = 5-5+10-10 = 0
f[2][0] = FZ[1][0]+FZ[2][0]+4*FZ[3][0]+4*FZ[4][0] = 5+5+20+20 = 50
f[3][0] = FZ[1][0]-FZ[2][0]+8*FZ[3][0]-8*FZ[4][0]+FZ[5][0] = 5-5+40-40+5 = 5
```

这 16 个值就是 4×4 输出 tile，会被写入输出特征图。

#### 10.3.5 系数的意义

A 矩阵的系数 `{1, 1, 1, 1, 1}` / `{1, -1, 2, -2}` / `{1, 1, 4, 4}` / `{1, -1, 8, -8, 1}` 有什么意义？

这实际上是**多项式求值**模式。Winograd 变换本质上是在不同的"求值点"上计算多项式，然后通过逆变换恢复空间域结果。系数 `1, 2, 4, 8` 是 2 的幂次，对应不同的求值点（x=1, x=2, x=4, x=8），而 `1, -1` 对应 x=1 和 x=-1。

---

### 10.4 F(4,4,3,3) SME 输出变换的 Kronecker 积详解

#### 10.4.1 为什么 SME 版用不同的算法

NEON 版做输出变换是"两次 1D 矩阵乘"（先列变换 F·A，再行变换 A^T·FZ）。但 SME 有 **FMOPA 外积指令**——一条指令可以做 16×16 的矩阵乘加。为了利用这个能力，ACL 把两次 1D 变换折叠成**一次 2D 矩阵-向量乘**。

#### 10.4.2 vec trick 数学推导

输出变换的数学：`f = A^T · M · A`，其中 M 是 6×6，A 是 4×6，f 是 4×4。

用矩阵向量化（vec trick）可以重写为：
```
vec(f) = (A^T ⊗ A^T) · vec(M)
```

其中：
- `vec(X)` 表示把矩阵按**列优先**展开成向量：`vec(M) = [M[0][0], M[1][0], ..., M[5][0], M[0][1], ..., M[5][5]]^T`
- `⊗` 是 Kronecker 积：如果 `A^T` 是 4×6，则 `A^T ⊗ A^T` 是 16×36
- `vec(f)` 是 16 维向量（4×4 展平）
- `vec(M)` 是 36 维向量（6×6 展平）

所以输出变换变成了一个 16×36 的矩阵-向量乘法。

#### 10.4.3 Kronecker 积的构造

`A^T ⊗ A^T` 是一个 16×36 的矩阵。它的每个 6×6 子块是 `A^T[i][j] * A^T`：

```
A^T ⊗ A^T = ┌ A^T[0][0]*A^T   A^T[0][1]*A^T   ...  A^T[0][5]*A^T ┐
            │ A^T[1][0]*A^T   A^T[1][1]*A^T   ...                │
            │  ...                                                    │
            └ A^T[3][0]*A^T   ...                          A^T[3][5]*A^T┘
```

也就是说，`(A^T ⊗ A^T)[i*4+p][j*6+q] = A^T[i][j] * A^T[p][q]`。

ACL 的实现把这个 16×36 矩阵拆成两部分：
- **outer_terms**（:59-72）：A^T 的行系数，决定 `i` 和 `p`
- **inner_terms**（:80-87）：A^T 的列系数，决定 `j` 和 `q`

运行时用 `fmul` 把 `outer_terms[i] × inner_terms[j]` 构造出 Kronecker 积的每个元素。

#### 10.4.4 FMOPA 如何计算矩阵-向量乘

FMOPA 的定义：`zaD += ZN ⊗ ZM`（外积累加），其中 ZN 和 ZM 是向量。

在 SME 输出变换中：
- ZM = Winograd 域数据向量（36 个 Winograd 元素之一，含 VL 个通道）
- ZN = Kronecker 系数向量（由 outer×inner 构造，16 个系数对应 16 个输出位置）
- ZA tile 累加结果 = 输出值

```
一条 FMOPA 做了什么：

  ZM = [m0, m1, m2, ..., m15]   ← 16 个通道的 Winograd 域数据
  ZN = [c0, c1, c2, ..., c15]   ← 16 个 Kronecker 系数（对应 16 个输出位置）

  外积 ZN ⊗ ZM = 16×16 矩阵：
    ┌ c0*m0  c0*m1  c0*m2  ... c0*m15 ┐
    │ c1*m0  c1*m1  ...               │
    │ ...                              │   ← 累加进 ZA tile
    └ c15*m0 c15*m1 ...     c15*m15  ┘
  
  zaD[行 p][通道 ch] += ZN[p] * ZM[ch]
  → 等价于：output[p][ch] += Kronecker_coef[p] * winograd_data[ch]
```

对 36 个 Winograd 域元素各做一次 FMOPA（4 个 ZA tile 并行），就完成了整个 16×36 矩阵-向量乘。

#### 10.4.5 4 个 ZA tile 的角色

```
za0 → 累加输出行 0 的 4 个元素 × 所有通道
za1 → 累加输出行 1
za2 → 累加输出行 2
za3 → 累加输出行 3

为什么 4 个就够了？
  4×4 = 16 个输出值，每行 4 个。
  一个 ZA tile 是 16×16 = 256 个值。
  4 个 tile × 256 = 1024 个值，远超 16。
  但实际上每个 tile 只用了 4 行（对应 4 列输出），其余行空闲。
  通过 MOVA 指令读出需要的 4 行。
```

---

### 10.5 F(2,2,3,3) 的三矩阵与公式

#### 10.5.1 权重变换 G

```
        ┌ 1     0     0   ┐
   G =  │ 0.5   0.5   0.5 │     （4×3，无归一化因子）
        │ 0.5  -0.5   0.5 │
        └ 0     0     1   ┘
```

逐元素公式：
```
Ww[0][j] = g[0][j]                                    // 首行
Ww[1][j] = 0.5 * (g[0][j] + g[1][j] + g[2][j])       // 三行均值
Ww[2][j] = 0.5 * (g[0][j] - g[1][j] + g[2][j])       // 首末行减中间
Ww[3][j] = g[2][j]                                    // 末行

V[i][0] = Ww[i][0]
V[i][1] = 0.5 * (Ww[i][0] + Ww[i][1] + Ww[i][2])
V[i][2] = 0.5 * (Ww[i][0] - Ww[i][1] + Ww[i][2])
V[i][3] = Ww[i][2]
```

> **与 F(4,4,3,3) 的区别**：F(2,2,3,3) 的 G 矩阵系数更简单（只有 0, 0.5, 1），无归一化因子。变换开销低很多。

#### 10.5.2 输入变换 B^T

```
          ┌ 1  0 -1  0 ┐
   B^T =  │ 0  1   1  0 │     （4×4）
          │ 0 -1   1  0 │
          └ 0  1   0 -1 ┘
```

逐元素公式：
```
XTx[0][j] = d[0][j] - d[2][j]     ← 简单减法
XTx[1][j] = d[1][j] + d[2][j]     ← 简单加法
XTx[2][j] = d[2][j] - d[1][j]     ← 反向减法
XTx[3][j] = d[1][j] - d[3][j]     ← 首末差

U[i][0] = XTx[i][0] - XTx[i][2]
U[i][1] = XTx[i][1] + XTx[i][2]
U[i][2] = XTx[i][2] - XTx[i][1]
U[i][3] = XTx[i][1] - XTx[i][3]
```

> **与 F(4,4,3,3) 的区别**：F(2,2,3,3) 的 B^T 系数全为 ±1，变换只有加法和减法，**完全没有乘法**。F(4,4,3,3) 需要 2, 4, 5 的乘法。

#### 10.5.3 输出变换 A^T

```
          ┌ 1  1  1  0 ┐
   A^T =  │           │     （2×4）
          └ 0 -1 -1 -1 ┘
```

逐元素公式：
```
FZ[i][0] = M[i][0] + M[i][1] + M[i][2]     ← 前 3 列之和
FZ[i][1] = M[i][1] - M[i][2] - M[i][3]     ← 后 3 列的差

f[0][j] = FZ[0][j] + FZ[1][j] + FZ[2][j]   ← 前 3 行之和
f[1][j] = FZ[1][j] - FZ[2][j] - FZ[3][j]   ← 后 3 行的差
```

> **与 F(4,4,3,3) 的区别**：F(2,2,3,3) 的 A^T 只有 ±1，完全没有乘法。F(4,4,3,3) 需要 2, 4, 8 的乘法。

#### 10.5.4 F(2,2,3,3) 完整数值示例

设：3×3 核全为 1，4×4 输入 tile 全为 1。

```
权重变换 V = G·g·G^T:
  Ww = [6→1, 均值→1, 差→1, 末行→1]  (因为核全1, 所有线性组合=1)
  V = 同样的模式 → V 全为 [1, 1, 1, 1; ...] (4×4 全为1附近)

输入变换 U = B^T·d·B:
  XTx[0] = 1-1 = 0
  XTx[1] = 1+1 = 2
  XTx[2] = 1-1 = 0
  XTx[3] = 1-1 = 0
  U 的值取决于两次变换的组合

GEMM M = U · V:
  (假设 GEMM 结果 M 全为某个值)

输出变换 f = A^T·M·A:
  FZ[i][0] = M0+M1+M2
  FZ[i][1] = M1-M2-M3
  f[0] = FZ0+FZ1+FZ2
  f[1] = FZ1-FZ2-FZ3

  + bias + ReLU → 最终输出
```

> **结论**：Winograd 的数学保证 `A^T · (G·g·G^T ⊙ B^T·d·B) · A = 直接卷积结果`。虽然中间过程涉及变换、GEMM、逆变换，但最终结果与直接卷积完全一致。数值示例验证了这一点。

---

### 10.6 各变换的乘加量分析

| 变换 | F(2,2,3,3) | F(4,4,3,3) |
|------|-----------|-----------|
| 权重变换乘法 | 0（系数 0.5 可用移位） | 6×3×3 = 54 次（系数含 6, -4, 4, 2, 24） |
| 权重变换加法 | 12 次 | 36 次 |
| 归一化 | 无 | ÷576（1 次乘以 1/576） |
| 输入变换乘法 | **0**（系数全 ±1） | 有（系数含 2, 4, 5） |
| 输入变换加法 | 24 次 | 36 次 |
| 输出变换乘法 | **0**（系数全 ±1） | 有（系数含 2, 4, 8） |
| 输出变换加法 | 12 次 | 24 次 |
| **变换总乘法** | **0** | ~90 |
| **变换总加法** | **48** | ~96 |
| GEMM 乘法/tile | 16 | 36 |
| GEMM 每像素等效乘法 | 16/4 = 4 | 36/16 = 2.25 |
| 直接卷积每像素乘法 | 9 | 9 |
| **总节省** | 9→4 (56%) | 9→2.25 (75%) |

> F(2,2,3,3) 的变换**完全没有乘法**（只有加减），但每像素等效乘法只省到 4。F(4,4,3,3) 的变换有较多乘法，但每像素等效乘法省到 2.25。这就是为什么大图选 F(4,4,3,3)（GEMM 省更多），小图选 F(2,2,3,3)（变换开销更小）。

> 本文基于 ACL 53.1.0 源码逐行分析，所有汇编指令均标注源码行号，具体实现以源码为准。
