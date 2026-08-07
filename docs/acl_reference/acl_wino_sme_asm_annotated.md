# ACL Winograd SME 内联汇编逐行注释

> 源文件:
> - `ComputeLibrary-53.1.0/.../input_transforms/sme_fp32_mla_6x6.cpp` (363 行)
> - `ComputeLibrary-53.1.0/.../output_transforms/sme_fp32_mopa_4x4_3x3.cpp` (892 行)
>
> SME 是 Arm 的矩阵扩展，核心是 **ZA tile 寄存器**（二维累加器）和 **FMOPA 指令**（外积累加）。

---

## 1. SME 核心概念

### 1.1 ZA Tile 寄存器

ZA 是一个二维寄存器阵列，大小 = VL × VL（SVE-512 时 = 16×16 = 256 个 float）。
划分为 4 个 tile（za0-za3），每个 = VL × VL/4 = 16×4 = 64 个 float。

```
ZA tile 布局 (SVE-512, VL=16):
┌──────────────────────────┐
│ za0 (16×4 = 64 floats)   │  ← tile 0
├──────────────────────────┤
│ za1 (16×4 = 64 floats)   │  ← tile 1
├──────────────────────────┤
│ za2 (16×4 = 64 floats)   │  ← tile 2
├──────────────────────────┤
│ za3 (16×4 = 64 floats)   │  ← tile 3
└──────────────────────────┘
       ↑ 列 0..3 (每个 Z 列)
```

### 1.2 FMOPA 指令

```
FMOPA zaD.s, p/M, p/M, zN.s, zM.s
→ zaD[row][col] += zN[row] * zM[col]  (对所有活跃 row,col)
```

一条 FMOPA = VL × VL = 256 次 FMA（SVE-512 时）。
对比 SVE FMLA = VL 次 FMA（16 次）。
**FMOPA 吞吐量是 FMLA 的 16 倍。**

### 1.3 MOVA 指令

```
MOVA zD.s, p/M, zaNh.s[xR]
→ zD[row] = zaN[row][xR]  (读出 ZA tile 的第 xR 列)
```

### 1.4 流式模式 (Streaming Mode)

SME 指令必须在 SMSTART 和 SMSTOP 之间执行：
```
SMSTART ZA  → 进入流式模式，激活 ZA tile
  ... SME 指令 ...
SMSTOP      → 退出流式模式，ZA tile 内容丢失
```

---

## 2. SME 输入变换 (`sme_fp32_mla_6x6.cpp`)

### 2.1 关键发现：输入变换不用 FMOPA/ZA

**SME 输入变换 = SVE 指令 + SMSTART/SMSTOP 包裹。**

ACL 的 `sme_fp32_mla_6x6.cpp` 中没有任何 FMOPA、MOVA 或 ZA tile 操作。
它只用了 SVE 的 `fmla/fmls/fmad/fmsb/fmul/fadd/fsub` 指令，只是用 SMSTART/SMSTOP 包裹。

**原因**：输入变换是 `B^T · d · B`，本质是矩阵-矩阵乘，不是外积累加。
FMOPA 适合矩阵-矩阵乘（GEMM），不适合变换计算。
SME 在输入变换中的唯一作用是**流式模式 SVE**，可能获得更高的 SVE 吞吐量。

### 2.2 逐行注释

```asm
// === 进入流式模式 ===
".inst 0xd503477f  // SMSTART ZA\n"
// 0xd503477f = SMSTART ZA 的 .inst 编码
// 作用：进入流式模式 SVE，激活 ZA tile

// === 常量设置（与 SVE 版完全相同）===
"fmov z16.s, #4.0\n"                    // z16 = 4.0
"ptrue p1.b\n"                          // p1 = 全真
"ld1rqw { z2.s }, p1/Z, [%x[B_values]]\n"  // z2 = [1,2,4,5] 重复

// === 指针设置（与 SVE 版完全相同）===
// 6 个输入行指针 + 6 个输出行指针 + 列偏移
// ... 同 sve_fp32_6x6.cpp

// === 通道循环 ===
// 与 SVE 版完全相同的指令序列
// fmul/fmla/fmls/fmad/fmsb/fadd/fsub
// ld1w/st1w with whilelt 谓词

// === 退出流式模式 ===
".inst 0xd503467f  // SMSTOP\n"
// 0xd503467f = SMSTOP 的 .inst 编码
// 作用：退出流式模式，ZA tile 内容丢失
```

### 2.3 与 SVE 版的差异

| 方面 | SVE (`sve_fp32_6x6.cpp`) | SME (`sme_fp32_mla_6x6.cpp`) |
|------|--------------------------|------------------------------|
| SMSTART/SMSTOP | 无 | 有 |
| FMOPA/MOVA/ZA | 无 | 无（不用） |
| 指令序列 | SVE 指令 | **完全相同**的 SVE 指令 |
| 性能 | 基准 | 可能更快（流式模式 SVE 吞吐更高） |

**结论**：SME 输入变换就是 SVE 输入变换加了流式模式包裹，代码完全相同。

---

## 3. SME 输出变换 (`sme_fp32_mopa_4x4_3x3.cpp`)

### 3.1 核心思路：Kronecker 积 + FMOPA

输出变换 `f = A^T · M · A` 可以用 **vec trick** 重写为：

```
vec(f) = (A^T ⊗ A^T) · vec(M)
```

其中 ⊗ 是 Kronecker 积，`(A^T ⊗ A^T)` 是 16×36 矩阵。

**用 FMOPA 实现矩阵-向量乘**：
- 加载 36 个 Winograd 域元素（每个含 VL 通道）
- 对每个元素，构造 Kronecker 系数向量（16 个系数）
- FMOPA: `zaD += kronecker_coef ⊗ winograd_data`（一条指令做 VL×VL 外积）
- 4 个 ZA tile（za0-za3）对应 4 行输出
- MOVA 读出结果 + fmin/fmax (ReLU) + st1w 存储

### 3.2 A^T 矩阵

F(4,4,3,3) 的 A^T 矩阵（6×4）：
```
A^T = [1   0   0   0]    ← 行 0
      [1   1   1   1]    ← 行 1
      [1  -1   1  -1]    ← 行 2
      [1   2   4   8]    ← 行 3
      [1  -2   4  -8]    ← 行 4
      [0   0   0   1]    ← 行 5
```

### 3.3 Kronecker 系数分解

完整矩阵 `(A^T ⊗ A^T)` 是 16×36，太大了。
ACL 将其分解为两个小矩阵的运行时组合：

**outer_terms**（32 个 float，2 个 Z 寄存器）：
```c
static const float outer_terms[32] = {
    // A^T 的前 4 列（每行 4 个系数）
     1, 1,  1, 1,    // A^T 行 0: [1, 1, 1, 1]
     0, 1, -1, 2,    // A^T 行 1
     0, 1,  1, 4,    // A^T 行 2
     0, 1, -1, 8,    // A^T 行 3
    // A^T 的后 2 列（含 padding 确保四字对齐）
     1, 0,  0, 0,    // A^T 行 0 后半
    -2, 0,  0, 0,    // A^T 行 1 后半
     4, 0,  0, 0,    // A^T 行 2 后半
    -8, 1,  0, 0     // A^T 行 3 后半
};
```

**inner_terms**（24 个 float，6 个 ld1rqw）：
```c
static const float inner_terms[24] = {
    1,  0, 0,  0,    // A^T 列 0: [1, 0, 0, 0, 0, 0]
    1,  1, 1,  1,    // A^T 列 1: [1, 1, 1, 1, 1, 0]
    1, -1, 1, -1,    // A^T 列 2
    1,  2, 4,  8,    // A^T 列 3
    1, -2, 4, -8,    // A^T 列 4
    0,  0, 0,  1     // A^T 列 5
};
```

**运行时组合**：用 `fmul z_coef = z_inner * z_outer[idx]` 构造 Kronecker 系数，
然后用 FMOPA 累加。

### 3.4 寄存器分配

```
z6, z7     = outer_terms（2 个 ld1w 加载，每个 VL float）
z9, z8, z15, z4, z3, z2 = inner_terms（6 个 ld1rqw 加载，每个重复 1 quad）
z11, z5    = 工作系数向量（fmul 构造的 Kronecker 系数）
z1         = 1.0（用于 bias 的 FMOPA）
z10        = output_max（用于 fmin）
z12        = output_min（用于 fmax）
z31-z16    = Winograd 域数据 / MOVA 读出结果

p5         = 全真谓词（FMOPA/MOVA 用）
p8         = bias 谓词（有 bias 时全真，无 bias 时全假）
p4, p3, p2, p1 = 通道谓词（whilelt 生成，处理 4 组 VL 宽度的通道）

x25, x24, x23, x22 = 4 组通道偏移（0, VL, 2*VL, 3*VL）
x15 = 0,  x14 = 12,  x13 = 4,  x12 = 8  (MOVA 行索引)
x21 = Winograd 数据行指针
x20, x8 = 输出列指针

ZA tiles:
  za0 = 输出行 0 的累加器
  za1 = 输出行 1 的累加器
  za2 = 输出行 2 的累加器
  za3 = 输出行 3 的累加器
```

### 3.5 逐行注释

#### 3.5.1 初始化（行 108-131）

```asm
// 加载 outer_terms 和 inner_terms
"ldr x20, [%x[params], %[ofs_ot]]\n"      // x20 → outer_terms 指针
".inst 0xd503477f  // SMSTART ZA\n"        // 进入流式模式
"ptrue p5.b\n"                              // p5 = 全真

// 加载 output_min/max
"ld1rw { z12.s }, p5/Z, [%x[params], %[ofs_amin]]\n"  // z12 = act_min
"ld1rw { z10.s }, p5/Z, [%x[params], %[ofs_amax]]\n"  // z10 = act_max

"pfalse p8.b\n"                             // p8 = 全假（默认无 bias）

// 加载系数到 Z 寄存器
"ldr x8, [%x[params], %[ofs_it]]\n"         // x8 → inner_terms 指针
"ld1w { z6.s }, p5/Z, [x20]\n"              // z6 = outer_terms[0..VL-1]
"ld1w { z7.s }, p5/Z, [x20, #1, MUL VL]\n"  // z7 = outer_terms[VL..2*VL-1]

// ld1rqw: 加载 1 quadword 并在整个向量中重复
"ld1rqw { z9.s }, p5/Z, [x8]\n"       // z9 = [1,0,0,0] 重复
"ld1rqw { z8.s }, p5/Z, [x8, #16]\n"  // z8 = [1,1,1,1] 重复
"ld1rqw { z15.s }, p5/Z, [x8, #32]\n" // z15 = [1,-1,1,-1] 重复
"ld1rqw { z4.s }, p5/Z, [x8, #48]\n"  // z4 = [1,2,4,8] 重复
"ld1rqw { z3.s }, p5/Z, [x8, #64]\n"  // z3 = [1,-2,4,-8] 重复
"ld1rqw { z2.s }, p5/Z, [x8, #80]\n"  // z2 = [0,0,0,1] 重复

// 构造初始 Kronecker 系数
"fmul z11.s, z9.s, z6.s[0]\n"   // z11 = inner[0] * outer[0] (A^T row 0, col 0 的系数)
"fmul z5.s, z9.s, z6.s[1]\n"    // z5  = inner[0] * outer[1] (A^T row 0, col 1 的系数)

// 检查是否有 bias
"cbz %x[bptr], 1f\n"            // bptr == 0? 跳过 bias 谓词设置
"ptrue p8.s\n"                   // p8 = 全真（有 bias）

"1:\n"
// 清零所有 ZA tile
".inst 0xc00800ff  // zero {zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7}\n"

// z1 = 1.0（用于 bias 的 FMOPA: za += 1.0 ⊗ bias）
"fmov z1.s, #1.0\n"

// 初始化通道偏移
"mov x25, #0x0\n"                // x25 = 0 (第一组 VL 通道)
"cntw x24\n"                     // x24 = VL (第二组)
"cntw x23, ALL, MUL #2\n"        // x23 = 2*VL (第三组)
"cntw x22, ALL, MUL #3\n"        // x22 = 3*VL (第四组)
```

#### 3.5.2 通道谓词 + 加载数据（行 135-160）

```asm
// 为 4 组通道生成谓词
"whilelt p4.s, x25, %x[n_channels]\n"   // p4: 通道 0..VL-1 活跃？
"whilelt p3.s, x24, %x[n_channels]\n"   // p3: 通道 VL..2*VL-1 活跃？
"whilelt p2.s, x23, %x[n_channels]\n"  // p2: 通道 2*VL..3*VL-1 活跃？
"whilelt p1.s, x22, %x[n_channels]\n"  // p1: 通道 3*VL..4*VL-1 活跃？

// 加载 36 个 Winograd 元素（4 组 VL 通道，前 4 行）
"ld1w { z31.s }, p4/Z, [%x[inptr], x25, LSL #2]\n"  // M[0][0], 通道 0..VL-1
"ld1w { z30.s }, p3/Z, [%x[inptr], x24, LSL #2]\n"  // M[0][0], 通道 VL..2*VL-1
"ld1w { z29.s }, p2/Z, [%x[inptr], x23, LSL #2]\n"  // M[0][0], 通道 2*VL..3*VL-1
// ... 继续加载 M[0][1], M[1][0], M[1][1], M[2][0] 等

// x21 = inptr + matrix_stride (指向 M[1][0])
"add x21, %x[inptr], %x[matrix_stride], LSL #2\n"
```

#### 3.5.3 Bias FMOPA（行 159-169）

```asm
// 如果有 bias，用 FMOPA 把 bias 加到每个 ZA tile
"ld1w { z0.s }, p0/Z, [%x[bptr], x25, LSL #2]\n"   // z0 = bias[0..VL-1]
".inst 0x8080b420  // fmopa za0.s, p5/M, p5/M, z1.s, z0.s\n"
// za0 += z1(=1.0) ⊗ z0(=bias) → za0 每个元素都加上 bias
// 这等效于：output[0][*] += bias（因为 1.0 ⊗ bias = bias 广播到整行）

"ld1w { z0.s }, p0/Z, [%x[bptr], x24, LSL #2]\n"   // z0 = bias[VL..2*VL-1]
".inst 0x8080b421  // fmopa za1.s, p5/M, p5/M, z1.s, z0.s\n"
// za1 += bias（同样广播）
// ... 对 za2, za3 重复
```

#### 3.5.4 主循环：36 个 Winograd 元素的 FMOPA 累加（行 170-509）

这是最核心的部分。对 6×6=36 个 Winograd 元素，每个做 4 次 FMOPA（对应 4 个输出行）。

```asm
"2:\n"  // 主循环开始

// === 第一组 Winograd 元素（M[0][0..5] + M[1][0..5] + ...）===

// 用 z11(当前 Kronecker 系数) 和 z31(当前 Winograd 数据) 做 FMOPA
".inst 0x809fb560  // fmopa za0.s, p5/M, p5/M, z11.s, z31.s\n"
// za0 += z11 ⊗ z31 → 16×16 外积累加到 za0
// z11 = Kronecker 系数（对应输出行 0 的某个位置）
// z31 = Winograd 域数据 M[i][j]（VL 个通道）

".inst 0x809eb561  // fmopa za1.s, p5/M, p5/M, z11.s, z30.s\n"
// za1 += z11 ⊗ z30 → 同样的 Kronecker 系数，但数据是下一组通道

// ... 继续对 za2, za3 做 FMOPA

// 构造下一个 Kronecker 系数
"fmul z11.s, z9.s, z6.s[2]\n"
// z11 = inner_terms[当前] * outer_terms[下一个 idx]
// 切换到 A^T 的下一个行系数

// 加载下一批 Winograd 数据
"ld1w { z31.s }, p4/Z, [x21, x25, LSL #2]\n"
// z31 = M[下一行][0][通道 0..VL-1]

// ... 这个模式重复 36 次（对应 6×6 个 Winograd 元素）
// 每次用不同的 Kronecker 系数（由 outer × inner 构造）
```

**Kronecker 系数构造模式**：

```asm
// z9 = inner_terms[0] = [1, 0, 0, 0]
// z8 = inner_terms[1] = [1, 1, 1, 1]
// z15 = inner_terms[2] = [1, -1, 1, -1]
// z4 = inner_terms[3] = [1, 2, 4, 8]
// z3 = inner_terms[4] = [1, -2, 4, -8]
// z2 = inner_terms[5] = [0, 0, 0, 1]

// z6 = outer_terms[0..VL-1] = [1, 1, 1, 1, 0, 1, -1, 2, 0, 1, 1, 4, 0, 1, -1, 8]
// z7 = outer_terms[VL..2*VL-1] = [1, 0, 0, 0, -2, 0, 0, 0, 4, 0, 0, 0, -8, 1, 0, 0]

// Kronecker 系数 = inner[i] * outer[j]
"fmul z11.s, z9.s, z6.s[0]"   // z11 = inner * outer[0] = [1,0,0,0] * 1
"fmul z5.s, z9.s, z6.s[1]"    // z5  = inner * outer[1] = [1,0,0,0] * 1
"fmul z11.s, z9.s, z6.s[2]"   // z11 = inner * outer[2] = [1,0,0,0] * 1
"fmul z5.s, z9.s, z6.s[3]"    // z5  = inner * outer[3] = [1,0,0,0] * 1
// ... 切换 inner_terms 和 outer_terms 的索引
"fmul z11.s, z9.s, z7.s[0]"   // 使用 z7（outer 后半）
"fmul z11.s, z8.s, z6.s[0]"   // 切换 inner 到 z8
// ... 共 6 (inner) × 2 (z6/z7) × 4 (element) = 48 种组合
//     但只有 36 个实际使用（对应 6×6 Winograd tile）
```

#### 3.5.5 MOVA 读出 + ReLU + 存储（行 510-594）

```asm
// === 从 za0 读出 4×4 输出 tile ===

// x15 = 0, x14 = 12 (0xC), x13 = 4, x12 = 8
// 这些是 MOVA 的行索引参数

// MOVA: 从 ZA tile 读出一行到 Z 向量
".inst 0xc082741f  // mova z31.s, p5/M, za0h.s[x15]\n"
// z31 = za0 的第 x15(=0) 列 → 输出 tile 的 (0,0) 元素（VL 个通道）

".inst 0xc082541c  // mova z28.s, p5/M, za0h.s[x14]\n"
// z28 = za0 的第 x14(=12) 列 → 输出 tile 的 (0,1) 元素

// ReLU: fmin(先上限) → fmax(后下限)
"fmin z31.s, p5/M, z31.s, z10.s\n"   // z31 = min(z31, act_max)
// ... 后续 fmax
"fmax z31.s, p5/M, z31.s, z12.s\n"  // z31 = max(z31, act_min)

// 存储
"st1w { z31.s }, p0, [%x[output], x25, LSL #2]\n"
// output[0][0][通道 0..VL-1] = z31

// 继续读出 (0,1), (0,2), (0,3)
".inst 0xc082743b  // mova z27.s, p5/M, za0h.s[x15, #1]\n"
// z27 = za0 的第 (x15+1=1) 列 → 输出 (0,1)

// ... 16 个 MOVA 读出 4×4 输出
```

**MOVA 行索引的含义**：

```
x15 = 0  → tile 行 0, 列偏移 0 → 输出 (0, 0)
x15+1 = 1  → tile 行 0, 列偏移 1 → 输出 (0, 1)
x15+2 = 2  → tile 行 0, 列偏移 2 → 输出 (0, 2)
x15+3 = 3  → tile 行 0, 列偏移 3 → 输出 (0, 3)
x14 = 12 → tile 行 3, 列偏移 0 → 输出 (1, 0)  (12 = 3*4)
x14+1 = 13 → tile 行 3, 列偏移 1 → 输出 (1, 1)
x13 = 4  → tile 行 1, 列偏移 0 → 输出 (2, 0)  (4 = 1*4)
x12 = 8  → tile 行 2, 列偏移 0 → 输出 (3, 0)  (8 = 2*4)
```

#### 3.5.6 za1/za2/za3 读出（行 595-843）

对 za1, za2, za3 重复相同的 MOVA + fmin/fmax + st1w 模式。
每个 ZA tile 对应一组 VL 个通道的 4×4 输出。

```asm
// 检查是否还有更多通道
"whilelt p0.s, x24, %x[n_channels]\n"  // 第二组 VL 通道还有吗？
"b.none 3f\n"                           // 没有，跳到结束

// 读出 za1（第二组 VL 通道的输出）
".inst 0xc082749f  // mova z31.s, p5/M, za1h.s[x15]\n"
// ... 同样的 MOVA + fmin/fmax + st1w 模式

// 检查 za2
"whilelt p0.s, x23, %x[n_channels]\n"
"b.none 3f\n"
// 读出 za2

// 检查 za3 + 准备下一轮
"whilelt p0.s, x22, %x[n_channels]\n"
"b.none 3f\n"
// 读出 za3 + 前进通道偏移 + 清零 ZA + 加载下一轮数据
```

#### 3.5.7 循环回跳（行 838-880）

```asm
// 读出 za3 的最后几个元素 + 存储
"st1w { z16.s }, p0, [x20, x22, LSL #2]\n"

// 前进通道偏移（4 组各前进 4*VL）
"incw x25, ALL, MUL #4\n"   // x25 += 4*VL
"incw x24, ALL, MUL #4\n"   // x24 += 4*VL
"incw x23, ALL, MUL #4\n"   // x23 += 4*VL
"incw x22, ALL, MUL #4\n"   // x22 += 4*VL

// 清零 ZA tile（准备下一轮累加）
".inst 0xc00800ff  // zero {zad0-zad7}\n"

// 重新设置数据指针
"add x21, %x[inptr], %x[matrix_stride], LSL #2\n"

// 重新生成谓词 + 加载下一轮数据
"whilelt p1.s, x22, %x[n_channels]\n"
"ld1w { z28.s }, p1/Z, [%x[inptr], x22, LSL #2]\n"
// ... 加载所有 36 个 Winograd 元素（下一轮通道）

// Bias FMOPA
".inst 0x8080b420  // fmopa za0.s, p5/M, p5/M, z1.s, z0.s\n"
// ... 4 次 bias FMOPA

// 回跳到主循环
"b.any 2b\n"   // 如果有任何活跃 lane，跳回循环

"3:\n"  // 循环结束
".inst 0xd503467f  // SMSTOP\n"   // 退出流式模式
```

---

## 4. 优化技术总结

### 4.1 4 组通道并行

每次循环处理 **4 × VL 个通道**（SVE-512 时 = 64 个通道）。
4 个 ZA tile 同时累加，一次循环产出 4 × 4 × VL = 4 × 4 × 16 = 256 个输出元素。

对比 SVE 版每次循环只处理 VL = 16 个通道，SME 版吞吐量是 **4 倍**。

### 4.2 FMOPA 替代两次 1D 矩阵乘

NEON/SVE 版的输出变换是两次 1D 矩阵乘（先 FZ = F·Z，再 f = Z^T·FZ），共 16×6 + 16×6 = 192 次乘加。

SME 版用 Kronecker 积 + FMOPA，直接做 16×36 的矩阵-向量乘。
36 条 FMOPA 指令，每条做 VL×VL = 256 次 FMA → 总共 36 × 256 = 9216 次 FMA（SVE-512）。
对比 SVE 版 192 × 16 = 3072 次 FMA。

**SME 多做了 3 倍 FMA 运算量，但每条 FMOPA 的吞吐量远高于 FMLA**，
且 4 个 ZA tile 并行，实际性能更高。

### 4.3 Kronecker 系数即时构造

不预存 16×36 = 576 个 Kronecker 系数（太占寄存器），而是在运行时用 `fmul` 构造：
```
z11 = inner_terms[i] * outer_terms[j][idx]
```

32 + 24 = 56 个 float 的预存系数，替代 576 个——节省 10 倍存储。

### 4.4 指令调度

FMOPA 指令之间穿插 `ld1w`（加载下一批数据）和 `fmul`（构造下一个系数）：
```
fmopa za0     // 累加
fmopa za1     // 累加
ld1w z31      // 加载（与 FMOPA 并行）
fmopa za2     // 累加
fmul z11      // 构造系数（与 FMOPA 并行）
fmopa za3     // 累加
add x21       // 指针前进（与 FMOPA 并行）
```

### 4.5 MOVA + fmin/fmax 交错

MOVA 读出后立即做 fmin（ReLU 上限），不等所有 MOVA 完成：
```
mova z31     // 读出 (0,0)
mova z28     // 读出 (0,1)（与上面并行）
fmin z31     // ReLU (0,0)（与 MOVA z28 并行）
mova z27     // 读出 (0,2)
fmin z28     // ReLU (0,1)
fmax z31     // ReLU (0,0) 完成
```

### 4.6 循环完全展开（36 个 Winograd 元素）

与 NEON/SVE 版相同：循环体内没有任何内层循环，36 个 Winograd 元素的 FMOPA 全部展开为直线代码。
循环只用于通道维度（每次 4×VL 个通道）。

---

## 5. 三种 ISA 对比总结

| 方面 | NEON | SVE | SME |
|------|------|-----|-----|
| **输入变换** | | | |
| 代码量 | 1140 行 | 361 行 | 363 行 |
| Tail 处理 | 三层降级 | whilelt 谓词 | whilelt 谓词 |
| 每迭代通道 | 4 | VL(16) | VL(16) |
| FMOPA/ZA | 不用 | 不用 | 不用（仅 SMSTART/STOP 包裹） |
| **输出变换** | | | |
| 代码量 | 242 行(C++) | 242 行(C++) | 892 行(asm) |
| 算法 | 两次 1D 矩阵乘 | 两次 1D 矩阵乘 | Kronecker + FMOPA |
| 每迭代通道 | 4 | VL(16) | 4×VL(64) |
| 指令/迭代 | ~192 FMA | ~192 FMLA | 36 FMOPA (=9216 FMA) |
| **权重变换** | | | |
| 实现 | C++ intrinsics | C++ intrinsics (NEON) | C++ intrinsics (NEON) |
| 说明 | 非热路径，不需要汇编 | 同左 | 同左 |
