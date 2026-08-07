# ACL Winograd SVE 内联汇编逐行注释

> 源文件: `ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/winograd/input_transforms/sve_fp32_6x6.cpp`
>
> SVE 输入变换 kernel，F(4,4,3,3) 配置。

---

## 1. SVE 与 NEON 的关键差异

| 方面 | NEON (`a64_fp32_6x6.cpp`) | SVE (`sve_fp32_6x6.cpp`) |
|------|--------------------------|--------------------------|
| 向量宽度 | 固定 128-bit (4 float) | 可伸缩 VL (16 float @ SVE-512) |
| 寄存器 | v0-v31 (Q/D/S 三层) | z0-z31 + p0-p15 |
| Tail 处理 | 三层降级 (4→2→1 通道) | 谓词掩码 `whilelt`（一份代码） |
| 代码量 | 1140 行 | 361 行（仅为 NEON 的 1/3） |
| 循环结构 | 完全展开（无内层循环） | `whilelt` + `bne` 回跳循环 |
| 系数加载 | `ldr q0` (4 float) | `ld1rqw` (1 个 quad 重复到全向量) |

**核心优势**：SVE 的谓词寄存器 `p0-p15` 可以在一条指令中控制哪些 lane 激活，
因此不需要写三份代码处理尾部——只需在循环入口用 `whilelt` 生成谓词。

---

## 2. 数学背景

与 NEON 版完全相同的算法：`U = B^T · d · B`

B_values = `{1.0, 2.0, 4.0, 5.0}` 是缩放系数，与 NEON 的 pcoeffs 相同。
z16 = 4.0 是 `fmad`/`fmsb` 的乘法操作数。

### SVE FMA 指令语义

```
FMLA  Zda, Pn/M, Zn, Zm   →  Zda[active] += Zn * Zm
FMLS  Zda, Pn/M, Zn, Zm   →  Zda[active] -= Zn * Zm
FMAD  Zda, Pn/M, Zm, Zn   →  Zda[active] = Zda * Zm + Zn  (fused)
FMSB  Zda, Pn/M, Zm, Zn   →  Zda[active] = Zda * Zm - Zn  (fused)
FMUL  Zd,  Pn/M, Zn, Zm   →  Zd[active] = Zn * Zm
FNEG  Zd,  Pn/M, Zn       →  Zd[active] = -Zn
FADD  Zd,  Pn/M, Zn, Zm   →  Zd[active] = Zn + Zm
FSUB  Zd,  Pn/M, Zn, Zm   →  Zd[active] = Zn - Zm
```

关键：`p1/M` 表示 "merging" 模式——非活跃 lane 保持原值不变（而非清零）。

---

## 3. 寄存器分配

```
z2    = B_values [1.0, 2.0, 4.0, 5.0]（通过 ld1rqw 重复填充）
z16   = 4.0（常量，用于 fmad/fmsb）
p0    = 通道谓词（whilelt 生成）
p1    = 全真谓词（ptrue）

输入行指针:
  input_row_0 → 行 0
  x16 = 行 0 + row_stride      → 行 1
  x14 = 行 0 + 2*row_stride    → 行 2
  x12 = 行 0 + 3*row_stride    → 行 3
  x10 = 行 0 + 4*row_stride    → 行 4
  x28 = 行 0 + 5*row_stride    → 行 5

输出行指针:
  output_row_0 → 行 0
  x15 = 行 0 + out_stride      → 行 1
  x13 = 行 0 + 2*out_stride    → 行 2
  x11 = 行 0 + 3*out_stride    → 行 3
  x9  = 行 0 + 4*out_stride    → 行 4
  x27 = 行 0 + 5*out_stride    → 行 5

列偏移:
  input_col_1_stride  → 列 1
  x26 = 2 * col_stride → 列 2
  x24 = 3 * col_stride → 列 3
  x22 = 4 * col_stride → 列 4
  x20 = 5 * col_stride → 列 5
```

---

## 4. 逐行注释

### 4.1 初始化（行 41-68）

```asm
// 行 41: B_values = {1.0, 2.0, 4.0, 5.0}
const float B_values[4] = { 1.0f, 2.0f, 4.0f, 5.0f };

// 行 45: 进入内联汇编
__asm__ __volatile__(

// 行 46: z16 = 4.0（常量，用于 fmad z = z*4 + x 和 fmsb z = z*4 - x）
"fmov z16.s, #4.0\n"

// 行 47: p1 = 全真（所有 lane 激活），用于非谓词化的向量操作
"ptrue p1.b\n"

// 行 48: z2 = B_values 重复填充
// ld1rqw = "load 1 quadword and replicate"
// 加载 4 个 float [1,2,4,5]，在整个 Z 向量中重复填充
// SVE-512 时 z2 = [1,2,4,5, 1,2,4,5, 1,2,4,5, 1,2,4,5]
"ld1rqw { z2.s }, p1/Z, [%x[B_values]]\n"

// 行 49-66: 设置 6 个输入行指针和 6 个输出行指针
// 与 NEON 版完全相同的指针算术，只是用 LSL #2（左移 2 位 = ×4 字节）
"add x16, %x[input_row_0], %x[input_row_stride], LSL #2\n"
// x16 = input + 1*row_stride*4 = 行 1

"add x14, %x[input_row_0], %x[input_row_stride], LSL #3\n"
// x14 = input + 2*row_stride*4 = 行 2  (LSL #3 = ×8 = 2×4)

"add x12, x14, %x[input_row_stride], LSL #2\n"
// x12 = 行 2 + 1*row_stride*4 = 行 3

"add x10, %x[input_row_0], %x[input_row_stride], LSL #4\n"
// x10 = input + 4*row_stride*4 = 行 4  (LSL #4 = ×16 = 4×4)

"add x28, x10, %x[input_row_stride], LSL #2\n"
// x28 = 行 4 + 1*row_stride*4 = 行 5

// 列偏移同理
"lsl x26, %x[input_col_1_stride], #0x1\n"  // x26 = 2*col_stride
"add x24, x26, %x[input_col_1_stride]\n"   // x24 = 3*col_stride
"lsl x22, %x[input_col_1_stride], #0x2\n"  // x22 = 4*col_stride
"add x20, x22, %x[input_col_1_stride]\n"   // x20 = 5*col_stride

// 行 67: 生成通道谓词
// whilelt p0.s, XZR(=0), num_channels
// 比较 0 < num_channels，生成活跃掩码
// 如果 num_channels >= VL，全部活跃；否则前 n_channels 个 lane 活跃
"whilelt p0.s, XZR, %x[num_channels]\n"

// 行 68: 如果没有通道（num_channels=0），跳到结尾
"beq 2f\n"
```

### 4.2 通道循环——行 0 变换（行 69-157）

这是循环体的第一部分，处理输入行 0-2，计算 U[0][0..5] 和中间量。

```asm
"1:"  // 通道循环开始

// === 加载输入行 0 的 6 个元素 ===
// d[0][0..5]，每个元素包含 VL 个通道
"ld1w { z31.s }, p0/Z, [%x[input_row_0]]\n"
// z31 = d[0][0]（VL 个通道，谓词 p0 控制尾部）

"decw %x[num_channels]\n"
// num_channels -= VL（SVE 的 cntw 返回 VL 元素数）
// decw 按 VL 递减（SVE-512 时每次 -16）

"ld1w { z28.s }, p0/Z, [%x[input_row_0], %x[input_col_1_stride], LSL #2]\n"
// z28 = d[0][1]（列 1）

// === 开始列变换 B^T·d ===
// z13 = z28 * z2.s[1] = d[0][1] * 2
"fmul z13.s, z28.s, z2.s[1]\n"

"ld1w { z27.s }, p0/Z, [%x[input_row_0], x26, LSL #2]\n"  // z27 = d[0][2]
"ld1w { z11.s }, p0/Z, [%x[input_row_0], x24, LSL #2]\n"  // z11 = d[0][3]

// z13 = -z13 = -2*d[0][1]
"fneg z13.s, p1/M, z13.s\n"

"ld1w { z7.s }, p0/Z, [%x[input_row_0], x22, LSL #2]\n"   // z7 = d[0][4]

// z15 = z7 - z27 = d[0][4] - d[0][2]
"fsub z15.s, z7.s, z27.s\n"

// z31 = z31 * z16 + z7 = d[0][0] * 4 + d[0][4]
// FMAD: Zda = Zda * Zm + Zn = z31 * 4 + z7
"fmad z31.s, p1/M, z16.s, z7.s\n"

"ld1w { z3.s }, p0/Z, [%x[input_row_0], x20, LSL #2]\n"   // z3 = d[0][5]

// z13 = z13 + z11 * z2.s[1] = -2*d[0][1] + 2*d[0][3]
// FMLA: Zda += Zn * Zm_by_element = z13 + z11 * 2
"fmla z13.s, z11.s, z2.s[1]\n"

"ld1w { z12.s }, p0/Z, [x14]\n"  // z12 = d[2][0]（行 2 列 0）

// 前进行 0 指针到下一组通道
"incb %x[input_row_0]\n"
// incb = "increment by bytes in current vector granularity"
// 即指针 += VL * 4 字节

// z31 = z31 - z27 * z2.s[3] = (4*d[0][0] + d[0][4]) - 5*d[0][2]
// FMLS: Zda -= Zn * Zm_by_element
"fmls z31.s, z27.s, z2.s[3]\n"

"ld1w { z14.s }, p0/Z, [x14, %x[input_col_1_stride], LSL #2]\n" // z14 = d[2][1]

// z25 = z15 - z13 = (d[0][4]-d[0][2]) - (-2*d[0][1]+2*d[0][3])
//      = 2*d[0][1] - d[0][2] - 2*d[0][3] + d[0][4]
"fsub z25.s, z15.s, z13.s\n"

// z8 = z13 + z15 = (-2*d[0][1]+2*d[0][3]) + (d[0][4]-d[0][2])
//    = -2*d[0][1] - d[0][2] + 2*d[0][3] + d[0][4]
"fadd z8.s, z13.s, z15.s\n"

"ld1w { z24.s }, p0/Z, [x14, x26, LSL #2]\n"  // z24 = d[2][2]

// FMSB: Zda = Zda * Zm - Zn = z27 * 4 - z7 = 4*d[0][2] - d[0][4]
"fmsb z27.s, p1/M, z16.s, z7.s\n"

// ... 继续类似模式处理 d[2][*]

// === 存储 U[0][0..5] ===
"st1w { z31.s }, p0, [%x[output_row_0]]\n"
// U[0][0] = z31 = 4*d[0][0] - 5*d[0][2] + d[0][4]

"st1w { z17.s }, p0, [%x[output_row_0], %x[output_col_1_stride], LSL #2]\n"
// U[0][1] = z17

"st1w { z27.s }, p0, [%x[output_row_0], x25, LSL #2]\n"     // U[0][2]
"st1w { z8.s }, p0, [%x[output_row_0], x23, LSL #2]\n"      // U[0][3]
"st1w { z25.s }, p0, [%x[output_row_0], x21, LSL #2]\n"     // U[0][4]
"st1w { z28.s }, p0, [%x[output_row_0], x8, LSL #2]\n"      // U[0][5]

// 前进输出指针
"incb %x[output_row_0]\n"
```

### 4.3 通道循环——行 1-5 变换（行 159-346）

完全相同的模式，只是输入行指针从 `input_row_0` 变为 `x16`（行 1）、`x12`（行 3）、`x10`（行 4）、`x28`（行 5）。

每组 6 个 `ld1w` 加载 → `fmul/fmla/fmls/fmad/fmsb/fadd/fsub` 计算 → 6 个 `st1w` 存储。

### 4.4 循环回跳（行 347-349）

```asm
// 重新生成通道谓词（检查是否还有剩余通道）
"whilelt p0.s, XZR, %x[num_channels]\n"
// 如果 num_channels > 0，p0 至少有 1 个活跃 lane

// 如果有任何活跃 lane，跳回循环开始
"bne 1b\n"

"2:"  // 循环结束
```

**与 NEON 的关键差异**：
- NEON: `cmp #4; b.ge 1b`（固定比较 4）
- SVE: `whilelt p0, 0, n` → `bne 1b`（谓词驱动，自动适应 VL）

---

## 5. 优化技术总结

### 5.1 谓词化 Tail 处理（替代三层降级）

**SVE 的核心优势**：用一份代码处理任意通道数。

```asm
// 循环入口
whilelt p0.s, 0, n_channels    // 生成谓词
bne 1b                          // 有活跃 lane 就继续

// 循环体内
ld1w z, p0/Z, [ptr]            // p0 控制哪些 lane 加载
st1w z, p0, [ptr]              // p0 控制哪些 lane 存储
// 非活跃 lane 不访存，不引发越界

// 循环递减
decw n_channels                // n_channels -= VL
```

对比 NEON 需要 1140 行（三层完全展开），SVE 只需 361 行（一份循环）。

### 5.2 指令调度（与 NEON 相同模式）

加载和计算交错执行：
```asm
ld1w z31    // 加载 d[0][0]
decw n      // 递减计数器（与加载并行）
ld1w z28    // 加载 d[0][1]
fmul z13    // 计算（用已加载数据）
ld1w z27    // 加载 d[0][2]（与计算并行）
fneg z13    // 计算
ld1w z11    // 加载 d[0][3]
fsub z15    // 计算（用 z7 和 z27）
fmad z31    // 计算（用 z31 和 z7）
ld1w z3     // 加载 d[0][5]
fmla z13    // 计算
```

### 5.3 系数向量复用

```asm
ld1rqw { z2.s }, p1/Z, [B_values]
// 加载 [1,2,4,5] 并在整个 Z 向量中重复
// SVE-512: z2 = [1,2,4,5,1,2,4,5,1,2,4,5,1,2,4,5]

// 后续通过 by-element 索引：
fmul z, zN, z2.s[0]   // 乘 1.0
fmla z, zN, z2.s[1]   // 乘 2.0
fmls z, zN, z2.s[2]   // 乘 4.0
fmls z, zN, z2.s[3]   // 乘 5.0
```

### 5.4 循环完全展开（6 行 × 6 列）

与 NEON 版相同：循环体内没有行/列循环，36 个元素的变换全部展开为直线代码。
循环只用于通道维度（每次 VL 个通道）。

### 5.5 谓词模式选择

代码中使用了两种谓词模式：
- `p0/Z`（zeroing）：非活跃 lane 写零 → 用于加载（避免脏数据）
- `p1/M`（merging）：非活跃 lane 保持原值 → 用于计算（避免污染中间结果）
- `p0`（无后缀，store 用）：非活跃 lane 不写 → 用于存储（避免越界写）

---

## 6. SVE 指令速查

| 指令 | 语义 | 本文件中的用途 |
|------|------|--------------|
| `fmov z, #imm` | 向量广播立即数 | z16 = 4.0 |
| `ptrue p.b` | 全真谓词 | p1 = 全部活跃 |
| `ld1rqw z, p, [x]` | 加载 1 quad 并重复填充 | z2 = B_values |
| `ld1w z, p, [x]` | 谓词化加载 | 加载输入数据 |
| `st1w z, p, [x]` | 谓词化存储 | 存储输出数据 |
| `whilelt p.s, a, b` | 生成 a < b 的谓词 | 通道 tail 处理 |
| `incb x` | 指针 += VL 字节 | 前进到下一组通道 |
| `decw x` | 计数器 -= VL 元素 | 递减通道计数 |
| `fmul z, zn, zm[idx]` | by-element 乘 | z13 = z28 * 2.0 |
| `fneg z, p/M, zn` | 谓词化取反 | z13 = -z13 |
| `fmla z, p/M, zn, zm[idx]` | by-element 乘加 | z13 += z11 * 2.0 |
| `fmls z, p/M, zn, zm[idx]` | by-element 乘减 | z31 -= z27 * 5.0 |
| `fadd z, p/M, zn, zm` | 谓词化加 | z8 = z13 + z15 |
| `fsub z, p/M, zn, zm` | 谓词化减 | z25 = z15 - z13 |
| `fmad z, p/M, zm, zn` | 融合乘加 z=z*m+n | z31 = z31*4 + z7 |
| `fmsb z, p/M, zm, zn` | 融合乘减 z=z*m-n | z27 = z27*4 - z7 |
| `bne label` | 条件分支 | 循环回跳 |
| `beq label` | 条件分支 | 跳过空循环 |
