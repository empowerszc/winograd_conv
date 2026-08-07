# ACL Winograd NEON C++ Intrinsics 逐行注释

> 源文件目录: `ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/winograd/`
>
> 本文档覆盖使用 C++ NEON intrinsics 实现的 kernel（非内联汇编）：
> - 输入变换: `input_transforms/arm_fp32_4x4.cpp` (F(2,2,3,3))
> - 输出变换: `output_transforms/arm_fp32_4x4_3x3.cpp` (F(4,4,3,3))
> - 输出变换: `output_transforms/arm_fp32_2x2_3x3.cpp` (F(2,2,3,3))
> - 权重变换: `weight_transforms/arm_fp32_4x4_3x3.cpp` (F(4,4,3,3))
> - 权重变换: `weight_transforms/arm_fp32_2x2_3x3.cpp` (F(2,2,3,3))
>
> F(4,4,3,3) 输入变换使用内联汇编（a64_fp32_6x6.cpp），见 `acl_wino_neon_asm_annotated.md`。

---

## 1. 通用模式：三层降级

所有 NEON C++ intrinsic kernel 都使用相同的三层降级策略：

```cpp
for (; n_channels >= 4; n_channels -= 4) {
    // 用 float32x4_t (128-bit Q 寄存器)，4 通道并行
}
for (; n_channels >= 2; n_channels -= 2) {
    // 用 float32x2_t (64-bit D 寄存器)，2 通道并行
}
for (; n_channels; n_channels--) {
    // 用 float (32-bit 标量)，1 通道
}
```

三层代码逻辑完全相同，只是 intrinsics 后缀不同（`q` → 无后缀）。

---

## 2. 输入变换 F(2,2,3,3) — `arm_fp32_4x4.cpp`

### 2.1 数学公式

```
U = B^T · d · B
```

B^T 矩阵（4×4，从代码逆推）：

```
B^T = [ 1   0   0   0]    →  XTx[0][j] = x[0][j] - x[2][j]
      [ 0   1   1   0]    →  XTx[1][j] = x[1][j] + x[2][j]
      [ 0  -1   1   0]    →  XTx[2][j] = x[2][j] - x[1][j]
      [ 0   1   0  -1]    →  XTx[3][j] = x[1][j] - x[3][j]
```

B = B^T 的转置：

```
B  = [ 1   0   0   0]
      [ 0   1  -1   1]
      [ 0   1   1   0]
      [ 0   0   0  -1]
```

但实际代码中，行变换和列变换使用相同的系数模式（因为 B 和 B^T 结构对称）。

### 2.2 4-通道层逐行注释

```cpp
// 加载 4×4 输入 tile（16 个元素，每个 4 通道）
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        x[i][j] = vld1q_f32(x_ptrs[i][j]);  // 加载 4 个 float 到 Q 寄存器
        x_ptrs[i][j] += 4;                   // 指针前进 4 个 float
    }
}

// === 第一步：列变换 XTx = B^T · x ===
for (int j = 0; j < 4; j++) {
    // B^T[0] = [1, 0, -1, 0] → XTx[0][j] = x[0][j] - x[2][j]
    XTx[0][j] = vsubq_f32(x[0][j], x[2][j]);
    // vsubq_f32 = NEON 128-bit 浮点减法

    // B^T[1] = [0, 1, 1, 0] → XTx[1][j] = x[1][j] + x[2][j]
    XTx[1][j] = vaddq_f32(x[1][j], x[2][j]);
    // vaddq_f32 = NEON 128-bit 浮点加法

    // B^T[2] = [0, -1, 1, 0] → XTx[2][j] = x[2][j] - x[1][j]
    XTx[2][j] = vsubq_f32(x[2][j], x[1][j]);

    // B^T[3] = [0, 1, 0, -1] → XTx[3][j] = x[1][j] - x[3][j]
    XTx[3][j] = vsubq_f32(x[1][j], x[3][j]);
}

// === 第二步：行变换 U = XTx · B ===
for (int i = 0; i < 4; i++) {
    // B[0] = [1, 0, 0, 0]^T → U[i][0] = XTx[i][0] - XTx[i][2]
    U[i][0] = vsubq_f32(XTx[i][0], XTx[i][2]);

    // B[1] = [0, 1, -1, 1]^T → U[i][1] = XTx[i][1] + XTx[i][2]
    U[i][1] = vaddq_f32(XTx[i][1], XTx[i][2]);

    // B[2] = [0, 1, 1, 0]^T → U[i][2] = XTx[i][2] - XTx[i][1]
    U[i][2] = vsubq_f32(XTx[i][2], XTx[i][1]);

    // B[3] = [0, 1, 0, -1]^T → U[i][3] = XTx[i][1] - XTx[i][3]
    U[i][3] = vsubq_f32(XTx[i][1], XTx[i][3]);
}

// === 存储 4×4 输出 tile ===
for (int i = 0, m = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++, m++) {
        vst1q_f32(outptr + m*matrix_stride, U[i][j]);
        // vst1q_f32 = NEON 128-bit 存储
    }
}
outptr += 4;  // 前进 4 个通道
```

### 2.3 2-通道层差异

```cpp
// 用 float32x2_t (64-bit D 寄存器)
x[i][j] = vld1_f32(x_ptrs[i][j]);     // 加载 2 个 float
XTx[0][j] = vsub_f32(x[0][j], x[2][j]); // 64-bit 减法
vst1_f32(outptr + m*matrix_stride, U[i][j]); // 存储 2 个 float
outptr += 2;
```

### 2.4 1-通道层

```cpp
// 用标量 float
x[i][j] = *(x_ptrs[i][j]++);
XTx[0][j] = x[0][j] - x[2][j];  // 标量减法
*(outptr + m*matrix_stride) = U[i][j];
outptr++;
```

### 2.5 优化分析

- **加法/减法树**：B^T 的非零元素都是 ±1，所以只需 `vadd/vsub`，无需乘法
- **对称性**：B 和 B^T 结构相同，行变换和列变换用相同模式
- **中间存储**：XTx 是 4×4 矩阵，存在寄存器中（4×4=16 个 Q 寄存器，刚好不超过 32 个）
- **指针预计算**：`x_ptrs[i][j]` 在循环外预计算，减少循环内的地址算术

---

## 3. 输出变换 F(2,2,3,3) — `arm_fp32_2x2_3x3.cpp`

### 3.1 数学公式

```
f = A^T · M · A
```

A^T 矩阵（4×2）：

```
A^T = [1   0   0   0]    →  f[0][j] = FZ[0][j] + FZ[1][j] + FZ[2][j]
      [0   1  -1  -1]    →  f[1][j] = FZ[1][j] - FZ[2][j] - FZ[3][j]
```

其中 FZ = M · A（中间量）：

```
FZ[i][0] = M[i][0] + M[i][1] + M[i][2]     (A[0] = [1,1,1,0]^T)
FZ[i][1] = M[i][1] - M[i][2] - M[i][3]     (A[1] = [0,1,-1,-1]^T)
```

### 3.2 4-通道层逐行注释

```cpp
// 加载 4×4 Winograd 域 tile
for (int i = 0, m = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++, m++) {
        F[i][j] = vld1q_f32(inptr + m*matrix_stride);
    }
}

// === 第一步：FZ = F · A（列变换，每行独立）===
for (int i = 0; i < 4; i++) {
    // FZ[i][0] = F[i][0] + F[i][1] + F[i][2]
    FZ[i][0] = vaddq_f32(vaddq_f32(F[i][0], F[i][1]), F[i][2]);
    // 先加前两个，再加第三个

    // FZ[i][1] = F[i][1] - F[i][2] - F[i][3]
    FZ[i][1] = vsubq_f32(vsubq_f32(F[i][1], F[i][2]), F[i][3]);
    // 先减第二个，再减第三个
}

// === 第二步：f = A^T · FZ（行变换，每列独立）===
for (int j = 0; j < 2; j++) {
    // f[0][j] = FZ[0][j] + FZ[1][j] + FZ[2][j]
    f[0][j] = vaddq_f32(vaddq_f32(FZ[0][j], FZ[1][j]), FZ[2][j]);

    // f[1][j] = FZ[1][j] - FZ[2][j] - FZ[3][j]
    f[1][j] = vsubq_f32(vsubq_f32(FZ[1][j], FZ[2][j]), FZ[3][j]);
}

// === Bias + ReLU + 存储 ===
if (bptr != nullptr) {
    b = vld1q_f32(bptr);   // 加载 4 通道 bias
    bptr += 4;
} else {
    b = vdupq_n_f32(0.0f); // 无 bias 时填零
}

for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
        // f[i][j] + bias, 然后 clamp [min, max]
        const auto y = vmaxq_f32(
            vminq_f32(vaddq_f32(f[i][j], b), vdupq_n_f32(output_max)),
            vdupq_n_f32(output_min)
        );
        vst1q_f32(outptr + i*output_row_stride + j*output_col_stride, y);
    }
}
```

### 3.3 优化分析

- **全加法/减法**：A^T 和 A 的元素都是 ±1，无乘法
- **2×2 输出 tile**：只需 4 个存储（对比 F(4,4) 的 16 个存储）
- **bias 广播**：同一组通道的 4 个输出共用一个 bias 向量

---

## 4. 输出变换 F(4,4,3,3) — `arm_fp32_4x4_3x3.cpp`

### 4.1 数学公式

```
f = A^T · M · A = Z^T · (M · Z) = Z^T · FZ
```

A^T 矩阵（6×4）：

```
A^T = [1   0   0   0]
      [1   1   1   1]
      [1  -1   1  -1]
      [1   2   4   8]
      [1  -2   4  -8]
      [0   0   0   1]
```

Z（= A 的转置 = A^T^T，4×6）：

```
Z = A^T 的转置 = [1  1  1  1  1  0]
                 [0  1 -1  2 -2  0]
                 [0  1  1  4  4  0]
                 [0  1 -1  8 -8  1]
```

### 4.2 4-通道层逐行注释

```cpp
// 加载 6×6 Winograd 域 tile（36 个元素）
for (int i = 0, m = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++, m++) {
        F[i][j] = vld1q_f32(inptr + m*matrix_stride);
    }
}

// === 第一步：FZ = F · Z（列变换）===
for (int i = 0; i < 6; i++) {
    // FZ[i][0] = F[i][0] + F[i][1] + F[i][2] + F[i][3] + F[i][4]
    // A^T 列 0 = [1,1,1,1,1,0]，取前 5 个（第 6 个 ×0 省略）
    FZ[i][0] = vaddq_f32(
        vaddq_f32(vaddq_f32(F[i][0], F[i][1]), vaddq_f32(F[i][2], F[i][3])),
        F[i][4]
    );
    // 嵌套 vaddq: 先两两配对加，再合并

    // FZ[i][1] = F[i][1] - F[i][2] + 2*(F[i][3] - F[i][4])
    // A^T 列 1 = [0,1,-1,2,-2,0]
    FZ[i][1] = vmlaq_n_f32(
        vsubq_f32(F[i][1], F[i][2]),   // F[i][1] - F[i][2]
        vsubq_f32(F[i][3], F[i][4]),   // × (F[i][3] - F[i][4])
        2.0f                            // × 2
    );
    // vmlaq_n_f32(a, b, n) = a + b * n（NEON 乘加标量）

    // FZ[i][2] = F[i][1] + F[i][2] + 4*(F[i][3] + F[i][4])
    // A^T 列 2 = [0,1,1,4,4,0]
    FZ[i][2] = vmlaq_n_f32(
        vaddq_f32(F[i][1], F[i][2]),
        vaddq_f32(F[i][3], F[i][4]),
        4.0f
    );

    // FZ[i][3] = F[i][1] - F[i][2] + 8*(F[i][3] - F[i][4]) + F[i][5]
    // A^T 列 3 = [0,1,-1,8,-8,1]
    FZ[i][3] = vaddq_f32(
        vmlaq_n_f32(
            vsubq_f32(F[i][1], F[i][2]),
            vsubq_f32(F[i][3], F[i][4]),
            8.0f
        ),
        F[i][5]
    );
}

// === 第二步：f = Z^T · FZ（行变换，相同系数）===
for (int j = 0; j < 4; j++) {
    // f[0][j] = FZ[0][j] + FZ[1][j] + FZ[2][j] + FZ[3][j] + FZ[4][j]
    f[0][j] = vaddq_f32(
        vaddq_f32(vaddq_f32(FZ[0][j], FZ[1][j]), vaddq_f32(FZ[2][j], FZ[3][j])),
        FZ[4][j]
    );

    // f[1][j] = FZ[1][j] - FZ[2][j] + 2*(FZ[3][j] - FZ[4][j])
    f[1][j] = vmlaq_n_f32(
        vsubq_f32(FZ[1][j], FZ[2][j]),
        vsubq_f32(FZ[3][j], FZ[4][j]),
        2.0f
    );

    // f[2][j] = FZ[1][j] + FZ[2][j] + 4*(FZ[3][j] + FZ[4][j])
    f[2][j] = vmlaq_n_f32(
        vaddq_f32(FZ[1][j], FZ[2][j]),
        vaddq_f32(FZ[3][j], FZ[4][j]),
        4.0f
    );

    // f[3][j] = FZ[1][j] - FZ[2][j] + 8*(FZ[3][j] - FZ[4][j]) + FZ[5][j]
    f[3][j] = vaddq_f32(
        vmlaq_n_f32(
            vsubq_f32(FZ[1][j], FZ[2][j]),
            vsubq_f32(FZ[3][j], FZ[4][j]),
            8.0f
        ),
        FZ[5][j]
    );
}

// === Bias + ReLU + 存储 4×4 输出 tile ===
// ... 同 F(2,2) 的 bias + clamp 模式
```

### 4.3 优化分析

- **系数 {1, 2, 4, 8}**：A^T 列的系数是 2 的幂，`vmlaq_n_f32` 用立即数乘法高效
- **两步分解**：先 FZ=F·Z（6×6→6×4），再 f=Z^T·FZ（6×4→4×4）
  - 中间矩阵 FZ 是 6×4（24 个 Q 寄存器），刚好不超过 32
- **对称性**：Z^T 和 Z 的行/列系数相同，两步用相同公式
- **嵌套 vadd**：`(a+b) + (c+d)` 而非 `((a+b)+c)+d`，减少依赖链长度

---

## 5. 权重变换 F(2,2,3,3) — `arm_fp32_2x2_3x3.cpp`

### 5.1 数学公式

```
V = G · g · G^T
```

G 矩阵（4×3，F(2,2,3,3) 的权重变换矩阵）：

```
G = [ 1      0      0   ]
    [ 1/2   1/2    1/2  ]
    [ 1/2  -1/2    1/2  ]
    [ 0      0      1   ]
```

### 5.2 4-通道层逐行注释

```cpp
// 加载 3×3 卷积核
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        w[i][j] = vld1q_f32(inptr + i*ld_weight_row + j*ld_weight_col);
    }
}

// === 第一步：Ww = G · g（列变换）===
for (int j = 0; j < 3; j++) {
    // G[0] = [1, 0, 0] → Ww[0][j] = w[0][j]
    Ww[0][j] = w[0][j];

    // G[1] = [1/2, 1/2, 1/2] → Ww[1][j] = 0.5*(w[0][j] + w[1][j] + w[2][j])
    Ww[1][j] = vmulq_n_f32(
        vaddq_f32(vaddq_f32(w[0][j], w[1][j]), w[2][j]),
        0.5f
    );
    // vmulq_n_f32(a, n) = a * n（标量乘法）

    // G[2] = [1/2, -1/2, 1/2] → Ww[2][j] = 0.5*(w[0][j] - w[1][j] + w[2][j])
    Ww[2][j] = vmulq_n_f32(
        vaddq_f32(vsubq_f32(w[0][j], w[1][j]), w[2][j]),
        0.5f
    );

    // G[3] = [0, 0, 1] → Ww[3][j] = w[2][j]
    Ww[3][j] = w[2][j];
}

// === 第二步：V = Ww · G^T（行变换，相同系数）===
for (int i = 0; i < 4; i++) {
    V[i][0] = Ww[i][0];

    V[i][1] = vmulq_n_f32(
        vaddq_f32(vaddq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]),
        0.5f
    );

    V[i][2] = vmulq_n_f32(
        vaddq_f32(vsubq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]),
        0.5f
    );

    V[i][3] = Ww[i][2];
}

// === 存储 4×4 变换后权重 ===
for (int i = 0, m = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++, m++) {
        vst1q_f32(outptr + m*matrix_stride, V[i][j]);
    }
}
```

### 5.3 优化分析

- **G 的系数是 1/2**：只需 `vadd/vsub + vmul_n 0.5`，无复杂乘法
- **行 0 和行 3 是直接复制**：`Ww[0][j] = w[0][j]`，无计算
- **对称性**：G 和 G^T 结构相同，两步用相同公式

---

## 6. 权重变换 F(4,4,3,3) — `arm_fp32_4x4_3x3.cpp`

### 6.1 数学公式

```
V = G · g · G^T / 576
```

G 矩阵（6×3，整数缩放版）：

```
G_scaled = [ 6   0   0]     原始 G 行 0 × 6
           [-4  -4  -4]     原始 G 行 1 × (-4) × 9
           [-4   4  -4]     原始 G 行 2 × (-4) × 9
           [ 1   2   4]     原始 G 行 3 × 90
           [ 1  -2   4]     原始 G 行 4 × 90
           [ 0   0  24]     原始 G 行 5 × 24
```

归一化因子 = 576 = 24²（因为最大缩放系数是 24）。

### 6.2 4-通道层逐行注释

```cpp
// 加载 3×3 卷积核
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        w[i][j] = vld1q_f32(inptr + i*ld_weight_row + j*ld_weight_col);
    }
}

// === 第一步：Ww = G · g（列变换）===
for (int j = 0; j < 3; j++) {
    // G[0] = [6, 0, 0] → Ww[0][j] = 6 * w[0][j]
    Ww[0][j] = vmulq_n_f32(w[0][j], 6.0);

    // G[1] = [-4, -4, -4] → Ww[1][j] = -4 * (w[0][j] + w[1][j] + w[2][j])
    Ww[1][j] = vmulq_n_f32(
        vaddq_f32(vaddq_f32(w[0][j], w[1][j]), w[2][j]),
        -4.0
    );

    // G[2] = [-4, 4, -4] → Ww[2][j] = 4 * (w[1][j] - w[0][j] - w[2][j])
    Ww[2][j] = vmulq_n_f32(
        vsubq_f32(vsubq_f32(w[1][j], w[0][j]), w[2][j]),
        4.0
    );

    // G[3] = [1, 2, 4] → Ww[3][j] = w[0][j] + 2*w[1][j] + 4*w[2][j]
    Ww[3][j] = vmlaq_n_f32(
        vmlaq_n_f32(w[0][j], w[1][j], 2.0f),  // w[0][j] + 2*w[1][j]
        w[2][j], 4.0f                          // + 4*w[2][j]
    );
    // vmlaq_n_f32(a, b, n) = a + b * n

    // G[4] = [1, -2, 4] → Ww[4][j] = w[0][j] - 2*w[1][j] + 4*w[2][j]
    Ww[4][j] = vmlaq_n_f32(
        vmlsq_n_f32(w[0][j], w[1][j], 2.0f),  // w[0][j] - 2*w[1][j]
        w[2][j], 4.0f                          // + 4*w[2][j]
    );
    // vmlsq_n_f32(a, b, n) = a - b * n

    // G[5] = [0, 0, 24] → Ww[5][j] = 24 * w[2][j]
    Ww[5][j] = vmulq_n_f32(w[2][j], 24.0f);
}

// === 第二步：V = Ww · G^T / 576（行变换，相同系数 + 归一化）===
for (int i = 0; i < 6; i++) {
    const float recip576 = 1.0f / 576.0f;
    // 预计算 1/576，用乘法替代除法

    // V[i][0] = 6 * Ww[i][0] / 576
    V[i][0] = vmulq_n_f32(vmulq_n_f32(Ww[i][0], 6.0), recip576);

    // V[i][1] = -4 * (Ww[i][0] + Ww[i][1] + Ww[i][2]) / 576
    V[i][1] = vmulq_n_f32(
        vmulq_n_f32(
            vaddq_f32(vaddq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]),
            -4.0
        ),
        recip576
    );

    // V[i][2] = 4 * (Ww[i][1] - Ww[i][0] - Ww[i][2]) / 576
    V[i][2] = vmulq_n_f32(
        vmulq_n_f32(
            vsubq_f32(vsubq_f32(Ww[i][1], Ww[i][0]), Ww[i][2]),
            4.0
        ),
        recip576
    );

    // V[i][3] = (Ww[i][0] + 2*Ww[i][1] + 4*Ww[i][2]) / 576
    V[i][3] = vmulq_n_f32(
        vmlaq_n_f32(vmlaq_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f),
        recip576
    );

    // V[i][4] = (Ww[i][0] - 2*Ww[i][1] + 4*Ww[i][2]) / 576
    V[i][4] = vmulq_n_f32(
        vmlaq_n_f32(vmlsq_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f),
        recip576
    );

    // V[i][5] = 24 * Ww[i][2] / 576
    V[i][5] = vmulq_n_f32(vmulq_n_f32(Ww[i][2], 24.0f), recip576);
}

// === 存储 6×6 变换后权重 ===
for (int i = 0, m = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++, m++) {
        vst1q_f32(outptr + m*matrix_stride, V[i][j]);
    }
}
```

### 6.3 优化分析

- **整数缩放**：G 的浮点系数（1/2, 1/4, 1/90 等）乘以整数变成整数（6, -4, 1, 24），避免浮点除法
- **1/576 预计算**：用 `1.0f / 576.0f` 预计算倒数，后续用乘法 `vmulq_n_f32` 替代除法
- **vmlaq/vmlsq 链**：`vmlaq_n_f32(vmlaq_n_f32(a, b, 2), c, 4)` = a + 2b + 4c，
  用两条 FMA 指令替代两条乘法 + 两条加法
- **对称性**：G 和 G^T 结构相同，两步用相同公式

---

## 7. NEON Intrinsic 速查表

| Intrinsic | 语义 | 用途 |
|-----------|------|------|
| `vld1q_f32` | 加载 4 个 float 到 Q 寄存器 | 加载输入/权重 |
| `vld1_f32` | 加载 2 个 float 到 D 寄存器 | 加载（2 通道层） |
| `vst1q_f32` | 存储 4 个 float | 存储输出 |
| `vst1_f32` | 存储 2 个 float | 存储（2 通道层） |
| `vaddq_f32` | 4 元素加法 | B^T/A^T 系数 ±1 |
| `vsubq_f32` | 4 元素减法 | B^T/A^T 系数 ±1 |
| `vmulq_n_f32` | 4 元素 × 标量 | G 矩阵整数系数 |
| `vmlaq_n_f32` | 4 元素 += × 标量 | FMA：a + b*n |
| `vmlsq_n_f32` | 4 元素 -= × 标量 | FMS：a - b*n |
| `vmaxq_f32` | 4 元素取最大 | ReLU 上限 |
| `vminq_f32` | 4 元素取最小 | ReLU 下限 |
| `vdupq_n_f32` | 广播标量到 4 元素 | bias=0, min, max |
| `vpfalse_b` | (无) | NEON 无谓词 |

---

## 8. 三层降级对比

| 方面 | 4-通道层 | 2-通道层 | 1-通道层 |
|------|---------|---------|---------|
| 寄存器 | `float32x4_t` (Q) | `float32x2_t` (D) | `float` (S) |
| 加载 | `vld1q_f32` | `vld1_f32` | `*(ptr++)` |
| 存储 | `vst1q_f32` | `vst1_f32` | `*ptr = val` |
| 加法 | `vaddq_f32` | `vadd_f32` | `a + b` |
| 减法 | `vsubq_f32` | `vsub_f32` | `a - b` |
| 乘标量 | `vmulq_n_f32` | `vmul_n_f32` | `a * n` |
| FMA | `vmlaq_n_f32` | `vmla_n_f32` | `a + b * n` |
| FMS | `vmlsq_n_f32` | `vmls_n_f32` | `a - b * n` |
| 指针步进 | `+= 4` | `+= 2` | `+= 1` |
| 后缀 | `q` | 无 | 无（标量） |

**降级规则**：NEON 的 Q 寄存器固定 128-bit = 4 float。
当剩余通道 < 4 时无法填满 Q 寄存器，改用 D（64-bit = 2 float）或标量。
SVE 用谓词掩码避免此问题，只需一份代码。
