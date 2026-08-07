# 性能分析与优化进展

## 优化历程（Case 0: 4×192×40×40, NHWC, SVE）

| 优化阶段 | t1(ms) | t8(ms) | t16(ms) | t32(ms) | vs oneDNN t32 |
|---------|--------|--------|---------|---------|-------------|
| NCHW 基线 | 22.5 | 14.6 | 14.2 | 14.0 | 3.89x |
| +NHWC 布局 | 16.3 | ~14 | ~14 | ~14 | ~3.89x |
| +GEMM 并行 + schedule(2) + raw malloc | 21.6 | 10.3 | 9.5 | 9.1 | 2.53x |
| +合并 OpenMP 区域 | 18.9 | 7.9 | 7.0 | 6.7 | **1.86x** |
| oneDNN 参考 | 18.0 | 4.9 | 4.0 | 3.6 | 1.0x |

**合并 OpenMP 区域效果**：t8 从 10.3→7.9ms（23%），t32 从 9.1→6.7ms（26%）。

## 当前性能数据（合并 OpenMP 区域后）

| Case | Shape | t1 | t8 | t16 | t32 | t38 | t8加速 |
|------|-------|-----|-----|-----|-----|-----|--------|
| 0 | 4×192×40×40 | 18.9 | 7.9 | 7.0 | 6.7 | 6.6 | 2.39x |
| 1 | 4×96×80×80 | 25.3 | 7.4 | 5.9 | 5.3 | 5.0 | 3.42x |
| 2 | 4×48×160×160 | 69.6 | 14.3 | 10.6 | 8.9 | 8.2 | 4.87x |
| 3 | 4×192×20×20 | 8.6 | 2.9 | 2.3 | 2.3 | 2.4 | 2.96x |
| 4 | 4×384×80×80 | 54.7 | 12.5 | 8.2 | 6.3 | 5.5 | 4.38x |
| 5 | 4×768×40×40 | 32.3 | 8.4 | 6.3 | 5.6 | 5.3 | 3.84x |

## 剩余差距分析

### Case 0 分解（t32 = 6.7ms vs oneDNN 3.6ms = 1.86x 差距）

串行 timing（1 线程）显示：
- 权重变换：3.18ms（串行，不随线程数变化）
- GEMM：1.99ms（已并行，32 线程时 ~0.06ms）
- 输入变换：2.79ms（已并行，32 线程时 ~0.09ms）
- 输出变换：1.94ms（已并行，32 线程时 ~0.06ms）
- Scatter/Gather：2.25ms（已并行，32 线程时 ~0.07ms）
- Tile 提取/写回：1.50ms（已并行，32 线程时 ~0.05ms）

**理论 32 线程时间**：3.18 + 0.06 + 0.09 + 0.06 + 0.07 + 0.05 + barrier开销 ≈ 3.51 + overhead
**实际 32 线程时间**：6.7ms
**未解释开销**：~3.2ms

### 差距来源排序

| 来源 | 预估影响 | 可解决性 |
|------|---------|---------|
| 1. 权重变换（3.18ms 串行） | 47% of t32 | 优化变换实现（模板→手写） |
| 2. OpenMP barrier 开销 | ~0.8ms | 用 task 代替 for |
| 3. GEMM 内核质量 | ~0.5-1ms | 换 arm_gemm |
| 4. 每线程缓冲区分配 | ~0.5ms（首次） | 用栈/mempool |
| 5. NUMA 远程访问 | 不可量化 | numactl |
| 6. std::fill(d_tile, 0) | ~0.1ms | 只清零 padding 行 |

## 下一步优化建议

### 立即尝试（不改代码）

```bash
# NUMA 交织 + 线程绑定
numactl --interleave=all env OMP_PROC_BIND=spread OMP_PLACES=cores \
  ./bench_winograd --sve --nhwc --threads 32 shapes.csv

# 单 NUMA 绑定（38 核）
numactl --cpunodebind=0 --membind=0 \
  ./bench_winograd --sve --nhwc --threads 38 shapes.csv
```

### 代码优化（按收益排序）

#### A. 权重变换优化（预期 -1.5ms, 22%）

当前权重变换用 `weight_transform_neon<6>()` 模板 + NEON intrinsics。ACL 用手写汇编完全展开。

当前每次 OC 循环做一次 `dispatch_weight_transform`，内部有 NEON 4 通道 + scalar tail 两条路径。可以：
- 改用 ACL 的 `arm_fp32_4x4_3x3.cpp` 的手写公式（直接展开每个 G 矩阵行）
- 跳过中间 `Ww` 缓冲区，直接写 V

#### B. 去掉 `std::fill(d_tile, 0)` 全量清零（预期 -0.3ms）

当前每 tile 清零整个 `d_tile`（6×6×IC = 6912 floats）。实际只有 padding 行/列需要清零（第 0 行、第 5 行、第 0 列、第 5 列）。可以只清零 padding 部分。

#### C. 用 `#pragma omp task` 替代 `#pragma omp for`（预期 -0.5ms）

当前 2 个 barrier（输入→GEMM, GEMM→输出）强制所有线程同步。用 task 可以让先完成的线程开始下一阶段：
```
输入变换(tile 0..K) → GEMM(ts 0..K) 重叠 输入变换(tile K+1..N)
```

#### D. GEMM 内核优化（预期 -0.5ms, 需换库）

OpenBLAS 对小矩阵（100×192×192）可能不如 arm_gemm JIT。但实现成本高。
