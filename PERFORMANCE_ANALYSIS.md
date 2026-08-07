# 性能分析与优化进展

## 当前性能 vs oneDNN（Case 0: 4×192×40×40）

| 线程数 | 本项目(ms) | oneDNN(ms) | 差距 |
|--------|-----------|-----------|------|
| 1 | 21.6 | 18.0 | 1.20x |
| 8 | 10.3 | 4.9 | 2.10x |
| 16 | 9.5 | 4.0 | 2.38x |
| 32 | 9.1 | 3.6 | 2.53x |

**1 线程差距小（1.2x），但多线程后差距拉大（2.5x）**。oneDNN 多线程扩展性更好。

## 各 Case 多线程扩展性分析

| Case | Shape | Tiles | t1 | t8 | t8加速 | t16/t8 | t32/t16 |
|------|-------|-------|-----|-----|--------|--------|---------|
| 0 | 4×192×40×40 | 100 | 21.6 | 10.3 | 2.10x | 1.09x | 1.04x |
| 1 | 4×96×80×80 | 400 | 25.4 | 7.5 | 3.39x | 0.80x | 0.89x |
| 2 | 4×48×160×160 | 1600 | 58.9 | 11.6 | 5.08x | 1.38x | 1.23x |
| 3 | 4×192×20×20 | 25 | 8.3 | 2.5 | 3.34x | 0.81x | 1.88x↓ |
| 4 | 4×384×80×80 | 400 | 55.7 | 12.9 | 4.33x | 1.50x | 1.28x |
| 5 | 4×768×40×40 | 100 | 32.7 | 9.1 | 3.59x | 1.30x | 1.12x |

### 关键发现

- **Case 2（1600 tiles）扩展最好**：t8=5.08x，t32 仍在提升
- **Case 3（25 tiles）8线程后变差**：tile 太少，线程调度开销 > 计算
- **Case 4（400 tiles, IC=384）扩展好**：每 tile 工作量大，并行效率高
- **Case 0（100 tiles, IC=192）扩展差**：tile 数和每 tile 工作量都不够大

## 差距来源分析

### 1. GEMM 内核质量（最大差距来源）

oneDNN 用 **arm_gemm JIT** 编译的 SVE GEMM 内核，针对具体矩阵大小自动生成最优指令序列。OpenBLAS 的 GEMM 内核是通用的，可能没有针对这些大小优化。

每个 GEMM 是 `n_tiles × OC × IC` 的小矩阵乘（如 100×192×192）。arm_gemm JIT 会对这种小矩阵做寄存器阻塞 + 软件流水，而 OpenBLAS 可能对大矩阵更优化。

**预期差距**：GEMM 部分可能慢 2-3x。

### 2. 变换实现效率

我们的 `transform_2d` 用泛型模板 + NEON/SVE intrinsics。ACL 用手写汇编，完全展开，无分支。

| 方面 | 本项目 | ACL/oneDNN |
|------|--------|-----------|
| 1D 变换 | 模板循环 + NEON | 手写汇编，完全展开 |
| 2D tmp | thread_local malloc | 栈上 alloca 或预分配 |
| tile 提取 | NEON vld1q 循环 | 内联到变换中 |
| scatter/gather | 独立 NEON 拷贝 | 合并到变换中 |

### 3. OpenMP 并行效率

每个 batch 有 **3 个 OpenMP 并行区域**（输入变换、GEMM、输出变换），每个有 fork/join 开销。4 batches = 12 次 fork/join。

oneDNN 用 NEScheduler 做细粒度任务队列，避免重复 fork/join。

### 4. NUMA 效应（920F 16 NUMA）

32 线程可能跨 NUMA 分布。`thread_local` 内存在一个 NUMA 分配，其他 NUMA 的线程访问时延迟高。

## 进一步优化建议

### 立即可做（无需改代码）

```bash
# 1. NUMA 交织：让内存均匀分布到所有 NUMA 节点
numactl --interleave=all ./bench_winograd --sve --nhwc --threads 32 shapes.csv

# 2. 线程绑定：防止线程迁移跨 NUMA
OMP_PROC_BIND=spread OMP_PLACES=cores ./bench_winograd --sve --nhwc --threads 32 shapes.csv

# 3. 两者结合
numactl --interleave=all env OMP_PROC_BIND=spread OMP_PLACES=cores ./bench_winograd --sve --nhwc --threads 32 shapes.csv
```

### 代码优化

#### 优化 A：合并 3 个 OpenMP 区域为 1 个

当前每 batch 有 3 次 fork/join（输入→GEMM→输出），改为 1 次：

```cpp
#pragma omp parallel
{
    // per-thread buffers
    #pragma omp single
    { /* weight transform (serial, 1 thread) */ }

    // Phase 1: all threads do input transform
    #pragma omp for
    for (tiles...) { input_transform(); }

    // Phase 2: all threads do GEMM (no barrier needed if #pragma omp for)
    #pragma omp for
    for (36 GEMMs...) { gemm(); }

    // Phase 3: all threads do output transform
    #pragma omp for
    for (tiles...) { output_transform(); }
}  // 1 次 fork/join per batch
```

减少 fork/join 从 12 次/batch → 4 次/batch。

#### 优化 B：GEMM 混合并行（OpenMP + OpenBLAS 多线程）

当前 OpenBLAS 单线程，36 个 GEMM 分到 N 线程。当 N>36 时有线程空闲。

改为：让 OpenBLAS 用 K 线程，OpenMP 分 36/K 个 GEMM：

```cpp
int nt = omp_get_max_threads();
int blas_threads = std::max(1, nt / 4);  // OpenBLAS 用总线程的 1/4
int outer_parallel = std::max(1, 36 / blas_threads);

openblas_set_num_threads(blas_threads);
#pragma omp parallel for num_threads(outer_parallel) schedule(dynamic)
for (int ts_idx = 0; ts_idx < 36; ts_idx++) {
    winograd_gemm(...);  // OpenBLAS 内部用 blas_threads 线程
}
openblas_set_num_threads(1);
```

但这有风险：OpenBLAS 内部线程可能与其他 OpenMP 线程冲突。需要测试。

#### 优化 C：用 arm_gemm 替换 OpenBLAS

oneDNN 用 arm_gemm 的 JIT 内核，针对 AArch64 SVE 优化。可以：

1. 编译 arm_gemm 库（ACL 的子库）
2. 替换 `cblas_sgemm` 为 `arm_gemm` 调用
3. 或直接用 oneDNN 的 GEMM 内核

这是最接近 oneDNN 性能的方式，但实现复杂度最高。

#### 优化 D：pipeline 重叠

当前是严格串行的 3 阶段：
```
输入变换(tile 0..N) → GEMM(ts 0..35) → 输出变换(tile 0..N)
```

可以重叠：
```
输入变换(tile 0..K) → GEMM(ts 0) + 输入变换(tile K+1..2K) → GEMM(ts 1) + ...
```

用 `#pragma omp task` + `taskwait` 实现生产者-消费者流水线。实现复杂但可以隐藏延迟。

### 优化优先级

| 优化 | 预期收益 | 难度 | 立即可做 |
|------|---------|------|---------|
| numactl + OMP_PROC_BIND | 10-20% | 零 | ✓ |
| 合并 OpenMP 区域 | 5-10% | 小 | |
| GEMM 混合并行 | 10-30%（需测试） | 中 | |
| arm_gemm 替换 OpenBLAS | 50-100%（GEMM 部分） | 大 | |
| pipeline 重叠 | 20-40% | 大 | |
