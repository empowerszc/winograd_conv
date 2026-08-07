# Winograd 卷积优化分析

> 基于 NHWC 布局 + SVE + OpenBLAS + OpenMP 的实测数据分析

## 1. 当前性能数据

### 标准模式（NHWC, SVE, OpenMP + OpenBLAS 单线程）

| Case | Shape (N,IC,IH,IW) | (OC,IC) | t1(ms) | t8(ms) | t16(ms) | t32(ms) | t38(ms) | t8加速比 |
|------|---------------------|---------|--------|--------|---------|---------|---------|---------|
| 0 | (4,192,40,40) | (192,192) | 22.5 | 14.6 | 14.2 | 14.0 | 13.9 | 1.54x |
| 1 | (4,96,80,80) | (96,96) | 32.0 | 16.4 | 15.4 | 14.9 | 14.8 | 1.95x |
| 2 | (4,48,160,160) | (48,48) | 61.6 | 26.0 | 24.0 | 21.7 | 21.5 | 2.37x |
| 3 | (4,192,20,20) | (192,192) | 8.4 | 6.5 | 6.4 | 6.2 | 6.2 | 1.30x |
| 4 | (4,384,80,80) | (96,384) | 79.6 | 39.1 | 36.2 | 34.9 | 34.7 | 2.04x |
| 5 | (4,768,40,40) | (96,768) | 43.0 | 25.0 | 23.7 | 23.2 | 23.1 | 1.72x |

### 观察

- 8→16 线程几乎无提升（Case 0: 14.6→14.2ms, Case 1: 16.4→15.4ms）
- 16→32→38 线程微弱提升
- 最大加速比仅 2.37x（Case 2），远低于 8 线程的理论上限

### 串行细粒度计时（NHWC, SVE, 1 线程, timing 模式）

| 阶段 | Case 0 | Case 4 | Case 5 |
|------|--------|--------|--------|
| Weight 变换 | 3.18ms (19%) | 0.99ms (2%) | 1.90ms (7%) |
| Tile 提取 | 1.10ms (7%) | 9.14ms (17%) | 4.59ms (18%) |
| 输入变换 | 2.79ms (17%) | 21.71ms (41%) | 10.17ms (39%) |
| Scatter | 0.83ms (5%) | 4.76ms (9%) | 2.44ms (9%) |
| GEMM | 1.99ms (12%) | 7.69ms (15%) | 4.86ms (19%) |
| Gather | 1.42ms (9%) | 1.58ms (3%) | 0.49ms (2%) |
| 输出变换 | 1.94ms (12%) | 4.48ms (9%) | 1.11ms (4%) |
| 输出写回 | 0.40ms (2%) | 1.35ms (3%) | 0.29ms (1%) |

---

## 2. 为什么 8 线程后加速停滞

### 2.1 OpenMP 只并行了 tile 循环

当前 `#pragma omp parallel` 只在输入/输出变换的 tile 循环上。GEMM 是串行的（OpenBLAS 设为单线程）。

```
输入变换 (并行) → GEMM (串行!) → 输出变换 (并行)
```

根据 Amdahl 定律：

| Case | 并行部分(t1) | 串行GEMM(t1) | 并行占比 | 8线程理论上限 |
|------|-------------|-------------|---------|-------------|
| 0 | 20.5ms | 2.0ms | 91% | 4.6ms |
| 4 | 72.0ms | 7.7ms | 90% | 16.7ms |
| 5 | 39.0ms | 4.9ms | 88% | 9.8ms |

理论上 8 线程应该加速到 ~5ms（Case 0），但实际是 14.6ms。说明 **OpenMP 并行效果远不如理论**。

### 2.2 并行效果不佳的原因

**原因 A：每 tile 工作量太小**

- Case 0: 100 tiles, IC=192 → 每 tile ~2μs 工作
- OpenMP 线程调度 + cache coherence 开销可能与之相当
- `schedule(dynamic)` 有任务窃取开销

**原因 B：`thread_local static std::vector tmp` 在 transform_2d 内部**

```cpp
// transform_2d_neon / transform_2d_sve 中：
thread_local static std::vector<float> tmp;
tmp.resize(OUT_SIZE * IN_SIZE * channels);
```

- 首次调用：堆分配 + 清零（27KB for IC=192）
- 后续调用：`resize()` 检查大小，即使不变也有函数调用开销
- `thread_local` 在 OpenMP 线程上的初始化/销毁有额外开销

**原因 C：GEMM 串行是硬瓶颈**

GEMM 占总时间的 12-19%，在 8 线程后并行部分缩短，GEMM 占比增大。但由于 OpenBLAS 被设为单线程，无法通过增加线程改善。

**原因 D：内存带宽饱和**

即使 NHWC 使 tile 提取变成了连续访问，16 个线程同时读取不同 tile 的数据，仍然会竞争内存带宽。920F 有 16 NUMA，跨 NUMA 访问延迟高。

---

## 3. 优化建议详细展开

### 优化 1：prepare/execute 分离（预期 -19%~-2%）

**问题**：每次调用 `winograd_convolution()` 都重新做权重变换。

```
当前流程（每次调用）：
  权重变换(3.18ms) → 输入变换 → GEMM → 输出变换
                      ↑ 这部分才是 per-batch 的工作

应该的流程：
  prepare: 权重变换（一次性）
  execute: 输入变换 → GEMM → 输出变换（每次调用）
```

**实现方案**：

```cpp
struct WinogradContext {
    std::vector<float> V;          // 预变换的权重
    int OC, IC, TS;
    ISALevel isa;
    Layout layout;
};

WinogradContext prepare(const float* wei, int OC, int IC,
                        bool is_f44, ISALevel isa) {
    WinogradContext ctx;
    ctx.OC = OC; ctx.IC = IC;
    ctx.TS = is_f44 ? 6 : 4;
    ctx.isa = isa;
    // 权重变换：V = G * g * G^T / norm
    int TS = ctx.TS;
    ctx.V.resize(TS * TS * OC * IC, 0.0f);
    for (int oc = 0; oc < OC; oc++) {
        std::vector<float> g(9 * IC);
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    g[(kh * 3 + kw) * IC + ic] =
                        wei[((oc * IC + ic) * 3 + kh) * 3 + kw];
        std::vector<float> V_oc(TS * TS * IC);
        dispatch_weight_transform(g.data(), V_oc.data(), IC, is_f44, isa);
        for (int m = 0; m < TS * TS; m++)
            for (int ic = 0; ic < IC; ic++)
                ctx.V[m * OC * IC + oc * IC + ic] = V_oc[m * IC + ic];
    }
    return ctx;
}

void execute(WinogradContext& ctx,
             const float* src, const float* bias, float* dst,
             int N, int IH, int IW) {
    // 直接用 ctx.V，跳过权重变换
    // 输入变换 → GEMM → 输出变换
}
```

**收益**：
- Case 0: -3.18ms（19% → 0%）
- Case 5: -1.90ms（7% → 0%）
- 大 IC 的 case 收益最大

### 优化 2：GEMM 并行化（预期 GEMM 部分加速到接近 0）

**问题**：GEMM 循环是串行的（OpenBLAS 设为单线程避免冲突）。

```cpp
// 当前（串行）
for (int ts_idx = 0; ts_idx < NM; ts_idx++) {  // 36 次 GEMM
    winograd_gemm(U_slice, V_slice, M_slice, n_tiles, OC, IC);
}
```

**方案 A：GEMM 循环加 `#pragma omp for`**

```cpp
#pragma omp parallel for schedule(dynamic)
for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
    const float* U_slice = U.data() + ts_idx * n_tiles * IC;
    const float* V_slice = V.data() + ts_idx * OC * IC;
    float* M_slice = M_buf.data() + ts_idx * n_tiles * OC;
    winograd_gemm(U_slice, V_slice, M_slice, n_tiles, OC, IC);
}
```

每次 GEMM 是独立的（不同 ts_idx），不冲突。36 个 GEMM 分布到 16 线程。

**方案 B：OpenBLAS 多线程（需要在 GEMM 循环外用）**

```cpp
// 在 GEMM 循环前设多线程，循环后设回单线程
openblas_set_num_threads(num_threads);
for (int ts_idx = 0; ts_idx < NM; ts_idx++) {
    winograd_gemm(...);  // OpenBLAS 内部多线程
}
openblas_set_num_threads(1);  // 恢复单线程（避免与 tile 并行冲突）
```

但方案 B 有风险：OpenBLAS 线程与 tile 循环的 OpenMP 线程可能冲突。方案 A 更安全。

**收益**：
- Case 0: GEMM 1.99ms → ~0.13ms（16 线程）
- Case 4: GEMM 7.69ms → ~0.48ms
- Case 5: GEMM 4.86ms → ~0.30ms

### 优化 3：合并 scatter/gather 到变换（预期 -5%~-9%）

**问题**：变换后先写到 `U_tile`，再 scatter 到 `U`。多了一次拷贝。

```
当前：d_tile → [transform_2d] → U_tile → [scatter copy] → U
优化：d_tile → [transform_2d, 直接写 U] → U
```

**实现方案**：修改 `transform_2d` 接受输出指针 + stride，直接写到目标位置：

```cpp
// 当前
dispatch_input_transform(d_tile.data(), U_tile.data(), IC, is_f44, isa);
// 然后 scatter U_tile → U（36 * IC 次拷贝）

// 优化后：直接 transform 到 U 的正确位置
// U[ts_idx][tile_idx][ic]，stride = n_tiles * IC
// 需要修改 transform_2d 支持 non-trivial output stride
```

这需要修改 `transform_2d` 的接口，让输出 stride ≠ channel_stride。改动较大但收益明确。

**收益**：
- Case 0: -0.83ms（scatter）-1.42ms（gather）= -2.25ms
- Case 4: -4.76ms -1.58ms = -6.34ms

### 优化 4：消除 `thread_local static std::vector tmp`（预期 -2%~-5%）

**问题**：`transform_2d` 内部用 `thread_local static std::vector` 做 tmp 缓冲区。

```cpp
// 当前
thread_local static std::vector<float> tmp;
tmp.resize(OUT_SIZE * IN_SIZE * channels);
```

每次调用 `resize()` 即使不重新分配也有开销（检查容量、设置 size）。

**方案**：用 C `malloc`/`free` 或 `alloca` 替代，或传入外部预分配 buffer：

```cpp
// 方案 A：传入 buffer
template <int OUT_SIZE, int IN_SIZE>
void transform_2d_neon(..., float* tmp_buffer) {
    // 直接用 tmp_buffer，不分配
}

// 方案 B：用 raw malloc（比 std::vector 快）
static thread_local float* tmp = nullptr;
static thread_local size_t tmp_size = 0;
if (tmp_size < needed) {
    free(tmp);
    tmp = (float*)malloc(needed * sizeof(float));
    tmp_size = needed;
}
```

### 优化 5：增大 tile 工作量以改善 OpenMP 效率

**问题**：每 tile ~2μs，OpenMP 调度开销相对较大。

**方案**：每个 OpenMP 迭代处理多个 tile：

```cpp
#pragma omp parallel
{
    // per-thread buffers
    #pragma omp for schedule(dynamic, 4)  // 每次处理 4 个 tile
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int tr = tile_idx / n_tile_cols;
        int tc = tile_idx % n_tile_cols;
        // ... 处理 tile
    }
}
```

`schedule(dynamic, 4)` 让每次分配 4 个 tile 给一个线程，减少调度开销 4 倍。

---

## 4. 综合预期收益

| 优化 | Case 0 预期 | Case 4 预期 | 难度 |
|------|------------|------------|------|
| prepare/execute | -3.18ms (19%) | -0.99ms (2%) | 中 |
| GEMM 并行 | -1.86ms (12%) | -7.21ms (15%) | 小 |
| 合并 scatter/gather | -2.25ms (14%) | -6.34ms (12%) | 大 |
| 去掉 thread_local | -0.5ms (3%) | -1.0ms (2%) | 小 |
| tile 批处理 | -1.0ms (6%) | -3.0ms (6%) | 小 |
| **合计** | **-8.8ms (54%)** | **-18.5ms (35%)** | |

实施优化 1+2+5（难度低）后的预期：

| Case | 当前 t8 | 优化后 t8 预期 | 加速比 |
|------|--------|-------------|--------|
| 0 | 14.6ms | ~7ms | 2x |
| 4 | 39.1ms | ~18ms | 2x |
| 5 | 25.0ms | ~12ms | 2x |

---

## 5. 更多线程时性能更优的机会

当前 8→16→32 线程提升停滞的原因：

1. **GEMM 串行**：解决后，更多线程可以并行更多 GEMM（36 个 GEMM 分布到 32 线程）
2. **tile 工作量太小**：用 `schedule(dynamic, 4)` 批处理后，每批 ~8μs，线程调度开销占比下降
3. **NUMA 效应**：920F 有 16 NUMA，线程应绑定到数据所在 NUMA（`numactl --interleave=all` 或 `omp_set_affinity`）
4. **内存带宽**：NHWC 已优化为连续访问，但 16+ 线程同时读不同 tile 仍可能饱和带宽。可以用 `sched_affinity` 让线程访问本地 NUMA 的数据

### 推荐的运行方式

```bash
# NUMA 交织（让内存均匀分布在所有 NUMA 节点）
numactl --interleave=all ./bench_winograd --sve --nhwc --threads 32 shapes.csv

# 或绑定到单个 NUMA（38 核，减少跨 NUMA 访问）
numactl --cpunodebind=0 --membind=0 ./bench_winograd --sve --nhwc --threads 38 shapes.csv
```
