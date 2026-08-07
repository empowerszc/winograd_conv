# Winograd 卷积性能分析与优化记录

> 本文档记录完整的硬件信息、ACL/oneDNN 参考性能、本项目优化历程、当前差距分析和下一步优化方向。

---

## 1. 目标硬件

| 项目 | 信息 |
|------|------|
| CPU | 华为鲲鹏 920F（920 Pro） |
| 架构 | Armv9 |
| ISA | NEON + SVE-512 + SME（无 SME2） |
| 核心 | 16 NUMA × 38 cores = 608 cores |
| 编译器 | Clang（AArch64 交叉编译/原生） |
| NUMA 工具 | `numactl -N .. -m ..` |
| 运行方式 | 测试时已使用 `numactl` 绑定 NUMA 节点 |

---

## 2. ACL/oneDNN 参考性能

### oneDNN 算子性能（Case 0: 4×192×40×40, 3×3 conv, stride=1, pad=1）

| 线程数 | oneDNN(ms) |
|--------|-----------|
| 1 | 18.0 |
| 4 | 6.8 |
| 8 | 4.9 |
| 16 | 4.0 |
| 32 | 3.6 |

oneDNN 使用 ACL 的 Winograd 实现：
- 变换：NEON/SVE 手写汇编（完全展开，无分支，指令调度优化）
- GEMM：arm_gemm JIT 编译的 SVE 内核（寄存器阻塞 + 软件流水 + 针对矩阵大小自动调优）
- 数据布局：Winograd 专用布局（matrix_stride 参数，避免 scatter/gather）
- 多线程：NEScheduler 细粒度任务队列

### ACL 版本选择

| ACL 版本 | 性能 | 原因 |
|---------|------|------|
| 23.11 | **更优** | GEMM 选择列表中 SVE kernel 在 SME kernel 之前 |
| 53.1.0 | 较差 | 新增 SME GEMM kernel 拦截 SVE 路径，920F 有 SME 但无 SME2 |

920F 有 SME 但可能没有 SME2，53.1.0 选择了不成熟的 SME GEMM kernel，导致性能下降。23.11 回退到成熟的 SVE GEMM kernel。

---

## 3. 优化历程

### 3.1 各阶段性能数据（Case 0: 4×192×40×40, NHWC, SVE）

| 阶段 | t1(ms) | t8(ms) | t16(ms) | t32(ms) | t38(ms) | vs oneDNN t32 |
|------|--------|--------|---------|---------|---------|-------------|
| ① NCHW 基线 | 22.5 | 14.6 | 14.2 | 14.0 | 13.9 | 3.89x |
| ② +NHWC 布局 | 16.3 | ~14 | ~14 | ~14 | ~14 | ~3.89x |
| ③ +GEMM并行+schedule(2)+raw malloc | 21.6 | 10.3 | 9.5 | 9.1 | 9.0 | 2.53x |
| ④ +合并OpenMP区域 | 18.9 | 7.9 | 7.0 | 6.7 | 6.6 | **1.86x** |
| ⑤ +优化B+C(跳过fill+nowait) | 待测 | 待测 | 待测 | 待测 | 待测 | — |
| oneDNN 参考 | 18.0 | 4.9 | 4.0 | 3.6 | — | 1.0x |

### 3.2 优化 ② 后的细粒度计时（NHWC, SVE, 1 线程）

| 阶段 | Case 0 (192×192) | Case 4 (384×96) | Case 5 (768×96) |
|------|-----------------|----------------|-----------------|
| 权重变换 | 3.18ms (19%) | 0.99ms (2%) | 1.90ms (7%) |
| Tile 提取(NHWC) | 1.10ms (7%) | 9.14ms (17%) | 4.59ms (18%) |
| 输入变换 B^T·d·B | 2.79ms (17%) | 21.71ms (41%) | 10.17ms (39%) |
| Scatter(NEON拷贝) | 0.83ms (5%) | 4.76ms (9%) | 2.44ms (9%) |
| GEMM | 1.99ms (12%) | 7.69ms (15%) | 4.86ms (19%) |
| Gather(NEON拷贝) | 1.42ms (9%) | 1.58ms (3%) | 0.49ms (2%) |
| 输出变换 A^T·M·A | 1.94ms (12%) | 4.48ms (9%) | 1.11ms (4%) |
| 输出写回(NHWC) | 0.40ms (2%) | 1.35ms (3%) | 0.29ms (1%) |

### 3.3 NCHW vs NHWC 对比（Case 0, timing 模式）

| 阶段 | NCHW(ms) | NHWC(ms) | 加速比 |
|------|---------|---------|--------|
| Tile 提取 | 4.20 | 1.10 | **3.8x** |
| 输出写回 | 2.95 | 0.40 | **7.4x** |
| 总计 | ~22 | ~16.3 | **1.35x** |

### 3.4 优化 ③+④ 后的多线程扩展性（所有 Case）

| Case | Shape (N,IC,IH,IW) | (OC,IC) | t1 | t8 | t16 | t32 | t38 | t8加速 |
|------|---------------------|---------|-----|-----|-----|-----|-----|--------|
| 0 | 4,192,40,40 | 192,192 | 18.9 | 7.9 | 7.0 | 6.7 | 6.6 | 2.39x |
| 1 | 4,96,80,80 | 96,96 | 25.3 | 7.4 | 5.9 | 5.3 | 5.0 | 3.42x |
| 2 | 4,48,160,160 | 48,48 | 69.6 | 14.3 | 10.6 | 8.9 | 8.2 | 4.87x |
| 3 | 4,192,20,20 | 192,192 | 8.6 | 2.9 | 2.3 | 2.3 | 2.4 | 2.96x |
| 4 | 4,384,80,80 | 96,384 | 54.7 | 12.5 | 8.2 | 6.3 | 5.5 | 4.38x |
| 5 | 4,768,40,40 | 96,768 | 32.3 | 8.4 | 6.3 | 5.6 | 5.3 | 3.84x |

---

## 4. 已实施优化清单

| # | 优化 | 描述 | 效果 |
|---|------|------|------|
| 1 | NHWC 布局 | `Layout::NHWC`，tile 提取用 `vld1q_f32` NEON 连续加载 | TileExt 3.8x, OutWrite 7.4x |
| 2 | GEMM 并行 | `#pragma omp for schedule(dynamic)` 并行 36 个 GEMM | GEMM 从串行→并行 |
| 3 | 合并 OpenMP 区域 | 3 个 `#pragma omp parallel` → 1 个，3 阶段 3 个 `#pragma omp for` | fork/join 12→4/batch |
| 4 | `schedule(dynamic, 2)` | 每 2 tile 一批 | 调度开销减半 |
| 5 | `thread_local static float*` + `malloc` | 替代 `std::vector::resize()` | 消除 resize 开销 |
| 6 | 内部 tile 跳过 `memset` | 64% tiles 无 padding | 清零开销 → 0 |
| 7 | 预计算有效行列范围 | `ti_start/ti_end` 消除 `if` 分支 | 内层循环无分支 |
| 8 | 输出 `nowait` | 消除最后阶段的多余 barrier | -1 barrier/batch |
| 9 | OpenBLAS `openblas_set_num_threads(1)` | 避免 GEMM 线程与 OpenMP 冲突 | 消除 double free |

---

## 5. 剩余差距分析

### 5.1 Case 0 分解（t32 = 6.7ms vs oneDNN 3.6ms = 1.86x）

串行 timing（1 线程）各阶段时间 → 32 线程理论时间：

| 阶段 | 串行(ms) | 32线程理论(ms) | 说明 |
|------|---------|-------------|------|
| 权重变换 | 3.18 | 3.18 (串行) | 不随线程数变化 |
| Tile 提取 | 1.10 | ~0.03 | 100 tiles / 32 threads |
| 输入变换 | 2.79 | ~0.09 | 并行 |
| Scatter | 0.83 | ~0.03 | 并行 |
| GEMM | 1.99 | ~0.06 | 36 GEMMs / 32 threads |
| Gather | 1.42 | ~0.04 | 并行 |
| 输出变换 | 1.94 | ~0.06 | 并行 |
| 输出写回 | 0.40 | ~0.01 | 并行 |
| **理论合计** | 12.65 | **3.50** | |
| **实际 t32** | | **6.70** | |
| **未解释开销** | | **3.20** | barrier + 缓冲区分配 + NUMA + 其他 |

### 5.2 差距来源排序

| 排名 | 来源 | 预估影响(t32) | 占比 | 可解决性 |
|------|------|-------------|------|---------|
| 1 | 权重变换（串行） | 3.18ms | 47% | 用 ACL 手写公式优化 |
| 2 | OpenMP barrier | ~0.8ms | 12% | 重构数据布局为 U[tile][ts][ic] |
| 3 | 每线程缓冲区分配 | ~0.5ms | 7% | 栈/mempool |
| 4 | GEMM 内核质量 | ~0.5-1ms | 7-15% | 换 arm_gemm JIT |
| 5 | NUMA 远程访问 | 不可量化 | — | 已用 numactl |
| 6 | 变换实现效率 | ~0.3ms | 4% | 用 ACL 汇编式变换 |

### 5.3 多线程扩展性分析

| Case | Tiles | t8加速比 | 8→16提升 | 16→32提升 | 扩展性 |
|------|-------|---------|---------|---------|--------|
| 0 | 100 | 2.39x | 1.13x | 1.04x | 差（tile 少，barrier 占比大） |
| 1 | 400 | 3.42x | 0.80x | 0.89x | 中 |
| 2 | 1600 | 4.87x | 1.38x | 1.23x | 好（tile 多） |
| 3 | 25 | 2.96x | 0.81x | 1.88x↓ | 差（tile 太少） |
| 4 | 400 | 4.38x | 1.50x | 1.28x | 好（IC=384, 每 tile 工作大） |
| 5 | 100 | 3.84x | 1.30x | 1.12x | 中（IC=768 补偿 tile 少） |

**关键发现**：
- tile 数 > 400 时扩展性好（Case 1, 2, 4）
- tile 数 ≤ 100 时 8 线程后停滞（Case 0, 3, 5）
- IC 大的 case 即使 tile 少也能扩展（Case 5: IC=768）

---

## 6. 下一步优化方向

### 6.1 权重变换优化（预期 -1.5ms, 占 t32 的 22%）

**当前实现**：`weight_transform_neon<6>()` 模板循环 + NEON intrinsics

**ACL 实现**（见 `docs/acl_reference/acl_wino_neon_intrinsics_annotated.md`）：
```cpp
// ACL 的 arm_fp32_4x4_3x3.cpp 直接展开每个 G 矩阵行：
Ww[0][j] = vmulq_n_f32(w[0][j], 6.0);           // G[0]=[6,0,0]
Ww[1][j] = vmulq_n_f32(vaddq_f32(...), -4.0);    // G[1]=[-4,-4,-4]
Ww[2][j] = vmulq_n_f32(vsubq_f32(...), 4.0);     // G[2]=[-4,4,-4]
// ... 直接写 V，跳过 Ww 中间缓冲区
```

**优化方案**：
- 用 ACL 公式直接展开 G 矩阵行，跳过 `Ww` 中间缓冲区
- 权重变换并行化（`#pragma omp for` over OC）
- 跳过 per-OC 的 `std::vector g` 分配（预分配连续 g 数组）

### 6.2 重构数据布局消除 barrier（预期 -0.8ms）

**当前**：`U[ts_idx][tile_idx][ic]`，每个 GEMM 需要所有 tile 的 U → barrier 必须

**重构**：`U[tile_idx][ts_idx][ic]`，每个 tile 的 GEMM 独立 → 无 barrier

```cpp
// 重构后：每 tile 独立 pipeline，无 barrier
#pragma omp parallel for schedule(dynamic, 2)
for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
    input_transform(tile_idx);   // U[tile][ts][ic]
    gemm(tile_idx);              // M[tile][oc] = sum_ts U[tile][ts][ic] * V[ts][oc][ic]
    output_transform(tile_idx);  // f[tile][oc]
}
// 0 barriers per batch!
```

**代价**：GEMM 形状从 (n_tiles×OC×IC) 变为 (36×OC×IC)，矩阵更小，OpenBLAS 可能效率降低。需要测试。

### 6.3 GEMM 内核替换（预期 -0.5ms）

OpenBLAS 对小矩阵（100×192×192）可能不如 arm_gemm JIT。可编译 arm_gemm 库替换。

### 6.4 变换实现优化（预期 -0.3ms）

当前用泛型模板 `transform_1d_neon<>` + `transform_2d_neon<>`。ACL 用手写汇编完全展开。

参考 `docs/acl_reference/acl_wino_sve_asm_annotated.md` 中 ACL 的 SVE 汇编实现。

### 6.5 优先级排序

| 优化 | 预期收益(t32) | 难度 | 依赖 |
|------|-------------|------|------|
| 权重变换手写公式 + 并行 | -1.5ms (22%) | 中 | ACL 参考文档 |
| 重构数据布局消除 barrier | -0.8ms (12%) | 大 | 改变 GEMM 形状 |
| arm_gemm 替换 OpenBLAS | -0.5ms (7%) | 大 | 编译 arm_gemm |
| 变换汇编化 | -0.3ms (4%) | 大 | ACL SVE 汇编参考 |
| **合计** | **-3.1ms** | | t32: 6.7→3.6ms ≈ oneDNN |

---

## 7. 测试 shape 列表

以下 shape 从用户提供的 CSV 中筛选（stride=1, group=1, 3×3, pad=1）：

| # | Input (N,IC,IH,IW) | Weight (OC,IC,3,3) | Count | Tiles |
|---|---------------------|---------------------|-------|-------|
| 0 | 4,192,40,40 | 192,192 | 24 | 100 |
| 1 | 4,96,80,80 | 96,96 | 17 | 400 |
| 2 | 4,48,160,160 | 48,48 | 8 | 1600 |
| 3 | 4,192,20,20 | 192,192 | 16 | 25 |
| 4 | 4,384,80,80 | 96,384 | 1 | 400 |
| 5 | 4,768,40,40 | 96,768 | 1 | 100 |
