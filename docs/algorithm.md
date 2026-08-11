# Winograd 卷积算法详解

> 本文档解释本仓库 `winograd_conv` 的**当前实现**：Winograd 数学原理、三步变换、数据布局、缓冲与内存策略、OpenMP 并行结构、ISA 内核实现、GEMM 选择、正确性与数值精度。
> 面向想深入理解实现细节的读者；运行与使用说明见 `README.md`，性能历程见 `PERFORMANCE_ANALYSIS.md`，AI 协作约束见 `AGENTS.md`。

---

## 1. 一句话总览

把「每个输出像素 9 次乘法（3×3 kernel）」的直卷，换成 **3 个小的线性变换 + 36 个更小的 GEMM**：

```
直接卷积:  dst[oh][ow] = Σ_{ic,kh,kw} src[oh+kh-1][ow+kw-1][ic] · wei[oc][ic][kh][kw]
                             每像素 9 次乘法（F(4,4,3,3) 输出 16 像素时 9×16=144 次）

Winograd:  V = G·g·Gᵀ/576            （权重变换，一次性）
           U = Bᵀ·d·B                （输入变换，每 tile）
           M = U ⊙ V 按通道收缩       （GEMM，每 Winograd 域元素）
           f = Aᵀ·M·A + bias + ReLU  （输出变换，每 tile）
                             每像素等效 ~2.25 次乘法（但多了变换运算）
```

Winograd 的收益只在「乘法省下来的钱 > 变换增加的钱」时成立——这正是本项目性能特征分析的基础。

---

## 2. Winograd 数学原理

### 2.1 一维出发点

Winograd 最小滤波算法（Lavin & Fast, 2016）把「长度 r=3 的滤波器与长度 m 的输入段做相关」变成「对 m+r-1 个点上做逐点乘法」：

给定 m+r-1 个采样点 α₀…α_{m+r-2}，用拉格朗日插值表示卷积多项式：

```
F(m, r): 一次 1D 卷积 = Bᵀ（输入变换）→ 逐点乘（m+r-1 次）→ Aᵀ（输出变换）
```

- **乘法数**：直接法 m×r 次；Winograd 只需 m+r-1 次。
  - F(2,3)：4 次 vs 6 次
  - F(4,3)：6 次 vs 12 次
- 代价是两次线性变换（Bᵀ 与 Aᵀ），以及条件数放大（见 §11）。

### 2.2 二维推广

二维卷积是二维可分结构，用 **Kronecker 积** 直接张量化：

```
F(m×m, r×r) = (Aᵀ ⊗ Aᵀ) · [ (G⊗G)·g ] ⊙ [ (Bᵀ⊗Bᵀ)·d ] · (B⊗B 侧对应)
```

实现上（本项目）利用可分性拆成两次 1D 变换（先列后行，见 §6），矩阵写为：

| 变换 | 公式 | 本项目实现 |
|------|------|-----------|
| 权重 | V = G·g·Gᵀ/norm | `weight_transform_*` |
| 输入 | U = Bᵀ·d·B | `input_transform_*` = `transform_2d`(Bᵀ) |
| 输出 | f = Aᵀ·M·A | `output_transform_*` = `transform_2d`(A) |

### 2.3 配置

| | F(2,2,3,3) | F(4,4,3,3) |
|---|---|---|
| 采样点数 / tile | 4 | 6 |
| 输出 tile (OT) | 2×2 | 4×4 |
| 输入 tile (TS) | 4×4 | 6×6 |
| Winograd 域元素 NM | 16 | 36 |
| 每像素等效乘法 | 4 | 2.25 |
| 矩阵系数 | 全 {0,±1}（仅加减） | {±1,±2,±4,±5,±8}（需乘） |
| 归一化 | 1 | 1/576（G 内已 ×24） |

`WinogradConfig::F22_33()` / `F44_33()` 定义于 `winograd_config.hpp`；矩阵定义于 `winograd_matrices.hpp`。

### 2.4 矩阵验证

矩阵不靠「肉眼对抄」，用 Winograd 多项式恒等式数值验证（`test_winograd.cpp` 内）：

```
∀ i,a,b:  Σ_j Aᵀ[i][j] · Bᵀ[j][a] · G[j][b] = C · δ(a, i+b)
```

即「输出 = 输入 + 滤波器的时序对齐」。历史教训：F44_Bᵀ 曾错 5 个元素，因该恒等式成立范围覆盖不到而漏检；最终用数值求解器从 Aᵀ 和 G 反解出正确的 Bᵀ（见 AGENTS.md 历史修复）。

---

## 3. 端到端数据流与缓冲布局

### 3.1 流水线（`src/winograd_conv.cpp` 的 `winograd_convolution()`）

```
输入 src[N][·]  ──►  权重变换 V[36·OC·IC]（一次性）
                     │  （单 parallel region 内完成）
                     ▼
  每个 batch n:
    Phase 1  输入侧：tile 提取 → d_tile[36·IC] → U_tile[36·IC] → scatter → U[36·tiles·IC]
    Phase 2  GEMM：  对 36 个 ts：M[ts] = U[ts]·V[ts]  (n_tiles×OC×IC)  → M_buf[36·tiles·OC]
    Phase 3  输出侧：gather → M_tile[36·OC] → f_tile[16·OC] (+bias+ReLU) → writeback → dst[N][·]
```

**缓冲尺寸**（均为每线程或共享，`thread_local`）：

| 缓冲 | 尺寸 | 属主 | 用途 |
|------|------|------|------|
| V | TS²·OC·IC | 共享（master） | 变换后权重 |
| U | TS²·tiles·IC | 共享 | 变换后输入，scatter 目标 |
| M_buf | TS²·tiles·OC | 共享 | GEMM 输出 |
| d_tile | TS²·IC | 每线程 | 原始 tile |
| U_tile | TS²·IC | 每线程 | 输入变换输出 |
| M_tile | TS²·OC | 每线程 | 输出变换输入 |
| f_tile | OT²·OC | 每线程 | 输出变换输出 |
| g_wt / V_oc_wt | 9·IC / TS²·IC | 每线程 | 权重变换中间量 |

**为什么 U 的布局是 `[ts][tile][ic]` 而不是 `[tile][ts][ic]`**：Phase 2 对每个 ts 调 `winograd_gemm(U+ts·tiles·IC, V+ts·OC·IC, ...)`。GEMM 的 K 维是 IC、M 维是 tiles、N 维是 OC，`[ts][tile][ic]` 让 U[ts] 是一块连续 `tiles×IC` 矩阵，可直接喂给 cblas_sgemm/arm_gemm。代价是 Phase 1 每个 tile 的 36 个 ts 要 scatter 到 36 个不连续位置（`U + (ts·tiles+tile)·IC`），Phase 3 再从 36 个位置 gather 回来——这就是 scatter/gather 两个拷贝阶段存在的根本原因。若改成 `[tile][ts][ic]` 可免 barrier、免 scatter/gather，但 GEMM 变 36×更小矩阵，需重测（见 PERFORMANCE_ANALYSIS.md §6.2）。

**内存规模（每 batch，float）**：Case 0（IC=OC=192, tiles=100）：U=2.8MB、M_buf=2.8MB、V=5.3MB。Case 5（IC=768, tiles=100）：U=11MB。920F **无 L3、L2 仅 768KB/核**，U 远超 L2——这是「展平 N 个 batch」实验（内存 ×N）在 Case 4 上反而 +76% 的根因（cache thrashing，见 AGENTS.md）。

### 3.2 布局约定

- **NCHW**：`src[((n·IC+ic)·IH+ih)·IW+iw]`。tile 提取/写回是标量、跨行非连续（慢）。
- **NHWC**：`src[((n·IH+ih)·IW+iw)·IC+ic]`。IC 个通道**连续**，tile 提取/写回用 `copy_f32()` 一次拷一整个通道向量（SVE-512 时 16 float/指令）。实测 tile 提取 3.8×、写回 7.4× 快于 NCHW（PERFORMANCE_ANALYSIS.md §3.3）。
- 变换内部统一 `[tile 元素][channel]`（channel 连续）——三种 ISA 都按此布局做 4/16/64 通道向量化。

---

## 4. 缓冲与内存策略（A1/A2 落地）

`winograd_conv.cpp` 匿名 namespace：

```cpp
struct Scratch { float* ptr; size_t cap; ~Scratch(){ free(ptr); } };
inline float* scratch_f32(size_t n, Scratch& s) {
    if (s.cap < n) { free(s.ptr); s.ptr = malloc(n*sizeof(float)); s.cap = n; }
    return s.ptr;
}
thread_local Scratch sU, sM, sV;                 // 共享缓冲
thread_local Scratch sd_tile, sU_tile, sM_tile, sf_tile, sg_wt, sV_oc_wt;  // 每线程 tile 缓冲
```

设计要点：
1. **增长式 + 不清零**：容量不足才 realloc；malloc 不初始化。U/M_buf/V 在读取前必被完整覆写（scatter 全覆盖、GEMM beta=0、权重变换全覆盖），因此无需 memset——省掉每调用最多 ~33MB 的无谓清零（A1）。
2. **跨调用复用**：`thread_local` 让同一线程的缓冲在多次调用间存活，消除 benchmark/推理循环里每调用的堆分配 churn（A2）。
3. **⚠️ 别名陷阱**：`Scratch` 与 `scratch_f32(n, Scratch&)` 的签名强制「一个逻辑缓冲一个 Scratch」。若共用一个 Scratch，所有缓冲拿到同一个指针，互相踩内存，正确性全挂。这是本项目踩过的坑，签名就是为了让编译器/阅读者不可能再犯。

---

## 5. OpenMP 并行结构

单个 `#pragma omp parallel` 区域覆盖**权重变换 + 全部 batch**，只 fork/join 一次：

```
#pragma omp parallel
  #pragma omp for schedule(dynamic,4)     // 权重变换，over OC
  for oc: g_wt ← wei[oc]; V[oc] = G·g·Gᵀ/576
  ← 隐式 barrier（V 完成才能进 batch 循环）
  for n in batches:
    #pragma omp for collapse(2) schedule(dynamic,2)   // Phase 1 输入侧
    for tr,tc: extract → d_tile → U_tile → scatter → U
    ← 隐式 barrier（U 完整后才能 GEMM）
    #pragma omp for schedule(dynamic)                  // Phase 2 GEMM
    for ts: M[ts] = gemm(U[ts], V[ts])
    ← 隐式 barrier（M 完整后才能输出变换）
    #pragma omp for collapse(2) schedule(dynamic,2) nowait   // Phase 3 输出侧
    for tr,tc: gather → M_tile → f_tile → writeback
    ← 下一 batch 的 Phase 1 入口 barrier 兼作本 batch 的收尾同步
```

- **barrier 总数** = 1（权重）+ 2×N（每 batch 两个：输入→GEMM、GEMM→输出）= 1+2N。`nowait` 让 Phase 3 结束时无需专门等待——它由下一个 `omp for` 的隐式 barrier 同步。
- **为什么权重变换和 batch 循环合并**：历史教训——权重变换独立 `#pragma omp parallel` 会多一次 fork/join（~1-2ms），小 OC 时抵消并行收益（Case 0 反而变慢），大 IC 时受益（-19~-30%）。合并进主 region 两全（AGENTS.md 教训 ⑧/⑦）。
- **为什么 `schedule(dynamic,2)`**：tile 之间计算量不均（边缘 tile 有 padding 分支、不同 ts 的 GEMM 大小一致但 tile 提取量不同），动态调度 2 个一组平衡负载。
- **OpenBLAS 单线程**：`openblas_set_num_threads(1)`——并行度在 tile/ts 层由 OpenMP 提供；若 OpenBLAS 自己开线程会与 OpenMP 竞争，曾造成 double free。

---

## 6. 变换内核：NEON / SVE / SME

### 6.1 泛型 2D 变换（`transform_1d_neon`/`_sve` + `transform_2d`）

2D 变换 `out = M·in·Mᵀ` 拆成两次 1D：

```
Step 1（列变换）: tmp[:,j] = M · in[:,j]      （对 IN_SIZE 列各做一次 1D）
Step 2（行变换）: out[i,:] = tmp[i] · Mᵀ      （对 OUT_SIZE 行各做一次 1D）
```

每个 1D 是 `out[o] = Σ_k M[o][k]·in[k]`，实现按输出行 o 展开：先初始化 `out[o] = M[o][0]·in[0]`（系数 0 则直接 memset），再对 k=1..IN-1 做 `fma(M[o][k], in[k])`。M 的零系数自动跳过（`if (matrix[o][k]==0) continue`）。

**通道维并行度**：
- **NEON**：128-bit，一次 4 个 float。尾通道用 3 段降级（4→2→1），因为 Q 寄存器固定宽度没有谓词。
- **SVE**：`svwhilelt_b32` 谓词 + `svld1/svst1/svmla_n_x`，VL 自适应。SVE-512 时一次 16 float，且无需降级代码。
- **SME 输出变换**：用 Kronecker 展开 `(Aᵀ⊗Aᵀ)·vec(M)` + `FMOPA` 外积累加（157 条 `.inst` FMOPA + 69 条 MOVA），一次处理 4×VL=64 通道。这是唯一用矩阵 tile 的地方。

### 6.2 输入/输出变换的差异

- `input_transform`：U = Bᵀ·d·B，输入/输出都是 TS×TS。
- `output_transform`：f = Aᵀ·M·A，6×6 → 4×4（A 是 4×6），输出前加 bias + `clamp(act_min, act_max)`（ReLU 就是 min=0）。bias+ReLU 做在变换函数内部，避免端到端重复加。

### 6.3 `copy_f32`（A3）

4 处内存拷贝（tile 提取、scatter、gather、写回）共用：

```cpp
#if defined(__ARM_FEATURE_SVE)   // 编译期由 -march 决定
    svwhilelt_b32 谓词循环，svld1/svst1 一次 16 float
#else
    vld1q/vst1q 一次 4 float + 标量 tail
#endif
```

注意这是**编译期**选择（SVE 构建自动启用），与运行时 `isa_level()` 覆盖无关——因为 SVE 指令需要 `-march=armv8.6-a+sve2` 编译。

---

## 7. GEMM 内核（编译期三选一）

`winograd_gemm(U, V, M, n_tiles, OC, IC)` 计算 `M[tile][oc] = Σ_ic U[tile][ic]·V[oc][ic]`，每 ts 一次，共 36 次：

| 选项 | 实现 | 说明 |
|------|------|------|
| `-DUSE_ARM_GEMM` | `GemmHybrid<gemm_wide,float,float>` | ACL JIT，与 oneDNN 相同内核；`M=U·Vᵀ` |
| `-DUSE_OPENBLAS` | `cblas_sgemm(NoTrans, Trans)` | `V` 是 `[OC][IC]`，作转置 B |
| 默认 | naive 三重循环 | 正确性验证用；fp32 串行累加，误差 O(IC) |

**形状**：M 维 = n_tiles（100~1600），N 维 = OC（48~192），K 维 = IC（48~768）。对小 K 矩阵 OpenBLAS 不如 arm_gemm JIT——这正是 Case 4/5（大 IC）GEMM 受限、下一步换 arm_gemm 的原因。

---

## 8. ISA 调度

`winograd_convolution()` 内部：
```cpp
ISALevel isa = isa_level();              // 全局默认
if (env WINOGRAD_ISA) isa = parse_isa(env);   // 环境变量覆盖
```
`dispatch_*_transform()` 按 `isa` 选 NEON/SVE/SME 实现。注意：**SVE/SME 分支代码本身仍受编译守卫**（`#if defined(__ARM_FEATURE_SVE)`），运行时 dispatch 只是让同一个二进制能切到 NEON 回退路径。

---

## 9. tile 边界处理（正确性关键）

tile 覆盖 `ih ∈ [tr·OT-1, tr·OT-1+TS)`。为防越界：

1. **边缘 tile 先 memset 清零**：`is_edge = (tr==0||tr==last||tc==0||tc==last)` 时清 d_tile，padding 自动为 0。
2. **行/列裁剪**（非整 tile 时只拷贝有效区）：
   ```cpp
   ih_begin = tr·OT - 1;
   ti_start = (ih_begin<0) ? -ih_begin : 0;
   ti_end   = (ih_begin+TS > IH) ? (IH - ih_begin) : TS;
   ```
   历史 bug：旧公式「最后一 tile 用 TS-1」只对**偶数** IH/IW 成立；奇数维（如 IH=7）会读到下一通道第 0 行甚至缓冲外，导致正确性测试 F(2,2) N=2 IC=16 IH=7 挂。当前公式对奇偶维度统一正确，偶数维行为不变。

---

## 10. 端到端正确性验证

- `test_winograd.cpp`：变换级调试 + 端到端对比 `direct_convolution_3x3`（fp32 参考，容差 1e-3），22/22 通过。
- `bench_winograd.cpp --verify`：每个 case 对比 **fp64 直接卷积**（`direct_convolution_3x3_f64`，OpenMP 按 batch 并行），判据**相对容差** `err < 1e-4×max|ref| + 1e-5`。见 §11。

---

## 11. 数值精度（为什么误差可控）

### 11.1 误差来源

F(4,4) fp32 误差的主源是 **GEMM 按 IC 串行累加**（naive 内核），再被输出变换放大：
- 输出变换 A 矩阵系数最高 ±8 → 放大 ~8-16×。
- 输入变换 Bᵀ 系数最高 ±5 → 贡献较小。

### 11.2 关键事实：相对误差恒定

误差 ∝ IC（累加项数），输出幅值也 ∝ IC（每个输出 = Σ IC×9 个乘积），因此**相对误差恒定在 ~2-6e-6**，所有 9 个 case 一致。这是 fp32 的正常水平（fp32 机器精度 1.2e-7；6912 项累加的随机游走误差 ~1e-5）。

> 早期用固定绝对容差 1e-3 导致大 IC case（IC=384/768）误报 FAIL——那是**指标问题**，不是实现 bug。IC=768 时输出幅值 ~1700，1e-2 绝对误差相对只有 ~6e-6。

### 11.3 验证方法学的正确做法

- 参考用 **fp64**（接近精确数学），测 Winograd 的**真实误差**，而不是两个 fp32 实现的差值（fp32 参考自身就有 ~1e-4 误差）。
- 判据用**相对容差**，随问题规模缩放。

### 11.4 若需更严格精度（性价比排序）

| 方法 | 效果 | 代价 |
|------|------|------|
| GEMM fp64/Kahan 累加 | 消除主要误差源 | 打在性能关键路径 |
| 变换 fp64 | 消除变换放大 | 破坏 NEON/SVE 向量化 |
| 换 F(2,2) | 矩阵全 ±1 无放大 | 改变被测算法 |

换 arm_gemm 后其 blocked 累加比 naive 串行更准，误差会自然改善。

---

## 12. 性能特征总结

当前实现「8/9 case 快于 oneDNN」的关键设计点：

1. **单 OpenMP 区域 + 每调用零分配**（A2）：微基准对固定开销极敏感。
2. **SVE-512 内存拷贝**（A3）：tile 提取/scatter/gather/写回 4→16 float/指令。
3. **NHWC 布局**：通道连续，变换全程 channel-vectorized。
4. **U/M_buf/V 免清零**（A1）：消除大内存 memset。
5. **per-batch 内存复用**：U 每 batch 复用，避免大 IC 时内存膨胀超出 L2（920F 无 L3）。
6. **GEMM 并行**：36 个 ts 交给 36×更细的 OpenMP 任务，而非 GEMM 库内部开线程。

局限性（为什么 Case 2 仍慢、为什么 16 线程后平台化）：
- barrier 开销 vs tile 数量；小 IC 时 GEMM 矩阵太小（OpenBLAS 低效）。
- 无 L3、768KB L2：大 U 缓冲走主存，高线程时内存带宽受限。
