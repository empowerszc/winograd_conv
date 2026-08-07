# ACL 23.11 vs 53.1.0 Winograd 卷积性能差异深度分析

> **背景**：在华为鲲鹏 920F（Armv9 + SVE-512 + SME）服务器上测试发现，**ACL 23.11 的 acl:wino 性能好于 53.1.0**。两者都走 Winograd 算法（acl:wino），变换选择也相同（SME 版 F(4,4,3,3)），但 GEMM 内核选择不同。
>
> 本文逐层对比两个版本的 Winograd 实现，定位性能差异的根因，便于分享与查阅。

---

## 目录

- [1. 对比总览：哪些相同，哪些不同](#1-对比总览哪些相同哪些不同)
- [2. Winograd 变换层：几乎完全一致](#2-winograd-变换层几乎完全一致)
- [3. GEMM 内核选择层：性能差异的根因](#3-gemm-内核选择层性能差异的根因)
- [4. 根因详解：SME GEMM 内核的引入](#4-根因详解sme-gemm-内核的引入)
- [5. 其他差异](#5-其他差异)
- [6. 根因总结与验证建议](#6-根因总结与验证建议)
- [7. SVE GEMM 内核实现差异详解](#7-sve-gemm-内核实现差异详解)

---

## 1. 对比总览：哪些相同，哪些不同

| 层次 | 23.11 vs 53.1.0 | 影响 |
|------|:-:|------|
| Winograd 变换注册表（input/output/weight） | **完全一致** | 无 |
| `winograd_implementations.hpp`（匹配逻辑） | **完全一致** | 无 |
| `winograd.hpp`（WinogradImpl 结构） | **完全一致** | 无 |
| `CpuWinogradConv2d.h`（类结构） | **完全一致** | 无 |
| `CpuWinogradConv2d.cpp`（configure/run/prepare） | **2 行差异** | TransformedWeights 生命周期 |
| SME 输入变换 `sme_fp32_mla_6x6.cpp` | **完全一致** | 无 |
| SVE 输入变换 `sve_fp32_6x6.cpp` | **完全一致** | 无 |
| NEON 输入变换 `a64_fp32_6x6.cpp` | **完全一致** | 无 |
| SME 输出变换 `sme_fp32_mopa_4x4_3x3.cpp` | **37 行差异**（注释+1条MOV） | 微小 |
| NEON 输出变换 `arm_fp32_4x4_3x3.cpp` | **2 行差异** | 极小 |
| NEON 输出变换 `arm_fp32_2x2_3x3.cpp` | **2 行差异** | 极小 |
| 权重变换 `arm_fp32_4x4_3x3.cpp` | **完全一致** | 无 |
| **SVE GEMM 内核（.hpp 头文件）** | **全部有差异**（6-33 行） | API 重构，详见第 7 节 |
| **SVE GEMM 内核（.cpp 实现）** | **全部有差异**（176-2415 行） | 汇编重写，详见第 7 节 |
| **SVE 支持/变换/合并文件** | **全部有差异**（7-51 行） | 详见第 7 节 |
| **`gemm_fp32.cpp`（GEMM 内核选择列表）** | **★325 行差异** | **★性能根因之一** |
| `gemm_implementation.hpp`（GEMM 框架） | 160 行差异（重构） | 间接影响 |
| arm_gemm 文件总数 | 433 → 540（+107） | 新增 SME 内核 |

> **结论**：Winograd 变换层几乎完全一致，**性能差异来自 GEMM 内核选择层**——53.1.0 新增了 SME（非 SME2）GEMM 内核，改变了 GEMM 内核选择路径。

---

## 2. Winograd 变换层：几乎完全一致

### 2.1 变换注册表完全一致

三个 fp32 注册表（`input_transforms_fp32.cpp`、`output_transforms_fp32.cpp`、`weight_transforms_fp32.cpp`）逐行对比：

| 文件 | 23.11 行数 | 53.1.0 行数 | 差异 |
|------|:-:|:-:|------|
| `input_transforms_fp32.cpp` | 77 | 77 | 2 行（仅 include 守卫和注释） |
| `output_transforms_fp32.cpp` | 76 | 76 | 6 行（仅 include 守卫和注释） |
| `weight_transforms_fp32.cpp` | 74 | 74 | **完全一致** |

两个版本在 920F（SME 机器）上选中**完全相同**的变换三元组：
- 输入：`sme_fp32_mla_6x6`（SME）
- 输出：`sme_fp32_mopa_4x4_3x3`（SME）
- 权重：`arm_fp32_4x4_3x3`（NEON）

### 2.2 SME 输出变换的 37 行差异

`sme_fp32_mopa_4x4_3x3.cpp` 的差异**不影响指令编码**，仅影响注释和一条 MOV：

| | 23.11 | 53.1.0 |
|---|---|---|
| ZA tile 行索引 | `XZR`（零寄存器，硬件恒 0） | `x15`（需 `mov x15, #0` 设置） |
| `mov x15, #0` 指令 | **无**（不需要） | **有**（多 1 条指令） |
| `mova` 指令的 `.inst` 编码 | `0xc082741f` | `0xc082741f`（**相同**） |
| clobber 列表 | 不含 x15 | 含 x15 |

**翻译**：23.11 用 `XZR`（寄存器 31，硬件恒为 0）做 ZA tile 行索引，无需 setup；53.1.0 改用 `x15`（需先 `mov x15, #0`），多 1 条 MOV + 占 1 个 GPR。但 `.inst` 编码相同，实际 MOVA 指令行为一致。**差异极小，不足以解释性能差距。**

### 2.3 变换层结论

Winograd 的三步（输入变换 → GEMM → 输出变换）中，变换部分两个版本**实质一致**。性能差异一定来自中间的 **GEMM 步骤**。

---

## 3. GEMM 内核选择层：性能差异的根因

### 3.1 Winograd 域 GEMM 的重要性

F(4,4,3,3) 的 Winograd 需要跑 **36 个 batched GEMM**（`n_multis = (4+3-1)² = 36`），每个 GEMM 的维度是 M=tiles, N=OC, K=IC。GEMM 通常是 Winograd 的**性能瓶颈**——变换是 O(tiles × channels) 的加法，而 GEMM 是 O(tiles × OC × IC) 的乘加。

ACL 的 Winograd 用 `CpuGemm` → `arm_gemm` 库执行这些 GEMM。`arm_gemm` 有一个**有序的内核选择列表**（`gemm_fp32.cpp`），按顺序尝试每个内核的 `is_supported()` 和 `estimate()`，**第一个匹配的内核被选中**。

### 3.2 `gemm_fp32.cpp` 选择列表对比

两个版本的 fp32 GEMM 内核选择列表有 **325 行差异**——这是最核心的差异。

#### 23.11 的选择顺序（`gemm_fp32.cpp:84-403`）

```
1. gemv_batched           (M==1 && nbatches>1)
2. [BF16 fast-mode 内核]   (fast_mode && has_bf16)
3. [SME2 内核]             (has_sme2())        ← 仅 SME2，不含纯 SME
   - sme2_gemv_fp32_mla_16VL
   - sme2_interleaved_nomerge_fp32_mopa_*
4. [SVE BF16 内核]         (fast_mode && has_bf16)
5. [SVE F32MM 内核]        (has_svef32mm)
   - sve_interleaved_fp32_mmla_8x3VL
6. [SVE 内核]              (has_sve())
   - sve_hybrid_fp32_mla_8x1VL   (N<12)
   - sve_hybrid_fp32_mla_6x4VL   ★ 主力
   - sve_interleaved_fp32_mla_8x3VL
7. [A64 NEON 内核]         (fallback)
```

#### 53.1.0 的选择顺序（`gemm_fp32.cpp:107-436`）

```
1. gemv_batched           (M==1 && nbatches>1)
2. [BF16 fast-mode 内核]   (fast_mode && has_bf16)
3. [SME2 内核]             (has_sme2())
   - sme2_gemv_fp32_mla_16VL
   - sme2_interleaved_nomerge_fp32_mopa_*
4. ★[SME 内核] NEW!       (has_sme())          ← 53.1.0 新增！23.11 没有
   - sme_gemv_fp32_mla_8VL
   - sme_interleaved_nomerge_fp32_mopa_1VLx4VL
   - sme_interleaved_nomerge_fp32_mopa_4VLx1VL
   - sme_interleaved_nomerge_fp32_mopa_2VLx2VL  ★ fallback（estimate=nullptr）
5. [SVE 内核]              (has_sve())
   - sve_hybrid_fp32_mla_6x4VL
   - sve_interleaved_fp32_mla_8x3VL
   - sve_interleaved_fp32_mmla_8x3VL  (has_svef32mm)
6. [A64 NEON 内核]         (fallback)
```

### 3.3 关键差异：53.1.0 新增了 SME（非 SME2）GEMM 内核

53.1.0 在 SME2 内核之后、SVE 内核之前，**新增了一组纯 SME GEMM 内核**（`#ifdef ARM_COMPUTE_ENABLE_SME`，`gemm_fp32.cpp:58-69, 192-240`）：

| 内核 | 守卫条件 | estimate 条件 |
|------|----------|--------------|
| `sme_gemv_fp32bf16fp32_dot_8VL` | `has_sme() && M==1 && nbatches==1` | nullptr |
| `sme_gemv_fp32_mla_8VL` | `has_sme() && M==1 && nbatches==1` | nullptr |
| `sme_interleaved_nomerge_bf16fp32_mopa_1VLx4VL` | `fast_mode && has_sme() && has_sme_b16f32()` | `N>=8*VL \|\| M<=VL \|\| ...` |
| `sme_interleaved_nomerge_bf16fp32_mopa_4VLx1VL` | `fast_mode && has_sme() && has_sme_b16f32()` | `N<=VL \|\| ...` |
| `sme_interleaved_nomerge_bf16fp32_mopa_2VLx2VL` | `fast_mode && has_sme() && has_sme_b16f32()` | nullptr |
| `sme_interleaved_nomerge_fp32_mopa_1VLx4VL` | `has_sme() && has_sme_f32f32()` | `M<=VL \|\| ...` |
| `sme_interleaved_nomerge_fp32_mopa_4VLx1VL` | `has_sme() && has_sme_f32f32()` | `N<=VL \|\| ...` |
| `sme_interleaved_nomerge_fp32_mopa_2VLx2VL` | `has_sme() && has_sme_f32f32()` | **nullptr（总匹配）** |

**23.11 完全没有这些纯 SME GEMM 内核**——只有 SME2 内核（`#ifdef ARM_COMPUTE_ENABLE_SME2`）。

---

## 4. 根因详解：SME GEMM 内核的引入

### 4.1 在 920F 上的实际选择路径

920F 特性：SVE-512 + **SME**（支持 `has_sme()`=true, `has_sme_f32f32()`=true）。假设 **没有 SME2**（`has_sme2()`=false，这在很多 SME 实现中成立）。

以 Winograd GEMM（F(4,4,3,3)，case 1：N=4, IC=192, OC=192, 40×40）为例：
- M=100 tiles, N=192(OC), K=192(IC), nbatches=4, fast_mode=true, SVE-512 VL=16

**23.11 的选择路径**：
```
1. gemv_batched: M==1? NO (M=100) → skip
2. BF16 fast-mode: fast_mode && has_bf16? → 尝试 BF16 内核
   → 如果 BF16 内核 estimate 不够好 → 继续
3. SME2 内核: has_sme2()? NO → skip（全部跳过）
4. SVE BF16 内核: 尝试
5. SVE F32MM 内核: has_svef32mm()? → 尝试 sve_interleaved_fp32_mmla_8x3VL
6. SVE 内核: has_sve()? YES → ★选中
   - sve_hybrid_fp32_mla_6x4VL (M=100, N=192 → estimate 匹配)
   或 sve_interleaved_fp32_mla_8x3VL
```
→ **23.11 选中成熟的 SVE GEMM 内核**（`sve_hybrid_fp32_mla_6x4VL` 或 `sve_interleaved_fp32_mmla_8x3VL`），这些内核经过多年优化，有 A64FX 专用变体（`a64fx.cpp`）。

**53.1.0 的选择路径**：
```
1. gemv_batched: M==1? NO → skip
2. BF16 fast-mode: 尝试
3. SME2 内核: has_sme2()? NO → skip
4. ★SME 内核: has_sme()? YES → 尝试！
   - sme_interleaved_nomerge_fp32_mopa_1VLx4VL:
     estimate: M<=VL(=16)? NO (M=100). 2*VL<M<=3*VL? 32<100<=48? NO → skip
   - sme_interleaved_nomerge_fp32_mopa_4VLx1VL:
     estimate: N<=VL(=16)? NO (N=192). 2*VL<N<=3*VL? 32<192<=48? NO → skip
   - sme_interleaved_nomerge_fp32_mopa_2VLx2VL:
     estimate: nullptr → ★总匹配！选中！
```
→ **53.1.0 选中新的 SME GEMM 内核** `sme_interleaved_nomerge_fp32_mopa_2VLx2VL`，而非 SVE 内核。

### 4.2 为什么 SME GEMM 内核更慢

`GemmInterleavedNoMerge<cls_sme_interleaved_nomerge_fp32_mopa_2VLx2VL>` 与 `GemmHybridIndirect<cls_sve_hybrid_fp32_mla_6x4VL>` 或 `GemmInterleaved<cls_sve_interleaved_fp32_mmla_8x3VL>` 相比，可能更慢的原因：

| 因素 | SVE 内核（23.11） | SME 内核（53.1.0） |
|------|-------------------|-------------------|
| 成熟度 | 多年优化，有 A64FX 专用变体 | 53.1.0 新增，首次引入 |
| 实现类 | `GemmHybridIndirect` / `GemmInterleaved` | `GemmInterleavedNoMerge`（不同类） |
| 计算模式 | SVE FMLA（乘加） | SME FMOPA（外积累加） |
| Tile 大小 | 8×3VL=8×48 或 6×4VL=6×64 | 2VL×2VL=32×32 |
| SMSTART/SMSTOP | 无（SVE 不需要） | 每次调用需 enter/exit streaming mode |
| NoMerge 特性 | — | 不合并多个小 GEMM 的结果，可能增加写回次数 |

**最可能的原因**：
1. **新代码未充分优化**：SME GEMM 内核是 53.1.0 首次引入，缺乏针对 Winograd GEMM 形状（小 M、中 N/K、36 batched）的调优
2. **NoMerge 实现的开销**：`GemmInterleavedNoMerge` 对每个 batch 元素（36 个 Winograd 矩阵）独立处理，不合并结果，可能导致更多的内存写回和更少的权重复用
3. **SMSTART/SMSTOP 开销**：SME 流式模式的进入/退出有固定开销，对大量小 GEMM 可能不够 amortized
4. **Tile 大小不匹配**：2VL×2VL=32×32 的 tile 对 M=100、N=192 的 GEMM 不是最优（padding 浪费多）

### 4.3 对其他 case 的影响

以你的 11 个 case 中 stride=1 的 6 个为例（这些在显式 winograd 模式下走 acl:wino）：

| Case | M(tiles) | N(OC) | K(IC) | 53.1.0 选中的 SME 内核 |
|------|----------|-------|-------|----------------------|
| 1 (40×40, 192) | 100 | 192 | 192 | `2VLx2VL`（fallback） |
| 3 (80×80, 96) | 400 | 96 | 96 | `2VLx2VL`（fallback） |
| 6 (160×160, 48) | 1600 | 48 | 48 | `2VLx2VL`（fallback） |
| 8 (20×20, 192) | 25 | 192 | 192 | `1VLx4VL`（M=25<=16? NO; 32<25? NO → fallback `2VLx2VL`） |
| 10 (80×80, 96oc) | 400 | 96 | 384 | `2VLx2VL`（fallback） |
| 11 (40×40, 96oc) | 100 | 96 | 768 | `2VLx2VL`（fallback） |

> 大多数 case 都落入 `2VLx2VL` 的 fallback（因为 M 和 N 都远大于 VL=16，不满足 1VLx4VL/4VLx1VL 的形状条件）。这个 fallback 内核是**总匹配**（estimate=nullptr）但**不一定最优**的通用 SME GEMM 内核。

---

## 5. 其他差异

### 5.1 TransformedWeights 生命周期

| | 23.11 | 53.1.0 |
|---|---|---|
| `CpuWinogradConv2d.cpp:312` | `MemoryLifetime::Persistent` | `MemoryLifetime::Prepare` |
| 含义 | 变换后权重常驻内存，永不释放 | prepare 后可释放，可能需要重新 prepare |

**影响**：`Persistent` 让变换后权重一直留在内存中，可能更好的 cache 局部性；`Prepare` 允许内存管理器回收，可能导致重复 prepare 或 cache miss。但这通常是次要因素。

### 5.2 `gemm_implementation.hpp` 框架重构

53.1.0 给 `GemmImplementation` 模板增加了 `OutputStage` 参数（三参数版 `GemmImplementation<Top, Tret, OutputStage>`），重构了 `is_supported`/`do_instantiate` 逻辑。这改变了 GEMM 内核的注册和选择框架，但选择逻辑的核心（按列表顺序尝试）不变。

### 5.3 arm_gemm 文件数变化

| | 23.11 | 53.1.0 |
|---|---|---|
| 文件总数 | 433 | 540（+107） |
| 新增 | — | SME 内核、SME transforms、lhs2VL 变体等 |

53.1.0 新增了大量 SME 相关文件（`sme_gemv_*`、`sme_interleaved_nomerge_*`、`sme_transpose_interleave_*`），丰富了 SME 生态但也改变了 GEMM 选择路径。

---

## 6. 根因总结与验证建议

### 6.1 根因一句话

**53.1.0 在 GEMM 选择列表中新增了纯 SME（非 SME2）GEMM 内核，排在 SVE 内核之前。920F 有 SME 但可能没有 SME2，导致 53.1.0 选中新的（但未充分优化的）SME GEMM 内核 `sme_interleaved_nomerge_fp32_mopa_2VLx2VL`，而 23.11 跳过 SME 直接使用成熟的 SVE GEMM 内核（如 `sve_hybrid_fp32_mla_6x4VL`），后者经过多年优化且对 Winograd GEMM 形状更友好。**

### 6.2 性能差异链路图

```
                    23.11                          53.1.0
                      │                               │
         has_sme2()? ─┤ NO                   has_sme2()? ─┤ NO
                      │                               │
     无 SME GEMM 内核  │              ★有 SME GEMM 内核  │
     (只有 SME2 内核)  │              (53.1.0 新增)      │
                      │                               │
           跳过 → SVE 内核                选中 SME 内核  │
           sve_hybrid_fp32_mla_6x4VL      sme_interleaved_nomerge_fp32_mopa_2VLx2VL
           sve_interleaved_fp32_mmla     GemmInterleavedNoMerge
           GemmHybridIndirect             (新代码, NoMerge, 2VL×2VL tile)
           (成熟, 有a64fx变体)                         │
                      │                               │
                      ▼                               ▼
               ★ 性能更好                     ★ 性能更差
               (多年优化)                     (新代码, 未充分调优)
```

### 6.3 验证建议

1. **确认 SME2 是否可用**：在 920F 上检查 `cat /proc/cpuinfo` 或 ACL 的 `CPUInfo::has_sme2()`。如果 `has_sme2()=true`，则两个版本都走 SME2 内核（无差异），根因另在他处。如果 `has_sme2()=false`，则上述分析成立。

2. **查看实际选中的 GEMM 内核**：启用 ACL 日志（`ARM_COMPUTE_LOG_LEVEL=INFO`），观察 Winograd GEMM 步骤选中的内核名。23.11 应显示 `sve_hybrid_fp32_mla_6x4VL` 或 `sve_interleaved_*`，53.1.0 应显示 `sme_interleaved_nomerge_fp32_mopa_2VLx2VL`。

3. **强制 SVE 内核**：在 53.1.0 中通过 `GemmConfig` 的 `weight_format_filter` 或修改 `gemm_fp32.cpp` 临时注释掉 `#ifdef ARM_COMPUTE_ENABLE_SME` 段，强制走 SVE 内核，验证性能是否恢复到 23.11 水平。

4. **Profile GEMM 步骤**：用 perf 或 Arm Streamline 对比两个版本的 GEMM 步骤耗时占比。如果 53.1.0 的 GEMM 步骤明显更慢，则确认根因。

5. **对比变换步骤**：分别测量输入变换 + 输出变换的耗时（排除 GEMM），确认两版本变换性能一致（预期一致）。

### 6.4 修复建议

| 方案 | 描述 | 代价 |
|------|------|------|
| **A. 回退到 23.11** | 直接使用 23.11 版本的 ACL | 最简单，但失去 53.1.0 的其他改进 |
| **B. 禁用 SME GEMM** | 编译 53.1.0 时不定义 `ARM_COMPUTE_ENABLE_SME`（仅对 GEMM） | 保留 53.1.0 框架，GEMM 走 SVE，但 SME Winograd 变换也可能被禁 |
| **C. 修改选择顺序** | 在 `gemm_fp32.cpp` 中把 SME GEMM 内核移到 SVE 之后 | 保留 SME 变换 + SVE GEMM，最佳方案 |
| **D. 优化 SME GEMM** | 给 SME GEMM 内核增加 `estimate` 函数，让小 M/中 N 的 Winograd 形状不选 SME | 最彻底，但需 ACL 上游修复 |

> **推荐方案 C**：在 `gemm_fp32.cpp` 中把 `#ifdef ARM_COMPUTE_ENABLE_SME` 段的 GEMM 内核列表移到 `#ifdef ARM_COMPUTE_ENABLE_SVE` 段之后，让 SVE 内核优先于 SME GEMM 内核被尝试。这样保留了 SME 的 Winograd 变换（它们走单独的选择路径，不受影响），但 GEMM 用成熟的 SVE 内核。

---

## 附录：关键文件索引

### 23.11（`D:\300Code\ComputeLibrary-23.11\ComputeLibrary-23.11\`）

| 文件 | 关键内容 |
|------|----------|
| `src/cpu/operators/CpuWinogradConv2d.cpp:312` | TransformedWeights = `Persistent` |
| `src/core/NEON/kernels/arm_gemm/gemm_fp32.cpp:84-403` | GEMM 选择列表（**无 SME 内核**） |
| `src/core/NEON/kernels/arm_gemm/gemm_implementation.hpp` | GEMM 框架（2 参数版） |
| `src/core/NEON/kernels/arm_gemm/`（433 文件） | 含 A64FX 专用 SVE 内核 |

### 53.1.0（`D:\300Code\ComputeLibrary-53.1.0\`）

| 文件 | 关键内容 |
|------|----------|
| `src/cpu/operators/CpuWinogradConv2d.cpp:312` | TransformedWeights = `Prepare` |
| `src/core/NEON/kernels/arm_gemm/gemm_fp32.cpp:107-436` | GEMM 选择列表（★新增 SME 内核 :192-240） |
| `src/core/NEON/kernels/arm_gemm/gemm_implementation.hpp` | GEMM 框架（3 参数版，+OutputStage） |
| `src/core/NEON/kernels/arm_gemm/`（540 文件） | 新增 `sme_gemv_*`、`sme_interleaved_nomerge_*` |

### 共同（两版本一致）

| 文件 | 关键内容 |
|------|----------|
| `src/core/NEON/kernels/convolution/winograd/winograd_implementations.hpp` | 变换匹配逻辑（完全一致） |
| `src/core/NEON/kernels/assembly/winograd.hpp` | WinogradImpl 结构（完全一致） |
| `src/core/NEON/kernels/convolution/winograd/input_transforms/sme_fp32_mla_6x6.cpp` | SME 输入变换（完全一致） |
| `src/core/NEON/kernels/convolution/winograd/output_transforms/sme_fp32_mopa_4x4_3x3.cpp` | SME 输出变换（37 行差异，不影响指令编码） |

> **注意**：SVE GEMM 内核文件（arm_gemm/kernels/sve_*）在两版本间**全部有差异**，不在"共同"之列——详见第 7 节。

---

## 7. SVE GEMM 内核实现差异详解

> 前文指出 53.1.0 新增 SME GEMM 内核拦截了 SVE 选择路径（根因之一）。但即使强制走 SVE 路径，两个版本的 **SVE GEMM 内核实现本身也完全不同**——这是性能差异的第二个层面。

### 7.1 对比范围与方法

对 `arm_gemm/` 下所有含 "sve" 的文件（36 个 .hpp + .cpp）逐文件 diff。**结果：全部有差异，无一相同。**

### 7.2 差异总览

| 文件类别 | 差异行数 | 性质 |
|----------|---------|------|
| SVE 内核 .hpp 头文件 | 6-33 行 | API 重构（模板参数、守卫宏） |
| SVE 内核 generic.cpp（fp32） | 176-2415 行 | ★汇编重写 |
| SVE 内核 a64fx.cpp | 238-831 行 | ★汇编重写 |
| SVE 支持（transform-sve 等） | 7-51 行 | 路径重构 + 间接参数 |
| std_transforms_sve.hpp | 21 行 | 模板参数拆分 + 新方法 |
| transforms/list-sve.hpp | 15 行 | 混入 SME 变换 + 新增/删除 |
| merges/list-sve.hpp | 12 行 | 重命名 + 新增 |

### 7.3 头文件（.hpp）差异：API 重构

以 `sve_hybrid_fp32_mla_6x4VL.hpp` 为例：

| | 23.11 | 53.1.0 |
|---|---|---|
| 守卫宏 | `#ifdef ARM_COMPUTE_ENABLE_SVE` | **无守卫**（始终编译） |
| 类型定义 | `typedef float operand_type;` | `typedef float lhs_operand_type;` `typedef float rhs_operand_type;` |
| 变换模板 | `StdTransformsSVE<operand_type, result_type, 6, 4, 1>` | `StdTransformsSVE<lhs_operand_type, rhs_operand_type, result_type, 6, 4, 1>` |
| out_height | `static unsigned int` | `static constexpr unsigned int` |

**翻译**：
- 53.1.0 **移除了 `#ifdef ARM_COMPUTE_ENABLE_SVE` 守卫**——SVE 内核类始终被编译（但 .cpp 仍有守卫），使得非 SVE 平台也能引用类定义
- `operand_type` 拆分为 `lhs_operand_type`（左操作数/A 矩阵）和 `rhs_operand_type`（右操作数/B 矩阵），支持 A/B 不同类型（如 bf16 输入 + fp32 累加）
- `StdTransformsSVE` 增加了 `TWeight` 模板参数

### 7.4 实现（.cpp）差异：汇编重写

以 `sve_hybrid_fp32_mla_6x4VL/generic.cpp`（23.11: 2091 行 → 53.1.0: 2090 行，2415 行 diff）为例，核心变化：

#### 7.4.1 守卫条件加严

```diff
- #ifdef ARM_COMPUTE_ENABLE_SVE
+ #if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__aarch64__)
```

53.1.0 增加 `__aarch64__` 检查，确保只在 64 位 ARM 上编译 SVE 内核。

#### 7.4.2 Include 路径重构

```diff
- #include "arm_gemm.hpp"
- #include "../../utils.hpp"
+ #include "arm_gemm/arm_gemm.hpp"
+ #include "arm_common/internal/utils.hpp"
```

53.1.0 重构了 include 路径体系，统一用 `arm_gemm/` 前缀和 `arm_common/` 命名空间。

#### 7.4.3 ★参数传递机制重构（最关键变化）

**23.11**：直接寄存器绑定——C 变量直接映射到汇编操作数：

```asm
      "mov x12, %x[bias]\n"          // bias 指针直接从 C 变量绑定
      "ldr x11, [%x[args_ptr], %[offsetof_N]]\n"
      "ldr x10, [%x[args_ptr], %[offsetof_B_ptr]]\n"
      "mov x9, %x[output_ptr]\n"     // output_ptr 直接绑定
```

**53.1.0**：间接 args 结构体——所有参数通过 `args_ptr` + offset 加载：

```asm
      "ldr x12, [%x[args_ptr], %[offsetof_N]]\n"
      "ldr x11, [%x[args_ptr], %[offsetof_B_ptr]]\n"
      "ldr x10, [%x[args_ptr], %[offsetof_bias]]\n"    // 从 args 结构体加载 bias
      "ldr x9, [%x[args_ptr], %[offsetof_output_ptr]]\n" // 从 args 结构体加载 output_ptr
```

C++ 侧也相应改变：

```cpp
// 23.11: 直接绑定
void *output_ptr;
output_ptr = (void *)(output_arg.indirect.ptr);
// asm约束: [bias] "r" (bias_ptr), [output_ptr] "r" (output_ptr)

// 53.1.0: 通过 kernel_args 结构体
struct kernel_args {
    void *output_ptr = {};
    const float *bias = {};
    // ... 其他参数
} ka;
ka.output_ptr = (void *)(output_arg.indirect.ptr);
ka.bias = bias;
// asm约束: [args_ptr] "r" (&ka), [offsetof_bias] "I" (offsetof(kernel_args, bias))
```

**影响分析**：
- 53.1.0 每个参数多一条 `ldr`（从 args 结构体加载），增加 prologue 开销
- 对大量小 GEMM（如 Winograd 的 36 个 batched GEMM），prologue 开销被放大
- 但对大 GEMM，prologue 占比极低，影响可忽略
- 寄存器分配也改变（bias 从 x12→x10，N 从 x11→x12 等），可能影响指令调度

#### 7.4.4 新增早期退出

```diff
+ "cbz %x[ablocks], 6f\n"    // 53.1.0 新增：如果 ablocks==0 直接跳到结尾
```

53.1.0 增加了空块检查，避免无效计算。这是正确性优化，对正常 case 无影响。

#### 7.4.5 格式微调

`#0x0` → `#0`、`int K) {` → `int K)\n{`、`addvl` 位置变化等——纯格式，不影响行为。

### 7.5 各 SVE fp32 内核差异量

| 内核 | .hpp diff | generic.cpp diff | a64fx.cpp diff | 对 Winograd 的影响 |
|------|:-:|:-:|:-:|------|
| `sve_hybrid_fp32_mla_6x4VL` | 6 行 | **2415 行** | **831 行** | ★主力 hybrid 内核，汇编大幅重写 |
| `sve_hybrid_fp32_mla_8x1VL` | 6 行 | **1375 行** | **735 行** | 小 N 用的 hybrid，大幅重写 |
| `sve_interleaved_fp32_mla_8x3VL` | 11 行 | 176 行 | 238 行 | ★Winograd 常用的 interleaved 内核 |
| `sve_interleaved_fp32_mmla_8x3VL` | 33 行 | 629 行 | —（无 a64fx 变体） | MMLA 内核，结构也变了 |
| `sve_ffhybrid_fp32_mla_6x4VL` | 6 行 | **2391 行** | **787 行** | Fixed-format hybrid，大幅重写 |
| `sve_ffinterleaved_fp32_mla_8x3VL` | 11 行 | 176 行 | 224 行 | Fixed-format interleaved |
| `sve_hybrid_fp32bf16fp32_mmla_4x6VL` | 8 行 | 1355 行 | — | BF16 fast-mode hybrid |
| `sve_hybrid_fp32bf16fp32_mmla_6x4VL` | 8 行 | 1715 行 | — | BF16 fast-mode hybrid |
| `sve_interleaved_bf16fp32_mmla_8x3VL` | 13 行 | 314 行 | — | BF16 interleaved |
| `sve_hybrid_bf16fp32_dot_6x4VL` | 8 行 | 2417 行 | — | BF16 dot hybrid |
| `sve_hybrid_bf16fp32_mmla_6x4VL` | 8 行 | 2225 行 | — | BF16 mmla hybrid |
| `sve_interleaved_bf16fp32_dot_8x3VL` | 13 行 | 178 行 | — | BF16 dot interleaved |

> **结论**：`generic.cpp` 的 diff 行数远超文件本身行数（如 2415 > 2091），说明几乎每行都被修改——这不是微调，是**整体重写**。

### 7.6 SVE 变换/合并/支持文件差异

| 文件 | diff 行数 | 关键变化 |
|------|:-:|------|
| `std_transforms_sve.hpp` | 21 | `TOperand`→`TInput,TWeight` 拆分；新增 `PrepareB_supports_transpose()`/`transposed` 参数 |
| `transforms/list-sve.hpp` | 15 | ★**混入 SME 变换**（`sme_transpose_interleave_8VL_*`）；新增 `sve_transpose_interleave_2VL*`；删除部分旧条目 |
| `merges/list-sve.hpp` | 12 | 重命名 `sve_merge_*_3VLx8` → `sve_merge_*_*_8x3VL`；新增 `sve_merge_fp16_fp16_8x3VL`、`sve_merge_fp32_bf16_8x3VL` |
| `transform-sve.cpp` | 7 | 路径重构 |
| `mergeresults-sve.cpp` | 8 | 路径重构 |
| `interleave_indirect-sve.cpp` | 51 | 较大差异，路径+逻辑重构 |
| `indirect-interleaves/list-sve.hpp` | 12 | 新增/调整 |

### 7.7 新增/删除的 SVE 文件

**53.1.0 新增**（23.11 没有）：

| 文件 | 用途 |
|------|------|
| `sve_ffhybrid_fp16fp32fp16_mla_6x4VL` | 新 fp16→fp32→fp16 混合精度 |
| `sve_ffhybrid_fp16fp32_mla_6x4VL` | 新 fp16→fp32 混合精度 |
| `sve_ffinterleaved_bf16fp32_dot_8x3VL` | 新 BF16 dot 交错 |
| `sve_hybrid_fp16fp32fp16_mla_6x4VL` | 新混合精度 hybrid |
| `sve_hybrid_fp16fp32_mla_6x4VL` | 新混合精度 hybrid |
| `sve_hybrid_u8s8qa_dot/mmla_4x4VL` | 新无符号×有符号 int8 |
| `sve_hybrid_u8s8s32_mmla_6x4VL` | 新无符号×有符号 int8 |
| `sve_interleaved_u8s8s32_mmla_8x3VL` | 新无符号×有符号 int8 交错 |
| `sve_transpose_interleave_2VL.hpp` | 新 2VL 交错变换 |
| `sve_transpose_interleave_2VL_2x4_fp32bf16.hpp` | 新 2VL bf16 交错 |
| `sve_merge_fp16_fp16_8x3VL.hpp` | 重命名+新合并 |
| `sve_merge_fp32_bf16_8x3VL.hpp` | 新 bf16 合并 |
| `sve_merge_fp32_fp32_8x3VL.hpp` | 重命名 |
| `sve_merge_s32_s32_8x3VL.hpp` | 重命名 |
| `sve_merge_u32_u32_8x3VL.hpp` | 重命名 |

**53.1.0 删除**（23.11 有）：

| 文件 | 说明 |
|------|------|
| `misc-sve.cpp` | 杂项 SVE 函数，已合并到其他文件 |
| `sve_transpose_interleave_8VL.hpp` | 合并到其他变换文件 |
| `sve_transpose_interleave_8VL_2x2.hpp` | 合并到其他变换文件 |
| `sve_merge_*_3VLx8.hpp`（4 个） | 重命名为 `*_8x3VL` 命名 |

### 7.8 SVE 差异对 Winograd 性能的影响

即使绕过 SME GEMM 内核拦截（强制走 SVE 路径），53.1.0 的 SVE 内核实现也不同于 23.11：

```
两层性能差异：
  层1（选择路径）：53.1.0 新增 SME GEMM 内核拦截 → 选中未优化的 SME 内核
  层2（SVE 实现）：即使走 SVE 路径，53.1.0 的 SVE 内核也已被重写

  层1 是主因（选中完全不同的 GEMM 类型）
  层2 是次因（同类型 GEMM 的实现差异）
```

SVE 内核重写的性能影响方向不确定（可能更好也可能更差），需实测。但参数传递重构（直接绑定→间接 args 结构体）对小 GEMM 有额外 prologue 开销，对 Winograd 的 36 个 batched 小 GEMM 不利。

### 7.9 验证 SVE 内核差异的方法

1. **编译时确认 SVE 内核选择**：在 53.1.0 中注释掉 `gemm_fp32.cpp` 的 `#ifdef ARM_COMPUTE_ENABLE_SME` 段，强制走 SVE 路径，对比与 23.11 的性能。
2. **查看 GEMM 内核名**：ACL 日志会输出选中的内核名（如 `sve_hybrid_fp32_mla_6x4VL`）。
3. **对比汇编**：用 `objdump -d` 反汇编两个版本的 GEMM kernel 函数，对比指令序列。
4. **Profile prologue**：用 perf 分析 GEMM kernel 的 prologue（参数加载）占比，确认间接 args 开销。

---

### 7.10 附录：内联汇编入门解读（面向不熟悉汇编的工程师）

> 本节帮助不熟悉 ARM 汇编的工程师理解第 7.4 节的内容。用通俗语言解释关键概念，配合图示和类比。

#### 7.10.1 什么是"内联汇编"

C++ 代码中可以嵌入 CPU 直接执行的汇编指令，语法是：

```cpp
__asm__ __volatile__(
    "mov x9, %x[output_ptr]\n"    // 汇编指令
    "ldr x11, [%x[args_ptr]]\n"   // 另一条汇编指令
    : /* 输出操作数 */
    : [output_ptr] "r" (output_ptr),   /* 输入操作数：C 变量 output_ptr 绑定到 %[output_ptr] */
      [args_ptr] "r" (&args_struct)
    : "cc", "memory", "x9", "x11"      /* clobber 列表：这些寄存器被汇编修改了 */
);
```

**关键概念**：
- `%x[name]` 是一个**占位符**——编译器会把 `%x[output_ptr]` 替换成存放 C 变量 `output_ptr` 值的那个物理寄存器名（比如 `x3`）
- `"r"` 约束告诉编译器："请把这个 C 变量放进一个通用寄存器，然后把寄存器名填到汇编里"
- `clobber` 列表告诉编译器："这段汇编会动这些寄存器，你别用它们存别的变量"

**类比**：就像你在写信时写"请把[收件人姓名]贴在信封上"，邮局（编译器）看到后会自动把 `[收件人姓名]` 替换成实际名字。`%x[output_ptr]` 就是这个占位符。

#### 7.10.2 AArch64 寄存器速查

| 寄存器 | 类比 | 用途 |
|--------|------|------|
| `x0`-`x30` | 31 个"口袋" | 通用寄存器，存指针/整数/地址（64 bit） |
| `x31` = `XZR` | "空口袋" | 零寄存器，读出恒为 0（特殊的） |
| `z0`-`z31` | 32 条"流水线" | SVE 向量寄存器，存 16 个 float（512 bit） |
| `p0`-`p15` | 16 个"开关阵列" | 谓词寄存器，控制哪些 lane 参与运算 |
| `v0`-`v31` = `q0`-`q31` | 128 bit NEON | 非 SVE 的旧向量寄存器 |

#### 7.10.3 关键指令"翻译"表

以文档中出现的指令为例，逐条翻译成人话：

| 汇编指令 | 人话翻译 | 类比 |
|----------|----------|------|
| `mov x12, %x[bias]` | "把 bias 指针的值复制到 x12 寄存器" | 从一个口袋复制到另一个口袋——**快**（1 周期） |
| `ldr x10, [%x[args_ptr], %[offsetof_bias]]` | "从内存地址 args_ptr+offset_of_bias 处读取一个值到 x10" | 去仓库取一件物品——**慢**（3-10 周期，要访存） |
| `cbz %x[ablocks], 6f` | "如果 ablocks==0，跳到标号 6" | 如果盒子是空的，直接跳过这步 |
| `whilelt p4.s, x20, x11` | "对每个 lane：如果 x20+i < x11，谓词 p4 的第 i 位=真" | 逐个检查"这个位置还需要处理吗？" |
| `ld1w {z31.s}, p4/Z, [addr]` | "从内存加载一个向量到 z31，只填 p4 为真的 lane，其余置零" | 按开关阵列选择性地装载流水线 |
| `fmla z31.s, p1/M, z16.s, z7.s` | "z31 = z31 + z16 × z7（只对 p1 为真的 lane）" | 批量乘加——一条指令做 16 次乘加 |
| `st1w {z31.s}, p0, [addr]` | "把 z31 存到内存，只存 p0 为真的 lane" | 按开关阵列选择性地写回 |
| `ptrue p5.b` | "谓词 p5 所有位都设为真" | 全部开关打开 |
| `incw x20` | "x20 += SVE 向量长度对应的元素数（如 16）" | 指针前进一大步 |

#### 7.10.4 两种参数传递方式的图解

**23.11：直接绑定（部分参数）**

```
C++ 侧：
  void *output_ptr = (void *)(output_arg.indirect.ptr);
  const float *bias = bias_ptr;
  // asm 约束：[output_ptr] "r" (output_ptr),  [bias] "r" (bias)

编译器把 C 变量直接放进寄存器：

  ┌─────────────┐         ┌──────────┐
  │ C 变量       │──mov──→ │ x9       │  output_ptr 直接进 x9
  │ output_ptr  │         └──────────┘
  └─────────────┘
  ┌─────────────┐         ┌──────────┐
  │ C 变量       │──mov──→ │ x12      │  bias 直接进 x12
  │ bias        │         └──────────┘
  └─────────────┘

  汇编：mov x9, %x[output_ptr]     →  编译后：mov x9, x3（寄存器间复制，1 周期）
        mov x12, %x[bias]           →  编译后：mov x12, x5（寄存器间复制，1 周期）

  ★ 总开销：2 条 mov（2 周期），不访存
```

**53.1.0：间接 args 结构体（全部参数）**

```
C++ 侧：
  struct kernel_args { void *output_ptr; const float *bias; size_t N; ... } ka;
  ka.output_ptr = (void *)(output_arg.indirect.ptr);
  ka.bias = bias;
  // asm 约束：[args_ptr] "r" (&ka),  [offsetof_output_ptr] "I" (offsetof(...,output_ptr))

编译器只绑定 &ka 到一个寄存器，汇编再从内存结构体加载：

  ┌──────────────┐         ┌──────────┐
  │ &ka          │──mov──→ │ x4       │  args_ptr 指向结构体
  └──────────────┘         └──────────┘
                                  │
                                  ▼ 指向内存中的结构体
  内存中的 kernel_args 结构体：
  ┌──────────────────────────────────┐
  │ offset 0:   output_ptr (void*)   │ ← ldr x9,  [x4, #offsetof_output_ptr]
  │ offset 8:   bias (float*)        │ ← ldr x10, [x4, #offsetof_bias]
  │ offset 16:  N (size_t)           │ ← ldr x12, [x4, #offsetof_N]
  │ ...                              │
  └──────────────────────────────────┘

  汇编：ldr x9,  [%x[args_ptr], %[offsetof_output_ptr]]  → 从内存读（3-10 周期）
        ldr x10, [%x[args_ptr], %[offsetof_bias]]         → 从内存读（3-10 周期）
        ldr x12, [%x[args_ptr], %[offsetof_N]]            → 从内存读（3-10 周期）

  ★ 总开销：3 条 ldr（9-30 周期），要访存
```

**类比**：
- **23.11 的直接绑定**就像"快递员直接把包裹递到你手上"——一步到位，快
- **53.1.0 的间接结构体**就像"快递员把包裹放到快递柜，你要自己去柜子取"——多一步，慢

#### 7.10.5 为什么对 Winograd 影响大

Winograd F(4,4,3,3) 需要跑 **36 个 batched 小 GEMM**。每个 GEMM 调用一次 GEMM kernel 函数：

```
一次 Winograd 前向：
  输入变换 → [36 个 GEMM] → 输出变换

每个 GEMM 调用 GEMM kernel：
  ┌─ prologue（参数加载）──┐   ┌─ 主计算循环 ────────┐   ┌─ epilogue ─┐
  │ mov/ldr 参数到寄存器    │ → │ fmla × 大量迭代     │ → │ st1w 写回   │
  │ ~10 条指令             │   │ ~数千条指令          │   │ ~几条指令   │
  └────────────────────────┘   └──────────────────────┘   └────────────┘
   23.11: ~2 周期(mov)         对大 GEMM：prologue 占比 <0.1% → 影响可忽略
   53.1.0: ~9-30 周期(ldr)     对小 GEMM：prologue 占比 可能 1-5% → 有影响

  ★ 36 个 GEMM × 额外 prologue 开销 = 累积可观的性能损失
```

对**大 GEMM**（如直接卷积的 im2col GEMM，M=N=K 都很大），prologue 的几条指令相对数万条 fmla 可忽略。但对 **Winograd 的 batched 小 GEMM**（每个 GEMM 的 M=tiles 可能只有几十到几百），prologue 占比上升，53.1.0 的 `ldr` 比 23.11 的 `mov` 多出的 7-28 周期 × 36 次，累积可达数百到上千周期。

#### 7.10.6 53.1.0 为什么改用间接结构体

这是一个合理的设计重构动机，尽管对 Winograd 有副作用：

| 动机 | 解释 |
|------|------|
| **减少寄存器占用** | 直接绑定每个参数占一个 asm 约束位，参数多时编译器寄存器压力大；间接结构体只需 1 个约束（`args_ptr`） |
| **支持更多参数** | 53.1.0 新增了 `bias`、`accumulate`、激活参数等，直接绑定会耗尽约束位 |
| **统一接口** | 所有 GEMM 内核用同一个 args 结构体模式，代码更统一 |
| **可扩展性** | 新增参数只需加结构体字段，不改 asm 约束签名 |

> **总结**：53.1.0 的间接参数传递是面向"通用性和可扩展性"的设计选择，对大 GEMM 无害，但对 Winograd 的 36 个小 GEMM 有额外的 prologue 开销。这是性能差异的**次要因素**（主要因素仍是 SME GEMM 内核拦截，见第 4 节）。
