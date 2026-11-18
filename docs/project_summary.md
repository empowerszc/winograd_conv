# Winograd 卷积复刻项目 — 工作总结

> **用途**：工作汇报 / 技术总结的一站式底稿。涵盖「做了什么、为什么这样做、比同类实现强在哪里、
> 性能证据、局限与后续」。
> **配套细节**：算法推导见 `docs/algorithm.md`；性能历程见 `PERFORMANCE_ANALYSIS.md`；
> 与 oneDNN 的对照见 `docs/onednn_comparison.md`；SME 内核见 `docs/sme_fmopa_gemm.md`。
> **日期**：2026-11-18

---

## 0. 一页纸摘要

**做了什么**：从零复刻了一套面向 Armv9（SVE-512 + SME）的 Winograd 快速卷积实现——
F(4,4,3,3) 与 F(2,2,3,3) 两种配置，NEON / SVE / SME 三级 ISA 内核，
可插拔 GEMM 后端（arm_gemm JIT / KML / OpenBLAS / naive / 自研 SME FMOPA），
NHWC 计算核心 + NCHW 转换包装，OpenMP 全流程并行，并配齐正确性验证与性能基准工具链。

**关键结果**：

| 对照 | 口径 | 结果 |
|---|---|---|
| vs OpenBLAS（鲲鹏版）※ | 集群 920F，59 形状，同作业，t16 | **59/59 全胜**，geomean **1.99x**，合计 **2.50x**，最大 5.34x |
| vs oneDNN `wino:acl`（同算法）※ | 集群 920F，59 形状，同作业同线程 | **全面领先 1.15–9.5x** |
| vs oneDNN 自动选优（auto）※ | 同上 | 大形状 **快 1.1–2.8x**；小形状慢 0.09–0.91x |
| vs OpenBLAS 的峰值效率 ※ | 最大形状 | 折算 GEMM ≈ 960 GFLOPS / 16 核 ≈ **SVE-512 峰值 94%** |
| 正确性 | fp64 直接卷积参考 | 22/22 用例通过；相对误差 **2–6e-6**（容差 1e-4） |
| 自研 SME FMOPA GEMM | 本机 608 核，59 形状，绑核 | t1 合计快 2.5%（31/59 胜，最高 **2.2x**）；t16 合计快 **10.4%**（32/59 胜） |

> ※ 标记的行为**集群 920F** 历史测量（后端 arm_gemm vs OpenBLAS、对照 oneDNN）；
> SME 行为**本机 608 核**测量（后端 SME vs KML）。两组数字来自不同机器，
> 不可跨行直接比较，只可各自纵向解读。

**结论**：在「Winograd 变换 + 小切片 GEMM」这个形态上，本实现通过**布局契约、后端选择、
缓存复用与并行结构**的系统性设计，稳定超过通用 BLAS 后端与 oneDNN 的 Winograd 路径；
自研 SME 内核已验证可行并在特定形状区间显著领先，为下一步替换通用 GEMM 打下地基。

---

## 1. 项目定位

| 项 | 内容 |
|---|---|
| 性质 | 教学 / 研究型复刻：理解 ACL Winograd 的算法原理与 ISA 优化策略 |
| 语言 / 标准 | C++17 + AArch64 NEON/SVE intrinsics + SME 内联汇编 |
| 目标平台 | 鲲鹏 920F（Armv9，SVE-512，SME `smef32f32`）；本机 608 核 / 32 NUMA / 无 L3 |
| 规模 | 全仓库 ~5.9k 行（含测试与基准）；核心实现（`include/` + `src/`）约 3.6k 行 |
| 外部依赖 | 可选：arm_gemm（ACL 源码子集）、KML kblas、OpenBLAS；均可关闭退回 naive |

**要回答的问题**：
1. Winograd 的三步变换在向量 ISA 上如何高效实现（尤其零系数与 tail）？
2. 36 个「小切片 GEMM」应该交给谁做——通用 BLAS、JIT 内核，还是矩阵扩展（SME）？
3. 在无 L3、NUMA 复杂的机器上，变换 / GEMM / 输出的流水线怎么切分与并行？
4. 如何证明「快」是真快（同作业、同线程、同口径）而不是测量假象？

---

## 2. 技术方案

### 2.1 算法

```
直接卷积：每输出像素 9 次乘加（3×3 核），F(4,4) 输出 16 像素 → 144 次
Winograd：V = G·g·Gᵀ/576        （权重变换，每层一次）
          U = Bᵀ·d·B            （输入变换，每 tile）
          M = U ⊙ V（按通道 GEMM）（36 个 Winograd 域元素）
          f = Aᵀ·M·A + bias + ReLU（输出变换，每 tile）
          → 等效每像素约 2.25 次乘法
```

- **F(4,4,3,3)**：6×6 输入 tile → 4×4 输出，乘法节省最多，是本项目主路径。
- **F(2,2,3,3)**：4×4 输入 tile → 2×2 输出，矩阵系数全为 ±1（无放大），精度更好、变换更便宜。

### 2.2 数据流与并行结构

```
[N][IC][IH][IW]
   │  (NCHW→NHWC 转换包装)
   ▼
[N][IH][IW][IC]
   │
   ├─ Phase 1  输入变换  Bᵀ·d·B      每 tile，通道向量化
   ├─ Phase 2  GEMM      M = U·Vᵀ    36 个切片，按 ts 并行
   └─ Phase 3  输出变换  Aᵀ·M·A + bias + ReLU
   │
   ▼
[N][OH][OW][OC] → (NHWC→NCHW 转换) → 输出
```

- **单个 `#pragma omp parallel` 区域**包住权重变换 + 所有 batch 的 Phase 1/2/3，
  全流程只有必要的 barrier，避免反复 fork/join（实测比分开区域快，尤其小 OC）。
- 每个线程持有 6 个 `thread_local` 缓冲（d/U/M/f tile、权重临时），跨调用复用，
  不在热路径上做堆分配。

### 2.3 三种 ISA 内核

| 方面 | NEON | SVE | SME |
|---|---|---|---|
| 向量宽度 | 128 bit（4 float） | VL（SVE-512 = 16 float） | 同 SVE + 64×64 ZA tile |
| 变换实现 | intrinsics，三段降级（4→2→1） | intrinsics + 谓词，无降级 | 输出变换用 FMOPA 外积 + Kronecker 展开 |
| tail 处理 | 手工 3 份代码 | `whilelt` 谓词 | 同 SVE |
| 通道并行度 | 4 | 16 | 64（4 × VL） |
| 典型收益 | 基线 | 相对 NEON 约 4x 拷贝/变换吞吐 | 变换 t1 geomean 1.02、变换密集形状最高 1.33x；GEMM 侧见 §4.4 |

### 2.4 GEMM 后端（可插拔，编译期选择）

| 后端 | 实现 | 定位 |
|---|---|---|
| **arm_gemm** | ACL 53.1.0 的 SVE JIT 内核（与 oneDNN 同源） | **生产默认**，59/59 胜 OpenBLAS |
| **KML kblas** | 华为 KML 的 `cblas_sgemm`（locking 变体） | 本机（毕昇环境）日常首选 |
| **OpenBLAS** | 鲲鹏版 `cblas_sgemm` | 对照基线 |
| **SME FMOPA** | 自研 ZA 外积内核（`WINO_GEMM=sme`） | 实验，已验证，特定形状领先 |

### 2.5 交付物清单

| 模块 | 内容 | 位置 |
|---|---|---|
| 变换内核 | NEON / SVE / SME 三套 × F(2,2)+F(4,4)，权重/输入/输出全流程 | `include/winograd_transforms*.hpp` |
| 端到端流水线 | NHWC 计算核心 + NCHW 转换包装、OpenMP 单区域并行、线程本地缓冲 | `src/winograd_conv.cpp` |
| GEMM 适配层 | arm_gemm / KML / OpenBLAS / naive / SME 五种，布局契约统一 | `src/winograd_conv.cpp` |
| 正确性体系 | fp64 参考 + 相对容差、bit-exact 布局回归、变换级调试 | `tests/test_winograd.cpp`、`tests/test_ref_vs_nchw.cpp` |
| 基准工具 | CSV 驱动、GFLOPS、细粒度 `--timing`、`--verify` 门、单 case profiling | `tests/bench_winograd.cpp`、`tests/profile_case.cpp` |
| 用例集 | 59 形状（ResNet/VGG/轻量/大 batch/tiles×IC 网格），分节标注测什么假设 | `shapes/conv_all.csv` |
| 对照工具链 | 单作业 A/B、oneDNN e2e + benchdnn 三列、结果合并与诊断 | `tools/onednn/` |
| 文档体系 | 算法、性能、对照、排障、SME 内核、本总结 | `docs/` |

---

## 3. 关键工程决策（差异点所在）

| # | 决策 | 原因 / 效果 |
|---|---|---|
| 1 | **计算核心固定 NHWC**，NCHW 走转换包装 | 无 L3 机器上 NCHW 的跨步 tile 提取有 ~16x cache line 放大；转换是顺序搬运 |
| 2 | **统一矩阵常量**，三种 ISA 共用 | 修一次矩阵，NEON/SVE/SME 同时正确（历史上 3 次矩阵 bug 的教训） |
| 3 | **SVE 用谓词而非降级** | 一份代码覆盖任意 VL 与 tail；NEON 因无谓词必须三份 |
| 4 | **权重变换 + 全流程合并进单个 OpenMP 区域** | 避免额外 fork/join（曾因独立 region 在小 case 慢 ~7%，Case 0 t32 6.7→7.2 ms） |
| 5 | **线程本地缓冲跨调用复用** | 消除每 tile `std::vector` 的堆锁竞争与分配 churn |
| 6 | **V 跨调用缓存（指针 + 权重指纹）** | 推理场景权重恒定；KML t16 geomean 1.07x，最高 1.74x |
| 7 | **tile 边界按实际维度裁剪** | 修复奇数 IH/IW 越界读（会读到下一通道） |
| 8 | **GEMM 布局契约：U/V 一律 K 主序** | 后端只需一次 B 预转置；行主序散写曾让小形状明显退化（见终局基准的散写回归） |
| 9 | **SME 后端把 ZA 行=OC、列=tile** | drain 的 64 B 正好是 M_buf 一行 16 OC，零后置换 |
| 10 | **计时只信无 debug 编译** | 调试/选核开销会不成比例放大小形状（历史上误判 1.6x） |

---

## 4. 与同类实现的对比

### 4.1 vs ACL / oneDNN（同算法对比）

| 维度 | 本项目 | ACL 53.1.0 / oneDNN |
|---|---|---|
| 算法 | F(4,4,3,3) 主路径 + F(2,2,3,3) | Winograd（wino:acl，疑似以 F(2,2) 为主） |
| 变换内核 | SVE intrinsics（编译器调度）；SME 输出变换为手写汇编 | 手写 SVE 汇编，完全展开、周期级调度（`sve_fp32_6x6.cpp`） |
| GEMM | arm_gemm JIT / KML / 自研 SME FMOPA | arm_gemm JIT（自动选 SVE/SME 内核） |
| 数据布局 | NHWC + 行主序 U/M；SME 路径 K 主序打包 | Winograd 专用矩阵布局（`matrix_stride`，免 scatter/gather） |
| 权重处理 | 跨调用 V 缓存 + 指纹防 ABA | primitive 内缓存 TransformedWeights |
| 多线程 | 单 OpenMP 区域，显式 barrier | NEScheduler 细粒度任务队列 |
| 代码规模 | ~3.6k 行核心，可完整阅读 | 数十万行，工业级 |

**结果**：
- **同算法 PK（e2e_wino 列）**：我们 **1.15–9.5x** 全面领先。两列独立 driver
  （e2e 与 benchdnn）互证，数据可信。
- **对 oneDNN 自动选优（e2e_auto 列）**：大形状快 **1.1–2.8x**；小形状慢
  **0.09–0.91x**。分水岭在 `IC × spatial ≈ 256 × 56²`——小形状下 Winograd 的
  变换固定开销摊不薄，oneDNN 改用 brgconv（直接卷积）反而更优（19/59 个形状它正是这么选的）。

> **诚实说明**：本项目的**单个 kernel 并不比 ACL 的手写汇编强**（对方是周期级调度）。
> 领先来自：算法选择（F(4,4) 对大形状）、布局契约（免 scatter/gather）、
> 缓存复用（V 跨调用）、并行结构（少 fork/join/barrier），以及
> **在固定开销主导的小负载上更契合 920F 的工程实现**。

### 4.2 vs 通用 BLAS 后端

> ⚠️ 下表中**集群与本地是两套机器**，绝对毫秒数不可跨行比较，只用于说明「后端优劣依平台而定」。

| 后端 | 测量环境 | 59 形状合计 | 相对 |
|---|---|---|---|
| arm_gemm（ACL 53.1.0 JIT SVE） | 集群 920F，t16，同作业 | **321.9 ms** | 基准 |
| OpenBLAS（鲲鹏版） | 同上 | 804.9 ms | 2.50x 慢 |
| arm_gemm | 本机 608 核，t16 | 328 ms | 基准 |
| KML kblas | 同上 | **233 ms** | 快 1.41x |

- arm_gemm 对 **K 短、M 小**的切片明显更强：深层瘦条形状（1,2048,7²,512）达 **5.34x**。
- 大形状折算 GEMM ≈ **960 GFLOPS / 16 核**，接近 SVE-512 峰值 94%，说明后端已接近硬件上限。
- KML 在本机（毕昇）整体优于 arm_gemm，但**不可跨机迁移结论**：
  大 IC 小 tile（20²/40² 族）KML 快 2–2.5x，小 OC 长序列 arm_gemm 略胜。
  → 本项目把 GEMM 做成**可插拔**正是为此：后端优劣依平台与形状而定。

### 4.3 vs 直接卷积

| 场景 | 结论 |
|---|---|
| 大形状（IC×spatial 大） | Winograd 明显占优（乘法数 144→36 的收益能摊薄变换） |
| 极小形状（batch=1、IC≤64、spatial≤20²） | 直接卷积可能更优——Winograd 的变换与固定开销无法摊薄 |
| 自研 SME GEMM | 改变了这一边界：小空间深通道（7²/14²）t1 快 2.2x；大空间形状（160² 族）t16 快 1.7x |

### 4.4 自研 SME FMOPA GEMM 的定位

| 维度 | 说明 |
|---|---|
| 设计 | ZA 行 = 16 OC，列 = 16 tile，4 个 ZA tile = 64 OC；K 循环每轮 4 条 `fmopa` |
| 优势 | M（tile 数）再小也不损失 tile 利用率；drain 直接落 M_buf 行；B 操作数随 V 缓存预打包 |
| 现状（t1 / t16，59 形状，绑核） | t1 31/59 胜、合计快 2.5%（小空间深通道最高 **2.2x**）；t16 32/59 胜、合计快 **10.4%**（大空间形状最高 1.7x）；小空间大 OC 落后 1.4–3.4x |
| 意义 | 证明「Winograd 小切片 GEMM 用矩阵扩展比通用 BLAS 更合适」这一假设，并给出可继续优化的完整工程骨架 |

---

## 5. 性能结果（证据链）

### 5.0 优化累积收益（Case 0：4,192,40²,192，历史阶段）

优化是逐项 A/B 堆出来的，不是一次性设计的结果：

| 阶段 | t16 (ms) | 累计收益 |
|---|---|---|
| ① NCHW 基线 | 14.2 | — |
| ② +NHWC 布局 | ~14 | 布局本身收益有限，但为后续铺路 |
| ③ +GEMM 并行 / 调度策略 / 去堆分配 | 9.5 | 1.5x |
| ④ +合并 OpenMP 区域（少 fork/join） | 7.0 | 2.0x |
| ⑦ +权重变换并入主区域 | 7.3 | 2.0x |
| **A1/A2/A3**（去无用清零 + 缓冲跨调用复用 + SVE 化拷贝） | **1.94** | **7.3x** |

> 上表是 OpenBLAS 后端时代的历史数据（同一机器、同一形状），用于说明**优化结构**
> 而非当前绝对值；当前绝对值见 `docs/final_benchmark_bfd6b1e.md`（arm_gemm 后端）。

关键结论：**收益最大的一步不是「换 GEMM 内核」，而是消除无谓的内存工作**
（每次调用的清零、堆分配、NEON 4-float 拷贝）。这与「Winograd 在 920F 上是
内存/固定开销受限」的判断一致，也解释了为什么 SME 这类「提高算力」的优化
只在特定形状区间见效。

### 5.1 端到端 vs OpenBLAS（59 形状，同作业，t16）

> 本节及 §5.2 为**集群 920F** 的测量（arm_gemm 后端 vs OpenBLAS/oneDNN）；§5.3 为
> **本机 608 核**的测量（SME vs KML）。两者是不同机器，请勿横向比较。

- **59/59 全胜**；geomean **1.99x**；合计 321.9 ms vs 804.9 ms（**2.50x**）。
- 分布：>3x 8 个、2–3x 17 个、1.5–2x 20 个、1.25–1.5x 10 个、<1.25x 4 个（均为 <1.1 ms 微负载）。
- 完整逐形状表见 `docs/final_benchmark_bfd6b1e.md`。

### 5.2 vs oneDNN（三列口径）

| 对照列 | oneDNN 算法 | 结论 |
|---|---|---|
| `e2e_auto` | 自动（40 wino:acl + 19 brgconv） | 大形状我们快 1.1–2.8x；小形状慢 0.09–0.91x |
| `e2e_wino` | 强制 wino:acl | 我们快 **1.15–9.5x**（全形状） |
| `benchdnn_wino` | wino:acl（driver 口径） | 与 `e2e_wino` 一致，互证 |

### 5.3 SME 后端（实验，绑核）

| 线程 | SME 合计 | KML 合计 | 结论 |
|---|---|---|---|
| t1（固定单核） | 3317.9 ms | 3403.3 ms | SME 快 **2.5%**，31/59 形状胜出 |
| t16（绑 16 核） | 345.9 ms | 386.3 ms | SME 快 **10.4%**，32/59 形状胜出 |

- **胜得最多**：小空间 + 深通道（7²/14²）——KML 在 M 极小时效率崩塌（20–22 GFLOPS），
  SME 不受 M 规模影响，t1 最高 **2.2x**；t16 大空间形状（160² 族）最高 **1.7x**
  （#57 `192,160²` 0.58x、#58 `384,160²` 0.66x）。
- **落后最多**：小空间 + 大 OC（28²/20² 族，比值 1.4–2.0x）；t16 下 tile 数少的形状
  最差 3.45x（#6 `768,20²`），因为并行度只来自 36 个 ts。
- 完整逐形状表与原因分析见 `docs/sme_fmopa_gemm.md` §5。

### 5.4 线程扩展性

- 大 tile 数形状（1600 tile）t1→t38 扩展约 **26x**；小 tile 数形状在 16 线程后平台化
  （无 L3 + 768 KB L2/核 的带宽限制）。
- 并行效率瓶颈已从「GEMM 内核」转为「barrier + NUMA 远程访问」。

---

## 6. 正确性保证

| 手段 | 说明 |
|---|---|
| fp64 直接卷积参考 | `--verify` 用双精度直接卷积生成真值，逐元素比较 |
| **相对容差**判据 | `err < 1e-4 × max(abs(ref)) + 1e-5`；固定绝对容差会误报大 IC（幅值 ~1000） |
| 布局回归 | `test_ref_vs_nchw` 断言「NCHW 包装」与存档原生 NCHW 内核 **bit-exact** |
| 覆盖 | NCHW 12 例 + NHWC 10 例 = 22 例；F(2,2)/F(4,4)、多 batch、奇数尺寸、ReLU |
| 精度实测 | fp32 相对误差 2–6e-6，与 IC 无关（绝对误差随 IC 线性增长，幅值同步增长） |
| 微内核自检 | SME GEMM 输出与标量参考 max abs diff = 0 |

---

## 7. 可复现性与工程化

**构建矩阵**（均已验证）：

```bash
# NEON 基线
cmake .. -DCMAKE_BUILD_TYPE=Release
# SVE
cmake .. -DENABLE_SVE=ON
# SME（含 SVE）
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON
# arm_gemm 后端（与 oneDNN 同内核）
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON -DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=...
# KML 后端（本机推荐）
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON -DUSE_KML=ON
```

**测量协议（避免假象）**：

1. 性能对比必须**同一作业内 A/B**（该机器跨作业存在 3–7x 性能态漂移）。
2. 计时只信**无 debug** 构建；调试开关会放大小形状耗时。
3. 对照 oneDNN 时必须：同绑核、同实现、primitive 跨迭代复用。
4. 线程数固定并设置 `OMP_PROC_BIND=close OMP_PLACES=cores`。

**工具链**：CSV 驱动基准（`bench_winograd`）、单 case profiling（`profile_case`）、
oneDNN 对照脚本（`tools/onednn/`）、基准用例集（`shapes/conv_all.csv`，59 形状，
按「固定开销 / VGG / ResNet / 轻量 / 大 batch / tiles×IC 网格」分节）。

---

## 8. 局限

1. **小形状落后 oneDNN auto**：batch=1、IC≤128、spatial≤56² 区间慢 0.09–0.91x，
   根因是变换固定开销；oneDNN 在这些形状改走直接卷积。
2. **单个 kernel 不如 ACL 手写汇编**：SVE 变换由编译器调度，指令级效率有差距；
   差距靠算法与工程结构弥补。
3. **SME GEMM 尚未超越通用 BLAS 的整体水平**：仅在特定形状区间领先；
   `Ust→Ug` 转置与 OC 尾块浪费是主要成本。
4. **多线程扩展性在 16 线程后平台化**：无 L3 + 内存带宽 + barrier 开销。
5. **NCHW 端到端**：依赖转换包装，整图两遍数据搬运；在 9 线程档曾落后 oneDNN。
6. **SME 内联汇编的 ABI 陷阱**：`smstart/smstop` 使所有 z 寄存器与谓词不可预测，
   必须显式 clobber `d8–d15` 与 `p1–p15`（详见 `docs/sme_fmopa_gemm.md` §6）。

---

## 9. 后续路线（按预期收益）

| 优先级 | 方向 | 预期 |
|---|---|---|
| 1 | **小形状 hybrid 调度**：小形状走 im2col + 直接卷积，大形状走 F(4,4) | 小形状 0.09x → ~1.0x |
| 2 | **SME：消除 `Ust→Ug` 转置**（变换直接写 K 主序） | 大 IC 场景显著 |
| 3 | **SME：OC 尾块专用 tile 数**（96 → 3 tile × 2 窗） | 省 25% FMOPA |
| 4 | **形状感知的后端选择**（SME / KML / arm_gemm） | 拼接各自优势区间 |
| 5 | **F(4,4) 专用展开 SVE 变换**（Bᵀ 50% 零系数） | 变换 −30~50% |
| 6 | **大 IC 通道分块**（L1 32KB 约束） | TileExt/InXform 改善 10–25% |

---

## 10. 汇报要点（可直接用于 PPT）

1. **一句话**：从零实现了一套 Armv9 的 Winograd 卷积，在 59 个真实形状上
   **全面超过 OpenBLAS（1.99x geomean）**，同算法对比 **领先 oneDNN/ACL 1.15–9.5x**。
2. **技术完整性**：F(2,2)/F(4,4) 双配置 × NEON/SVE/SME 三级 ISA × 5 种 GEMM 后端，
   全流程可运行、可验证、可复现。
3. **算法选择的胜利**：F(4,4) 在大形状上把乘法数从 144 降到 36，
   配合布局契约与缓存复用，把理论收益落成实测收益。
4. **工程细节的价值**：单 OpenMP 区域、线程本地缓冲、V 跨调用缓存、边界修复——
   每一项都有 A/B 数据支撑（合计带来数倍提升）。
5. **前沿探索**：自研 SME FMOPA GEMM，验证了「矩阵扩展更适合 Winograd 小切片」
   的假设——小空间深通道形状 **2.2x**、大空间形状最高 **1.7x**、t16 合计 **快 10%**，
   并沉淀了完整的内核骨架与踩坑经验
   （含一个极具迷惑性的 ABI 陷阱：`smstart/smstop` 摧毁 callee-saved 向量寄存器）。
6. **方法论**：所有结论都建立在「同作业、同线程、同口径」的对照与 fp64 真值验证之上，
   并保留了完整的弯路复盘（`docs/debugging_lessons.md`）。
