# 与 ACL Winograd 实现的差异与优势

> **用途**：汇报 / 展示底稿。回答三个问题——
> ① 这个项目和 ACL 的 Winograd 到底哪里同、哪里不同？
> ② 差异带来了什么可量化的好处？
> ③ 哪些地方我们还不如它（不藏）。
>
> **事实基础**：ACL 侧全部结论来自本仓库内的 `ComputeLibrary-53.1.0/` 源码逐行核对
> （`src/core/NEON/kernels/convolution/winograd/`，8 056 行），不是二手描述。
> 本项目侧来自 `include/`、`src/winograd_conv.cpp`。
> **配套文档**：`docs/project_summary.md`（工作总结）、`docs/sme_fmopa_gemm.md`（SME 内核）、
> `docs/acl_reference/`（ACL 源码分析）、`docs/onednn_comparison.md`（对照方法论）。
> **日期**：2026-11-18

---

## 0. 一页纸结论

| 结论 | 内容 |
|---|---|
| **同源** | 算法、变换矩阵、tile 划分、Winograd 域布局、输出变换融合 bias+ReLU、SME 输出变换实现——**逐项一致**，本项目是从 ACL 的思路复刻出来的 |
| **最大的结构差异** | ACL 是「**按形状贪心选实现**」的库（tile 大小、是否 SME、是否回退到直接卷积都由运行时判据决定）；本项目是「**固定 F(4,4) 主路径 + 可插拔后端**」的研究实现 |
| **最大的实现差异** | ACL 的变换是**手写全展开汇编**（寄存器驻留、无中间内存）；本项目是**泛型 SVE intrinsics**（两趟、走 thread_local 中间缓冲） |
| **结果** | 同算法 PK（oneDNN 强制 `wino:acl`）：本项目 **1.15–9.5x**；对 oneDNN 自动选优：大形状 **1.1–2.8x**、小形状 **0.09–0.91x** |
| **诚实结论** | 领先**不来自**「单 kernel 比 ACL 强」——恰恰相反，我们的变换内核比 ACL 手写汇编慢；领先来自**算法选择更彻底、GEMM 调用组织、缓存生命周期、并行结构** |
| **代码规模** | 核心 3 636 行（含全部三阶段 × 三种 ISA × 两种 F 配置）vs ACL Winograd 目录 8 056 行（不含 driver） |

**一句话**：同一套数学，两种工程哲学。ACL 用「手写内核 + 形状自适应」把每一档都做到工业级；
本项目用「泛型内核 + 系统级取舍」在多数形状上更快，并把差距与原因都留在了可复现的数据里。

---

## 1. 同源：哪些东西确实是从 ACL 借鉴的

这一节很重要——汇报时先说「不是另起炉灶」，后面的差异才站得住。

| 维度 | ACL 53.1.0 | 本项目 | 是否一致 |
|---|---|---|---|
| 算法 | `Y = Aᵀ·[(G·g·Gᵀ) ⊙ (Bᵀ·d·B)]·A` | 同 | ✅ 逐字相同 |
| 配置 | F(4,4,3,3)（SME 可用时首选） | F(4,4,3,3) 主路径 + F(2,2,3,3) 可选 | ✅ 主路径相同 |
| G 矩阵 | `{6,0,0}, {-4,-4,-4}, {-4,4,-4}, {1,2,4}, {1,-2,4}, {0,0,24}` | 同 | ✅ 逐元素相同 |
| Bᵀ 矩阵 | 首行 `{4,0,-5,0,1,0}`（`pcoeffs={1,2,4,5}`） | 同 | ✅ 逐元素相同 |
| A 矩阵 | `{1,1,1,1,1,0} / {0,1,-1,2,-2,0} / {0,1,1,4,4,0} / {0,1,-1,8,-8,1}` | 同 | ✅ 逐元素相同 |
| 归一化 | 权重变换末尾乘 `recip576` | `norm = 1/576` | ✅ 相同 |
| tile 划分 | 输入 6×6、输出 4×4、步长 4，tile 数 `ceil(OH/4)×ceil(OW/4)` | 同 | ✅ 相同 |
| Winograd 域布局 | `[n_multis=36][tile][channel]`（`input_ld_row = roundup(IC,4)`） | `U[ts][tile][IC]`（不 padding） | ✅ 同构，补齐策略不同 |
| 输出变换 | 内含 bias 加法 + `min/max` clamp | 同 | ✅ 相同 |
| SME 输出变换 | `sme_fp32_mopa_4x4_3x3`（Kronecker 展开 + FMOPA） | 直接移植该实现（助记符重写并逐位校验） | ✅ 同源 |
| ISA 分层 | NEON / SVE / SME 三档 | 同 | ✅ 相同 |
| GEMM | arm_gemm（ACL 自带） | 就地编译 ACL 53.1.0 的 fp32-SVE 子集 | ✅ 同一份源码 |

> 结论：**数学层面 100% 同源，差别全在工程实现与调度策略**。所以下面的对比是
> 「同一个算法的两种工程写法」，不是「两种算法比谁快」。

---

## 2. 逐层差异

### 2.1 算法与 tile 选择：贪心自适应 vs 固定主路径

| | ACL | 本项目 |
|---|---|---|
| 选型机制 | `get_implementation()` 按**注册顺序贪心选最大输出 tile**，再倒推匹配的权重/输入变换 | 编译期/运行期固定 F(4,4)；F(2,2) 需显式切换 |
| 降级条件 | `arm_fp32_4x4_3x3` 带 `LargerShape` 约束（输入 > 4×4），小图会落到 F(2,2,3,3)；但 SME 版 `sme_fp32_mopa_4x4_3x3` 排在最前且**无此约束** | 无降级 |
| 上游门禁 | oneDNN 的 wino 路径在 `ih/iw > 112`、`ic/oc < 64`、`threads > 28` 任一成立时**主动回退** brgconv | 无门禁，全形状走 Winograd |
| 边界处理 | `TransformUnpadded`：有效区拷进 per-thread workspace 补零 | 边界 tile 整体 memset + 按实际范围裁剪（内部 tile 跳过 memset） |

> **口径提醒**：ACL 选哪个 F 取决于**它的构建是否启用 SME**。本项目集群侧的 oneDNN 对照
> 日志未能确认实际选中项（历史文档记为「疑似 F(2,2)」）。因此下面「同算法 PK 1.15–9.5x」
> 应按「我们的 F(4,4) vs oneDNN 当时的实际选择」来读，而不是默认对方一定是 F(2,2)。

**差异后果**：本项目在所有形状上都吃到 F(4,4) 的乘法节省（每输出像素 2.25 次乘 vs
F(2,2) 的 4 次），但**极小形状**没有 F(2,2)/直接卷积的退路。这正是我们在
`batch=1、IC≤128、spatial≤56²` 区间落后 oneDNN auto 的原因——oneDNN 在那 19 个形状上
改走 brgconv（直接卷积），变换固定开销为 0。

### 2.2 变换内核：手写全展开汇编 vs 泛型 intrinsics

| | ACL | 本项目 |
|---|---|---|
| 形式 | 内联汇编，**完全展开**，系数编码进指令 | C++ 模板 + SVE intrinsics，矩阵是**运行时数组** |
| 规模（fp32） | NEON 输入 1 140 行、SVE 输入 361 行、SME 输入 363 行、SME 输出 892 行（fp32 变换合计 **5 118 行**） | 变换头文件合计 **1 691 行**（NEON 390 / SVE 285 / SME 1 016），覆盖 F22+F44+三阶段 |
| 中间结果 | 全部驻留寄存器（z0–z31 分配满） | 两趟：列变换写 `thread_local` tmp，再行变换 |
| 零系数 | 展开时直接消去（无运行期判断） | 循环里 `if (matrix[o][k] == 0.0f) continue;` 跳过 |
| tail | 谓词 `whilelt` | 谓词 `whilelt`（同） |
| 系数加载 | `ld1rqw` 一次性广播 | 每行每系数 `svmul_n_f32` / `svmla_n_f32` + 逐次 load/store |

**差异后果**：我们少写数倍代码（1 691 行覆盖 ACL 5 118 行的功能面），但**指令级效率明显低**：
实测 `transform_1d_sve` 约 **52 cyc/向量操作**，理想值 ~2（见 `AGENTS.md` 任务清单）。
这是本项目最实在的一处落后，也是路线图第 6 项的由来。

### 2.3 数据布局与 scatter/gather：直写 GEMM 序 vs 二次搬运

```
ACL：  变换 kernel ──[matrix_stride 直接寻址]──► Winograd 域缓冲（GEMM 可直接消费）
本项目：变换 kernel ──► 线程局部 U_tile ──copy_f32──► U[ts][tile][IC]
                                              ↑ 这一趟就是 scatter
```

| | ACL | 本项目 |
|---|---|---|
| 布局形状 | `[multis][patches][IC]`，`ld_row = roundup(IC,4)` | `[ts][tile][IC]`，不 padding |
| 写入方式 | kernel 里 `outptr + m*matrix_stride`，**一步到位** | 先写 `U_tile`，再 36 次 `copy_f32` 散入 `U` |
| 通道补齐 | 补到 4 的倍数（NEON 4 宽写） | 不补，SVE 谓词处理 tail |
| 代价 | 0 次额外遍历 | 输入侧 +1 遍全量读写，输出侧 +1 遍 gather |

**差异后果**：结构上我们多两趟内存往返（Case 0 串行计时：scatter 6%、gather 15%）。
我们也试过用 tile-major 布局彻底消灭它们，**实测被证伪**：KML −14%、arm_gemm −16%
（`lda` 从 `IC` 变成 `36·IC`，局部性损失大于拷贝收益，代码内已留注释）。

### 2.4 Winograd 域 GEMM：一次 batched 调用 vs 可插拔后端

| | ACL | 本项目 |
|---|---|---|
| 调用形态 | 一次 `arm_gemm::gemm`，`GemmArgs(M=n_patches, N=OC, K=IC, nmulti=36, batches=N)` | 按后端而定：arm_gemm 路径 = **每线程领一段连续 ts 切片，合并成一次 batched 调用**；KML/OpenBLAS = 逐切片 `cblas_sgemm`；SME = 自研 ZA 内核 |
| 线程 | arm_gemm 内部按 M/N 分块 | 外层 OpenMP 分切片，内层单线程 GEMM |
| 可选后端 | 只有 arm_gemm | 5 种（arm_gemm / KML / OpenBLAS / naive / SME FMOPA），编译期 + 环境变量切换 |
| B 操作数 | `[multis][IC][roundup(OC,4)]`（**OC 最内**），由 `weight_ld_row` 决定 | `[ts][OC][IC]`（**IC 最内**，即 B 存成 N×K），arm_gemm 路径按需打包；SME 路径另存 `Vkt[ts][k][OC64]` |

**差异后果**：
- 我们避免了「36 次 GEMM 对象构造 + 36 次 B 预转置」——把切片折叠进 `nmulti` 后，
  固定开销只付一次。这是 59/59 反超 OpenBLAS 的关键之一（修复前 47/59）。
- 但**可插拔**才是这一层的真正价值：换平台/换形状能立刻 A/B（本机 KML 233 ms
  < arm_gemm 328 ms，与集群结论相反，见 §3-A2）。

### 2.5 权重与缓存：primitive 生命周期 vs 跨调用指纹缓存

| | ACL / oneDNN | 本项目 |
|---|---|---|
| 权重变换时机 | `prepare()` 一次：permute(OHWI→HWIO) → 权重变换 → `gemm->prepare()`（reorder） | 首次调用时变换，随后进入 **V 跨调用缓存** |
| 缓存 key | primitive 实例（生命周期由框架管理） | `权重指针 + 64 点 FNV-1a 指纹 + shape + F44 标记 + ISA` |
| 并发 | oneDNN 的 ACL 封装每次 `run()` **全程持锁**（`resource_mapper` 限制） | 只在缓存 miss 与条目发布时持锁，热路径无锁 |
| SME 的 B 打包 | arm_gemm 内部 interleave | `Vkt`（`[ts][k][OC64]`）与 `V` 在**同一把锁内**发布，读者不会看到半成品 |

**差异后果**：本项目的缓存生命周期贴合「裸函数 API」（没有 primitive 概念时也能复用），
且用指纹防了指针复用的 ABA（新层权重恰好落在旧层地址且 shape 相同）。
代价：容量只有 24 条、无淘汰策略，59 形状的用例集第 25 个之后每次重打包 `Vkt`
（t1 占总时间 0.3%–15%，已在文档中标注为口径折扣）。

### 2.6 多线程结构：细粒度任务队列 vs 单并行区域

| | ACL | 本项目 |
|---|---|---|
| 调度 | `NEScheduler` 细粒度任务；变换 kernel **内部**再切线程（`tile_i = thread_id; += n_threads`）；GEMM 由 arm_gemm 自己分 | 单个 `#pragma omp parallel` 覆盖权重变换 + 所有 batch 的 Phase1/2/3，阶段间显式 barrier |
| fork/join | 每段一次调度 | 全程 **1 次** |
| 缓冲 | per-thread workspace（scratchpad 管理，`Temporary` 生命周期） | 6 个 `thread_local` 缓冲跨调用复用，热路径零堆分配 |
| 并发安全 | `execute_forward` 全程 `lock_guard`，同一实例不能并发 run | 无全局锁 |

**差异后果**：小形状上我们的固定开销显著更低（这正是 920F 上 8/9 形状反超的历史原因）；
代价是结构刚性——阶段之间不能流水重叠。另一面：ACL 侧因 `resource_mapper` 整段串行，
而我们的热路径完全可重入。

### 2.7 SME 的用法：变换 + 通用 SME GEMM vs 变换移植 + 专用 ZA GEMM

| | ACL 53.1.0 | 本项目 |
|---|---|---|
| SME 输入变换 | 有（`sme_fp32_mla_6x6`，363 行，FMLA 为主） | **无**：输入变换保持纯 SVE（历史教训：把含 `thread_local` malloc 的 SVE 路径包进 `SMSTART` 曾破坏堆） |
| SME 输出变换 | `sme_fp32_mopa_4x4_3x3`（892 行） | **移植同一实现**（改写成助记符 + 逐位校验编码） |
| SME GEMM | arm_gemm 自带 `sme_interleaved_nomerge_fp32_mopa_{1VLx4VL, 2VLx2VL, 4VLx1VL}` 等内核（需 interleave 打包）；本项目构建时用 `cfg.filter="sve_"` 强制走 SVE 内核 | **自研**：ZA 行 = 16 OC、列 = 16 tile、za0–3 = 4 个 OC 块；K 主序直取，drain 的 64 B 切片正好是 `M_buf` 的一行 16 OC |

**差异后果**：
- 我们的 SME GEMM **免掉了 interleave 打包**，且 drain 零置换（ACL 的通用内核需要先把
  A/B 打包成 interleaved 布局）。
- 但只做了 4-tile 一档、只对 `OC ≥ 16` 生效，且**尚未与 arm_gemm 的 SME 内核同机对打**
  （这是明确的验证缺口，见 §4-K4）。

### 2.8 覆盖面：本项目有意收窄

| 维度 | ACL 53.1.0 | 本项目 |
|---|---|---|
| 数据类型 | fp32 + fp16（+ int8 走其它路径） | 仅 fp32 |
| 卷积形态 | 2D + 1D 变体、3×3 / 5×5、F(2,2)/F(4,4) | 仅 2D 3×3、F(2,2)/F(4,4) |
| GEMM 内核数 | 数十个（按 M/N/K/线程数自动选） | 5 个后端 |
| 框架集成 | primitive / 内存池 / 调度器 / 校验层 | 裸函数 API + CSV 驱动基准 |

这是**教学研究定位**的自觉取舍：把面收窄，才能把每一处取舍讲清楚并留下数据。

---

## 3. 优势提炼（每条都给机制、证据、适用边界）

### A1 算法选择更彻底：全形状 F(4,4)，不降级、不设门禁

- **机制**：不像 ACL 的非 SME 路径那样在小图上被 `LargerShape` 逼回 F(2,2)，也不像
  oneDNN 那样在 `ih/iw>112`、`ic/oc<64`、`threads>28` 时回退 brgconv。
- **证据**：同算法 PK（oneDNN 强制 `--winograd`，即 `wino:acl`）**1.15–9.5x**，
  且 `e2e_wino` 与 `benchdnn_wino` 两个独立 driver 互相验证。
- **边界**：① 极小形状反而输——oneDNN 在 19/59 个形状上选 brgconv 是对的
  （我们的 0.09–0.91x 就在这一档），所以这条优势的真实含义是
  「**在值得做 Winograd 的形状上做得更彻底**」；② 对方实际选了哪个 F 未从日志确认（见 §2.1 口径提醒）。

### A2 GEMM 后端可插拔：把「哪个 BLAS 更强」变成可测量的问题

- **机制**：编译期 5 选 1（arm_gemm / KML / OpenBLAS / naive / SME），统一 K 主序布局契约；
  运行期 `WINO_GEMM=sme` 可切换。
- **证据**：
  - 集群 920F：arm_gemm vs OpenBLAS **59/59 全胜**，合计 **2.50x**、几何平均 **1.99x**、
    算数平均 **2.17x**、中位数 **1.77x**，范围 1.10–5.34x。
  - 本机 608 核：**KML 233 ms < arm_gemm 328 ms** —— 与集群结论相反。
- **价值**：正因为可插拔，才发现「后端优劣依平台而定」这一事实。换成单一后端，这个结论
  根本看不到，也不会知道该不该迁移别人的数字。

### A3 batched 调用摊薄固定开销

- **机制**：arm_gemm 路径下每个线程把自己的连续切片折叠成**一次** `nmulti` 调用，
  对象构造 / 内核选择 / B 预转置只付一次；切片数不能被线程数整除时用
  `b = i*NM/nth` 保证所有线程都有活。
- **证据**：修复布局契约与分批后，从 **47/59 → 59/59**；原先输掉的小通道窄 M/K
  （1.6–2.4x）与深层 7² 族（1.9–5.3x）**全部翻赢**。
- **边界**：切片数（36）小于线程数时，折叠收益消失，此时 SM E 路径的 `omp for (ts)`
  也会暴露并行度不足（20² 族最差 3.45x）。

### A4 单 OpenMP 区域 + 线程本地缓冲：把固定开销压到最低

- **机制**：权重变换并入主 region（避免额外 fork/join）；6 个 `thread_local` 缓冲跨调用复用；
  去掉 U/M_buf/V 的无效清零；4 处拷贝 SVE 化（16 float/指令）。
- **证据**（Case 0：4,192,40²,192，t16）：14.2 ms → **1.94 ms**（**7.3x**）。
  其中「权重变换放独立 region」反而慢 ~7%（t32 6.7→7.2 ms），回退并合并才恢复。
- **边界**：阶段间不重叠；`--timing` 串行模式不代表并行后的占比。

### A5 缓存生命周期贴合裸 API，且有防 ABA 设计

- **机制**：`key = 权重指针 + 64 点 FNV-1a 指纹 + shape + F44 标记 + ISA`；命中直接跳过
  权重变换；SME 的 `Vkt` 与 `V` 同锁发布。
- **证据**：KML t16 geomean **1.07x**，最高 **1.74x**（纯推理场景，权重恒定）。
- **边界**：`kMaxEntries=24` 无淘汰 —— 59 形状用例集第 25 个之后不再命中，需改 LRU。

### A6 自研 SME GEMM 与 Winograd 切片形状对齐

- **机制**：ZA 行 = 16 个 OC、列 = 16 个 tile、`za0..3` = 4 个 OC 块。
  选这个方向是因为 `M_buf` 是 `[ts][tile][OC]` 行主序，drain 出的 64 B 正好是
  「一行里的 16 个连续 OC」——**零 shuffle、零转置**；反过来（OC 在列）就必须做转置。
- **证据**（绑核口径，59 形状）：
  - t1：合计 **3317.9 vs 3403.3 ms（快 2.5%）**，31/59 胜，最高 **2.2x**（7²/14² 深通道）。
  - t16：合计 **345.9 vs 386.3 ms（快 10.4%）**，32/59 胜，160² 族最高 **1.7x**。
  - 微内核逐元素对比标量参考 **max abs diff = 0**。
- **边界**：小空间 + 大 OC/IC 落后 1.4–3.4x（`28²` 的尾组只有 1 个有效 tile，
  ZN 利用率 6%）；且未与 arm_gemm 的 SME 内核同机对打。

### A7 代码规模与可读性

| | 本项目 | ACL 53.1.0 |
|---|---|---|
| 核心实现 | **3 636 行**（6 个头文件 + 1 个 .cpp，含三阶段 × 三 ISA × 两配置） | Winograd 目录 **8 056 行**（仅变换与 driver 头，不含调度器/内存池/校验层） |
| 单文件最大 | `winograd_conv.cpp` 1 516 行（流水线 + 5 个 GEMM 后端 + SME 内核） | `a64_fp32_6x6.cpp` 1 140 行（仅一个输入变换） |

可完整通读 → 适合教学；每处取舍都能指到代码位置与文档段落。

### A8 独立真值验证体系

| 手段 | 说明 |
|---|---|
| fp64 直接卷积参考 | `--verify` 用双精度直接卷积生成真值，逐元素比较 |
| 相对容差判据 | `err < 1e-4 × max(abs(ref)) + 1e-5`（固定绝对容差会误报大 IC，幅值 ~1000） |
| 布局回归 | `test_ref_vs_nchw` 断言「NCHW 包装」与存档原生 NCHW 内核 **bit-exact** |
| 微内核自检 | SME GEMM 输出 vs 标量参考 `max abs diff = 0` |
| 实测精度 | fp32 相对误差 **2–6e-6**，与 IC 无关 |

> 这不是说 ACL 没有测试（它有完整的 validation 层），而是说：**本项目的每个性能数字
> 背后都有一个独立于实现的 fp64 真值门**，避免了「两边一起错所以看着对」。

---

## 4. 诚实的差距

| # | 差距 | 具体 | 影响 | 怎么补 |
|---|---|---|---|---|
| K1 | **单 kernel 效率不如 ACL 手写汇编** | 我们的 `transform_1d_sve` 约 52 cyc/向量操作（理想 ~2）；ACL 全展开、寄存器驻留、零系数在编译期消去 | 变换阶段占比高的形状（小 IC、大空间）吃亏 | F(4,4) 专用展开内核，参考 `sve_fp32_6x6.cpp`（361 行）；预期变换 −30~50% |
| K2 | **多两趟内存往返** | 输入侧 scatter + 输出侧 gather | 旧计时 ~5% + ~12% | 让变换直接写 GEMM 序（ACL 的 `matrix_stride` 做法）；tile-major 布局已验证不可行 |
| K3 | **小形状无 hybrid 退路** | 无「小形状走直接卷积」判据 | `batch=1、IC≤128、spatial≤56²` 慢 0.09–0.91x | im2col + arm_gemm 直接卷积路径（路线图第 1 项） |
| K4 | **SME GEMM 未与 ACL 的 SME GEMM 同机对打** | 本项目构建用 `cfg.filter="sve_"` 强制 SVE 内核；ACL 53.1.0 自带 `sme_..._fp32_mopa_*` | 无法断言「自研 SME 内核优于 ACL 的 SME 内核」——只能说它优于 **KML** | `WINO_GEMM_FILTER=""` 放开 filter，让 arm_gemm 自己选 SME 内核，同作业 A/B |
| K5 | **覆盖面窄** | 只做 fp32、2D、3×3、unit stride | 不能覆盖 fp16/int8/1D/5×5 场景 | 有意取舍，非缺陷 |
| K6 | **无框架层设施** | 无 primitive、无内存池、无调度器 | 集成进推理框架需要额外工作 | 目前定位是研究实现 |

> **汇报时建议主动说 K1 和 K4**：主动暴露短板的报告更可信，而且这两条都有明确的补法。

---

## 5. 汇报用的三张幻灯片

### 幻灯片 1 —— 同一个算法，两种工程哲学

| | ACL 53.1.0 | 本项目 |
|---|---|---|
| 定位 | 工业库，追求每档都最优 | 研究实现，追求可解释、可 A/B |
| 算法选择 | 运行时贪心选最大 tile + 形状门禁 | 固定 F(4,4) 主路径，不降级 |
| 变换内核 | 手写全展开汇编（5 118 行） | 泛型 SVE intrinsics（1 691 行） |
| 布局 | `matrix_stride` 直写 GEMM 序 | 行主序 + 一次 scatter |
| GEMM | arm_gemm（单选） | 5 种可插拔后端 |
| 多线程 | 调度器 + kernel 内细粒度切分 | 单 OpenMP 区域 + 显式 barrier |

### 幻灯片 2 —— 差异兑现成了什么

```
算法更彻底 ──► 同算法 PK 1.15–9.5x
GEMM 组织  ──► 47/59 → 59/59（batched 折叠）
并行结构   ──► Case 0: 14.2ms → 1.94ms（7.3x）
缓存复用   ──► t16 geomean 1.07x（最高 1.74x）
SME 专用   ──► t16 合计快 10.4%，最高 2.2x
```

### 幻灯片 3 —— 不藏的两句话

1. **我们的单 kernel 打不过 ACL 的手写汇编**（52 cyc/vec-op vs 理想 2），领先来自
   算法选择、调用组织、缓存与并行结构。
2. **小形状仍落后 oneDNN auto**（0.09–0.91x），根因是 Winograd 的固定变换开销摊不薄，
   补法是 hybrid 直接卷积路径——方向已被 oneDNN 自己的选择验证（它给 19/59 个形状选了 brgconv）。

---

## 6. 如何验证本文的每个结论

```bash
source /home/envs/HPCKit_26.1.RC1/HPCKit/26.1.RC1/setvars.sh
cd /home/z00889957/100Work/winograd_conv

# ① 同源：矩阵逐元素对照（本项目矩阵 vs ACL 源码系数）
sed -n '76,120p' include/winograd_matrices.hpp
grep -n "recip576\|pcoeffs" \
  ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/winograd/weight_transforms/arm_fp32_4x4_3x3.cpp \
  ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/winograd/input_transforms/a64_fp32_6x6.cpp

# ② 算法选择规则（贪心 + LargerShape 约束）
sed -n '236,300p' ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/winograd/winograd_implementations.hpp
sed -n '45,70p' ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/winograd/output_transforms_fp32.cpp

# ③ 布局：ACL 的 matrix_stride 计算
grep -n "input_ld_row\|input_ld_matrix\|weight_ld_row" \
  ComputeLibrary-53.1.0/src/core/NEON/kernels/convolution/winograd/winograd_implementations.hpp

# ④ ACL 是否有 SME GEMM 内核（K4 的依据）
ls ComputeLibrary-53.1.0/src/core/NEON/kernels/arm_gemm/kernels/ | grep fp32_mopa

# ⑤ 本项目侧：正确性 + 后端 A/B
./build_gcc_sme/test_winograd --nhwc          # 22/22
./build_gcc_sme/bench_winograd --sve --nhwc --threads 16 --verify shapes/conv_all.csv
WINO_GEMM=sme taskset -c 3 ./build_gcc_sme/bench_winograd --sve --nhwc --threads 1 \
    --warmup 2 --repeats 3 shapes/conv_all.csv
taskset -c 3 ./build_gcc_sme/bench_winograd --sve --nhwc --threads 1 \
    --warmup 2 --repeats 3 shapes/conv_all.csv
```

> ⚠️ **测量口径**：本机与集群是两套机器，毫秒数不可跨机比较。本机 SME 数据必须绑核
> （t1 用 `taskset -c 3`，t16 用 `OMP_PROC_BIND=close OMP_PLACES=cores`），
> 不绑核结论会翻转（见 `docs/sme_fmopa_gemm.md` §5.6）。

---

## 7. 术语与源码位置索引

| 术语 | 含义 | 本项目位置 | ACL 位置 |
|---|---|---|---|
| `n_multis` | Winograd 域矩阵个数，F(4,4) = 36 | `NM` 常量 | `winograd_implementations.hpp:299` |
| `matrix_stride` | 相邻 Winograd 元素之间的字节/元素跨度 | 等价于 `n_tiles * IC` | 各变换 kernel 参数 |
| `ld_row` | 一行（一个 tile/通道块）的元素跨度 | `IC`（行主序） | `input_ld_row = roundup(IC,4)` |
| FMOPA | `za[row][col] += Zm[row]·Zn[col]`，一次 256 次 FMA（VL=16） | `wino_sme::gemm_window` | `sme_fp32_mopa_4x4_3x3.cpp` |
| `smstart/smstop` | 进出流式模式；切换时**所有** z 寄存器与谓词变 UNPREDICTABLE | `docs/sme_fmopa_gemm.md` §6 | 同 |
| batched GEMM | 一次调用完成 36 个切片的 GEMM | `winograd_gemm_batched` | `arm_gemm::GemmArgs::nmulti` |
| LargerShape | ACL 的约束：输入必须大于输出 tile 才允许该实现 | —（我们不降级） | `winograd_implementations.hpp:74` |

---

## 附：一句话版（可直接抄进汇报摘要）

> 本项目从 ACL 的 Winograd 思路出发，保持**算法与变换矩阵完全一致**，在工程实现上做了四个
> 不同的选择：固定 F(4,4) 主路径而不按形状降级、泛型 SVE 内核而不手写展开汇编、
> 可插拔 GEMM 后端而不绑定单一库、单 OpenMP 区域而不引入调度器。结果是：同算法对比
> oneDNN 的 `wino:acl` 快 1.15–9.5x，对 oneDNN 自动选优在大形状快 1.1–2.8x（小形状仍慢
> 0.09–0.91x）；代价是单 kernel 指令效率不如 ACL 手写汇编、多两趟内存搬运，
> 这两点已列为明确的后续任务。
