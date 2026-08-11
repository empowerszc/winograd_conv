# 为什么当前实现快于 ACL 23.11（oneDNN 侧）——独立分析

> **日期**：2026-08-11
> **结论前置**：这个「8/9 case 快于 oneDNN」是 **t16、单次小形状调用** 微基准下的结果。它**不是**「我们的内核比 ACL 强」的证据，而是「**在固定开销主导的小负载下，我们的工程实现更契合 920F**」的证据。
>
> 关联文档：本文是 `../PERFORMANCE_ANALYSIS.md` 第 10 节的独立展开版，补充了双方实现身份的厘清、有效 GFLOPS 证据和具体例子。

---

## 目录

- [1. 先厘清对比对象：这是「Winograd vs Winograd」](#1-先厘清对比对象这是winograd-vs-winograd)
- [2. 数字总览与有效 GFLOPS 证据](#2-数字总览与有效-gflops-证据)
- [3. 胜负模式诊断（带具体例子）](#3-胜负模式诊断带具体例子)
- [4. 六个原因（按置信度排序）](#4-六个原因按置信度排序)
- [5. 诚实的一面：对手的内核仍然更强](#5-诚实的一面对手的内核仍然更强)
- [6. 建议与验证清单](#6-建议与验证清单)
- [7. ACL 23.11 SVE 实现细节文档索引](#7-acl-2311-sve-实现细节文档索引)
- [8. 验证与分析方式工具箱](#8-验证与分析方式工具箱如何证明证伪快于-acl-2311)

---

## 1. 先厘清对比对象：这是「Winograd vs Winograd」

对比的双方，**算法都是 Winograd**，这一点对正确解读数字至关重要。

### 1.1 oneDNN 侧：走 Winograd 路径，用 ACL 23.11 的 SVE 内核

oneDNN 在 920F 上对 fp32 3x3 卷积选择了 **Winograd 路径**（`acl:wino`），其内核来自 **ACL 23.11**：

| 层次 | oneDNN / ACL 23.11 实现 |
|------|------------------------|
| 算法 | Winograd（F(4,4,3,3) 为主，具体 tile size 由 CPUID+shape 启发式决定） |
| 输入变换 | SVE 手写内联汇编 `sve_fp32_6x6.cpp`（完全展开、无分支、指令调度优化） |
| 权重变换 | `arm_fp32_4x4_3x3.cpp` |
| 输出变换 | `sve_fp32_... / sme_fp32_mopa_4x4_3x3.cpp` |
| GEMM | arm_gemm **JIT 编译**的 SVE 内核（寄存器阻塞 + 软件流水 + 按矩阵大小自动调优） |
| 数据布局 | Winograd 专用布局（matrix_stride 参数，避免 scatter/gather） |
| 多线程 | NEScheduler 细粒度任务队列 |

### 1.2 我们侧：自研 F(4,4) Winograd

| 层次 | 本实现 |
|------|--------|
| 算法 | 固定 F(4,4,3,3)（TS=6, OT=4, NM=36） |
| 变换 | 编译器生成的 NEON/SVE/SME 代码（`copy_f32` 等用 `svld1/svst1`，编译期按 `__ARM_FEATURE_SVE` 选择） |
| GEMM | 编译期三选一：arm_gemm JIT > OpenBLAS > naive |
| 数据布局 | NHWC 通道连续 + per-batch U 复用 |
| 多线程 | 单个 OpenMP region + tile/ts 级并行，每 batch 2 barrier + `nowait` |

### 1.3 由此推出的关键推论

对手的**内核**（手写 SVE 汇编变换 + arm_gemm JIT GEMM）在技术上优于我们的**编译器生成代码 + naive/OpenBLAS GEMM**。既然内核不如对方，我们却 8/9 赢，**优势必然在「内核之外」的工程层**：每次调用的固定开销、调度粒度、内存/缓冲策略、数据布局。这是本文的核心论点，第 3、4 节用数字和例子支撑它。

> ⚠️ 因此，如果换到**大负载**（batch 增大 / 空间增大 / 整网络推理），固定开销被摊薄后，对手更好的内核**很可能反超**。第 6 节给出验证方法。

---

## 2. 数字总览与有效 GFLOPS 证据

### 2.1 9 case 对比表（t16, NHWC, SVE）

| Case | Shape (N,IC,IH,IW) | (OC,IC) | Tiles | 本项目(ms) | oneDNN(ms) | 比率 | 结果 |
|------|--------------------|---------|-------|-----------|-----------|------|------|
| 0 | 4,192,40,40 | 192,192 | 100 | 1.937 | 3.55 | 1.83x | ✓ |
| 1 | 4,96,80,80 | 96,96 | 400 | 2.515 | 4.22 | 1.68x | ✓ |
| 2 | 4,48,160,160 | 48,48 | 1600 | 4.586 | 4.06 | **0.88x** | ✗ 慢 1.13x |
| 3 | 4,192,20,20 | 192,192 | 25 | 0.799 | 2.83 | 3.54x | ✓ |
| 4 | 4,384,80,80 | 96,384 | 400 | 6.080 | 13.78 | 2.27x | ✓ |
| 5 | 4,768,40,40 | 96,768 | 100 | 4.055 | 9.09 | 2.24x | ✓ |
| 6 | 4,768,20,20 | 96,768 | 25 | 1.808 | 5.58 | 3.09x | ✓ |
| 7 | 4,96,40,40 | 96,96 | 100 | 0.893 | 1.90 | 2.13x | ✓ |
| 8 | 4,96,20,20 | 96,96 | 25 | 0.344 | 1.15 | 3.34x | ✓ |

**8/9 赢，赢幅 1.68~3.54x；唯一输的是 Case 2（tiles=1600 + IC=48）慢 1.13x。**

### 2.2 有效 GFLOPS：oneDNN 在小调用上是「固定开销受限」的铁证

两个实现都是 Winograd，所以「直接卷积等效 GFLOPS」（= 直接卷积 FLOP 数 / 耗时）对两者**同样放大**，可以直接对比。数字本身：

| Case | 等效 FLOP(G) | 本项目 GFLOPS | oneDNN GFLOPS |
|------|:-:|:-:|:-:|
| 0 | 4.25 | 2.19 | 1.20 |
| 1 | 4.25 | 1.69 | 1.01 |
| 2 | 4.25 | 0.93 | 1.05 |
| 3 | 1.06 | 1.33 | **0.37** |
| 4 | 17.0 | 2.79 | 1.23 |
| 5 | 8.49 | 2.09 | 0.93 |
| 6 | 2.12 | 1.17 | **0.38** |
| 7 | 1.06 | 1.19 | **0.56** |
| 8 | 0.265 | 0.77 | **0.23** |

**观察**：
- oneDNN 的有效 GFLOPS 横跨 0.23~1.23 T，且**在最小负载的 Case 3/6/7/8 上塌到 0.23~0.56 T**。这不是它内核的能力——同一套手写 SVE 内核在 Case 4 能跑到 1.23 T。**同一内核，负载越小越慢**，说明瓶颈是**每次调用的固定开销 + 调度**，不是内核本身。
- 我们这边有效 GFLOPS 相对稳定在 0.77~2.79 T，对负载变化不敏感——因为我们的固定开销趋近于零（A2 起效）。

> 注：对 F(4,4,3,3)，GEMM 每 tile 做 36（=NM）次乘加，直接卷积每 tile 做 4×4×3×3=144 次，故「有效 GFLOPS」≈ 实际 GEMM FLOP 的 4 倍（未计变换开销）。两个实现都是 Winograd、放大倍数一致，结论不受影响。

---

## 3. 胜负模式诊断（带具体例子）

### 3.1 例子 1：Case 3（最小负载，赢 3.54x）

- 形状 4×192×20×20，**只有 25 个 tile**，总输出 307K 元素，等效 FLOP 仅 1.06G。
- 我们的 0.799ms 中，固定开销（单 OpenMP region + 零分配 + `thread_local` 复用）占比很小。
- oneDNN 的 2.83ms 里，有 NEScheduler 任务图构建、kernel window 分配、workspace 准备、GEMM 内核选择、可能的格式转换——这些都是**每次调用都要付**的。0.37 T 的有效 GFLOPS 说明它的大部分时间花在了内核跑起来之前。
- **直观类比**：一辆赛车（手写 SVE 内核）每次起步前要花 3 分钟做检查才能上赛道（固定开销）；在 1 秒的圈速（小 conv）里这 3 分钟是灾难；在一小时的耐力赛（大负载）里可以被忽略。

### 3.2 例子 2：Case 2（唯一输，tiles=1600 + IC=48，慢 1.13x）

- **tile 最多（1600）+ 每 tile 工作量最小（IC=48）**。这是对我们最不利的极端：
  - IC=48 时 SVE-512 的通道向量化只用到 **3 个寄存器**（48/16），SVE 宽度优势荡然无存；
  - 1600 个 tile 经 `collapse(2) schedule(dynamic,2)` 切成 800 个任务块，每块工作极小，调度/调用/分支开销相对突出；
  - 每 batch 2 barrier、共 8 次 barrier 的代价在总时间里占比最大。
- 对手 arm_gemm JIT 恰好擅长**小 K、多 tile** 的 GEMM（寄存器阻塞吃满），这里它赢了。
- **这是「固定开销主导」的镜像证明**：负载小到极致 + tile 碎到极致时，我们的调度成本也压不住了，对手的调度器反而更稳。

### 3.3 例子 3：Case 4（我们自己的 GEMM 最弱，仍赢 2.27x）

- Case 4：IC=384、400 tile，是**纯 GEMM 型**负载。我们这里用的是 naive/OpenBLAS 单线程 GEMM（最弱的环节），U 缓冲 22MB 远超 L2。
- 即便如此我们仍赢 2.27x（6.080 vs 13.78ms）。如果这是「内核 vs 内核」的公平对决，我们的 naive GEMM 应该惨败——事实恰恰相反。
- **结论**：Case 4/5 的大赢幅不是我们内核赢，而是 **oneDNN 在 400 tile 的 Winograd 上整体低效**（有效 GFLOPS 只有 1.2 T，远低于其内核在大负载上的能力）。对手输在**每次调用围绕 tile 的调度、转换、format 处理链条太长**，而我们用粗粒度 OpenMP 把这个链条砍到最短。

---

## 4. 六个原因（按置信度排序）

### 4.1 固定开销：微基准对小调用最不利（高置信）

| | oneDNN/ACL | 本实现 |
|---|---|---|
| 每次调用 | NEScheduler 任务图、kernel window 分配、workspace 准备、GEMM 内核选择 | 单 OpenMP region、**零堆分配**、`thread_local` 缓冲跨调用复用（A2） |
| 单次调用时长 | 0.34~6ms（固定开销占比显著） | 同上，但固定开销占比趋近于零 |

证据：第 2.2 节的 GFLOPS 表——oneDNN 有效 GFLOPS 随负载缩小而崩塌。

### 4.2 并行结构：粗粒度 OpenMP vs 细粒度 NEScheduler（高置信）

- 我们：tile/ts 级 OpenMP，`schedule(dynamic,2)`，每 batch 2 barrier + `nowait`，16 线程下同步少、负载均衡。
- ACL：NEScheduler 细粒度任务队列 + kernel window 切分，为**多核大图**设计。对 25~1600 tile 的小调用，任务投递/取回的开销相对高。
- 反例佐证：Case 2（1600 tile）是我们调度最碎的地方，也是唯一输的地方——碎到一定程度，我们的粗粒度优势消失。

### 4.3 内存层次契合 920F（中-高置信）

- 920F **无 L3、L2 仅 768KB/核**。我们：NHWC 通道连续 + `copy_f32` SVE-512（16 float/指令）+ per-batch U 复用（不放大到 N×）+ 免清零——直接减少主存流量。
- ACL 的 Winograd 专用布局有 padding/对齐，workspace 更大，在 L2 极小的机器上吃亏。
- 反例佐证：Case 4/5（U 缓冲 22~44MB，远超 L2）我们只赢 2.2x 而非 3.5x——内存受限时优势缩小。

### 4.4 CPU 适配 / CPUID 内核选择（中置信，需验证）

- ACL 内核选择由 CPUID 数据库驱动。鲲鹏 920F 未必有 tuned entry，可能回退到通用 NEON 或非最优路径（已知 23.11 的 GEMM 选择比 53.1.0 好，但那只是「次优里选较好」——见第 7 节引用的 23.11 vs 53.1.0 文档）。
- 我们明确以 `-march=armv8.6-a+sve2` 为目标编译，全路径 SVE。
- ⚠️ **需验证**：`ONEDNN_VERBOSE` 或 ACL 日志，确认 920F 上 oneDNN 实际选中的变换族和 GEMM 内核族（SVE？NEON？哪个实现？）。

### 4.5 算法 tile size 差异（中置信，需验证）

- 已确认双方都走 **Winograd**（排除 im2col 猜测）。剩余差别在 **tile size**：oneDNN 按 CPUID+shape 启发式可能选 F(2,2,3,3)（TS=4, OT=2）或 F(4,4,3,3)；我们固定 F(4,4)。
- F(2,2) 乘法更少（2.25 vs 4）但 tile 更碎、变换更频繁；F(4,4) 乘法多但每 tile 工作大。特定形状下差异会被放大。
- ⚠️ **需验证**：`ONEDNN_VERBOSE=1` 打印每个 shape 实际跑的算法与 tile size。

### 4.6 测量公平性（需验证）

两个实现的线程绑定（numactl / OMP_PROC_BIND）、warmup/repeats、oneDNN primitive 是否跨迭代复用、构建 flag 是否一致，均未记录。若 oneDNN 数字里含 per-iteration primitive 重建或格式转换，会系统性放大我们的优势。**这是排除「假赢」的最后一道门。** 下面展开两条容易误读、但方向相反的关键点。

#### 4.6.1 缓冲复用是不是「背靠背重复测量才有效」的假优势？——不是

A1/A2 的复用收益来自 `thread_local` 缓冲**驻留在堆里**（进程/线程生命周期内持有），而不是来自「两次调用在时间上相邻」。中间隔任意多的 conv/ReLU/transformer 算子，缓冲的所有权不变、`malloc` 不会被再次调用。这是**位置持久性，不是时间相邻性**——交错只改变调用次序，不改变堆里的驻留。

生产框架正是这么做的，与算子执行次序解耦：TensorRT 整个 engine 一个 workspace tensor 分配一次、各算子用其中一段；cuDNN 用 per-handle workspace；oneDNN 的 workspace 和变换权重挂在 primitive 上。它们都跨其它算子存活。所以「同一个 conv op 被执行几百万次」的「次」跨的是 inference pass，两次 pass 之间跑完整张网络，缓冲依然在。

**唯一会被交错破坏的是缓存热度层**：连续测同一个 case 时缓冲在 L2/L1 是热的，交错执行会被其它算子冲出去。但这层收益对 oneDNN **完全对称**（它的 workspace 一样被冲掉），不是我们对它的优势，顶多让两边的稳态数字都比真实略乐观，不改变胜负方向。

**因此，真正的风险仍是「不对称」而非「复用」**：如果 oneDNN 那组数字是每次迭代重建 primitive / 重分 workspace 测出来的，而我们是复用，赢幅就被系统性放大——这才是该查的地方。

#### 4.6.2 反证：权重变换 V 我们每次调用都重算，反而比 oneDNN 吃亏

oneDNN 会把**变换后的权重（TransformedWeights）缓存到 primitive 上**（23.11 vs 53.1.0 分析中专门讨论过的「生命周期」差异），第 2 次起的迭代跳过权重变换；而我们的 `dispatch_weight_transform` 在卷积函数内**每次调用都执行**（`src/winograd_conv.cpp:377`），V 没有跨调用缓存。

含义：在真正公平的重复测量里，oneDNN 比我们多摊薄了一部分工作。这说明 **A1/A2 不是我们赢 oneDNN 的来源**（我们甚至在此吃亏），反过来加强了「我们赢在固定开销/调度/内存策略」的结论；如果给 V 也加跨调用缓存，我们还会更快。

#### 4.6.3 复用的两笔真实成本（不是作弊，是取舍）

- **内存驻留**：`thread_local` 缓冲只涨不缩，多 shape 后累计到「所有见过的最大值」；oneDNN 的 workspace 按 primitive 精确缩放。内存受限部署里这是我们吃亏的地方。
- **冷启动**：进程内**第一次**调该 conv 的那个线程要付一次分配。框架若是固定线程池，每线程第一次碰到该 op 后即持有，几轮 pass 后稳态；若每请求新建线程（serverless 式），则每请求付一次——但那也会同样惩罚任何 thread_local 设计。
- **若要量化**：加一个「冷启动（`--cold`）」计时模式，两边的第一次调用含 setup 一起计时。小 case 大概率我们仍赢（我们的 setup 极小），但赢幅会缩小，大 case 可能反转。

---

## 5. 诚实的一面：对手的内核仍然更强

不要被 8/9 的成绩误导。**在单次调用的内核执行层面，ACL 23.11 仍然更强**：

| 维度 | ACL 23.11（oneDNN） | 本实现 |
|------|--------------------|--------|
| 变换内核 | 手写内联汇编，完全展开、无分支、指令调度到周期级 | 编译器生成 + 少量 intrinsics，未做周期级调度 |
| GEMM | arm_gemm JIT：按矩阵形状现场编译、寄存器阻塞、软件流水 | 默认 naive / 单线程 OpenBLAS（可选 arm_gemm） |
| 调度 | NEScheduler：任务图 + kernel window，多核扩展性强 | OpenMP 粗粒度，线程多了之后扩展性平台化（见 AGENTS.md t32/t38 数据） |

**含义**：
- 我们对 oneDNN 的优势是**工程实现策略**带来的，换到大负载会缩水甚至反转；
- 对手的汇编内核和 JIT GEMM 是**现成的优化目标**——我们一旦接入 arm_gemm（`-DUSE_ARM_GEMM`），就在内核层面与对手对齐，同时保留我们的低固定开销优势。

---

## 6. 建议与验证清单

按性价比排序：

1. **`ONEDNN_VERBOSE=1` 确认 oneDNN 实际跑的算法/内核**（winograd? F2/F4? SVE/NEON/SME? 哪个 kernel 族）。这一步直接验证 4.4 和 4.5，几十秒。
2. **统一测量环境**：同 numactl 绑定、同 warmup/repeats、确认 primitive 跨迭代复用、记录构建 flag。排除 4.6 的假赢。
3. **量化冷启动（可选）**：给 bench 加 `--cold` 模式，测两边第一次调用（含 setup）的耗时——若目标是「单次冷启动」场景，稳态数字不适用，需用它替代。
4. **在更大负载上复测**（batch 增大、空间增大、或整网络）：
   - 若大负载下被 ACL 反超 → 证实优势来自固定开销而非内核质量，结论是「微基准赢、生产负载未必」；
   - 若大负载仍赢 → 说明我们的内存/调度策略在 920F 上有系统性优势，值得深挖。
5. **把 ACL 的优化项当作下一步目标**：手写展开变换（F）、arm_gemm JIT GEMM（1）、合并 scatter/gather（3）。这些是真正拉近距离、并让大负载也赢的路径。

---

## 7. ACL 23.11 SVE 实现细节文档索引

oneDNN 侧实现细节的逐行分析在 `docs/acl_reference/`，按需查阅：

| 文档 | 内容 |
|------|------|
| `acl_wino_sve_asm_annotated.md` | SVE 内联汇编输入变换（`sve_fp32_6x6.cpp`）逐行注释 + 指令速查 |
| `acl_wino_sme_asm_annotated.md` | SME 输入/输出变换（`sme_fp32_mla_6x6` + `sme_fp32_mopa_4x4_3x3`）逐行注释 |
| `acl_wino_neon_asm_annotated.md` | NEON 内联汇编（`a64_fp32_6x6.cpp`）逐行注释 |
| `acl_wino_neon_intrinsics_annotated.md` | NEON C++ intrinsics 变换逐行注释 |
| `acl_wino_implementation_details.md` | ACL Winograd 实现深度剖析（矩阵推导、数据布局、调度、arm_gemm 内核选择机制） |
| `acl_wino_transform_kernels_explained.md` | 变换 kernel 源码分析（10 章，含公式/数值示例/数据布局） |
| `acl_23.11_wino_transform_kernels_explained.md` | ACL 23.11 版变换 kernel 文档 |
| `acl_23.11_vs_53.1.0_wino_analysis.md` | **23.11 vs 53.1.0 版本对比**：为什么选 23.11 的 SVE 而非 53.1.0 的 SME（GEMM 内核选择列表差异）+ SVE 内核差异详解 + 汇编入门 |

**阅读建议**：先看 `acl_23.11_vs_53.1.0_wino_analysis.md` 的第 1~4 节（版本选择根因），再看 `acl_wino_sve_asm_annotated.md`（SVE 变换写法），需要 GEMM 细节时看 `acl_wino_implementation_details.md` 的 arm_gemm 内核选择章节。

---

## 8. 验证与分析方式工具箱（如何证明/证伪「快于 ACL 23.11」）

> 本文 §4 列了六个原因，多数标着「需验证」。这一节给的是**验证工具与方法**，按「先证真、再归因」的顺序组织。每个方法给可执行命令（920F 专用）和它能回答的问题。核心原则：**每次只改一个变量，先排除假赢，再逐层归因**。

### 8.1 第一道门：先排除「假赢」（测量公平性）

在归因之前，必须确认对比本身没有系统性偏置。若 oneDNN 数字含 per-iteration primitive 重建/格式转换，或两边线程绑定/warmup 不一致，赢幅就是假的。

| 方法 | 命令 / 操作 | 回答的问题 |
|------|------------|-----------|
| oneDNN 实际路径 | `ONEDNN_VERBOSE=1 <bench>` | 它真走了 Winograd？F2/F4？SVE/NEON/SME？哪个 kernel 族？（验证 §4.4/4.5） |
| 拓扑确认 | `lscpu` / `numactl --hardware` | 16 NUMA 绑定是否一致；我们 numactl 绑了而对方没绑就是不公平 |
| 统一环境 | 两边同 numactl / `OMP_PROC_BIND=spread` / 同 warmup/repeats | 排除线程亲缘差异 |
| primitive 复用 | oneDNN 基准循环内**复用**同一 primitive 再测 | 若 oneDNN 每次迭代重建 primitive，固定开销被重复计入 → 系统性放大我们优势 |
| 冷启动量化 | 给 bench 加 `--cold`，只测两边第一次调用（含 setup） | 稳态数字是否适用；小 case 大概率仍赢但赢幅缩小 |
| 构建 flag | 记录两边 `-march`、`-O`、ISA 选项 | 排除「我们 -O3 + SVE，对方 -O2 + NEON」这类不对称 |

### 8.2 第二道门：拆时间——把「1.94ms vs 3.55ms」拆成可归因的片段

关键问题是：对方的 3.55ms 里，多少是**内核计算**、多少是**跑起来之前的固定开销**？

| 方法 | 操作 | 回答的问题 |
|------|------|-----------|
| 我们侧逐段计时 | `./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --timing`（8 步） | 我们自己的时间花在哪 |
| oneDNN 侧单 op 计时 | `ONEDNN_VERBOSE=1` 打印每 primitive 时间；或用最小程序只跑一个 conv primitive | 对方单 conv 的时间 |
| 固定开销分离 | oneDNN 连续调用**复用** primitive vs **每次重建**，差值 = per-iter setup | §4.1 的直接证据：setup 占多大 |
| 缩放扫描 | 同 shape 把 N/空间放大 4×、16× | 固定开销摊薄后胜负是否反转（§6-4 的核心实验） |

### 8.3 perf 硬件计数器（920F 专用事件）

> 注意：920F **无 L3**，`LLC-*` 事件不存在；用 L1/L2 事件。鲲鹏 920 有 SPE（见 8.4）。`perf list` 可查当前内核暴露的事件名。

```bash
# 基础：周期/指令/IPC/分支
perf stat -e task-clock,cycles,instructions,branches,branch-misses \
  ./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --warmup 5 --repeats 1

# cache（无 L3 → L1/L2）
perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,\
  L2-dcache-loads,L2-dcache-load-misses \
  ./profile_case ...

# topdown（区分 frontend/backend/memory bound）
perf stat --topdown ./profile_case ...
# 或显式：-e topdown-fetch-bubbles,topdown-recovery-bubbles,\
#   topdown-retiring,topdown-bad-spec

# NUMA（16 node 特色）
perf stat --per-node ./profile_case ... && numastat
```

| 计数器 | 回答的问题 |
|--------|-----------|
| cycles / IPC | 谁更接近峰值；指令级效率 |
| instructions | 我们的**总指令数**是否显著少于对方（粗粒度、少 barrier、少 setup → 指令少） |
| L2 miss 率 | 谁的访存局部性差（无 L3，L2 miss 直接进 DRAM） |
| topdown 各桶 | 对方手写汇编应是 backend/retiring 高；若我们 frontend/memory-bound 说明瓶颈在访存而非计算 |
| NUMA 分布 | 16 NUMA 上线程分布是否均匀；跨 node 访问量 |

### 8.4 SPE：Arm 统计采样（920F 访存局部性的终极证据）

SPE 直接采样**内存访问地址**，能量化「NHWC 连续 vs 跨步」和缓存局部性——这是论证 §4.3「内存策略契合 920F」的最强工具。

```bash
# 确认 920F 支持 SPE
ls /sys/bus/event_source/devices/ | grep -i spe

# 采样内存事件（period 控制采样率，越小越密）
perf record -e arm_spe_0/ts_enable=1,pa_enable=0,min_latency=0,period=10000/ \
  -- ./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --warmup 5 --repeats 1

# 分析：按函数/地址的负载直方图
perf report --stdio
# 导出采样 → 画访存地址分布（连续块 vs 稀疏跨步一目了然）
perf script > spe.dump
```

回答：`nchw_to_nhwc`/NHWC 提取是否呈现长连续段（局部性好的证据）；跨步访问占比多少；每条指令的平均访存延迟（`perf report --itrace` 相关，或看 sample 里的 latency 字段）。

### 8.5 假设驱动的对照实验（方法学核心）

每个「原因」配一个**可证伪**的实验，预期→结果→修正结论三段式：

| 假设（§4） | 可证伪实验 | 预期结果与解读 |
|-----------|-----------|---------------|
| 4.1 固定开销占优 | batch/空间放大 4×、16× 复测 | 赢幅缩小甚至反转 → 证实优势在固定开销；大负载仍赢 → 另有系统性优势 |
| 4.2 并行结构占优 | 我们 threads 1→8→16→38 vs oneDNN 同 threads，画扩展曲线 | 我们斜率陡且平台化早（t16 后平）；对方线性到 38 → 调度粒度差异 |
| 4.4/4.5 对方路径非最优 | `ONEDNN_VERBOSE=1` 确认真实路径 | 若它没走 SVE/F4 → 「内核不如它」前提需修正 |
| 5 内核对齐后胜负 | 我们 `-DUSE_ARM_GEMM=ON`（借 arm_gemm JIT 内核对齐） | 仍赢 → 赢在工程层；赢幅更大 → GEMM 本就是短板 |
| 4.3 内存策略占优 | 我们 NHWC vs 我们 NCHW 包装（`test_ref_vs_nchw --bench`） | NHWC 明显快 → 布局贡献量化；SPE 佐证局部性 |
| 调度粒度 | 我们 `schedule(dynamic,2)` vs `static` vs 其他 | 粒度差异对胜负的贡献 |

### 8.6 理论算术层

| 方法 | 操作 | 回答的问题 |
|------|------|-----------|
| 有效 GFLOPS | 双方各自：直接卷积等效 FLOP / 耗时（§2.2 已有） | 对方固定开销占比（负载越小 GFLOPS 越低越印证 §4.1） |
| 峰值利用率 | 实测 cycles ÷（主频 × 核数 × 2 FMA） | 双方离硬件峰值多远；我们若已 >50% 就是访存/调度受限而非内核 |
| 带宽上限 | STREAM 实测机器带宽，对比我们 U/M/V 流量估算（字节/耗时） | Case 4/5（U 22~44MB）是否已贴带宽墙 |
| 指令预算 | `perf stat instructions`，除以有效元素数 | 每输出元素我们比对方少多少条指令 |

### 8.7 负载-胜负热力图（最强诊断图）

扫 **tiles 数**（25→1600）和 **IC**（48→768）网格，每个格点跑双方，把胜负比画成热力图：

- 分界线在哪个区域翻转，直接指出「我们的工程优势在哪种负载下失效」；
- 若翻转区与「tiles 碎 + IC 小」重合 → 印证 §3.2 Case 2 的调度成本解释；
- 若在整个网格都赢 → 我们的内存/调度策略在 920F 上有系统性优势，值得深挖成论文级结论。

**综合顺序建议**：先 8.1（排除假赢）→ 8.2（拆时间定位大头）→ 8.5 的缩放实验（固定开销假说）→ 8.4 SPE（内存局部性定论）→ 8.7 热力图（边界画清）。perf 计数器（8.3）在每一步穿插，为每个结论补硬件证据。
