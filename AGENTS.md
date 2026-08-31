# AGENTS.md

> 本文件为 AI 代理（如 opencode、Claude、Copilot 等）在此代码库中工作时提供指导。

## 项目概述

Winograd 卷积复刻项目：基于 ACL（Arm Compute Library）的算法思路，用 C++17 + AArch64 NEON/SVE intrinsics + SME 内联汇编，独立实现的 Winograd 快速卷积。

**目标**：教学与研究用途，帮助理解 ACL Winograd 实现的算法原理和 ISA 优化策略。

**目标平台**：AArch64（华为鲲鹏 920F / Armv9 + SVE-512 + SME）。
L1=32KB/核, L2=768KB/核, **无 L3**, 16 NUMA × 38 cores = 608 cores。

## 当前状态（2026-08-31）— 新 agent 先读这里

- **正确性**：`--verify`（fp64 直接卷积参考 + 相对容差 1e-4）全过。arm_gemm 后端布局契约
  已修复（V 按 K 主序散写，`WINO_GEMM_BTRANS=0` 可回退对照），详见「历史修复记录」末尾。
- **性能终局（arm_gemm vs OpenBLAS，同作业）**：**59/59 形状全胜**，geomean **1.99x**，
  合计 321.9 vs 804.9ms = **2.50x**；大形状 4,384,160²,384 3.02x ≈94% SVE-512 峰值。
  全量逐形状表：`docs/final_benchmark_bfd6b1e.md`。
- **与 oneDNN 对照（⚠️ 卡点，进行中）**：工具 `tools/onednn/`（ab_onednn.sh = 单作业
  A/B：ours / e2e / benchdnn WINO+auto / filter_sweep / 合并表）。e2e 此前系统性 OOM 的
  **根因 = conv PD 误传 dilates={1,1}**（不是 omp_set_num_threads——8244944 是错修），
  已按用户参考格式修复（4bdc3ee）。判读矩阵、探针、坑：**`docs/onednn_comparison.md`**。
  ⚠️ benchdnn 计时虚高三修终版（50f9f2a）：① 608 线程超订（sbatch --exclusive
  默认全核）但 **TBB lib 忽略 OMP_NUM_THREADS**——用 **`numactl -C 0-15` 绑核**
  才生效；② benchdnn 默认 corr 模式，PASSED (N ms) 是聚合时间，须 **`--mode=p`**
  才打印 `perf,` 行（%-time%=单次执行 min）；③ merge 只解析 perf 行（exec 兜底），
  无则整列 N/A。**⛔ 当前卡点（2026-08-31）：集群作业产出的 benchdnn 文件仍无
  perf 行 → merge 两列 `[NO-SRC]`**。新脚本已确认部署（grep mode=p=5、merge 有
  iw=ih 回退），手动单/batch `--mode=p` 均能出 perf 行；作业内失败原因未明
  （嫌疑：BIN find 探测选错二进制 / 读了过期文件 / 作业上下文），诊断清单与
  计划修复见 **`docs/onednn_comparison.md` §六**。新 agent 接手先看该节。
  出 perf 行后 wino/auto 列应与 e2e 列同量级；e2e 列=最终对照。
- **M=25 选核**：已闭环（`tools/filter_sweep.sh`），**auto 保持最优**，不改选核。
- **运行协议**：所有性能对比必须**单 sbatch 作业内 A/B**
  （`sbatch -w node03 --exclusive --wrap="bash tools/onednn/ab_onednn.sh"`）——node03
  跨作业有 3~7x 性能态，跨作业数字一律作废。ours 侧计时只信**无 debug**（E3/E5）读数，
  `WINO_GEMM_DEBUG=1`（E1）会放大小形状。
- **判读/交接**：完整作业日志可能过长——`ab_onednn.sh` 末尾自动生成
  **`build/SUMMARY.txt`**（数据行数 / impl 直方图 / 诊断矩阵 / 合并表），判读与交给
  其他机器上的 agent 只取这一份即可；判读清单见 `docs/onednn_comparison.md` §八。
- **git 边界**：集群代码是 scp 文件副本（**无 .git**，CRLF 由脚本 `sed` 自愈）。
  本地仓库 `README.md` 与 `swish_sve/` 是用户未提交改动——**绝不 stage**（只 `git add`
  具名文件）；`tools/diag_ab.sh` 是临时诊断脚本，闭环后删除。

## 架构

### 数据流

```
输入特征图 [N][IC][IH][IW]
    │
    ├── 权重变换 (一次性): wei[OC][IC][3][3] → V[36][OC*IC] = G·g·G^T/576
    │   调用 dispatch_weight_transform() → NEON/SVE/SME
    │
    ├── 输入变换 (每 tile): d_tile[6][6][IC] → U_tile[6][6][IC] = B^T·d·B
    │   调用 dispatch_input_transform() → NEON/SVE/SME
    │
    ├── GEMM (每 Winograd 元素): M[ts][tile][OC] = Σ U[ts][tile][IC]·V[ts][OC][IC]
    │   调用 winograd_gemm() (naive / OpenBLAS cblas_sgemm / arm_gemm JIT SVE)
    │
    └── 输出变换 (每 tile): M_tile[6][6][OC] → f_tile[4][4][OC] = A^T·M·A + bias + ReLU
        调用 dispatch_output_transform() → NEON/SVE/SME
```

### 文件职责

| 文件 | 职责 | 关键类型/函数 |
|------|------|--------------|
| `winograd_config.hpp` | 配置 + ISA 检测 | `WinogradConfig`, `ISALevel`, `detect_isa()` |
| `winograd_matrices.hpp` | 变换矩阵常量 | `F22_G/Bt/A`, `F44_G/Bt/A` |
| `winograd_transforms.hpp` | NEON 变换 | `transform_2d_neon<>`, `weight/input/output_transform_neon<>` |
| `winograd_transforms_sve.hpp` | SVE 变换 | `transform_2d_sve<>`, `weight/input/output_transform_sve<>` |
| `winograd_transforms_sme.hpp` | SME 变换 | `input_transform_sme<>`, `sme_output_transform_f44()` |
| `winograd_convolution.hpp` | 端到端接口 | `winograd_convolution()`, `dispatch_*()` |
| `winograd_conv.cpp` | 端到端实现 | `winograd_gemm()`, `direct_convolution_3x3()` |
| `test_winograd.cpp` | 正确性验证 | `run_test()`, 变换级调试, B^T 矩阵求解器 |
| `bench_winograd.cpp` | 性能基准 | CSV 解析, GFLOPS 测量, 细粒度计时, 多线程对比, `--verify` 逐 case 正确性验证（OpenMP 并行 **fp64** 直接卷积参考，相对容差） |
| `profile_case.cpp` | 单 case profiling | `--timing` 8 子步骤, `--verify`, `--perf` 命令生成, 9 case preset |
| `swish_sve.h` | SVE Swish/SiLU 实现 | `exp_sve_f32()` (fexpa+fscale), `sigmoid_sve_f32()`, `swish_fwd_sve()`, `swish_bwd_sve()` |
| `swish_sve_bench.cpp` | Swish 性能/正确性基准 | 标量 ref 对比, GB/s 吞吐, 多 alpha/范围测试 |

### ISA 调度

三种 ISA（NEON < SVE < SME）通过运行时 dispatch 选择：

```cpp
ISALevel isa = isa_level();  // 可被 --neon/--sve/--sme 或 WINOGRAD_ISA 环境变量覆盖
dispatch_weight_transform(g, V, IC, is_f44, isa);
dispatch_input_transform(d, U, IC, is_f44, isa);
dispatch_output_transform(M, f, OC, bias, act_min, act_max, is_f44, isa);
```

## 构建

```bash
# 重要：必须清除旧缓存
rm -rf build && mkdir build && cd build

# NEON only (默认)
cmake .. -DCMAKE_BUILD_TYPE=Release

# SVE
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SVE=ON

# SME (包含 SVE)
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_SME=ON

# OpenBLAS GEMM（推荐）
cmake .. -DUSE_OPENBLAS=ON

# arm_gemm GEMM（与 oneDNN 相同的 SVE JIT 内核，需 ACL 源码树，见 tools/build_arm_gemm.md）
cmake .. -DUSE_ARM_GEMM=ON -DENABLE_SVE=ON -DARM_GEMM_ROOT=/path/to/ComputeLibrary-23.11/ComputeLibrary-23.11

# 全部启用（OpenBLAS 版）
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON -DUSE_OPENBLAS=ON

# 全部启用（arm_gemm 版）
cmake .. -DENABLE_SME=ON -DENABLE_OPENMP=ON -DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=...

make -j && ./test_winograd
```

**关键**：
- `__ARM_FEATURE_SVE`/`__ARM_FEATURE_SME` 是编译器内部宏，由 `-march` 触发，不能手动 `#define`
- 修改 CMakeLists.txt 后必须 `rm -rf build` 清缓存
- GEMM 内核优先级：arm_gemm > OpenBLAS > naive（互斥，arm_gemm 优先）
- OpenBLAS 需要 `cblas.h` + `libopenblas`：`apt install libopenblas-dev`
- arm_gemm 从 ACL 23.11 源码树**就地编译 fp32-SVE 子集**（19 个源 + `src/arm_gemm_cpuinfo.cpp`），用 `include/arm_compute/core/CPP/CPPTypes.h` 自包含 shim 替代 CPUInfo，无需编译整个 ACL。构建细节见 `tools/build_arm_gemm.md`

## 测试

```bash
# 正确性验证
./test_winograd              # 全部测试
./test_winograd --sme        # 强制 SME（需要 -DENABLE_SME=ON 编译）
./test_winograd --nhwc       # 测试 NHWC 布局（12 NCHW + 10 NHWC = 22 tests）
./test_winograd --f44        # 只测 F(4,4,3,3)
./test_winograd --relu       # 测 ReLU

# 性能基准（读 CSV，自动过滤 stride=1 group=1 3x3）
./bench_winograd --sve --nhwc --threads 1,8,16,32,38 shapes.csv
./bench_winograd --timing --threads 16 --sve --nhwc shapes.csv  # 细粒度计时
./bench_winograd --sve --nhwc --verify shapes.csv               # 逐 case 正确性验证（fp64 参考 + 相对容差 1e-4，FAIL 中止）
cat shapes.csv | ./bench_winograd --neon

# NCHW 包装 vs 存档原生 NCHW 参考（必须 bit-exact，8 shape × F44+F22）
./test_ref_vs_nchw
# 加 --bench 同测两条路径耗时，判定转换包装是否净赚（ratio<1=包装快，geomean 汇总）
./test_ref_vs_nchw --bench --threads 16 --warmup 3 --repeats 10

# 单 case profiling（perf 友好）
./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16         # 标准模式
./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --timing  # 8 子步骤计时
./profile_case --ic 96 --ih 80 --iw 80 --oc 96 --verify                          # 正确性验证
./profile_case --ic 384 --ih 80 --iw 80 --oc 96 --perf                           # 生成 perf 命令
./profile_case --help                                                           # 9 case preset 一览

# perf profiling
perf stat -e cycles,instructions,cache-misses,cache-references \
  ./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --repeats 1
perf record -g -- ./profile_case --ic 192 --ih 40 --iw 40 --oc 192 --isa sve --threads 16 --repeats 10
perf script | flamegraph.pl > flame.svg

# NUMA 优化
numactl --interleave=all env OMP_PROC_BIND=spread OMP_PLACES=cores \
  ./bench_winograd --sve --nhwc --threads 32 shapes.csv

# Swish/SiLU SVE 性能/正确性基准
./swish_sve_bench
```

测试包含：
1. **变换级调试**：权重/输入/输出/全流水线，用已知输入验证
2. **端到端测试**：12 个用例（小图到大图），对比直接卷积，容差 1e-3

### Swish/SiLU SVE 实现要点

`swish_sve/swish_sve.h` 是独立的 header-only SVE Swish 实现，算法对齐 oneDNN 3.12.1：

- **算法链**：`swish → sigmoid → exp`
- **exp**：`fexpa` + `fscale`（SVE 专用硬件加速）+ 线性多项式修正
- **sigmoid**：对称性优化 $\sigma(-x)=1-\sigma(x)$，只对负数算 exp 避免溢出
- **alpha=1 快速路径**：SiLU 跳过 alpha 乘法
- **backward**：`fmls` + `fmla` 链式计算 $Q(1+R(1-Q))$，无栈访问（优于 oneDNN JIT 版）
- **VLA**：`svwhilelt_b32` 谓词处理 tail，适配 128/256/512-bit SVE
- **fexpa**：通过 inline assembly 包装（ACLE 标准未直接暴露）

## 代码约定

- **语言**：C++17
- **命名空间**：`winograd_conv`
- **模板参数**：`TILE_SIZE`（4 或 6）、`OUTPUT_TILE`（2 或 4）、`OUT_SIZE`/`IN_SIZE`
- **数据布局**：**计算核心只认 NHWC**（`[tile][channels]` 通道连续）；NCHW 是包装层——入口 `nchw_to_nhwc` 转换、算完 `nhwc_to_nchw` 转回（见设计决策 #13）。原生 NCHW 内核已存档在 `ref/winograd_conv_nchw_ref.cpp`
- **ISA 守卫**：`#if defined(__ARM_FEATURE_SVE)` / `#if defined(__ARM_FEATURE_SME)`
- **SME 汇编**：使用 `.inst` 编码（与 ACL 一致），不依赖编译器 SME 助记符支持
- **无注释**：代码中不加注释（除非用户明确要求）

## 关键设计决策

1. **统一矩阵定义**：所有 ISA（NEON/SVE/SME）共用 `winograd_matrices.hpp` 中的矩阵常量。修复矩阵只需改一处，三种 ISA 路径同时修复

2. **泛型 2D 变换**：`transform_2d_neon<OUT, IN>` 用同一矩阵做列变换和行变换，因为 Winograd 的 B 和 A 满足 `output = M * input * M^T`

2. **三层降级（NEON）**：NEON Q 寄存器固定 128-bit，无法用谓词处理 tail，写三份代码（4→2→1 通道）

3. **谓词化（SVE）**：`whilelt` 谓词自动处理 tail，一份代码替代三层降级

4. **Kronecker 积（SME）**：输出变换用 `(A^T ⊗ A^T) · vec(M)` 替代两次 1D 矩阵乘，用 FMOPA 外积累加实现

5. **权重变换用 NEON**：权重变换在 prepare 阶段做一次，非热路径，ACL 也只有 NEON 版

6. **bias + ReLU 在变换内部**：输出变换函数内部添加 bias 并 clamp，避免端到端函数重复添加

7. **矩阵验证方法**：用 Winograd 多项式条件 `sum_j A^T[i][j]·B^T[j][a]·G[j][b] = C·delta(a, i+b)` 数值验证矩阵正确性。当矩阵来源不可靠时，可用求解器从 A^T 和 G 反解出正确的 B^T

8. **OpenMP 并行策略**：单个 `#pragma omp parallel` 区域包含全部阶段：
   - 权重变换：`#pragma omp for schedule(dynamic, 4)` over OC
   - barrier（V 必须完成才能进入 batch 循环）
   - 每个 batch：Phase 1（输入变换）`#pragma omp for collapse(2) schedule(dynamic, 2)` → barrier → Phase 2（GEMM）`#pragma omp for schedule(dynamic)` → barrier → Phase 3（输出变换）`#pragma omp for collapse(2) schedule(dynamic, 2) nowait`
   - 权重 + 所有 batch 合并在 1 个 parallel region，仅 1 次 fork/join
   - 6 个 per-thread 缓冲区（d_tile, U_tile, M_tile, f_tile, g_wt, V_oc_wt）在入口一次性分配、全程复用
   - OpenBLAS 单线程 `openblas_set_num_threads(1)` 避免冲突

   **教训**：权重变换最初放在独立 `#pragma omp parallel` 中，导致 Case 0/1 变慢（额外 fork/join ~1-2ms 抵消权重并行收益）。修复方法是合并进主 parallel region。Case 4/5（大 IC）则受益明显（-19%~-30%）。

9. **GEMM 内核切换**：编译期选择，优先级 arm_gemm > OpenBLAS > naive
   - `USE_ARM_GEMM`：ACL 23.11 arm_gemm 现代 API（`GemmArgs` + `gemm()` 工厂），`cfg.filter="sve_"` 强制 SVE 内核，每次调用 `pretranspose_B_array` 预转置 V（即 pack B）+ `execute` 单线程
   - `USE_OPENBLAS`：`cblas_sgemm` 单线程，`-DUSE_OPENBLAS=ON`
   - 默认：naive 三重循环

10. **NHWC 布局优化**：`Layout` 枚举选择 NCHW 或 NHWC。NHWC 下 tile 提取用 `vld1q_f32` 连续加载（3.8x 快于 NCHW 标量）

11. **transform_2d 临时缓冲区**：`thread_local static float*` + `malloc`/`free`，仅容量不足时重新分配

12. **tile 提取优化**：内部 tile 跳过 `memset` 清零，预计算有效行列范围消除 `if` 分支

13. **NCHW 包装层（2026-08-11 布局重构）**：Winograd 计算与布局无关，NCHW/NHWC 只差 tile 提取与写回。NCHW 的提取/写回按 `H*W` 通道步长访问，在无 L3 机器上 ~16x 缓存行放大。因此：把计算核心固定为 NHWC（`winograd_convolution_nhwc_core`），NCHW 入口做 `nchw_to_nhwc` → 计算 → `nhwc_to_nchw`（cache-blocked 转置，16 通道块 + `copy_f32`）。转换是纯数据搬运，输出与旧内核 **bit-exact**。原生 NCHW 内核存档在 `ref/`，`test_ref_vs_nchw` 断言 bit-exact。转换方案是否净赚需在目标机实测（见 PERFORMANCE_ANALYSIS.md §12）。

## 已知限制

1. **GEMM 内核**：3 种可选（arm_gemm JIT SVE > OpenBLAS > naive）。**生产后端 = arm_gemm**（已在 920F 验证：59/59 全胜 OpenBLAS，见 `docs/final_benchmark_bfd6b1e.md`；构建指南 `tools/build_arm_gemm.md`）。OpenBLAS/naive 保留作对照与开发
2. **NCHW 包装 vs 原生**：NCHW 现在是「转换 + NHWC 计算」包装。计算是共享的，NCHW 输入的实际开销 = 转换（整图两遍全量数据搬运）vs 旧内核的跨步提取/写回。2026-08-20 在 920F（9T NCHW 端到端）实测**转换开销较小**，后续性能比较统一用 NCHW 端到端口径。若某些 shape 转换反而更慢，可回退用 `ref/winograd_conv_nchw_ref.cpp` 里的原生内核（`Layout::NCHW` 换成 `winograd_convolution_nchw_ref`）
3. **OpenMP 并行**：权重变换 + 输入/输出变换 + GEMM 均已并行。Phase 1-3 合并 1 区域，2 barrier（输入→GEMM, GEMM→输出），输出 `nowait`。权重变换独立并行区域
4. **多线程扩展性受限**：8 线程后加速停滞，剩余瓶颈：OpenMP barrier 开销、GEMM 内核质量（OpenBLAS vs arm_gemm JIT）、NUMA 远程访问（920F 16 NUMA）
5. **SME 仅 F(4,4,3,3) 输出变换**：F(2,2,3,3) 输出变换回退到 SVE
6. **`--timing` 模式是串行的**：用于分析各阶段时间占比，不代表并行后的实际性能
7. **9 线程 NCHW 端到端落后 oneDNN**：2026-08-20 实测多数 case（7/9）落后 `wino:acl`（仅 row 4/9 赢）。瓶颈在 **per-tile 内核效率**（GEMM/变换/scatter-gather/权重变换重算），不在固定开销——详见「性能优化记录」→「9 线程 NCHW 端到端实测」，差距归因见 `docs/why_faster_than_acl_23.11.md` §10

### 性能对比（NHWC, SVE, 16 线程，完整 9 case）

| Case | Shape | 本项目 t16(ms) | oneDNN t16(ms) | 结果 |
|------|-------|--------------|---------------|------|
| 0 | 4,192,40,40 | 7.20 | 3.55 | 慢 2.03x |
| 1 | 4,96,80,80 | 7.54 | 4.22 | 慢 1.79x |
| 2 | 4,48,160,160 | 10.39 | 4.06 | 慢 2.56x |
| 3 | 4,192,20,20 | 1.17 | 2.83 | **快 59%** ✓ |
| 4 | 4,384,80,80 | 6.80 | 13.78 | **快 51%** ✓ |
| 5 | 4,768,40,40 | 4.43 | 9.09 | **快 51%** ✓ |
| 6 | 4,768,20,20 | 2.30 | 5.58 | **快 59%** ✓ |
| 7 | 4,96,40,40 | 1.08 | 1.90 | **快 43%** ✓ |
| 8 | 4,96,20,20 | 0.50 | 1.15 | **快 56%** ✓ |

**6/9 case 超越 oneDNN**（快 43-59%）。慢的 case 是多 tile + 小 IC（barrier 开销占比大）。

> ⚠️ **适用范围**：16T NHWC 微基准（历史；oneDNN 侧为 **benchdnn WINO 口径**，见 `docs/why_faster_than_acl_23.11.md` §10.4）。NCHW 端到端、9T 等线程档的实测见「性能优化记录」→「9 线程 NCHW 端到端实测」。

### t32 对比

| Case | 本项目 t32(ms) | oneDNN t32(估) | 结果 |
|------|--------------|--------------|------|
| 0 | 7.0 | 3.6 | 慢 1.94x |
| 1 | 6.9 | ~3.5 | 慢 ~2.0x |
| 2 | 8.7 | ~3.0 | 慢 ~2.9x |
| 3 | 1.0 | ~2.5 | **快 ~2.5x** ✓ |
| 4 | 4.9 | ~12 | **快 ~2.4x** ✓ |
| 5 | 3.6 | ~8 | **快 ~2.2x** ✓ |
| 6 | 2.0 | ~5 | **快 ~2.5x** ✓ |
| 7 | 1.1 | ~1.5 | **快 ~1.4x** ✓ |
| 8 | 0.5 | ~0.8 | **快 ~1.6x** ✓ |

Case 3/4/5 已大幅超越 oneDNN（大 IC 时变换计算量大，OpenMP 并行效率高）。Case 0/1/2 仍慢（小 IC，barrier 开销占比大）。详见 `PERFORMANCE_ANALYSIS.md`

## 历史修复记录

| 日期 | 问题 | 根因 |
|------|------|------|
| 2026-08 | F44 全部失败 | F44_Bt 矩阵 5 个元素错误，用数值求解器从 A^T 和 G 反解出正确 B^T |
| 2026-08 | F22 全部失败 | F22_A 矩阵 [1][1] 符号错误（-1 应为 +1） |
| 2026-08 | F44 OOB 读取 | weight_transform_neon 冗余外层 k 循环导致 vld1q 越界读取（UB） |
| 2026-08 | segfault | transform_2d_neon CHANNELS 模板参数未传，默认 0，零大小数组 |
| 2026-08 | SME bias 翻倍 | 输出变换内部已加 bias，端到端函数又加一次 |
| 2026-08 | 权重变换标量 | dispatch_weight_transform 已定义但未调用 |
| 2026-08 | SVE 编译报错 | `svptest_first` 需要 2 参数；`static inline` 函数在模板中触发两阶段查找，改用宏 |
| 2026-08 | OpenMP 不生效 | CMake `PRIVATE`→`PUBLIC`；`omp_set_num_threads` 原为桩函数 |
| 2026-08 | double free | OpenBLAS 内部线程与 OpenMP 冲突，`openblas_set_num_threads(1)` |
| 2026-08 | OpenMP 无加速 | 每 tile 在循环内 `std::vector` 分配导致堆锁竞争 |
| 2026-08 | 权重并行后变慢 | 独立 `#pragma omp parallel` 增加额外 fork/join，合并进主 region 修复 |

## 性能优化记录

### 优化历程（Case 0: 4×192×40×40, NHWC, SVE）

| 阶段 | t1(ms) | t8(ms) | t16(ms) | t32(ms) | vs oneDNN t32 |
|------|--------|--------|---------|---------|-------------|
| ① NCHW 基线 | 22.5 | 14.6 | 14.2 | 14.0 | 3.89x |
| ② +NHWC 布局 | 16.3 | ~14 | ~14 | ~14 | ~3.89x |
| ③ +GEMM并行+schedule(2)+raw malloc | 21.6 | 10.3 | 9.5 | 9.1 | 2.53x |
| ④ +合并OpenMP区域 | 18.9 | 7.9 | 7.0 | 6.7 | 1.86x |
| ⑤ +优化B+C(跳过fill+nowait) | — | — | — | — | — |
| ⑥ +权重变换并行（独立region） | 20.2 | 8.5 | 7.4 | 7.2 | 2.00x ↑变慢！ |
| ⑦ +合并权重到主region | 19.9 | 8.2 | 7.3 | 7.0 | 1.94x |
| ⑧ +展平N个batch(9→3 barriers) | 20.2 | 7.3 | 6.4 | 6.0 | 1.67x Case0好, Case4+76%↓ |
| ⑨ 回退到⑦(per-batch) | 20.6 | 8.1 | 7.2 | 7.0 | 1.94x (最终版) |
| oneDNN 参考 | 18.0 | 4.9 | 4.0 | 3.6 | 1.0x |

**关键教训**：
- ⑥ 权重变换放独立 `#pragma omp parallel`，Case 0 变慢（t32: 6.7→7.2），Case 4/5 改善（-19~-30%）。原因：额外 fork/join ~1-2ms 抵消小 OC 的权重并行收益
- ⑦ 合并进主 region 修复，Case 0/1 恢复，Case 3/4/5 保持改善
- ⑧ 展平 N 个 batch 减少 barrier（9→3），但 U/M_buf 内存增加 N 倍。Case 0（U=11MB）改善 -14%，Case 4（U=88MB）严重劣化 +76%（cache thrashing，920F 无 L3、L2 仅 768KB）
- ⑨ 回退到⑦（per-batch），最终最优方案。Case 0/1/2 慢于 oneDNN 但 Case 3-8 超越 oneDNN

### A1/A2/A3 落地（2026-08-10）：SVE 化内存拷贝 + 消除无用清零 + 缓冲跨调用复用

| # | 优化 | 描述 |
|---|------|------|
| A1 | 消除 U/M_buf/V 无用清零 | `std::vector(..., 0.0f)` → `scratch_f32()`（malloc 不清零）。三个 buffer 在读取前必被完整覆写（scatter / GEMM beta=0 / 权重变换），原先每次调用对最大 ~33MB 做无谓 memset |
| A2 | per-thread 缓冲跨调用复用 | 6 个 per-thread 缓冲（d_tile/U_tile/M_tile/f_tile/g_wt/V_oc_wt）+ U/M_buf/V 改用 `thread_local` 增长式缓冲（`scratch_f32()`），消除 benchmark/推理循环每次调用的堆分配 churn（原先每次调用 6 vector/线程 × N 线程） |
| A3 | SVE 化 4 处内存拷贝 | tile 提取/scatter/gather/输出写回改用 `copy_f32()`：SVE-512 16 float/指令（vs NEON 4 float），谓词自动处理 tail；NEON 构建自动回退 4 float。**编译期选择**（由 -march 决定），与运行时 isa_level() 覆盖无关 |

**实现位置**：
- `copy_f32()`：`winograd_transforms.hpp`（`__ARM_FEATURE_SVE` 守卫 SVE 路径，NEON 回退）
- `scratch_f32(n, Scratch&)`：`winograd_conv.cpp` 匿名 namespace（thread_local 增长式，容量不足才 realloc）
- 4 处拷贝调用点：`winograd_conv.cpp` Phase 1 tile 提取/scatter、Phase 3 gather/写回；`profile_case.cpp` 与 `bench_winograd.cpp` 的 `--timing` 模式同步更新保持一致

**⚠️ 坑（2026-08-10 修复）**：`scratch_f32` 每个逻辑缓冲必须持有**独立的** `thread_local Scratch`（U/M/V 各一、6 个 tile 缓冲各一）。最初用一个 `Scratch` 服务所有缓冲，导致 9 个缓冲全部别名同一内存，正确性测试全挂（天文数字错误）。此坑已在代码注释中标注。

**⚠️ 坑（2026-08-10 修复，预存 bug）**：tile 提取的边界裁剪原为「最后一个 tile 行/列用 `TS-1`」，只对**偶数** IH/IW 成立。奇数维度（如 IH=7，最后一行 tile 读 ih=7）会越界读到**下一通道的第 0 行**（batch 最后通道甚至读到缓冲外），正确性测试 F(2,2) N=2 IC=16 IH=7 挂。已改为按实际边界裁剪：`ti_end = (ih_begin+TS > IH) ? (IH - ih_begin) : TS`（`ih_begin = tr*OT-1`），偶数维行为不变。同时修正了 test_winograd.cpp End-to-End Debug 的错误期望（tap (0,0) 的 delta 输出在 (1,1) 即 flat index 5，不是 1）。

**验证结果（2026-08-10 A1/A2/A3 后，920F 复测）**：正确性 12/12 通过。性能见下方「最终性能对比」。原预期「拷贝项各 -50~75%」已兑现且超出——Case 0/1/2 总时间 -3.0~3.7×。

**数值精度与 `--verify` 方法（2026-08-11）**：
- **误差来源**：F(4,4) fp32 误差主要来自 GEMM 按 IC 串行累加（`src/winograd_conv.cpp` naive 三重循环），再被输出变换 A 矩阵（系数最高 ±8）放大。误差随 IC 线性增长，但输出幅值也随 IC 线性增长 → **相对误差恒定 ~2-6e-6**，属 fp32 正常水平（非 bug）。
- **早期 1e-3 绝对容差误报大 IC case FAIL**：IC=384/768 时输出幅值 ~1000，1e-2 绝对误差相对只有 ~6e-6。固定绝对容差不随规模缩放是错误指标。
- **正确做法**：`--verify` 参考改用 **fp64 直接卷积**（`direct_convolution_3x3_f64`，OpenMP 按 batch 并行），判据改**相对容差** `err < 1e-4 × max|ref| + 1e-5`。fp64 参考测的是 Winograd 相对精确数学的真实误差。
- **若要进一步降误差**：GEMM fp64/Kahan 累加（打在性能关键路径）> 变换 fp64（破坏 SIMD，代价大）> 换 F(2,2)（矩阵全 ±1 无放大，但改变被测算法）。换 arm_gemm 后 blocked 累加也会比 naive 串行更准。

### 最终性能对比（完整 9 case, NHWC, SVE, 16 线程）— A1/A2/A3 后复测

| Case | Shape (N,IC,IH,IW) | (OC,IC) | Tiles | 旧 t16(ms) | 新 t16(ms) | 加速 | oneDNN t16(ms) | 结果 |
|------|---------------------|---------|-------|-----------|-----------|------|---------------|------|
| 0 | 4,192,40,40 | 192,192 | 100 | 7.20 | **1.937** | 3.72x | 3.55 | **快 1.83x** ✓ |
| 1 | 4,96,80,80 | 96,96 | 400 | 7.54 | **2.515** | 3.00x | 4.22 | **快 1.68x** ✓ |
| 2 | 4,48,160,160 | 48,48 | 1600 | 10.39 | **4.586** | 2.27x | 4.06 | 慢 1.13x |
| 3 | 4,192,20,20 | 192,192 | 25 | 1.17 | **0.799** | 1.46x | 2.83 | **快 3.54x** ✓ |
| 4 | 4,384,80,80 | 96,384 | 400 | 6.80 | **6.080** | 1.12x | 13.78 | **快 2.27x** ✓ |
| 5 | 4,768,40,40 | 96,768 | 100 | 4.43 | **4.055** | 1.09x | 9.09 | **快 2.24x** ✓ |
| 6 | 4,768,20,20 | 96,768 | 25 | 2.30 | **1.808** | 1.27x | 5.58 | **快 3.09x** ✓ |
| 7 | 4,96,40,40 | 96,96 | 100 | 1.08 | **0.893** | 1.21x | 1.90 | **快 2.13x** ✓ |
| 8 | 4,96,20,20 | 96,96 | 25 | 0.50 | **0.344** | 1.45x | 1.15 | **快 3.34x** ✓ |

**8/9 case 超越 oneDNN**（t16 几何平均加速 ~1.67x）。Case 0/1 从落后 2x 反超为领先；Case 2 从慢 2.56x 追到 11.5% 以内。

> ⚠️ **适用范围**：16T NHWC 微基准（历史；oneDNN 侧为 **benchdnn WINO 口径**而非端到端——见 `docs/why_faster_than_acl_23.11.md` §10.4，8/9 含金量打折）。NCHW 端到端实测（9T）只赢 2/9，见「9 线程 NCHW 端到端实测」与 why_faster §10。
- **改进分布三档**：拷贝/缓冲受限的 Case 0/1/2 加速 2.3~3.7x（A1/A2/A3 正中要害）；均衡 Case 3/6/7/8 加速 1.2~1.5x；**GEMM 受限的 Case 4/5 仅 ~1.1x**（IC=384/768，U 缓冲 22~44MB 远超 L2，瓶颈在 naive GEMM，A1/A2/A3 未触及）
- **高线程扩展性**：t1→t38 缩放 6.2~26.6x。Case 2（1600 tile）最好 26.6x；小 tile 数 Case 0/7/8 在 16 线程后平台化（~10x），无 L3 + 768KB L2 内存带宽受限
- **峰值**：Case 4 达 5108 GFLOPS（t38, ~5.1 TFLOPS）

### 最终多线程扩展性（完整 9 case, 最终版⑦）

| Case | Shape | Tiles | t1 | t8 | t16 | t32 | t38 | t8加速 | t32加速 |
|------|-------|-------|-----|-----|-----|-----|-----|--------|--------|
| 0 | 4,192,40,40 | 100 | 20.6 | 8.1 | 7.2 | 7.0 | 6.8 | 2.54x | 2.94x |
| 1 | 4,96,80,80 | 400 | 28.6 | 9.0 | 7.5 | 6.9 | 6.7 | 3.18x | 4.14x |
| 2 | 4,48,160,160 | 1600 | 67.0 | 13.8 | 10.4 | 8.7 | 8.0 | 4.85x | 7.70x |
| 3 | 4,192,20,20 | 25 | 8.4 | 1.7 | 1.2 | 1.0 | 1.1 | 4.94x | 8.40x |
| 4 | 4,384,80,80 | 400 | 53.5 | 11.1 | 6.8 | 4.9 | 4.1 | 4.82x | 10.92x |
| 5 | 4,768,40,40 | 100 | 31.2 | 6.6 | 4.4 | 3.6 | 3.5 | 4.73x | 8.67x |
| 6 | 4,768,20,20 | 25 | 17.7 | 3.2 | 2.3 | 2.0 | 1.8 | 5.48x | 8.85x |
| 7 | 4,96,40,40 | 100 | 9.0 | 1.6 | 1.1 | 1.1 | 1.1 | 5.63x | 8.18x |
| 8 | 4,96,20,20 | 25 | 2.9 | 0.7 | 0.5 | 0.5 | 0.5 | 4.38x | 5.79x |

### 细粒度计时数据（NHWC, SVE, 1 线程, timing 模式）

注意：timing 模式是串行的（不含 `#pragma omp`），OpenBLAS 可能用默认多线程

| 阶段 | Case 0 (192×192) | Case 4 (384×96) | Case 5 (768×96) |
|------|-----------------|----------------|-----------------|
| 权重变换 | 1.96ms (12%) | 1.05ms (2%) | 1.96ms (8%) |
| Tile 提取(NHWC) | 1.08ms (7%) | 9.14ms (20%) | 4.58ms (20%) |
| 输入变换 | 2.78ms (18%) | 20.83ms (45%) | 9.81ms (42%) |
| Scatter | 0.77ms (5%) | 4.97ms (11%) | 2.32ms (10%) |
| GEMM | 1.22ms (8%) | 1.62ms (3%) | 2.16ms (9%) |
| Gather | 1.82ms (12%) | 2.71ms (6%) | 0.43ms (2%) |
| 输出变换 | 2.02ms (13%) | 4.70ms (10%) | 1.17ms (5%) |
| 输出写回 | 0.38ms (2%) | 1.06ms (2%) | 0.28ms (1%) |

### 胜负模式分析

**赢的条件**（6/9 case, 快 43-59%）：
- IC ≥ 384（GEMM 矩阵大，OpenBLAS 高效，权重并行收益大）
- 或 tiles ≤ 100 且 IC ≥ 96（barrier 开销占比小）

**输的条件**（3/9 case, 慢 1.79-2.56x）：
- IC ≤ 192 且 tiles ≥ 100（GEMM K 维小 + barrier 开销占比大）
- Case 2 最差（IC=48, tiles=1600）：GEMM K=48 极小 + 8 个 barrier

### 9 线程 NCHW 端到端实测（2026-08-20）：多数 case 落后 oneDNN wino:acl

**测量**：NCHW 端到端计时、numactl 绑核、6 线程档（4/8/9/16/32/38）。onednn 走 ACL SVE Winograd（`wino:acl`，8/9 row）或直接卷积（`brgconv:sve_512`，1/9 row）；我们走 `winograd_convolution`（NCHW 包装层 → NHWC core）。转换开销实测较小（用户确认），差距在核心 per-tile 效率，不在包装层。（benchdnn 与端到端测速为何不同、怎么公平对照 → `docs/why_faster_than_acl_23.11.md` §10.4）

**读数约定**：加速比 = **onednn / 我们**，>1 表示我们快（曾误读为 <1 快，方向反了会得出完全相反的结论）。

| row | Count | onednn 实现 | 我们 ms (4/8/9/16/32/38T) | onednn ms (4/8/9/16/32/38T) | 加速比 (4/8/9/16/32/38T) |
|---|---|---|---|---|---|
| 1 | 24 | wino:acl | 5.733/3.303/2.975/2.113/1.749/1.605 | 4.398/1.864/1.886/1.463/1.661/1.139 | 0.77/0.56/0.63/0.69/0.95/0.71 |
| 2 | 17 | wino:acl | 8.571/4.701/4.284/2.915/2.137/1.876 | 4.145/3.101/2.758/2.051/1.308/1.127 | 0.48/0.66/0.64/0.70/0.61/0.60 |
| 3 | 8 | brgconv:sve_512 | 17.788/9.483/8.273/5.427/3.544/2.800 | 9.241/4.734/4.137/2.335/1.203/1.039 | 0.52/0.50/0.50/0.43/0.34/0.37 |
| 4 | 16 | wino:acl | 2.500/1.498/1.327/0.901/0.742/0.852 | 1.708/1.713/2.045/1.436/0.393/0.315 | 0.68/1.14/**1.54**/1.59/0.53/0.37 |
| 5 | 1 | wino:acl | 22.903/12.102/10.684/7.102/4.895/4.120 | 9.954/6.884/7.461/6.198/6.658/5.569 | 0.43/0.57/0.70/0.87/1.36/1.35 |
| 6 | 1 | wino:acl | 12.946/7.064/6.478/4.529/3.346/3.117 | 6.007/3.551/3.650/2.503/5.873/5.183 | 0.46/0.50/0.56/0.55/1.76/1.66 |
| 7 | 1 | wino:acl | 5.420/3.012/2.964/1.908/1.527/1.256 | 2.312/2.152/2.069/1.464/1.890/1.677 | 0.43/0.71/0.70/0.77/1.24/1.34 |
| 8 | 1 | wino:acl | 2.909/1.624/1.443/1.019/1.045/1.111 | 1.473/1.074/0.988/0.564/0.323/0.301 | 0.51/0.66/0.68/0.55/0.31/0.27 |
| 9 | 1 | wino:acl | 0.998/0.592/0.541/0.418/0.474/0.515 | 0.641/0.519/0.642/0.514/0.087/0.080 | 0.64/0.88/**1.19**/1.23/0.18/0.16 |

> ⚠️ shape（mb,ic,ih,iw,oc）**待补**——当前只能按 row 编号对应，无法映射 tiles/IC，也就无法精确判定每行是 GEMM/变换/内存受限。

**胜率分布**：

| 线程 | 赢的 row | 说明 |
|---|---|---|
| 4T | 无 | 全面落后 |
| 8T | 4 | |
| 9T | 4、9 | ← 当前关注点 |
| 16T | 4、9 | |
| 32/38T | 5、6、7 | 因 onednn 自身 32/38 退化（row 5/6/7 的 onednn 16→32 反而变慢） |

onednn 的 32/38 行为反常：row 8/9 塌缩 4-6x（0.514→0.087ms）、row 5/6/7 反而退化——两个方向暂不深究，聚焦 9T。

**9T 落后的结构性原因**（代码已核实）：
1. **GEMM**：我们 `openblas_set_num_threads(1)`（src/winograd_conv.cpp:331）+ `cblas_sgemm`（240-246）钉死单线程小矩阵；对方就是 arm_gemm JIT——`winograd_gemm` 的 arm_gemm 分支（233-239）已写好、默认未开，**同一套内核**。
2. **变换内核**：编译器生成 vs 对方手写 SVE 汇编（`sve_fp32_6x6.cpp`，完全展开无分支、周期级调度）。
3. **scatter/gather**：Phase 1 散入 U + Phase 3 从 M 收集 = 两笔全量内存遍历；对方 `matrix_stride` 布局让变换直接写进 GEMM 顺序，不做 scatter/gather。
4. **权重变换每调用重算**：对方把 TransformedWeights 缓存到 primitive。

**9T 定向优化**（按性价比）：
1. **arm_gemm JIT**：✅ 代码已就绪（现代 API 重写，`src/winograd_conv.cpp` + `include/arm_compute/core/CPP/CPPTypes.h` shim + `src/arm_gemm_cpuinfo.cpp` + CMake 编译 fp32-SVE 子集）。**待 920F 实测**：`tools/build_arm_gemm.md` 按步骤构建 + `WINO_GEMM_DEBUG=1` 确认 SVE 内核 + 与 OpenBLAS A/B。GEMM 受限 row 直接拉平。小优化：`gemm()` 每次调用构造一次（36 次/批），提为静态缓存按 shape 复用。
2. **缓存 V**（权重变换跨调用复用，key = 权重指针 + shape）：Count≥8 的 row 1-4 立省一次权重变换。
3. **变换内核手写/展开**（对应「新增优化思路 F」）。
4. **消除 scatter/gather**：变换直接写 GEMM 布局，砍掉 Phase 1 scatter + Phase 3 gather。

**待补数据**：端到端 row 7-9 的 shape 列（row 1-6 已由 benchdnn 输出确认 = Case 0-5）；转换开销数字（用户可提供）。

**benchdnn vs 端到端测速**（why_faster §10.4 全量对照；做 oneDNN 性能对照前必读）：
- 实测（2026-08-26，benchdnn 16T WINO vs 端到端 16T，同 shape）：Case 0-5 慢 **1.74~3.63x**（benchdnn 3.55/4.22/4.06/2.83/13.78/9.09 vs 端到端 1.46/2.05/2.34/1.44/6.20/2.50）。
- **关键**：benchdnn 数字与历史「oneDNN 16 线程性能参考」（PERFORMANCE_ANALYSIS §2）**逐位一致**，且**用户已确认**：历史参考就是 benchdnn 跑的、实现为 **`wino_acl`**（与端到端**同为 wino_acl，无 wino_dlb**）。所以 6/9、8/9 两表是「**我们端到端 vs oneDNN benchdnn**」的不对等比较，oneDNN 被测量方式拖慢 1.7-3.6x，**8/9 的含金量要打折**；9T NCHW 端到端（两边同口径、同实现）才是公平战场，7/9 落后真实。
- 归因（同为 wino_acl → **gap 纯执行环境**，实现选择已排除，按嫌疑）：① 端到端 numactl 绑核 vs benchdnn 未绑 → 16 NUMA 跨节点全 miss（决定性验证：同 numactl 重跑 benchdnn）；② 布局/重排（benchdnn plain 格式 vs 端到端 format_any 可能不同）；③ 缓存冷热 + 每行重建 primitive；④ 每 iteration 固定开销。
- 公平对照三条件：**同样 numactl 绑核 + 同样实现（已确认同为 wino_acl）+ 重复迭代复用 primitive**。

### 下一步优化方向（按预期收益排序）

详见 `PERFORMANCE_ANALYSIS.md` 和 `docs/acl_reference/`。

**针对慢 case（0/1/2）的优化**：

1. **arm_gemm 替换 OpenBLAS** ✅ **已完成（2026-08-28）**：59/59 全胜 OpenBLAS，
   geomean 1.99x（见 `docs/final_benchmark_bfd6b1e.md`）；后续小矩阵瓶颈已不在内核选择。
   - 背景：OpenBLAS 对小 K（48-192）矩阵效率不如 arm_gemm JIT；arm_gemm 针对具体矩阵
     大小自动生成最优 SVE 指令序列。
   - 使用：`-DUSE_ARM_GEMM=ON -DENABLE_SVE=ON -DARM_GEMM_ROOT=/path/to/ComputeLibrary-23.11/ComputeLibrary-23.11`
   - 构建/验证：`tools/build_arm_gemm.md`；内核选择机制参考 `docs/acl_reference/acl_wino_implementation_details.md`

2. **变换用 SVE intrinsics 优化**（预期 -0.5~1ms）
   - 当前 `transform_2d` 用泛型模板循环，ACL 用手写 SVE intrinsics 完全展开
   - 参考 `docs/acl_reference/acl_wino_sve_asm_annotated.md`：ACL 的 SVE 实现 361 行 vs 我们 ~100 行
   - 可以在 SVE 路径中针对 F(4,4,3,3) 写专用展开（类似 ACL 的 `sve_fp32_6x6.cpp`）

3. **合并 scatter/gather 到变换**（预期 -0.5ms）
   - 当前变换先写 U_tile，再 NEON 拷贝到 U。可改为变换直接写 U
   - 需要修改 `transform_2d` 支持非连续输出 stride

4. **权重变换手写公式**（预期 -0.5ms）
   - 当前用 `weight_transform_neon<6>` 模板循环，ACL 用直接展开 G 矩阵行
   - 参考 `docs/acl_reference/acl_wino_neon_intrinsics_annotated.md`

5. **启发式展平**（对 Case 0 有效，对 Case 4 无害）
   - 当 `NM * n_tiles * IC * 4 < 16MB` 时展平 N 个 batch（9→3 barriers）
   - 当内存大时保持 per-batch（避免 cache thrashing）

**对已超越 oneDNN 的 case（3-8, 16T NHWC）**：⚠️ 该结论仅限 16T NHWC 微基准口径。2026-08-20 NCHW 端到端实测（9T）多数 case 落后 oneDNN——**不再存在「已优无需优化」的 case**，全部优先执行 arm_gemm JIT（**已完成**，59/59 全胜，见「9 线程 NCHW 端到端实测」与 `docs/final_benchmark_bfd6b1e.md`）

## 扩展指南

### GEMM 内核切换

已有 3 种 GEMM 内核（编译期选择，互斥），实现见 `winograd_gemm` + `arm_gemm_driver`（`src/winograd_conv.cpp`）：

```cpp
#if defined(USE_ARM_GEMM)      // ACL 23.11 arm_gemm SVE JIT（与 oneDNN 相同）
    // 现代 API：GemmArgs → gemm() 工厂 → pretranspose_B_array → execute
    GemmConfig cfg; cfg.filter = "sve_";
    GemmArgs args(&CPUInfo::get(), M, N, K, ...);
    auto gemm = gemm<float,float>(args);
    gemm->pretranspose_B_array(pretrans, V, ldb, 0);
    gemm->set_pretransposed_B_data(pretrans);
    gemm->set_arrays(U, ...); ...
    gemm->execute(to_ndcoord(gemm->get_window_size()), {}, 0);
#elif defined(USE_OPENBLAS)     // OpenBLAS cblas_sgemm（单线程，与基线同构）
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ...);
#else                           // naive 三重循环
    for (...) ...
#endif
```

CMake 选项：
- `-DUSE_ARM_GEMM=ON -DENABLE_SVE=ON -DARM_GEMM_ROOT=/path/to/ComputeLibrary-23.11/ComputeLibrary-23.11`（优先级最高，与 oneDNN 相同内核）
- `-DUSE_OPENBLAS=ON`（当前使用，对小 K 矩阵不如 arm_gemm）
- 默认 naive（开发/验证用）

构建与 920F 验证步骤：`tools/build_arm_gemm.md`（含 WINO_GEMM_DEBUG 确认 SVE 内核名、--verify 正确性门、与 OpenBLAS 的 A/B 计时）。

### 下一步优化方向（按预期收益排序）

详见 `PERFORMANCE_ANALYSIS.md` 和 `docs/acl_reference/`。

> ⚠️ **最新重排**（2026-08-11）：在「内核不如 ACL」前提下，机会清单 Tier 0~3 见 `PERFORMANCE_ANALYSIS.md` 第 11 节。推荐路径：**arm_gemm JIT GEMM（现成）→ V 跨调用缓存（补亏空）→ tile 分块落 L2（920F 特性）→ Bᵀ 零系数跳过**。下方旧清单按原顺序保留。

**针对慢 case（0/1/2, 慢 1.79-2.56x）的优化**：

1. **arm_gemm 替换 OpenBLAS**（预期 Case 0/1/2 各 -1~2ms）
   - OpenBLAS 对小 K（48-192）矩阵效率不如 arm_gemm JIT
   - arm_gemm 针对具体矩阵大小自动生成最优 SVE 指令序列
   - 使用：`-DUSE_ARM_GEMM=ON -DARM_GEMM_ROOT=/path/to/gemm`
   - 参考：`docs/acl_reference/acl_wino_implementation_details.md` 中 arm_gemm 选择机制

2. **变换用 SVE intrinsics 优化**（预期 -0.5~1ms）
   - 当前 `transform_2d` 用泛型模板循环，ACL 用手写 SVE intrinsics 完全展开
   - 参考 `docs/acl_reference/acl_wino_sve_asm_annotated.md`：ACL 的 SVE 实现 361 行 vs 我们 ~100 行
   - 可以在 SVE 路径中针对 F(4,4,3,3) 写专用展开（类似 ACL 的 `sve_fp32_6x6.cpp`）

3. **合并 scatter/gather 到变换**（预期 -0.5ms）
   - 当前变换先写 U_tile，再 NEON 拷贝到 U。可改为变换直接写 U
   - 需要修改 `transform_2d` 支持非连续输出 stride

4. **权重变换手写公式**（预期 -0.5ms）
   - 当前用 `weight_transform_neon<6>` 模板循环，ACL 用直接展开 G 矩阵行
   - 参考 `docs/acl_reference/acl_wino_neon_intrinsics_annotated.md`

5. **启发式展平**（对 Case 0 有效，对 Case 4 无害）
   - 当 `NM * n_tiles * IC * 4 < 16MB` 时展平 N 个 batch（9→3 barriers）
   - 当内存大时保持 per-batch（避免 cache thrashing）

**对已超越 oneDNN 的 case（3-8, 16T NHWC, 快 43-59%）**：⚠️ 该结论仅限 16T NHWC 微基准口径。2026-08-20 NCHW 端到端实测（9T）多数 case 落后 oneDNN——不再有「已优无需优化」的 case，全部优先 arm_gemm JIT（**已完成**，59/59 全胜，见「9 线程 NCHW 端到端实测」与 `docs/final_benchmark_bfd6b1e.md`）

### 新增优化思路（基于 920F 硬件特性 + 性能数据）

以下是基于 920F 无 L3 cache、L2 仅 768KB/核的硬件特性，以及对 3 个慢 case 的分析，提出的新优化方向：

#### A. Tile 分块处理（cache 友好）

**问题**：920F 无 L3，L2 仅 768KB/核。当前一次性处理所有 tile，U（2.7-11MB）远超 L2，每个 tile 的输入变换和 scatter 都直接走主存。

**方案**：将 tile 分为 K 个一组（如 K=8），每组数据量 8×6×6×192×4=221KB 能放 L2。在组内完成 input→GEMM→output，减少主存访问。

```cpp
const int CHUNK = 8;
for (int chunk_start = 0; chunk_start < n_tiles; chunk_start += CHUNK) {
    int chunk_end = min(chunk_start + CHUNK, n_tiles);
    // 1. 输入变换：chunk_start..chunk_end
    // 2. GEMM：36 个 ts × (chunk_end-chunk_start) tiles
    // 3. 输出变换：chunk_start..chunk_end
}
```

**预期**：减少主存访问次数，改善 Case 2（1600 tiles, IC=48）的 cache 局部性。

#### B. GEMM 批量合并（减少调用次数）

**问题**：当前 36 次 `cblas_sgemm` 调用，每次 (n_tiles × IC × OC)。36 次调用有函数调用开销和 OpenMP 调度开销。

**方案**：将多个 ts_idx 的 GEMM 合并为一次大 GEMM。将 U[ts0..ts3][tile][ic] 重排为 U_batch[4*n_tiles][ic]，V[ts0..ts3][oc][ic] 重排为 V_batch[4*OC][ic]，一次 GEMM 得到 M_batch[4*n_tiles][4*OC]。但输出布局复杂，需要额外的重排。

更简单的方案：将 U 和 V 沿 K 维拼接（不改变 M/N），利用 cblas_sgemm 的 beta 参数做累加：

```cpp
// 不需要改布局，用 beta=1.0 累加多次 GEMM 结果到同一 M
for (int ts = 0; ts < 36; ts++) {
    cblas_sgemm(..., 1.0f, U_ts, V_ts, ts == 0 ? 0.0f : 1.0f, M, ...);
}
// 但这样不并行，不如当前的 #pragma omp for
```

此方案对大 IC case（GEMM 已经高效）帮助不大，但对小 IC case 可能减少调度开销。

#### C. SVE 替代 NEON 做 tile 提取/scatter/gather

> **✅ 已实施（2026-08-10）**：落地为 `copy_f32()`（见「性能优化记录」A3）。以下保留设计说明作为参考。

**问题**：当前 tile 提取、scatter、gather 用 NEON（128-bit, 4 float/指令）。920F 支持 SVE-512（16 float/指令）。

**方案**：在 SVE 编译路径下，将 NEON `vld1q_f32`/`vst1q_f32` 替换为 SVE `svld1_f32`/`svst1_f32`。tile 提取从 4 float/指令→16 float/指令，4× 加速。

```cpp
// 当前 NEON（4 float）
for (; ic + 4 <= IC; ic += 4)
    vst1q_f32(dp + ic, vld1q_f32(sp + ic));

// SVE 路径（16 float，谓词自动处理 tail）
if (isa >= ISALevel::SVE) {
    svbool_t pg = svwhilelt_b32(ic, IC);
    do {
        svst1_f32(pg, dp + ic, svld1_f32(pg, sp + ic));
        ic += svcntw();
        pg = svwhilelt_b32(ic, IC);
    } while (svptest_first(svptrue_b32(), pg));
}
```

**预期**：tile 提取从 1.10ms → ~0.28ms（Case 0），scatter/gather 也有类似加速。对 Case 2（1600 tiles）收益最大。（已实施，待 920F 复测确认实际收益）

#### D. 双缓冲（batch 间重叠）

**问题**：当前 batch 间串行：batch N 的输出完成后才开始 batch N+1 的输入。

**方案**：分配双份 U/M_buf，batch N 的输出变换与 batch N+1 的输入变换重叠：

```
batch 0: input → GEMM → output
batch 1:              input → GEMM → output  (与 batch 0 output 并行)
```

用 OpenMP task 实现：

```cpp
#pragma omp taskgroup
{
    #pragma omp task
    { process_batch(0, U_A, M_A); }
    #pragma omp task
    { process_batch(1, U_B, M_B); } // 与 batch 0 的 output 并行
}
```

**代价**：内存翻倍（2×U + 2×M），但对 Case 0（U=2.7MB）可接受。

#### E. 权重变换直接写 V（消除 V_oc 中间缓冲）

**问题**：当前权重变换先写 V_oc（per-OC），再拷贝到 V。多一次内存拷贝。

**方案**：修改 `weight_transform_neon` 支持输出 stride，直接写 V[m*OC*IC + oc*IC + ic]。但 V 的 stride 与 V_oc 不同，需要参数化 transform 函数。

#### F. 变换函数专用化（F(4,4,3,3) 硬编码）

**问题**：当前 `transform_1d_neon<6,6>` 是泛型模板，循环内有 `if (matrix[o][k] == 0.0f)` 分支判断。ACL 为 F(4,4,3,3) 写了专用代码，完全展开，无分支。

**方案**：为 F(4,4,3,3) 的输入变换（B^T 矩阵有大量零元素）和输出变换（A^T 矩阵有零元素）写专用函数，跳过零系数的行/列：

```cpp
// F(4,4) 专用输入变换，B^T[0] = [4,0,-5,0,1,0]
// 只需处理 k=0,2,4（跳过 k=1,3,5 的零系数）
void input_transform_f44_specialized(...) {
    // 列变换：6 列，每列只需 3 个非零行 × 3 个非零系数 = 9 FMA（vs 泛型 6×6=36）
}
```

**预期**：变换计算量减少 ~50%（F(4,4) 的 B^T 有 50% 零元素）。

### 运行方式

```bash
# NUMA 交织（让内存均匀分布在所有 NUMA 节点）
numactl --interleave=all ./bench_winograd --sve --nhwc --threads 32 shapes.csv

# 单 NUMA 绑定（38 核，减少跨 NUMA 访问）
numactl --cpunodebind=0 --membind=0 ./bench_winograd --sve --nhwc --threads 38 shapes.csv

# 线程绑定（防止线程迁移跨 NUMA）
OMP_PROC_BIND=spread OMP_PLACES=cores ./bench_winograd --sve --nhwc --threads 32 shapes.csv
```

### 添加新配置（如 F(6,6,3,3)）

1. 在 `winograd_matrices.hpp` 添加 `F66_G`、`F66_Bt`、`F66_A`
2. 在 `winograd_config.hpp` 添加 `WinogradConfig::F66_33()`
3. 在 `winograd_transforms.hpp` 添加 `weight/input/output_transform_f66_neon` 便捷包装
4. 在 `winograd_conv.cpp` dispatch 中添加 `is_f66` 分支

## 相关文档

- **算法详解**：`docs/algorithm.md` — 当前实现的分步讲解（Winograd 数学、三步变换、数据/缓冲布局、OpenMP 并行结构、NEON/SVE/SME 内核、GEMM 选择、tile 边界、数值精度）
- **快于 ACL 23.11 的分析**：`docs/why_faster_than_acl_23.11.md` — 独立分析：双方都是 Winograd+SVE，8/9 赢是固定开销主导的小负载微基准结果（含有效 GFLOPS 证据、3 个具体例子、6 个原因、验证清单）
- **终局基准（arm_gemm vs OpenBLAS）**：`docs/final_benchmark_bfd6b1e.md` — 59 形状逐行表 + 散写回归验证 + E1 debug 计时陷阱
- **oneDNN 对照**：`docs/onednn_comparison.md` — e2e/benchdnn 工具链、sbatch 运行协议、**e2e OOM 真根因（dilates 误传）与判读矩阵**、数据现状
- **排障经验**：`docs/debugging_lessons.md` — 本次 OOM 排障的**弯路与方法论**（错修如何被证伪、红鲱鱼识别、源码≠设备、杀手实验）
- **用例集与对比指南**：`shapes/README.md` — `conv_all.csv`（59 行）各节测哪个假设、CSV↔benchdnn 描述符规则、profiling 对照计数器
- **ACL 参考文档**：`docs/acl_reference/` — 从 oneDNN 源码树复制的 ACL Winograd 实现分析文档（8 个文件）。用于指导后续优化（SVE/SME 汇编变换、arm_gemm GEMM 内核、权重变换手写公式等）
- **性能分析**：`PERFORMANCE_ANALYSIS.md` — 完整优化历程（9 阶段）、9 case 对比数据、细粒度计时、差距分析、新优化思路（A-F）、数值精度分析（第 9 节）
- **arm_gemm 构建指南**：`tools/build_arm_gemm.md` — 920F 上把 Winograd GEMM 切到 arm_gemm fp32-SVE 的完整步骤（cmake/make/内核名确认/正确性门/与 OpenBLAS A/B）
