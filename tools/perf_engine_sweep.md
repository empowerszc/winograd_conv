# arm_gemm 引擎轮换基准（选核 vs 固定开销定位）

> 注（2026-08-27）：第二步「K 主序 V + nmulti 批量调用」已合入 master，
> 每 ts_idx 的固定开销已被摊薄；本脚本仍有效——它回答的是**选核**问题，
> 且与历史 A/B 数字比较时时间基数会整体下移。

2026-08-27 全量 A/B（16 线程，conv_all.csv，`tools/compare.sh`）结论：arm_gemm
后端在 IC≤48 时微赢 1.1x，中高 IC 全面输 OpenBLAS 1.2~2.6x，损失幅度随
IC 单调增长。当时两个主嫌疑：

1. **选核失效**：驱动 CPUInfo shim 报 A76、无 perf 参数表 → 周期估计全部溢出
   （内核表里大量 `est=18446744073709551615`），`find_implementation()` 实际按
   注册顺序碰运气。interleaved 内核对大 M 还要先把 A 整体交错一遍，选错时双重伤。
2. **每调用固定开销**：Phase 2 对 36 个 ts_idx 各建一次 GEMM 对象 + B 打包
   （+修复布局后的 Bᵀ 暂存拷贝），36 次叠加；K=IC 越大这层越贵。

## 操作

仓库根目录：

```bash
cd winograd_conv && git pull
sbatch --wrap="bash tools/sweep_focus.sh"
```

可选覆盖：`sbatch --wrap="THREADS=38 REPEATS=20 bash tools/sweep_focus.sh"`。
脚本对 6 个重损形状 × 4 种内核选择各跑一遍（三个 SVE 家族 + 不过滤自动选）。

## 判读

| 观察 | 结论 |
|---|---|
| 某个具体引擎显著快于 `filter='' (auto)` 的默认选择 | 选核问题 ⇒ 下一步给驱动加 `cfg.method` 静态直选/修 shim 的 perf 表 |
| 四个引擎时间接近、都明显慢于同形状的 OpenBLAS 数字 | 内核执行层本身追不上 OB（批量改造后仍慢即如此）——剩余差距来自内核微架构适配，考虑 Step-3 用性能表修选核 + 评估 interleaved 家族是否值得留 |
| interleaved_8x3VL 在大 M 形状格外慢 | 它每次调用先整体交错 A —— 也是「别选它」的证据 |

对照基线：先跑一次 sweep（生成 `focus_sweep.csv`），再用 OpenBLAS 构建
跑同一形状表（`WINO_GEMM_FILTER` 对 OB 构建无效果，可不管）：

```bash
sbatch --wrap="./build_oblas/bench_winograd --sve --nhwc --threads 16 \
  --warmup 3 --repeats 10 focus_sweep.csv"
```

把脚本完整 stdout 贴回即可分析。

## 首轮结果（2026-08-27，node03，16 线程）

> **⚠️ 本节跨后端结论已作废（2026-08-28）**：arm 与 OB 的数字来自**不同 sbatch
> 作业**，而 node03 实测存在 3~7x 的跨作业性能态（同一 OB 二进制同命令同顺序：
> 大形状 47.3ms → 241.3ms，见 diag_ab v1 的 E4）。本节 arm 取自正常态作业、
> OB 取自慢态作业，**「5/6 反超 OB」不成立**。可信保留的只有**同作业内**的
> 4 个引擎强制轮换的**相对排序**（6x4VL 大 M 最优 / 8x1VL 仅小 M / interleaved
> 垫底）。另一疑点：当时跑的很可能是改造前（c48761e）的二进制——1,2048 auto
> 10.05ms 与旧基线 11.6ms 持平，而全量新码同形状测到 49ms——本节连引擎排序
> 是否代表新码都待确认。新基线以 diag_ab.sh v2 的同作业 A/B 为准。

当时（误）判读为：两种预设嫌疑均不成立——
auto 选核只轻微次优（距最优 3~25%），且 arm_gemm 在 6 形状中 5 个反超 OpenBLAS：

| 形状 | 最优引擎 (ms) | auto (ms) | OB (ms) | vs OB |
|---|---|---|---|---|
| 4,384,160²,384 | 72.6 (6x4VL) | 76.1 | 215.9 | 快 2.97x |
| 4,384,40²,384 | 9.82 (6x4VL) | 10.14 | 18.32 | 快 1.87x |
| 4,768,20²,96 | 3.89 (8x1VL) | 4.39 | 4.50 | 快 1.15x |
| 1,2048,7²,512 | 10.05 (auto) | 10.05 | 6.02 | **慢 1.67x** |
| 4,512,28²,512 | 12.78 (6x4VL) | 12.83 | 18.18 | 快 1.42x |
| 4,96,20²,96 | 0.40 (6x4VL) | 0.52 | 2.98 | 快 7.5x |

- 6x4VL：大 M（n_tiles 大）形状最优；8x1VL：仅小 M 占优；
  **interleaved_8x3VL 全程垫底，任何形状都不应选中**。
- row 0 的 6x4VL 折算 GEMM ≈936 GFLOPS/16 核 ≈ SVE-512 峰值 ~70%。
- 唯一失守 row 3（MB=1、n_tiles=4）：M=4 瘦条 + 36 个 B 面板 ~151MB 一次性流读 +
  每线程仅 2~3 片，批量摊薄最弱；auto 已是最优（钉核更差）⇒ 结构性问题，
  绝对差 4ms，暂接受或后续给小 n_tiles 换后端。
