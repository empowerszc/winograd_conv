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
