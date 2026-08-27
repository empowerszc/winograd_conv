#!/usr/bin/env bash
# diag_ab.sh - 临时诊断脚本（E1~E4）：裁决全量 A/B 与 focus sweep 的矛盾。用完即删。
#
# 背景（2026-08-27）：
#   * 全量 conv_all 与 focus sweep 数字矛盾（同形状差 1.3~4.9x）
#   * 同一进程内重复形状两次测量漂移 1.8~2.3x（bench 已是 best-of-20 也救不回）
#   * OB 同一二进制两套数差 4.6x（47.3 全量末位 vs 215.9 sweep 首形状）
#   * 位置16 的 1,512,7,7,512 疑似真代码回归（同位旧代码 3.05 vs 新 14.6）
#
# 用法（build_oblas 保持旧二进制勿重编，保持基线可比）：
#   sbatch -w node03 --exclusive --wrap="bash tools/diag_ab.sh"
#
# 判读：
#   E1 快(≈旧代码)          -> 纯环境问题，E3 绑核应同步好转
#   E1 仍慢 4~5x            -> M=4 瘦条类真代码回归，需出补丁
#   E4 出 ~47ms 一档        -> sweep 的 215.9 是「新鲜进程首形状」异常，
#                              此前「5/6 反超 OB」的结论按 E3/E4 重判

set +e
cd "$(dirname "$0")/.."
T=16
echo "#### diag_ab: node=$(hostname) date=$(date) ####"

echo
echo "================ E1: 回归形状全新进程内在速度 + auto 选核可视化 ================"
printf '%s\n' \
  '4,384,160,160,384,3,3,1,1,1,1,0,0,1,5' \
  '1,512,7,7,512,3,3,1,1,1,1,0,0,1,5' \
  '1,2048,7,7,512,3,3,1,1,1,1,0,0,1,5' \
  '4,768,20,20,96,3,3,1,1,1,1,0,0,1,5' \
  '4,96,20,20,96,3,3,1,1,1,1,0,0,1,5' > /tmp/mini.csv
WINO_GEMM_DEBUG=1 ./build/bench_winograd --sve --nhwc --threads $T \
  --warmup 3 --repeats 20 /tmp/mini.csv 2>&1

echo
echo "================ E2: 最大形状放最前先跑（复现『跑过大形状后变慢』） ================"
printf '%s\n' \
  '4,384,160,160,384,3,3,1,1,1,1,0,0,1,5' \
  '1,512,7,7,512,3,3,1,1,1,1,0,0,1,5' \
  '1,2048,7,7,512,3,3,1,1,1,1,0,0,1,5' > /tmp/mini2.csv
./build/bench_winograd --sve --nhwc --threads $T --warmup 3 --repeats 20 /tmp/mini2.csv

echo
echo "================ E3: 绑核后全量 59 形状（判 NUMA/线程漂移） ================"
OMP_PROC_BIND=close OMP_PLACES=cores ./tools/compare.sh shapes/conv_all.csv

echo
echo "================ E4: OB 基线重测（裁决 47.3 vs 215.9；勿重编 build_oblas） ================"
./build_oblas/bench_winograd --sve --nhwc --threads $T --warmup 10 --repeats 50 shapes/conv_all.csv

echo
echo "================ 完成：把本作业的全部输出贴回 ================"
