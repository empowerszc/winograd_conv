#!/usr/bin/env bash
# oneDNN benchdnn conv 基准：对 shapes/conv_all.list（与 conv_all.csv 同 shape）
# 跑 --conv（fp32 即默认，3.12.1 已不认 --cfg）。默认 auto 算法（oneDNN 自选，
# 端到端口径），可选 --winograd 强制 --alg=WINO（ACL Winograd，本机有 ACL 的
# build 可用；不支持的 shape 会 SKIP）。
#
# 用法：
#   bash tools/onednn/run_benchdnn.sh [--bin BENCHDNN] [--winograd] [--threads N]
#   BENCHDNN_NUMACTL 可整体覆盖绑核参数（默认 "-C 0-$(N-1)"）。
# 输出：build/benchdnn_auto.txt / build/benchdnn_wino.txt（原始 stdout）
#
# ⚠️ 2026-08-29 三次修复（按时间）：
#   1) 必须设 OMP_NUM_THREADS=$THREADS：此前不设时 sbatch --exclusive 下 oneDNN
#      默认取全部 608 CPU → 小形状慢 100~900x。
#   2) 但用户实测 OMP_NUM_THREADS 对带 TBB 的 ACL build 无效（数值与修复前逐位
#      相同）——真正有效的是 numactl -C 绑核（CPU 亲和对 OpenMP/TBB 都生效）。
#      故改用 numactl -C 0-$(THREADS-1) 优先，OMP_NUM_THREADS 仅兜底。
#   3) 必须 --mode=p：默认是 corr 模式，不打印 perf, 行，PASSED (N ms) 是含
#      fill/ref/compare 的聚合时间（不可比）。--mode=p 打印
#      `perf,<engine>,<impl>,<name>,<prb>,<Gops>,<ctime>,<min-ms>,...`，
#      %-time% = 逐次 start/stamp 的最小单次执行 ms（与 ours/e2e 同口径）。
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ALG=()  # 空 = auto；非空 = --alg=WINO
BIN="${BENCHDNN:-}"
THREADS=16
while [ $# -gt 0 ]; do
    case "$1" in
        --winograd) ALG=(--alg=WINO); ;;
        --bin) BIN="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

# ---- 定位 benchdnn ----
if [ -z "$BIN" ]; then
    for cand in "${ONEDNN_ROOT:-}" /usr/local /opt /workspace/z00889957/000Libs /workspace; do
        hit=$(find "$cand" -maxdepth 5 -type f -name benchdnn 2>/dev/null | head -1 || true)
        if [ -n "$hit" ]; then BIN="$hit"; break; fi
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "error: benchdnn not found. Set BENCHDNN=<path> or use --bin." >&2
    exit 1
fi
echo "[benchdnn] binary: $BIN"
echo "[benchdnn] threads=$THREADS (numactl -C 0-$(($THREADS-1)))"

# ---- 刷新描述符清单 ----
bash tools/gen_benchdnn_list.sh >/dev/null

if [ ${#ALG[@]} -eq 0 ]; then
    OUT=build/benchdnn_auto.txt
    echo "[benchdnn] algo = auto (oneDNN default)"
else
    OUT=build/benchdnn_wino.txt
    echo "[benchdnn] algo = ${ALG[*]} (oneDNN Winograd via ACL)"
fi

# ---- 线程绑定：numactl 优先；不可用则退 OpenMP 环境变量 ----
NUMACTL_ARGS="${BENCHDNN_NUMACTL:--C 0-$((THREADS-1))}"
if command -v numactl >/dev/null 2>&1; then
    echo "[benchdnn] numactl: $NUMACTL_ARGS"
    RUNNER=(numactl $NUMACTL_ARGS)
else
    echo "[benchdnn] numactl unavailable -> fallback OMP_PROC_BIND/OMP_NUM_THREADS"
    RUNNER=()
fi

# --mode=p 出 perf 单次行；不带 --cfg=f32（3.12.1 已移除）。
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$THREADS \
    "${RUNNER[@]}" "$BIN" --conv --mode=p --reset "${ALG[@]}" \
    --batch=shapes/conv_all.list >"$OUT" 2>&1 \
    || { echo "[benchdnn] exit $?; raw output in $OUT" >&2; tail -20 "$OUT" >&2; exit 1; }

echo "[benchdnn] done -> $OUT"
N_PASS=$(grep -c 'r[0-9]*"' "$OUT" 2>/dev/null || echo 0)
N_PERF=$(grep -c '^perf,' "$OUT" 2>/dev/null || echo 0)
N_EXEC=$(grep -c ',primitive,exec,' "$OUT" 2>/dev/null || echo 0)
echo "[benchdnn] PASSED lines: $N_PASS / perf single-exec lines: $N_PERF / onednn exec lines: $N_EXEC"
if [ "$N_PERF" -eq 0 ] && [ "$N_EXEC" -eq 0 ] && [ "$N_PASS" -gt 0 ]; then
    echo "!! WARNING: 无 perf/exec 行——merge 将整列置 N/A。"
    echo "   已加 --mode=p；若仍无 perf, 行，贴 head -20 $OUT 回来。"
fi
