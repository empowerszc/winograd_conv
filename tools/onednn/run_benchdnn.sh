#!/usr/bin/env bash
# oneDNN benchdnn conv 基准：对 shapes/conv_all.list（与 conv_all.csv 同 shape）
# 跑 --conv（fp32 即默认，3.12.1 已不认 --cfg），绑核 16 线程。默认 auto 算法
# （oneDNN 自选，端到端口径），可选 --winograd 强制 oneDNN Winograd 路径
# （AArch64 原生 build 无 ACL 时可能全部 unimplemented，见 ab_onednn.sh 说明）。
#
# 用法：
#   bash tools/onednn/run_benchdnn.sh [--bin BENCHDNN] [--winograd] [--threads N]
# 输出：build/benchdnn_auto.txt / build/benchdnn_wino.txt（原始 stdout）
#
# ⚠️ 2026-08-29 修复：必须设 OMP_NUM_THREADS=$THREADS。此前不设时（sbatch --exclusive
#   下）oneDNN 默认取全部 608 个 CPU → 每个 primitive 起 608 线程 → 小形状慢
#   100~900x，与 ours/e2e（16 线程）完全不可比（曾误判为「benchdnn 近单线程」）。
#   另设 ONEDNN_VERBOSE=exec：每执行落一条 onednn_verbose,...exec,... 行，末字段 =
#   单次执行 ms；merge_onednn.sh 用它而非 PASSED 的 (N ms)（perf 循环聚合，不可当单次）。
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
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
echo "[benchdnn] threads=$THREADS (OMP_NUM_THREADS)"

# ---- 刷新描述符清单 ----
bash tools/gen_benchdnn_list.sh >/dev/null

if [ ${#ALG[@]} -eq 0 ]; then
    OUT=build/benchdnn_auto.txt
    echo "[benchdnn] algo = auto (oneDNN default)"
else
    OUT=build/benchdnn_wino.txt
    echo "[benchdnn] algo = ${ALG[*]} (oneDNN Winograd)"
fi

# 不带 --cfg=f32：oneDNN 3.12.1 的 conv driver 已不认 --cfg（默认 dt=f32 即 fp32）。
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$THREADS ONEDNN_VERBOSE=exec \
    "$BIN" --conv --reset "${ALG[@]}" --batch=shapes/conv_all.list >"$OUT" 2>&1 \
    || { echo "[benchdnn] exit $?; raw output in $OUT" >&2; tail -20 "$OUT" >&2; exit 1; }

echo "[benchdnn] done -> $OUT"
echo "[benchdnn] lines with rN label: $(grep -c 'r[0-9]*"' "$OUT" 2>/dev/null || echo 0)"
echo "[benchdnn] exec-time lines (onednn_verbose,primitive,exec): $(grep -c ',primitive,exec,' "$OUT" 2>/dev/null || echo 0)"
