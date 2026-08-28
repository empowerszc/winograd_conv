#!/usr/bin/env bash
# oneDNN benchdnn conv 基准：对 shapes/conv_all.list（与 conv_all.csv 同 shape）
# 跑 --conv（fp32 即默认，3.12.1 已不认 --cfg），绑核 16 线程。默认 auto 算法
# （oneDNN 自选，端到端口径），可选 --winograd 强制 oneDNN Winograd 路径
# （AArch64 原生 build 无 ACL 时可能全部 unimplemented，见 ab_onednn.sh 说明）。
#
# 用法：
#   bash tools/onednn/run_benchdnn.sh [--bin BENCHDNN] [--winograd]
# 输出：build/benchdnn_auto.txt / build/benchdnn_wino.txt（原始 stdout）
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ALG=()  # 空 = auto；非空 = --alg=WINO
BIN="${BENCHDNN:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --winograd) ALG=(--alg=WINO); ;;
        --bin) BIN="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
    echo "error: benchdnn 未找到。设 BENCHDNN=<路径> 或 --bin。" >&2
    exit 1
fi
echo "[benchdnn] binary: $BIN"

# ---- 刷新描述符清单 ----
bash tools/gen_benchdnn_list.sh >/dev/null

if [ ${#ALG[@]} -eq 0 ]; then
    OUT=build/benchdnn_auto.txt
    echo "[benchdnn] 算法 = auto（oneDNN 自选）"
else
    OUT=build/benchdnn_wino.txt
    echo "[benchdnn] 算法 = ${ALG[*]}（oneDNN Winograd）"
fi

# 不带 --cfg=f32：oneDNN 3.12.1 的 conv driver 已不认 --cfg（默认 dt=f32 即 fp32）。
OMP_PROC_BIND=close OMP_PLACES=cores \
    "$BIN" --conv --reset "${ALG[@]}" --batch=shapes/conv_all.list >"$OUT" 2>&1 \
    || { echo "[benchdnn] 退出码 $?；原始输出见 $OUT" >&2; tail -20 "$OUT" >&2; exit 1; }

echo "[benchdnn] done -> $OUT"
grep -c 'r[0-9]*"' "$OUT" | xargs echo "[benchdnn] 带 rN 标签的行数:"
