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
#   3) 原 --mode=p 出 perf 单次行，但集群 benchdnn build 始终无 perf 输出
#      （可能不支持 perf 模式）。改用 ONEDNN_VERBOSE=exec 让 oneDNN 库直接
#      打印 per-execution 计时行（onednn_verbose,...,exec,...,<ms>），merge
#      的 exec 行解析兜底。PASSED (N ms) 是含 fill/ref/compare 的聚合时间，不用。
# CRLF 自愈：SFTP 从 Windows 传文件可能带 \r\n，set -e 下 cd 路径含 \r 会静默失败。
sed -i 's/\r$//' "$0" 2>/dev/null
if grep -q $'\r' "$0"; then exec bash "$0"; fi
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ALG=()  # 空 = auto；非空 = --alg=WINO
BIN="${BENCHDNN:-}"
THREADS=16
while [ $# -gt 0 ]; do
    case "$1" in
        --winograd) ALG=(--alg=WINO); shift ;;
        --bin) BIN="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- 输出文件：先确定再清空，保证任何失败路径都不留旧文件 ----
# 若 gen_benchdnn_list / BIN 探测 / numactl 设置任一步失败（set -e 提前退出），
# 旧 OUT 残留会被 ab_onednn.sh 的 [ -f "$BD1" ] 照读 → merge [NO-SRC]。
# 提前清空确保失败时留空文件（merge 走 [NO-SRC]，不读过期数据）。
if [ ${#ALG[@]} -eq 0 ]; then
    OUT=build/benchdnn_auto.txt
    echo "[benchdnn] algo = auto (oneDNN default)"
else
    OUT=build/benchdnn_wino.txt
    echo "[benchdnn] algo = ${ALG[*]} (oneDNN Winograd via ACL)"
fi
: > "$OUT"

# ---- 定位 benchdnn：已知好路径优先，再 find 探测兜底 ----
# 旧版 find 循环 cand 顺序为 ONEDNN_ROOT /usr/local /opt /workspace/... /workspace，
# 若 /usr/local 或 /opt 先命中更旧的 benchdnn（不认 --mode=p）→ corr 模式 → 无 perf 行。
# 已知好路径 /workspace/z00889957/000Libs/oneDNN-3.12.1/build/tests/benchdnn/benchdnn
# （相对 000Libs 深度 5，-maxdepth 5 可命中）优先检查，避免 find 选错。
KNOWN_GOOD="/workspace/z00889957/000Libs/oneDNN-3.12.1/build/tests/benchdnn/benchdnn"
if [ -z "$BIN" ] && [ -x "$KNOWN_GOOD" ]; then
    BIN="$KNOWN_GOOD"
fi
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

# ---- 刷新描述符清单（非致命：list 已在仓库，刷新失败不阻断 benchdnn 运行）----
bash tools/gen_benchdnn_list.sh >/dev/null || true

# ---- 线程绑定：numactl 优先；不可用则退 OpenMP 环境变量 ----
NUMACTL_ARGS="${BENCHDNN_NUMACTL:--C 0-$((THREADS-1))}"
if command -v numactl >/dev/null 2>&1; then
    echo "[benchdnn] numactl: $NUMACTL_ARGS"
    RUNNER=(numactl $NUMACTL_ARGS)
else
    echo "[benchdnn] numactl unavailable -> fallback OMP_PROC_BIND/OMP_NUM_THREADS"
    RUNNER=()
fi

# ONEDNN_VERBOSE=exec 让 oneDNN 库打印 per-execution 计时行（merge 解析 exec 行兜底）。
# 不用 --mode=p：该 benchdnn build 多次实测均无 perf 行（可能不支持 perf 模式）。
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$THREADS \
    ONEDNN_VERBOSE=exec \
    "${RUNNER[@]}" "$BIN" --conv --reset "${ALG[@]}" \
    --batch=shapes/conv_all.list >"$OUT" 2>&1 \
    || { echo "[benchdnn] exit $?; raw output in $OUT" >&2; tail -20 "$OUT" >&2; exit 1; }

echo "[benchdnn] done -> $OUT"
N_PASS=$(grep -c 'r[0-9]*"' "$OUT" 2>/dev/null || echo 0)
N_PERF=$(grep -c '^perf,' "$OUT" 2>/dev/null || echo 0)
N_EXEC=$(grep -cE '(onednn|dnnl)_verbose,.*exec' "$OUT" 2>/dev/null || echo 0)
echo "[benchdnn] PASSED lines: $N_PASS / perf single-exec lines: $N_PERF / onednn exec lines: $N_EXEC"
if [ "$N_PERF" -eq 0 ] && [ "$N_EXEC" -eq 0 ] && [ "$N_PASS" -gt 0 ]; then
    echo "!! WARNING: 无 perf/exec 行——merge 将回退到 PASSED 聚合时间（含 fill/ref/compare，约 2x 执行时间）。"
    echo "   已设 ONEDNN_VERBOSE=exec；若仍无 exec 行，benchdnn 可能压制了库级 verbose。"
fi
# 把自检写进 OUT 末尾（# 注释行，merge 忽略）——下一轮 SUMMARY 直接可见
{
    echo
    echo "# run_benchdnn self-check: binary=$BIN algo=${ALG[*]:-auto} threads=$THREADS numactl=${NUMACTL_ARGS:-none}"
    echo "# PASSED=$N_PASS perf_lines=$N_PERF exec_lines=$N_EXEC"
} >> "$OUT"
