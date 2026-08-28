#!/usr/bin/env bash
# sweep_focus.sh - arm_gemm 引擎强制轮换基准（920F 排查「选错核 vs 固定开销」用）
#
# 背景：2026-08 全量 A/B 显示 arm_gemm 在中高 IC 形状输 OpenBLAS 1.6~2.6x，
# 且怀疑两点：① find_implementation 的周期估计全废（CPUInfo shim 无 perf 表，
# est 溢出），选核基本看注册顺序；② 每 ts_idx 一次 GEMM 调用的固定开销。
# 本脚本把 A/B 中损失最重的 6 个形状用 4 种内核选择各跑一遍：
#   若某个具体引擎明显快于默认选择 ⇒ 选核问题；
#   若四家都追不上 OpenBLAS ⇒ 固定开销/布局问题为主。
#
# 用法（在提交作业的目录，即 winograd_conv 仓库根目录）：
#   cd winograd_conv && git pull
#   sbatch --wrap="bash tools/sweep_focus.sh"
# 可选环境变量：THREADS(默认16) WARMUP(默认3) REPEATS(默认10)
#
# 输出：每引擎一段 "== filter=... ==" 后跟每个形状的一行计时表。

set -euo pipefail
cd "$(dirname "$0")/.."    # 仓库根

BIN=${BIN:-build/bench_winograd}
THREADS=${THREADS:-16}
WARMUP=${WARMUP:-3}
REPEATS=${REPEATS:-10}

[ -x "$BIN" ] || { echo "error: $BIN missing; build it first" >&2; exit 1; }

# A/B 里 arm_gemm 损失最重 + 1 个微输对照的形状（count 仅示意，时间以 repeats 计）
cat > focus_sweep.csv <<'EOF'
mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count
4,384,160,160,384,3,3,1,1,1,1,0,0,1,5
4,384,40,40,384,3,3,1,1,1,1,0,0,1,5
4,768,20,20,96,3,3,1,1,1,1,0,0,1,5
1,2048,7,7,512,3,3,1,1,1,1,0,0,1,5
4,512,28,28,512,3,3,1,1,1,1,0,0,1,5
4,96,20,20,96,3,3,1,1,1,1,0,0,1,5
EOF

run_one() {
    local label="$1"; shift
    echo "============================================================"
    echo "== $label"
    echo "============================================================"
    # WINO_GEMM_FILTER=""（空串存在但为空）= 不过滤，让 arm_gemm 自由选
    env ${FILTER+WINO_GEMM_FILTER="$FILTER"} \
        "$BIN" --sve --nhwc --threads "$THREADS" --warmup "$WARMUP" \
               --repeats "$REPEATS" focus_sweep.csv
}

# 三个 SVE 内核家族 + 自由选择（含 NEON 候选）
for eng in sve_hybrid_fp32_mla_6x4VL \
           sve_interleaved_fp32_mla_8x3VL \
           sve_hybrid_fp32_mla_8x1VL \
           ""; do
    if [ -z "$eng" ]; then
        unset FILTER || true
        run_one "filter='' (auto)"
    else
        FILTER="$eng"
        export FILTER
        run_one "filter='$eng'"
        unset FILTER || true
    fi
done

echo
echo "interpretation guide in tools/perf_engine_sweep.md; paste full output back."
