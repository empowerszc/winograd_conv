#!/usr/bin/env bash
# 内核 filter 实验：零代码改动，对同一组 shape 在多个 WINO_GEMM_FILTER 下计时，
# 判定 auto 选核在 M=25（20² 族）上是否次优。
#
# 动机（docs/final_benchmark_bfd6b1e.md 遗留项 2）：20² 族（n_tiles=25, M=25）
# 的 auto 选核选中 sve_interleaved_fp32_mla_8x3VL，而旧 sweep 曾观测
# 「interleaved 全程垫底」（但那是改造前二进制 + 跨作业，需同作业复测）。
#
# 用法：
#   bash tools/filter_sweep.sh [shapes.csv]      # 默认内置 9 形状
# 输出：build/filter_sweep_{auto,6x4VL,8x1VL,inter}.csv（各自 ours_ms 表）
#       并打印合并对比表（每行最优标 *）。
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

CSV="${1:-/tmp/filter_focus.csv}"
if [ ! -f "$CSV" ]; then
    CSV=/tmp/filter_focus.csv
    cat > "$CSV" <<'EOF'
# mb,ic,ih,iw,oc,kh,kw,sh,sw,ph,pw,dh,dw,grp,count   （focus：20² 族 M=25 + 对照）
4,48,20,20,48,3,3,1,1,1,1,0,0,1,50
4,96,20,20,96,3,3,1,1,1,1,0,0,1,50
4,192,20,20,192,3,3,1,1,1,1,0,0,1,50
4,384,20,20,384,3,3,1,1,1,1,0,0,1,50
4,768,20,20,96,3,3,1,1,1,1,0,0,1,10
4,96,80,80,96,3,3,1,1,1,1,0,0,1,20
4,384,160,160,384,3,3,1,1,1,1,0,0,1,5
1,512,7,7,512,3,3,1,1,1,1,0,0,1,100
1,2048,7,7,512,3,3,1,1,1,1,0,0,1,100
EOF
    echo "[filter] 生成默认 focus CSV: $CSV"
fi

declare -A TAG=( [auto]="auto" [6x4VL]="sve_hybrid_fp32_mla_6x4VL" \
                 [8x1VL]="sve_hybrid_fp32_mla_8x1VL" [inter]="sve_interleaved_fp32_mla_8x3VL" )
mkdir -p build

for tag in auto 6x4VL 8x1VL inter; do
    f="build/filter_sweep_${tag}.csv"
    echo "===== [filter] ${tag} (${TAG[$tag]}) ====="
    if [ "$tag" = auto ]; then
        unset WINO_GEMM_FILTER
    else
        export WINO_GEMM_FILTER="${TAG[$tag]}"
    fi
    # compare.sh 的 stdout 即 mb,ic,ih,iw,oc,ours_ms 表（首行 # run:）
    OMP_PROC_BIND=close OMP_PLACES=cores bash tools/compare.sh --threads 16 --isa sve "$CSV" \
        > "$f"
    cat "$f"
done

echo
echo "===== [filter] 合并对比（* = 该行最优）====="
awk -F, '
    /^# run:/ || $1 ~ /^#/ || $1 == "mb" { next }
    FILENAME ~ /filter_sweep_auto/  { a[$1","$2","$3","$4","$5] = $6; order[++n] = $1","$2","$3","$4","$5; next }
    FILENAME ~ /filter_sweep_6x4VL/ { b[$1","$2","$3","$4","$5] = $6; next }
    FILENAME ~ /filter_sweep_8x1VL/ { c[$1","$2","$3","$4","$5] = $6; next }
    FILENAME ~ /filter_sweep_inter/ { d[$1","$2","$3","$4","$5] = $6; next }
    END {
        printf "%-18s %10s %10s %10s %10s   %s\n", "shape", "auto", "6x4VL", "8x1VL", "inter", ""
        for (i = 1; i <= n; i++) {
            k = order[i]; best = 1e30; bt = ""
            t = a[k]; if (t+0 < best) { best = t+0; bt = "auto" }
            t = b[k]; if (t+0 < best) { best = t+0; bt = "6x4VL" }
            t = c[k]; if (t+0 < best) { best = t+0; bt = "8x1VL" }
            t = d[k]; if (t+0 < best) { best = t+0; bt = "inter" }
            star = (bt == "auto") ? "" : "  <-- best=" bt
            printf "%-18s %10s %10s %10s %10s   %s\n", k, a[k], b[k], c[k], d[k], star
        }
    }
' build/filter_sweep_auto.csv build/filter_sweep_6x4VL.csv \
  build/filter_sweep_8x1VL.csv build/filter_sweep_inter.csv
