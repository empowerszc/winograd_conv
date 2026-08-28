#!/usr/bin/env bash
# 合并三列到一张对照表（按 ours 文件行序）：
#   mb,ic,ih,iw,oc, ours_ms, onednn_e2e_ms, benchdnn_ms, onednn/ours, benchdnn/ours
# 输入：
#   1) 我们侧 CSV（tools/compare.sh 输出：首行 # run:，随后 mb,ic,ih,iw,oc,ours_ms）
#   2) oneDNN e2e CSV（run_onednn_e2e.sh 输出：mb,ic,ih,iw,oc,onednn_ms）
#   3) benchdnn 原始 stdout（run_benchdnn.sh 输出，每行含 _n"rN" 标签）
#   4) [可选] 源 shapes CSV（默认 shapes/conv_all.csv；benchdnn 的 rN = 该文件数据行号）
# 用法：
#   bash tools/onednn/merge_onednn.sh ours.csv onednn_e2e.csv benchdnn_raw.txt
# benchdnn 解析尽力而为（格式随版本变化）；解析失败列显示 N/A，
# 把 benchdnn 原始输出贴回会话即可人工合并。
set -euo pipefail
OURS="${1:?ours.csv}"; E2E="${2:?onednn_e2e.csv}"; BD="${3:?benchdnn_raw.txt}"
CSV="${4:-shapes/conv_all.csv}"

# benchdnn：逐行抓 rN 标签 + ms（容忍格式变化），预解析为 "row ms"
grep -E '_n"r[0-9]+' "$BD" | awk '{
    if (match($0, /r[0-9]+"/)) n = substr($0, RSTART+1, RLENGTH-2); else next
    ms = "N/A"
    if      (match($0, /[0-9]+\.[0-9]+ *ms/))  ms = substr($0, RSTART, RLENGTH)
    else if (match($0, /time: *[0-9]+\.[0-9]+/)) ms = substr($0, RSTART+6, RLENGTH-6)
    else if (match($0, /perf: *[0-9]+\.[0-9]+/)) ms = substr($0, RSTART+6, RLENGTH-6)
    gsub(/ms/, "", ms); gsub(/ /, "", ms)
    print n, ms
}' | sort -k1,1n > /tmp/merge_bd.txt

awk -F, -v bd="/tmp/merge_bd.txt" -v csv="$CSV" -v ours="$OURS" -v e2e="$E2E" '
    BEGIN { row = 0; n = 0 }   # 未初始化变量是空串，与数字 0 是不同的数组键！
    FILENAME == bd { split($0, t, " "); bdms[t[1]] = t[2]; next }
    FILENAME == csv {                       # 源 shape：rN = 数据行号（0 基）
        # 跳过 注释/空行/表头（表头可能是 # 注释形式，也可能是裸 mb, 行）
        if ($0 ~ /^#/ || $0 ~ /^mb,/ || FNR == 1 || NF == 0) next
        row2key[row] = $1","$2","$3","$4","$5; row++
        next
    }
    $0 ~ /^#/ || $0 ~ /^mb,/ { next }       # 我们的表头：第 1 行 # run:，第 2 行 mb,...
    FILENAME == ours { key = $1","$2","$3","$4","$5; om[key] = $6; order[++n] = key; next }
    { key = $1","$2","$3","$4","$5; em[key] = $6 }
    END {
        print "# mb,ic,ih,iw,oc, ours_ms, onednn_e2e_ms, benchdnn_ms, onednn/ours, benchdnn/ours"
        for (i = 1; i <= n; i++) {
            k = order[i]
            e = (k in em) ? em[k] : "N/A"
            b = "N/A"
            for (j = 0; j < row; j++) if (row2key[j] == k) { b = bdms[j]; break }
            r1 = (e != "N/A" && e+0 > 0) ? sprintf("%.2fx", e/om[k]) : "N/A"
            r2 = (b != "N/A" && b+0 > 0) ? sprintf("%.2fx", b/om[k]) : "N/A"
            printf "%s, %s, %s, %s, %s, %s\n", k, om[k], e, b, r1, r2
        }
    }
' /tmp/merge_bd.txt "$CSV" "$OURS" "$E2E"
