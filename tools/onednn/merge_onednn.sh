#!/usr/bin/env bash
# 合并到一张对照表（按 ours 文件行序）：
#   mb,ic,ih,iw,oc, ours_ms, onednn_e2e_ms, benchdnn_<A>_ms[, benchdnn_<B>_ms],
#   onednn/ours, benchdnn_<A>/ours[, benchdnn_<B>/ours]
# 输入：
#   1) 我们侧 CSV（tools/compare.sh 输出：首行 # run:，随后 mb,ic,ih,iw,oc,ours_ms）
#   2) oneDNN e2e CSV（run_onednn_e2e.sh 输出：mb,ic,ih,iw,oc,onednn_ms）
#   3) benchdnn 原始 stdout（主列，如 WINO）
#   4) [可选] 第二个 benchdnn 原始 stdout（如 auto）
#   5) [可选] 源 shapes CSV（默认 shapes/conv_all.csv；benchdnn 的 rN = 该文件数据行号）
# 用法：
#   bash tools/onednn/merge_onednn.sh ours.csv onednn_e2e.csv benchdnn_wino.txt [benchdnn_auto.txt]
# benchdnn 解析尽力而为（格式随版本变化）；解析失败列显示 N/A，
# 把 benchdnn 原始输出贴回会话即可人工合并。
set -euo pipefail
OURS="${1:?ours.csv}"; E2E="${2:?onednn_e2e.csv}"; BD_A="${3:?benchdnn_raw.txt}"
# 位置 4 有二义（第二个 benchdnn 文件 或 shapes CSV）：用内容嗅探判——只有
# 含 rN 标签（r[0-9]+"）的才算 benchdnn 原始输出，否则当 shapes CSV。
CSV="${4:-shapes/conv_all.csv}"
BD_B=""
if [ $# -ge 4 ] && [ -n "$4" ] && [ -f "$4" ] \
   && grep -qE 'r[0-9]+"' "$4" 2>/dev/null; then
    BD_B="$4"
    CSV="${5:-shapes/conv_all.csv}"
fi

name_of() { basename "$1" | sed 's/\.txt$//'; }

# benchdnn：逐行抓 rN 标签 + ms（容忍格式变化），预解析为 "row ms"
# 实测 3.12.1 输出：`0:PASSED (513 ms) __REPRO: --conv ...n"r0"`——时间在括号里、
# 无小数点、描述符是紧凑 n"r0"（无下划线）。同 rN 可能多行，保留首个带数字的。
pre_parse() {   # $1 raw benchdnn → $2 排序后的 "row ms"
    grep -E 'r[0-9]+"' "$1" | awk '{
        if (match($0, /r[0-9]+"/)) n = substr($0, RSTART+1, RLENGTH-2); else next
        ms = "N/A"
        if      (match($0, /\([0-9][0-9.]* *ms\)/))  ms = substr($0, RSTART+1, RLENGTH-2)
        else if (match($0, /[0-9]+\.[0-9]+ *ms/))    ms = substr($0, RSTART, RLENGTH)
        else if (match($0, /time: *[0-9]+\.[0-9]+/)) ms = substr($0, RSTART+6, RLENGTH-6)
        else if (match($0, /perf: *[0-9]+\.[0-9]+/)) ms = substr($0, RSTART+6, RLENGTH-6)
        gsub(/ms/, "", ms); gsub(/[() ]/, "", ms)
        if (ms != "N/A") bdms[n] = ms
        else if (!(n in bdms)) bdms[n] = "N/A"
    }
    END { for (n in bdms) print n, bdms[n] }' | sort -k1,1n > "$2" || true
    # 文件无 rN 标签（如全部 unimplemented）时 grep 无匹配退出 1，|| true 兜底
}

pre_parse "$BD_A" /tmp/merge_bd_a.txt
NAME_A=$(name_of "$BD_A")
if [ -n "$BD_B" ]; then
    pre_parse "$BD_B" /tmp/merge_bd_b.txt
    NAME_B=$(name_of "$BD_B")
else
    NAME_B=""
fi

# awk 文件参数：主列预解析 + 副列预解析（无副列则用不存在的路径）+ shapes + ours + e2e
AWK_FILES="/tmp/merge_bd_a.txt"
[ -n "$BD_B" ] && AWK_FILES="$AWK_FILES /tmp/merge_bd_b.txt"
AWK_FILES="$AWK_FILES $CSV $OURS $E2E"

awk -F, -v bd_a="/tmp/merge_bd_a.txt" \
       -v bd_b="$([ -n "$BD_B" ] && echo /tmp/merge_bd_b.txt || echo /__none__)" \
       -v name_a="$NAME_A" -v name_b="$NAME_B" \
       -v csv="$CSV" -v ours="$OURS" -v e2e="$E2E" '
    BEGIN { row = 0; n = 0 }   # 未初始化变量是空串，与数字 0 是不同的数组键！
    FILENAME == bd_a { split($0, t, " "); bdms_a[t[1]] = t[2]; next }
    FILENAME == bd_b { split($0, t, " "); bdms_b[t[1]] = t[2]; next }
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
        hdr = "# mb,ic,ih,iw,oc, ours_ms, onednn_e2e_ms, " name_a "_ms"
        if (name_b != "") hdr = hdr ", " name_b "_ms"
        hdr = hdr ", onednn/ours, " name_a "/ours"
        if (name_b != "") hdr = hdr ", " name_b "/ours"
        print hdr
        for (i = 1; i <= n; i++) {
            k = order[i]
            e = (k in em) ? em[k] : "N/A"
            a = "N/A"
            for (j = 0; j < row; j++) if (row2key[j] == k) { a = bdms_a[j]; if (a == "") a = "N/A"; break }
            b = "N/A"
            if (name_b != "") for (j = 0; j < row; j++) if (row2key[j] == k) { b = bdms_b[j]; if (b == "") b = "N/A"; break }
            r1 = (e != "N/A" && e+0 > 0) ? sprintf("%.2fx", e/om[k]) : "N/A"
            r2 = (a != "N/A" && a+0 > 0) ? sprintf("%.2fx", a/om[k]) : "N/A"
            r3 = (b != "N/A" && b+0 > 0) ? sprintf("%.2fx", b/om[k]) : "N/A"
            line = sprintf("%s, %s, %s, %s", k, om[k], e, a)
            if (name_b != "") line = line ", " b
            line = line sprintf(", %s, %s", r1, r2)
            if (name_b != "") line = line ", " r3
            print line
        }
    }
' $AWK_FILES
