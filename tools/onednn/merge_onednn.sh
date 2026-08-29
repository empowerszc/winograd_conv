#!/usr/bin/env bash
# 合并到一张对照表（按 ours 文件行序）：
#   mb,ic,ih,iw,oc, ours_ms, onednn_e2e_ms, benchdnn_<A>_ms[, benchdnn_<B>_ms],
#   onednn/ours, benchdnn_<A>/ours[, benchdnn_<B>/ours]
# 输入：
#   1) 我们侧 CSV（tools/compare.sh 输出：首行 # run:，随后 mb,ic,ih,iw,oc,ours_ms）
#   2) oneDNN e2e CSV（run_onednn_e2e.sh 输出：mb,ic,ih,iw,oc,onednn_ms）
#   3) benchdnn 原始 stdout（主列，如 WINO）
#   4) [可选] 第二个 benchdnn 原始 stdout（如 auto）
#   5) [可选] 源 shapes CSV（默认 shapes/conv_all.csv）
# 用法：
#   bash tools/onednn/merge_onednn.sh ours.csv onednn_e2e.csv benchdnn_wino.txt [benchdnn_auto.txt]
#
# benchdnn 时间来源（2026-08-29 修）：优先解析 onednn_verbose,...primitive,exec,... 行的
#   末字段（= 单次执行 ms，run_benchdnn.sh 已带 ONEDNN_VERBOSE=exec），每个 shape 取
#   **最小**单次执行时间（≈ best-of，与 ours/e2e 同口径）；仅当无 exec 行时才退回
#   PASSED 行里的 (N ms)（benchdnn perf 循环的聚合时间，可能=多迭代总和，不可当单次）。
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

# --- exec 行：shapekey(逗号) 空格 minms。行形如
#     onednn_verbose,v1,primitive,exec,cpu,convolution,<impl>,...,<descriptor>,<ms>
#   用 -F, 时末字段 = ms、倒数第二 = descriptor；descriptor 里抽 mb/ic/ih/iw/oc。 ---
pre_parse_exec() {   # $1 raw → $2 "key minms"（无 exec 行则空文件）
    awk -F, '
        /onednn_verbose,.*,primitive,exec,/ {
            ms = $NF; if (ms !~ /^[0-9]+([.][0-9]+)?$/) next
            desc = $(NF-1)
            mb = ic = ih = iw = oc = ""
            if (match(desc, /mb[0-9]+/))  mb = substr(desc, RSTART+2, RLENGTH-2)
            if (match(desc, /ic[0-9]+/))  ic = substr(desc, RSTART+2, RLENGTH-2)
            if (match(desc, /ih[0-9]+/))  ih = substr(desc, RSTART+2, RLENGTH-2)
            if (match(desc, /iw[0-9]+/))  iw = substr(desc, RSTART+2, RLENGTH-2)
            if (match(desc, /oc[0-9]+/))  oc = substr(desc, RSTART+2, RLENGTH-2)
            if (mb=="" || ic=="" || ih=="" || iw=="" || oc=="") next
            key = mb","ic","ih","iw","oc
            if (!(key in mn) || ms+0 < mn[key]) mn[key] = ms
        }
        END { for (k in mn) print k, mn[k] }
    ' "$1" | sort > "$2" || true
}

# --- PASSED 行：rN → ms（兜底；仅当无 exec 行时使用）---
pre_parse_pass() {   # $1 raw → $2 排序后的 "rN ms"
    grep -E 'r[0-9]+"' "$1" | awk '{
        if (match($0, /r[0-9]+"/)) n = substr($0, RSTART+1, RLENGTH-2); else next
        ms = "N/A"
        if      (match($0, /\([0-9][0-9.]* *ms\)/))  ms = substr($0, RSTART+1, RLENGTH-2)
        else if (match($0, /[0-9]+\.[0-9]+ *ms/))    ms = substr($0, RSTART, RLENGTH)
        else if (match($0, /time: *[0-9]+\.[0-9]+/)) ms = substr($0, RSTART+6, RLENGTH-6)
        else if (match($0, /perf: *[0-9]+\.[0-9]+/)) ms = substr($0, RSTART+6, RLENGTH-6)
        gsub(/ms/, "", ms); gsub(/[() ]/, "", ms)
        if (ms != "N/A") pass[n] = ms
        else if (!(n in pass)) pass[n] = "N/A"
    }
    END { for (n in pass) print n, pass[n] }' | sort -k1,1n > "$2" || true
}

pre_parse_exec "$BD_A" /tmp/merge_exec_a.txt
pre_parse_pass "$BD_A" /tmp/merge_pass_a.txt
NAME_A=$(name_of "$BD_A")
if [ -n "$BD_B" ]; then
    pre_parse_exec "$BD_B" /tmp/merge_exec_b.txt
    pre_parse_pass "$BD_B" /tmp/merge_pass_b.txt
    NAME_B=$(name_of "$BD_B")
else
    NAME_B=""
fi

# awk 文件参数：exec 主列 + pass 主列 (+ exec 副列 + pass 副列) + shapes + ours + e2e
AWK_FILES="/tmp/merge_exec_a.txt /tmp/merge_pass_a.txt"
[ -n "$BD_B" ] && AWK_FILES="$AWK_FILES /tmp/merge_exec_b.txt /tmp/merge_pass_b.txt"
AWK_FILES="$AWK_FILES $CSV $OURS $E2E"

awk -F, -v exa="/tmp/merge_exec_a.txt" -v psa="/tmp/merge_pass_a.txt" \
       -v exb="/tmp/merge_exec_b.txt" -v psb="/tmp/merge_pass_b.txt" \
       -v name_a="$NAME_A" -v name_b="$NAME_B" \
       -v csv="$CSV" -v ours="$OURS" -v e2e="$E2E" '
    # getbd：优先 exec 单次 ms；无 exec 时经 row2key→rN 查 PASSED 兜底。
    function getbd(key, exf, psf,  j) {
        if (key in exf) return exf[key]
        for (j = 0; j < row; j++)
            if (row2key[j] == key) return (j in psf) ? psf[j] : "N/A"
        return "N/A"
    }
    BEGIN { row = 0; n = 0 }
    # exec/pass 文件都是 "key 空格 ms"（无逗号），用 $0 切，不受 -F, 影响
    FILENAME == exa { split($0, t, " "); ex_a[t[1]] = t[2]; next }
    FILENAME == psa { split($0, t, " "); ps_a[t[1]] = t[2]; next }
    FILENAME == exb { split($0, t, " "); ex_b[t[1]] = t[2]; next }
    FILENAME == psb { split($0, t, " "); ps_b[t[1]] = t[2]; next }
    FILENAME == csv {                       # 源 shape：rN = 数据行号（0 基）
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
            a = getbd(k, ex_a, ps_a)
            b = "N/A"
            if (name_b != "") b = getbd(k, ex_b, ps_b)
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
