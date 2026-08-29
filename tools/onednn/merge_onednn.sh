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
# benchdnn 时间来源（2026-08-29 再修）：只认「单次执行」口径，优先级：
#   1) --mode=p 的 perf 行：`perf,<engine>,<impl>,<name>,<prb>,<Gops>,<ctime>,<min-ms>,...`
#      其中 %-time%（第 8 字段）= 逐次 start/stamp 的最小单次执行 ms
#      （见 oneDNN 源码 tests/benchdnn/utils/perf_report.hpp + dnnl_common.cpp
#       measure_perf_individual 的 t.start()/t.stamp() 逐次计时）。与 ours/e2e 同口径。
#   2) ONEDNN_VERBOSE=exec 的 onednn_verbose,...exec,... 行（末字段单次 ms）。
#   两者都无 → 整列 N/A。PASSED 的 (N ms) 是 corr 模式的聚合测试时间
#   （含 fill/ref/compare）且旧跑法未绑线程可能被 608 线程超订放大，一律不用。
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

# --- perf 行：perf,engine,impl,name,prb,Gops,ctime,min-ms,minGflops,avg-ms,avgGflops ---
#   $5=prb（mb4ic192ih40iw40_oc192oh40ow40_kh3kw3_sh1sw1_ph1pw1_n"r0"），$8=min-ms
pre_parse_perf() {   # $1 raw → $2 "key minms"
    awk -F, '
        /^perf,/ {
            ms = $8; if (ms !~ /^[0-9]+([.][0-9]+)?$/) next
            prb = $5
            mb = ic = ih = iw = oc = ""
            if (match(prb, /mb[0-9]+/))  mb = substr(prb, RSTART+2, RLENGTH-2)
            if (match(prb, /ic[0-9]+/))  ic = substr(prb, RSTART+2, RLENGTH-2)
            if (match(prb, /ih[0-9]+/))  ih = substr(prb, RSTART+2, RLENGTH-2)
            if (match(prb, /iw[0-9]+/))  iw = substr(prb, RSTART+2, RLENGTH-2)
            if (match(prb, /oc[0-9]+/))  oc = substr(prb, RSTART+2, RLENGTH-2)
            if (mb=="" || ic=="" || ih=="" || iw=="" || oc=="") next
            key = mb","ic","ih","iw","oc
            if (!(key in mn) || ms+0 < mn[key]) mn[key] = ms
        }
        END { for (k in mn) print k, mn[k] }
    ' "$1" | sort > "$2" || true
}

# --- exec 行：onednn_verbose,...primitive,exec,...,<desc>,<ms>（末字段单次 ms） ---
pre_parse_exec() {   # $1 raw → $2 "key minms"
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

# --- 单列时间源：perf 优先、exec 兜底；都空 → 空文件（整列 N/A） ---
build_src() {   # $1 raw → $2 "key minms"
    pre_parse_perf "$1" /tmp/merge_pf.txt
    if [ -s /tmp/merge_pf.txt ]; then cat /tmp/merge_pf.txt > "$2"; return 0; fi
    pre_parse_exec "$1" /tmp/merge_ex.txt
    if [ -s /tmp/merge_ex.txt ]; then cat /tmp/merge_ex.txt > "$2"; return 0; fi
    : > "$2"
}

build_src "$BD_A" /tmp/merge_src_a.txt
NAME_A=$(name_of "$BD_A")
[ -s /tmp/merge_src_a.txt ] && NO_SRC_A=0 || NO_SRC_A=1
if [ -n "$BD_B" ]; then
    build_src "$BD_B" /tmp/merge_src_b.txt
    NAME_B=$(name_of "$BD_B")
    [ -s /tmp/merge_src_b.txt ] && NO_SRC_B=0 || NO_SRC_B=1
else
    NAME_B=""
fi

AWK_FILES="/tmp/merge_src_a.txt"
[ -n "$BD_B" ] && AWK_FILES="$AWK_FILES /tmp/merge_src_b.txt"
AWK_FILES="$AWK_FILES $CSV $OURS $E2E"

awk -F, -v sra="/tmp/merge_src_a.txt" -v srb="/tmp/merge_src_b.txt" \
       -v name_a="$NAME_A" -v name_b="$NAME_B" \
       -v no_src_a="$NO_SRC_A" -v no_src_b="$NO_SRC_B" \
       -v csv="$CSV" -v ours="$OURS" -v e2e="$E2E" '
    # src 文件都是 "key 空格 ms"（key 含逗号），用 $0 切，不受 -F, 影响
    BEGIN { row = 0; n = 0; n_ea=n_naa=n_eb=n_nab=0 }
    FILENAME == sra { split($0, t, " "); ex_a[t[1]] = t[2]; next }
    FILENAME == srb { split($0, t, " "); ex_b[t[1]] = t[2]; next }
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
            a = "N/A"
            if (no_src_a == 0) { if (k in ex_a) { a = ex_a[k]; n_ea++ } else n_naa++ }
            b = "N/A"
            if (name_b != "" && no_src_b == 0) { if (k in ex_b) { b = ex_b[k]; n_eb++ } else n_nab++ }
            r1 = (e != "N/A" && e+0 > 0) ? sprintf("%.2fx", e/om[k]) : "N/A"
            r2 = (a != "N/A" && a+0 > 0) ? sprintf("%.2fx", a/om[k]) : "N/A"
            r3 = (b != "N/A" && b+0 > 0) ? sprintf("%.2fx", b/om[k]) : "N/A"
            line = sprintf("%s, %s, %s, %s", k, om[k], e, a)
            if (name_b != "") line = line ", " b
            line = line sprintf(", %s, %s", r1, r2)
            if (name_b != "") line = line ", " r3
            print line
        }
        printf "merge_summary: %d shapes; %s -> src=%d na=%d",
            n, name_a, n_ea, n_naa > "/dev/stderr"
        if (no_src_a == 1) printf " [NO-SRC: 无 perf/exec 行，整列 N/A]" > "/dev/stderr"
        if (name_b != "") {
            printf "; %s -> src=%d na=%d", name_b, n_eb, n_nab > "/dev/stderr"
            if (no_src_b == 1) printf " [NO-SRC: 整列 N/A]" > "/dev/stderr"
        }
        printf "\n" > "/dev/stderr"
    }
' $AWK_FILES
