#!/usr/bin/env bash
# ab_onednn.sh - 单作业 A/B：ours(arm) vs oneDNN 端到端 / benchdnn + filter_sweep。
# 与 diag_ab.sh 同协议：所有对比塞进【同一个 sbatch 作业】——node03 跨作业有
# 3~7x 性能态，跨作业对比全部作废，只信同作业。
#
# 用法（先传 0538b3f 之后的整树，含本脚本）：
#   sbatch -w node03 --exclusive --wrap="bash tools/onednn/ab_onednn.sh"
# 可选环境变量（--wrap 内可带）：
#   ONEDNN_ROOT=<含 include/dnnl.hpp 的目录>   默认探测 /workspace/z00889957/000Libs
#   BENCHDNN=<benchdnn 可执行路径>             默认按 ONEDNN_ROOT 常见路径探测
#   SKIP_AUTO=1   跳过 A4 benchdnn auto（默认 auto 与 WINO 都跑）
#   SKIP_FILTER=1 跳过 M=25 选核 filter_sweep
#   T=<线程数>     默认 16
#
# 产物（build/ 下）：
#   ours_cmp.csv            ours 全量 59 形状紧凑表（mb,ic,ih,iw,oc,ours_ms）
#   onednn_e2e.csv          oneDNN 端到端同口径计时
#   benchdnn_wino.txt       benchdnn raw（--alg=WINO，与我们同为 Winograd 算法族）
#   benchdnn_auto.txt       benchdnn raw（auto 算法，oneDNN 自选，兜底对照）
#   filter_sweep_{auto,6x4VL,8x1VL,inter}.csv   M=25 选核实验
# 末尾打印：合并表（ours/onednn_e2e/benchdnn_wino/benchdnn_auto 比率）+ filter_sweep 合并表。
#
# 判读：ours_ms 越小越好；onednn/ours、benchdnn_*/ours > 1 即我们快。
# 注意：oneDNN 的 --alg=WINO 依赖 ACL；本原生 build（无 ACL）可能全部 SKIP
#       （unimplemented）或退 ref，届时 benchdnn_wino 列为 N/A，以 auto 列为准。
#       oneDNN WINO 是 F(2,2) 族，我们 F(4,4,3,3)——仅"同为 Winograd 算法族"。

set +e
cd "$(dirname "$0")/../.."   # repo root

# --- CRLF 自愈：集群是 scp 文件副本，可能整树带 \r。先修本脚本再重执行。 ---
sed -i 's/\r$//' "$0" tools/*.sh tools/onednn/*.sh shapes/*.csv 2>/dev/null
if grep -q $'\r' "$0"; then exec bash "$0"; fi

T="${T:-16}"
BIND="OMP_PROC_BIND=close OMP_PLACES=cores"
echo "#### ab_onednn: node=$(hostname) date=$(date) threads=$T ####"

if [ ! -x build/bench_winograd ]; then
    echo "error: build/bench_winograd 不存在，先构建（见 AGENTS.md）" >&2
    exit 1
fi

# --- smoke.csv（状态探针：计算型 + 带宽型各一） ---
printf '%s\n' \
  'mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count' \
  '4,192,40,40,192,3,3,1,1,1,1,0,0,1,10' \
  '1,2048,7,7,512,3,3,1,1,1,1,0,0,1,10' > /tmp/smoke2.csv

probe_state() {
  echo "---- 状态探针 $(date +%T) ----"
  if ls /sys/devices/system/cpu/cpu0/cpufreq >/dev/null 2>&1; then
    echo "scaling_cur_freq (MHz) 分布:"
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null \
      | awk '{printf "%d\n", $1/1000}' | sort -n | uniq -c | tail -6
  else
    grep -m1 "model name" /proc/cpuinfo
    grep "MHz" /proc/cpuinfo | awk '{print $4}' | sort -n | uniq -c | tail -6
  fi
}

run_smoke() {
  echo "[smoke] $1"
  env $BIND "$1" --sve --nhwc --threads $T --warmup 2 --repeats 10 /tmp/smoke2.csv 2>/dev/null \
    | grep -E '^#|^[0-9]'
}

echo
echo "================ P0: 作业起始状态探针 ================"
probe_state
run_smoke ./build/bench_winograd

echo
echo "================ A1: ours 全量 59 形状（绑定对比基线） ================"
env $BIND bash tools/compare.sh shapes/conv_all.csv | tee build/ours_cmp.csv

echo
echo "================ A2: oneDNN 端到端（同形状同绑核，primitive 复用） ================"
env $BIND bash tools/onednn/run_onednn_e2e.sh

echo
echo "================ A3: benchdnn conv（--alg=WINO，与我们同为 Winograd，算法对齐） ================"
env $BIND bash tools/onednn/run_benchdnn.sh --winograd

if [ "${SKIP_AUTO:-0}" != "1" ]; then
    echo
    echo "================ A4: benchdnn conv auto（oneDNN 自选，兜底对照） ================"
    env $BIND bash tools/onednn/run_benchdnn.sh
fi

if [ "${SKIP_FILTER:-0}" != "1" ]; then
    echo
    echo "================ A5: M=25 选核 filter_sweep（auto/6x4VL/8x1VL/inter） ================"
    env $BIND bash tools/filter_sweep.sh
fi

echo
echo "================ A6: 合并表（ours / oneDNN e2e / benchdnn_wino / benchdnn_auto） ================"
if [ -f build/ours_cmp.csv ] && [ -f build/onednn_e2e.csv ]; then
    BD1=build/benchdnn_wino.txt; BD2=build/benchdnn_auto.txt
    [ -f "$BD1" ] || { BD1="$BD2"; BD2=""; }   # WINO 缺失（如未实现）则主列退 auto
    [ -f "$BD1" ] || BD1=""
    [ -f "$BD2" ] || BD2=""
    if [ -n "$BD1" ]; then
        echo "[A6] benchdnn 主列 = $BD1${BD2:+（副列 $BD2）}"
        if ! bash tools/onednn/merge_onednn.sh build/ours_cmp.csv build/onednn_e2e.csv \
                "$BD1" "$BD2" shapes/conv_all.csv 2>&1; then
            echo "!! merge 异常（退出码 $?），见上 stderr"
        fi
    else
        echo "（无 benchdnn 结果文件，先只出 ours vs e2e）"
        awk -F, '
          /^#/ || /^mb,/ { next }
          NR==FNR { a[$1","$2","$3","$4","$5]=$6; next }
          { k=$1","$2","$3","$4","$5; r=(a[k]+0>0) ? sprintf("%.2fx", $6/a[k]) : "N/A";
            printf "%s, %s, %s, %s\n", k, a[k], $6, r }
        ' build/ours_cmp.csv build/onednn_e2e.csv
    fi
else
    echo "缺 ours_cmp.csv 或 onednn_e2e.csv，跳过合并"
fi

echo
echo "================ P1: 作业收尾状态探针（与 P0 对比判漂移） ================"
probe_state
run_smoke ./build/bench_winograd

echo
echo "================ 完成：把本作业全部输出贴回 ================"
