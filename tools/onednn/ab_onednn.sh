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
#   FORCE_WINO=1   追加跑 benchdnn --alg=WINO（较慢，默认不跑）
#   SKIP_FILTER=1  跳过 M=25 选核 filter_sweep
#   T=<线程数>     默认 16
#
# 产物（build/ 下）：
#   ours_cmp.csv            ours 全量 59 形状紧凑表（mb,ic,ih,iw,oc,ours_ms）
#   onednn_e2e.csv          oneDNN 端到端同口径计时
#   benchdnn_auto.txt       benchdnn raw（auto 算法）
#   benchdnn_wino.txt       （FORCE_WINO=1 时）benchdnn raw（WINO 算法）
#   filter_sweep_{auto,6x4VL,8x1VL,inter}.csv   M=25 选核实验
# 末尾打印：三列合并表（ours/onednn_e2e/benchdnn 比率）+ filter_sweep 合并表。
#
# 判读：ours_ms 越小越好；onednn/ours、benchdnn/ours > 1 即我们快。

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
echo "================ A3: benchdnn conv（auto 算法，oneDNN 自选） ================"
env $BIND bash tools/onednn/run_benchdnn.sh

if [ "${FORCE_WINO:-0}" = "1" ]; then
    echo
    echo "================ A4: benchdnn conv（--alg=WINO 强制 ACL Winograd） ================"
    env $BIND bash tools/onednn/run_benchdnn.sh --winograd
fi

if [ "${SKIP_FILTER:-0}" != "1" ]; then
    echo
    echo "================ A5: M=25 选核 filter_sweep（auto/6x4VL/8x1VL/inter） ================"
    env $BIND bash tools/filter_sweep.sh
fi

echo
echo "================ A6: 三列合并（ours / oneDNN e2e / benchdnn） ================"
if [ -f build/ours_cmp.csv ] && [ -f build/onednn_e2e.csv ]; then
    if [ -f build/benchdnn_auto.txt ]; then
        if ! bash tools/onednn/merge_onednn.sh build/ours_cmp.csv build/onednn_e2e.csv \
                build/benchdnn_auto.txt shapes/conv_all.csv 2>&1; then
            echo "!! merge 异常（退出码 $?），见上 stderr"
        fi
    else
        echo "（benchdnn_auto.txt 缺失，先只出 ours vs e2e）"
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
