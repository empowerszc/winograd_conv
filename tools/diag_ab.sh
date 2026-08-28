#!/usr/bin/env bash
# diag_ab.sh (v2) - 临时诊断脚本：在【同一个 sbatch 作业】内完成全部裁决。用完即删。
#
# 背景（2026-08-28，v1 的三个结论）：
#   1. v1 的 mini.csv 少了 header 行 —— bench read_shapes 把第一行当 header 跳过，
#      最大的 4,384,160²,384 根本没跑。v2 已补 header。
#   2. 集群 checkout 的 tools/*.sh 带 CRLF（./tools/compare.sh 触发 bash\r）。
#      v2 开头 sed 自愈，并用 bash 显式调用。
#   3. node03 存在 3~7x 的跨作业性能态：同一 OB 二进制同命令，大形状 47.3ms(v1前)
#      → 241.3ms(v1)。跨作业对比全部作废 —— 所以 v2 把所有对比塞进一个作业。
#
# 用法（build_oblas 保持旧二进制勿重编；旧码对照自动从 git 历史构建）：
#   sbatch -w node03 --exclusive --wrap="bash tools/diag_ab.sh"
# 可选：ACL_DIR=/path/to/ComputeLibrary-53.1.0（E5 旧码构建需要，默认
#       /workspace/z00889957/000Libs/ComputeLibrary-53.1.0）；SKIP_E5=1 跳过旧码构建。
#
# 判读（v2）：
#   P0 vs P1  首尾 smoke 漂移 >20% => 本作业自身不稳，其余数字全部存疑
#   E1 vs E2  同作业 arm(新码) vs OB —— 5 个关键形状的**可信** A/B（含大形状）
#   E1 vs E5  同作业 新码 vs 旧码(c48761e) —— 裁决「深层 7x7 形状 62ms vs 旧读数
#             10~11ms」是真代码回归还是历史表配置差异；顺带量化 nmulti 批量的收益
#   E3 vs E4  同作业全量 59 形状 arm vs OB —— 新基线表
#   状态探针  若 scaling_cur_freq 明显低于标称，说明节点被降频/受扰，数字另说

set +e
cd "$(dirname "$0")/.."
T=16
BIND="OMP_PROC_BIND=close OMP_PLACES=cores"
ACL_DIR="${ACL_DIR:-/workspace/z00889957/000Libs/ComputeLibrary-53.1.0}"
OLD_COMMIT="${OLD_COMMIT:-c48761e}"

echo "#### diag_ab v2: node=$(hostname) date=$(date) ####"

# --- CRLF 自愈（幂等）：集群 checkout 可能整树带 \r，先修脚本与 CSV ---
sed -i 's/\r$//' tools/*.sh shapes/*.csv 2>/dev/null

# --- mini.csv（5 形状，第一行是 header！read_shapes 会跳过首行）---
printf '%s\n' \
  'mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count' \
  '4,384,160,160,384,3,3,1,1,1,1,0,0,1,5' \
  '1,512,7,7,512,3,3,1,1,1,1,0,0,1,5' \
  '1,2048,7,7,512,3,3,1,1,1,1,0,0,1,5' \
  '4,768,20,20,96,3,3,1,1,1,1,0,0,1,5' \
  '4,96,20,20,96,3,3,1,1,1,1,0,0,1,5' > /tmp/mini.csv

# --- smoke.csv（状态探针：一个计算型 + 一个带宽型）---
printf '%s\n' \
  'mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count' \
  '4,192,40,40,192,3,3,1,1,1,1,0,0,1,10' \
  '1,2048,7,7,512,3,3,1,1,1,1,0,0,1,10' > /tmp/smoke.csv

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
  env $BIND "$1" --sve --nhwc --threads $T --warmup 2 --repeats 10 /tmp/smoke.csv 2>/dev/null \
    | grep -E '^#|^[0-9]'
}

echo
echo "================ P0: 作业起始状态探针 ================"
probe_state
run_smoke ./build/bench_winograd
run_smoke ./build_oblas/bench_winograd

echo
echo "================ E1: arm(新码) mini 5 形状 + auto 选核可视化 ================"
WINO_GEMM_DEBUG=1 env $BIND ./build/bench_winograd --sve --nhwc --threads $T \
  --warmup 3 --repeats 20 /tmp/mini.csv 2>&1

echo
echo "================ E2: OB mini 同作业对照（与 E1 同 CSV 同绑核） ================"
env $BIND ./build_oblas/bench_winograd --sve --nhwc --threads $T \
  --warmup 3 --repeats 20 /tmp/mini.csv

echo
echo "================ E5: 旧码(c48761e) 同作业对照（裁决 MB=1 回归真伪） ================"
if [ "${SKIP_E5:-0}" = "1" ]; then
  echo "SKIP_E5=1，跳过"
elif [ ! -d "$ACL_DIR" ]; then
  echo "ACL_DIR 不存在：$ACL_DIR —— 跳过（可用 sbatch --wrap=\"ACL_DIR=... bash tools/diag_ab.sh\" 指定）"
else
  rm -rf /tmp/wc_old /tmp/wc_old.tar
  echo "[E5] 提交存在性: $(git log --oneline -1 "$OLD_COMMIT" 2>&1 | head -1)"
  # v2.1: worktree add 在集群上失败过（错误被吞）；改用 git archive 导出纯源码树
  if ! git archive "$OLD_COMMIT" > /tmp/wc_old.tar 2>/tmp/e5_archive.err; then
    echo "git archive 失败（浅克隆可能没有该提交？），错误："
    cat /tmp/e5_archive.err
  else
    mkdir -p /tmp/wc_old
    tar -xf /tmp/wc_old.tar -C /tmp/wc_old
    cmake -S /tmp/wc_old -B /tmp/wc_old/build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SVE=ON -DENABLE_OPENMP=ON -DUSE_ARM_GEMM=ON \
      -DARM_GEMM_ROOT="$ACL_DIR" > /tmp/e5_cmake.log 2>&1
    if [ $? -ne 0 ]; then
      echo "cmake 失败，日志尾部："; tail -5 /tmp/e5_cmake.log
    else
      cmake --build /tmp/wc_old/build -j 32 > /tmp/e5_build.log 2>&1
      if [ $? -ne 0 ]; then
        echo "构建失败，日志尾部："; tail -10 /tmp/e5_build.log
      else
        echo "[E5] 旧码 mini 5 形状："
        env $BIND /tmp/wc_old/build/bench_winograd --sve --nhwc --threads $T \
          --warmup 3 --repeats 20 /tmp/mini.csv
      fi
    fi
  fi
fi

echo
echo "================ E3: 绑核全量 59 形状 arm（新基线，arm 侧） ================"
env $BIND bash tools/compare.sh shapes/conv_all.csv

echo
echo "================ E4: 绑核全量 59 形状 OB（新基线，OB 侧） ================"
env $BIND ./build_oblas/bench_winograd --sve --nhwc --threads $T \
  --warmup 10 --repeats 50 shapes/conv_all.csv

echo
echo "================ P1: 作业收尾状态探针（与 P0 对比判漂移） ================"
probe_state
run_smoke ./build/bench_winograd
run_smoke ./build_oblas/bench_winograd

rm -rf /tmp/wc_old /tmp/wc_old.tar
echo
echo "================ 完成：把本作业全部输出贴回 ================"
