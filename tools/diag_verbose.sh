#!/usr/bin/env bash
# 临时诊断脚本：找出集群 benchdnn 3.12.1 的正确 verbose flag + e2e 编译问题。
# 用法：bash tools/diag_verbose.sh
# 闭环后删除此脚本（AGENTS.md 铁律：diag 脚本不提交）。
set +e
cd "$(dirname "$0")/../.."   # repo root

BIN=/workspace/z00889957/000Libs/oneDNN-3.12.1/build/tests/benchdnn/benchdnn
DESC='mb4ic192ih40iw40_oc192oh40ow40_kh3kw3_sh1sw1_ph1pw1_n"r0"'

echo "======== 1. -v4 产出了多少 onednn_verbose 行？ ========"
echo "  (0 = -v4 完全无效；>0 = 有 verbose 但可能没 exec 行)"
grep -c 'onednn_verbose' build/benchdnn_wino.txt 2>/dev/null
echo

echo "======== 2. benchdnn --help 里的 verbose 选项 ========"
"$BIN" --help 2>&1 | grep -i -E 'verbose|mode|perf|-v' || echo "  (no match)"
echo

echo "======== 3. 逐 flag 试单 case，看哪个出 primitive,exec 行 ========"
for FLAG in "-v4" "--verbose=4" "--verbose=exec" "--verbose=all" "--mode=p -v4"; do
    OUT=$(numactl -C 0-15 "$BIN" --conv $FLAG --reset --alg=wino "$DESC" 2>&1)
    N_EXEC=$(echo "$OUT" | grep -c 'primitive,exec,cpu,convolution,')
    N_VERBOSE=$(echo "$OUT" | grep -c 'onednn_verbose')
    SAMPLE=$(echo "$OUT" | grep 'primitive,exec,cpu,convolution,' | head -1)
    echo "  flag='$FLAG'  exec_lines=$N_EXEC  verbose_lines=$N_VERBOSE"
    [ -n "$SAMPLE" ] && echo "  sample: ${SAMPLE:0:120}..."
done
echo

echo "======== 4. 也试 env var 方式 ========"
for ENVVAR in "ONEDNN_VERBOSE=primitive:exec" "ONEDNN_VERBOSE=all" "ONEDNN_VERBOSE=exec"; do
    OUT=$(numactl -C 0-15 env $ENVVAR "$BIN" --conv --reset --alg=wino "$DESC" 2>&1)
    N_EXEC=$(echo "$OUT" | grep -c 'primitive,exec,cpu,convolution,')
    N_VERBOSE=$(echo "$OUT" | grep -c 'onednn_verbose')
    echo "  env='$ENVVAR'  exec_lines=$N_EXEC  verbose_lines=$N_VERBOSE"
done
echo

echo "======== 5. e2e 编译问题：ldd + ROOT 探测 ========"
echo "-- ldd build/onednn_e2e --"
ldd build/onednn_e2e 2>&1 | grep -iE 'dnnl|omp|tbb' || echo "  (no dnnl/omp/tbb in deps — 编译没链接到 libdnnl)"
echo
echo "-- run_onednn_e2e.sh ROOT/LIBS_DIR 探测 --"
bash -x tools/onednn/run_onednn_e2e.sh 2>&1 | grep -E '^\+ (ROOT=|LIBS_DIR=|CXX=|\$CXX|error)' | head -10
echo

echo "======== done ========"
