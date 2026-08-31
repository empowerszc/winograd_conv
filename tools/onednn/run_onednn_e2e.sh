#!/usr/bin/env bash
# oneDNN 端到端 conv 基准：编译 tools/onednn/onednn_e2e.cpp 并对 shapes CSV 计时。
# 与 tools/compare.sh 同口径（best-of-repeats、16 线程、绑核），产出
#   mb,ic,ih,iw,oc,onednn_ms
# 供 tools/onednn/merge_onednn.sh 与我们的 ours_ms 合并。
#
# 用法：
#   bash tools/onednn/run_onednn_e2e.sh [--root ONEDNN_ROOT] [--threads T]
#            [--warmup N] [--repeats M] [--winograd] [shapes.csv]
#   ONEDNN_ROOT 默认：$ONEDNN_ROOT 环境变量 → 常见路径探测。
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

THREADS=16; WARMUP=3; REPEATS=20; ALG="--auto"; CSV="shapes/conv_all.csv"; ROOT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --root)     ROOT="$2";   shift 2 ;;
        --threads)  THREADS="$2"; shift 2 ;;
        --warmup)   WARMUP="$2"; shift 2 ;;
        --repeats)  REPEATS="$2"; shift 2 ;;
        --winograd) ALG="--winograd"; shift ;;
        -h|--help)  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          CSV="$1"; shift ;;
    esac
done

# ---- 定位 oneDNN（必须同时有 include/dnnl.hpp 和 lib/libdnnl.*） ----
if [ -z "$ROOT" ] && [ -n "${ONEDNN_ROOT:-}" ]; then ROOT="$ONEDNN_ROOT"; fi
if [ -z "$ROOT" ]; then
    # 遍历所有命中，优先挑带 lib 的完整安装（避免挑到只有头文件的空壳目录）
    while IFS= read -r h; do
        r="${h%/include/dnnl.hpp}"
        for ld in "$r/lib64" "$r/lib"; do
            if [ -f "$ld/libdnnl.so" ] || ls "$ld"/libdnnl.so.* >/dev/null 2>&1 \
               || [ -f "$ld/libdnnl.a" ]; then
                ROOT="$r"; break 2
            fi
        done
    done < <(find /usr/local /opt /workspace/z00889957/000Libs /workspace \
               -maxdepth 3 -name dnnl.hpp 2>/dev/null)
fi
if [ -z "$ROOT" ] || [ ! -f "$ROOT/include/dnnl.hpp" ]; then
    echo "error: oneDNN not found (need include/dnnl.hpp + lib/libdnnl.*). Set ONEDNN_ROOT=<dir> or use --root." >&2
    exit 1
fi
echo "[onednn] root: $ROOT"

# ---- 编译 ----
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    for c in g++ clang++ armclang++; do command -v "$c" >/dev/null && { CXX="$c"; break; }; done
fi
LIBS_DIR=""
for ld in "$ROOT/lib64" "$ROOT/lib"; do
    [ -d "$ld" ] && { LIBS_DIR="$ld"; break; }
done
if [ -z "$LIBS_DIR" ]; then
    echo "error: no lib/lib64 dir under $ROOT" >&2; exit 1
fi
mkdir -p build
: > build/onednn_e2e.csv   # 清空：避免编译/运行失败时 ab_onednn.sh 读到上一轮残留 CSV（1423 行异常根因）
echo "[onednn] compiling with $CXX (libs: $LIBS_DIR) ..."
$CXX -O3 -std=c++17 -fopenmp -I"$ROOT/include" \
    tools/onednn/onednn_e2e.cpp \
    -L"$LIBS_DIR" -Wl,-rpath,"$LIBS_DIR" -ldnnl -o build/onednn_e2e

# ---- verbose 探针（首个形状、独立进程、1 轮）：ONEDNN_VERBOSE 输出会混进 stdout，
#      所以单独起一次进程、单独文件，不污染数据流；用于定位 PD 创建失败时 oneDNN
#      到底选了哪个实现、失败发生在哪。用 --shape 模式（不带探针）且 e2e 只在
#      ONEDNN_VERBOSE 未设置时 set_verbose(1) ⇒ env 的 all 级不会被覆盖。
probe_shp5=""
[ -f "$CSV" ] && probe_shp5=$(awk 'NR>1 && $0 !~ /^#/ {split($0,a,","); print a[1]","a[2]","a[3]","a[4]","a[5]; exit}' "$CSV")
if [ -n "$probe_shp5" ]; then
    echo "[onednn] verbose probe on first shape $probe_shp5 (1 iter) -> build/probe_verbose.txt"
    ONEDNN_VERBOSE=all ./build/onednn_e2e --shape "$probe_shp5" "$THREADS" 1 1 $ALG \
        > build/probe_verbose.txt 2>&1 || true
fi

# ---- 运行（与 our 侧同绑核；线程由 OMP_NUM_THREADS env 控制，与 benchdnn 同口径）----
# 关键修复（2026-08-29）：conv PD 构造不再显式传 dilates={1,1}（CSV 是 dh=dw=0），
# 改用「带 bias、不带 dilates」的重载（oneDNN 内部 dilates 置 0）。此前错误 dilation
# 在集群 build 里直达 impl → 系统性 PD 创建 OOM（[PD-ALL] 全 oom），与线程数/ISA
# 无关（nothr=608 线程、SVE off 都一样 oom）——8244944 的 omp_set_num_threads
# 根因结论是错的，已更正。
echo "[onednn] threads=$THREADS warmup=$WARMUP repeats=$REPEATS alg=$ALG csv=$CSV"
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$THREADS \
    ./build/onednn_e2e "$CSV" "$THREADS" "$WARMUP" "$REPEATS" $ALG 2>build/onednn_e2e.err \
    | tee build/onednn_e2e.csv

# ---- 自查：数据行数应等于形状数；异常时把 stderr 打出来（编译过了也可能 parse 失败
#      或全部形状抛 dnnl::error）----
count_rows() { awk '!/^#/ && !/^mb,/ && NF {c++} END {print c+0}' "$1"; }
nrows=$(count_rows build/onednn_e2e.csv)
echo "[onednn] data rows: $nrows"

# ---- 兜底 A：OMP_NUM_THREADS=$THREADS 下仍 OOM？→ 去掉 OMP_NUM_THREADS
#      （=benchdnn 同配置，已被证明可建 PD）确认库本身仍好。此档线程数为 oneDNN
#      默认（非 $THREADS），merge 时不可与 16 线程列直接比——仅作「库是否可用」证据。
if [ "$nrows" -lt 10 ]; then
    echo "[onednn] OMP_NUM_THREADS=$THREADS gave only $nrows rows; retrying WITHOUT OMP_NUM_THREADS (benchdnn-equivalent env) ..."
    OMP_PROC_BIND=close OMP_PLACES=cores \
        ./build/onednn_e2e "$CSV" "$THREADS" "$WARMUP" "$REPEATS" $ALG \
        2>build/onednn_e2e_nothr.err | tee build/onednn_e2e_nothr.csv
    nrows2=$(count_rows build/onednn_e2e_nothr.csv)
    if [ "$nrows2" -ge 10 ]; then
        cp build/onednn_e2e_nothr.csv build/onednn_e2e.csv
        cp build/onednn_e2e_nothr.err build/onednn_e2e.err
        nrows=$nrows2
        echo "[onednn] NOTE: promoted run WITHOUT OMP_NUM_THREADS ($nrows rows) - thread count = oneDNN default (not $THREADS); merge comparison may be unfair"
    fi
fi

# ---- 兜底 B：仍不够 → 关 SVE 重试（3.12.1 AArch64 的 ONEDNN_MAX_CPU_ISA 合法值：
#      default / advanced_simd(纯 NEON，无 SVE) / sve_128 / sve_256 / sve_512）。
if [ "$nrows" -lt 10 ]; then
    echo "[onednn] still insufficient ($nrows rows); retrying with ONEDNN_MAX_CPU_ISA=advanced_simd (SVE off) ..."
    OMP_PROC_BIND=close OMP_PLACES=cores \
        ONEDNN_MAX_CPU_ISA=advanced_simd \
        ./build/onednn_e2e "$CSV" "$THREADS" "$WARMUP" "$REPEATS" $ALG \
        2>build/onednn_e2e_nosve.err | tee build/onednn_e2e_nosve.csv
    nrows_ns=$(count_rows build/onednn_e2e_nosve.csv)
    if [ "$nrows_ns" -ge 10 ]; then
        cp build/onednn_e2e_nosve.csv build/onednn_e2e.csv
        cp build/onednn_e2e_nosve.err build/onednn_e2e.err
        nrows=$nrows_ns
        echo "[onednn] SVE-off retry produced $nrows rows -> promoted to build/onednn_e2e.csv"
    else
        echo "[onednn] SVE-off retry also insufficient ($nrows_ns rows); see build/onednn_e2e_nosve.err"
    fi
fi

# ---- 兜底 C：仍不够 → 逐 shape 起独立进程（每个进程只建一个 conv PD，彻底绕开
#      进程内跨 PD 状态污染；即便有残留状态问题也能出全量数据）。
if [ "$nrows" -lt 10 ]; then
    echo "[onednn] still insufficient ($nrows rows); retrying per-shape fresh processes ..."
    : > build/onednn_e2e_ps.csv
    : > build/onednn_e2e_ps.err
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        case "$line" in \#*) continue ;; esac
        case "$line" in mb,*) continue ;; esac
        shp5=$(awk -F, '{print $1","$2","$3","$4","$5}' <<<"$line")
        OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$THREADS \
            ./build/onednn_e2e --shape "$shp5" "$THREADS" "$WARMUP" "$REPEATS" $ALG \
            >> build/onednn_e2e_ps.csv 2>> build/onednn_e2e_ps.err
    done < "$CSV"
    nrows_ps=$(count_rows build/onednn_e2e_ps.csv)
    if [ "$nrows_ps" -ge 10 ]; then
        cp build/onednn_e2e_ps.csv build/onednn_e2e.csv
        cp build/onednn_e2e_ps.err build/onednn_e2e.err
        nrows=$nrows_ps
        echo "[onednn] per-shape fresh processes produced $nrows rows -> promoted"
    else
        echo "[onednn] per-shape processes also insufficient ($nrows_ps rows); see build/onednn_e2e_ps.err"
    fi
fi

# ---- 单 PD 诊断矩阵（各自独立进程）：直接验证根因 = 旧 dilates 参数 ----
echo "[onednn] ---- 单 PD 诊断矩阵（各独立进程；验证修复）----"
: > build/diag.txt
for dc in eltwise_nchw eltwise_any \
          conv_direct_nchw_nodil conv_direct_any_nodil conv_direct_any_dil1; do
    OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=$THREADS \
        ./build/onednn_e2e --diag "$dc" "$THREADS" >> build/diag.txt 2>&1 || true
done
cat build/diag.txt
echo "[onednn] 判读：conv_direct_any_nodil=ok 且 conv_direct_any_dil1=FAIL ⇒ 根因确认为 dilates 旧 bug，修复生效；"
echo "        eltwise_any=FAIL 属预期（该 build 不支持 eltwise+format_any，非数据路径）。"

if [ -s build/onednn_e2e.err ]; then
    echo "!!! onednn_e2e stderr (parsed N shapes + per-shape skip reason):"
    cat build/onednn_e2e.err
fi
if [ "$nrows" -lt 10 ]; then
    echo "!!! too few data rows - see stderr above (parsed 0 = CSV parse issue; else primitive threw)"
fi
echo "[onednn] via alg histogram (which oneDNN alg actually ran per shape):"
grep -o 'via alg=[a-z]*' build/onednn_e2e.err 2>/dev/null | sort | uniq -c || true
echo "[onednn] impl histogram (pd.impl_info_str(): which impl actually ran - OOM 定位关键):"
grep -o 'impl=[^ ]*' build/onednn_e2e.err 2>/dev/null | sort | uniq -c || true
# 库级探针（版本一致性 + eltwise 非 conv PD + [thr] env 线程数确认）+ verbose info
# 行（ISA/线程检测）。按 err 文件逐一列出：主运行 / 兜底 A(no OMP_NUM_THREADS) /
# 兜底 B(SVE off) / 兜底 C(per-shape)。探针现在在数据流之后。
if [ -s build/onednn_e2e.err ]; then
    echo "[onednn] ---- 探针（按 err 文件逐一列出：主运行 / nothr / nosve / per-shape）----"
    for ef in build/onednn_e2e.err build/onednn_e2e_nothr.err build/onednn_e2e_nosve.err build/onednn_e2e_ps.err; do
        [ -s "$ef" ] || continue
        echo "[onednn] ---- $ef ----"
        grep -E '^\[env\]|^\[thr\]|^\[smoke\]|^\[smoke-any\]|^\[capi|^\[heap\]|^\[preOMP\]|^dnnl_verbose,info|^parsed ' "$ef" | head -12
    done
fi
if [ -s build/probe_verbose.txt ]; then
    echo "[onednn] ---- build/probe_verbose.txt (ONEDNN_VERBOSE=all, default ISA, first shape) ----"
    head -120 build/probe_verbose.txt
    echo "[onednn] ---- end probe ----"
fi
echo "[onednn] full result: build/onednn_e2e.csv (stdout is the file)"
