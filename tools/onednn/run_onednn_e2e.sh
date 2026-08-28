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
echo "[onednn] compiling with $CXX (libs: $LIBS_DIR) ..."
$CXX -O3 -std=c++17 -fopenmp -I"$ROOT/include" \
    tools/onednn/onednn_e2e.cpp \
    -L"$LIBS_DIR" -Wl,-rpath,"$LIBS_DIR" -ldnnl -o build/onednn_e2e

# ---- verbose 探针（首个形状，1 轮）：ONEDNN_VERBOSE 输出会混进 stdout，
#      所以单独起一次进程、单独文件，不污染数据流；用于定位 PD 创建 OOM 时
#      oneDNN 到底选了哪个实现（brgconv:sve_512？）、失败发生点在哪。
probe_shp=""
[ -f "$CSV" ] && probe_shp=$(awk 'NR>1 && $0 !~ /^#/ {print; exit}' "$CSV")
if [ -n "$probe_shp" ]; then
    printf 'mb,ic,ih,iw,oc,kh,kw,stride_h,stride_w,pad_h,pad_w,dil_h,dil_w,grp,count\n%s\n' \
        "$probe_shp" > build/probe_shape.csv
    echo "[onednn] verbose probe on first shape (1 iter) -> build/probe_verbose.txt"
    ONEDNN_VERBOSE=all ./build/onednn_e2e build/probe_shape.csv "$THREADS" 1 1 $ALG \
        > build/probe_verbose.txt 2>&1 || true
fi

# ---- 运行（与 our 侧同绑核）----
echo "[onednn] threads=$THREADS warmup=$WARMUP repeats=$REPEATS alg=$ALG csv=$CSV"
OMP_PROC_BIND=close OMP_PLACES=cores \
    ./build/onednn_e2e "$CSV" "$THREADS" "$WARMUP" "$REPEATS" $ALG 2>build/onednn_e2e.err \
    | tee build/onednn_e2e.csv

# ---- 自查：数据行数应等于形状数；异常时把 stderr 打出来（编译过了也可能 parse 失败
#      或全部形状抛 dnnl::error）----
nrows=$(awk '!/^#/ && !/^mb,/ && NF {c++} END {print c+0}' build/onednn_e2e.csv)
echo "[onednn] data rows: $nrows"

# ---- SVE 兜底：默认 ISA 下 PD 创建系统性 OOM（疑 brgconv:sve_512 的 pd
#      is_initialized 失败 → primitive_desc_t::create 返回 out_of_memory）→ 关 SVE
#      重试。3.12.1 AArch64 的 ONEDNN_MAX_CPU_ISA 合法值：
#      default / advanced_simd(纯 NEON，无 SVE) / sve_128 / sve_256 / sve_512。
if [ "$nrows" -lt 10 ]; then
    echo "[onednn] default ISA gave only $nrows rows; retrying with ONEDNN_MAX_CPU_ISA=advanced_simd (SVE off) ..."
    OMP_PROC_BIND=close OMP_PLACES=cores \
        ONEDNN_MAX_CPU_ISA=advanced_simd \
        ./build/onednn_e2e "$CSV" "$THREADS" "$WARMUP" "$REPEATS" $ALG \
        2>build/onednn_e2e_nosve.err | tee build/onednn_e2e_nosve.csv
    nrows_ns=$(awk '!/^#/ && !/^mb,/ && NF {c++} END {print c+0}' build/onednn_e2e_nosve.csv)
    if [ "$nrows_ns" -ge 10 ]; then
        cp build/onednn_e2e_nosve.csv build/onednn_e2e.csv
        cp build/onednn_e2e_nosve.err build/onednn_e2e.err
        nrows=$nrows_ns
        echo "[onednn] SVE-off retry produced $nrows rows -> promoted to build/onednn_e2e.csv"
    else
        echo "[onednn] SVE-off retry also insufficient ($nrows_ns rows); see build/onednn_e2e_nosve.err"
    fi
fi

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
echo "[onednn] full result: build/onednn_e2e.csv (stdout is the file)"
