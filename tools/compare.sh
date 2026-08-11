#!/usr/bin/env bash
# compare.sh - bench our Winograd F(4,4,3,3) over a shapes CSV and emit a
# compact `mb,ic,ih,iw,oc,ours_ms` table for side-by-side comparison with
# oneDNN benchdnn numbers.
#
# Deliberately NOT a benchdnn wrapper: we do not parse benchdnn output and do
# not generate .dnn files. Data creation and measurement follow the repo's own
# bench_winograd (NHWC, fp32, best-of-repeats). Run the SAME shapes through
# benchdnn yourself (see shapes/README.md for the mb..count -> descriptor
# rule), then paste both tables side by side.
#
# Usage:
#   tools/compare.sh [options] [shapes.csv]
#   tools/compare.sh --threads 32 --isa sve --warmup 3 --repeats 20 \
#                    shapes/conv_all.csv
#
# Options:
#   --threads T    OpenMP thread count (default 16)
#   --isa X        neon | sve | sme | auto  (default sve; auto = let the
#                  binary detect)
#   --warmup N     warmup iterations  (default 3)
#   --repeats N    timed iterations    (default 20)
#   --bin PATH     bench_winograd binary (default build/bench_winograd)
#   -h, --help     show this help
#
# Output: prints a `# run:` comment line, then the reordered CSV
#   mb,ic,ih,iw,oc,ours_ms
# and writes the full bench_winograd CSV to build/compare_ours.csv.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

THREADS=16
ISA=sve
WARMUP=3
REPEATS=20
BIN="build/bench_winograd"
CSV="shapes/conv_all.csv"

usage() {
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --isa)     ISA="$2";     shift 2 ;;
        --warmup)  WARMUP="$2";  shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --bin)     BIN="$2";     shift 2 ;;
        -h|--help) usage ;;
        *) CSV="$1"; shift ;;
    esac
done

if [ ! -x "$BIN" ]; then
    echo "error: bench binary not found: $BIN (build first, see AGENTS.md)" >&2
    exit 1
fi
if [ ! -f "$CSV" ]; then
    echo "error: shapes CSV not found: $CSV" >&2
    exit 1
fi

BENCH_ARGS=(--nhwc --warmup "$WARMUP" --repeats "$REPEATS" --threads "$THREADS")
[ "$ISA" != "auto" ] && BENCH_ARGS+=("--$ISA")
OUT="build/compare_ours.csv"

# Run the bench, keep the full CSV as an artifact, suppress its stdout noise
# (OpenMP diagnostics + per-shape table).
"$BIN" "${BENCH_ARGS[@]}" --output "$OUT" "$CSV" >/dev/null

# Reorder the bench output CSV to mb,ic,ih,iw,oc,ours_ms. The ours_ms column
# is located by header name (t<threads>_ms) so this stays correct regardless
# of thread-count position.
awk -F, -v ms="t${THREADS}_ms" -v thr="$THREADS" -v isa="$ISA" -v wu="$WARMUP" -v rp="$REPEATS" '
    NR == 1 {
        for (i = 1; i <= NF; i++) if ($i == ms) msidx = i;
        printf "# run: threads=%s isa=%s warmup=%s repeats=%s\n", thr, isa, wu, rp;
        print "mb,ic,ih,iw,oc,ours_ms"
        next
    }
    { printf "%s,%s,%s,%s,%s,%s\n", $2, $3, $4, $5, $6, $msidx }
' "$OUT"
