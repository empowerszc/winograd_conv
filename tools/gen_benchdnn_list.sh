#!/usr/bin/env bash
# Generate a oneDNN benchdnn conv descriptor list from shapes/conv_all.csv.
#
# Every CSV row is stride=1/group=1/3x3/pad=1, so the benchdnn descriptor is
#   mb{mb}_ic{ic}_ih{ih}iw{iw}_oc{oc}_oh{oh}ow{ow}_kh3kw3_sh1sw1_ph1pw1
# with oh=ih, ow=iw (pad1/s1/k3 keeps output size equal to input).
# The `_n"r<row>"` suffix labels each descriptor with its 0-based row in
# conv_all.csv, so the benchdnn output can be merged back onto the our-side
# numbers row by row (see docs/timing_breakdown_920f.md 附录 A).
#
# Usage:
#   ./tools/gen_benchdnn_list.sh [src_csv] [out_list]
#   default: shapes/conv_all.csv -> shapes/conv_all.list
#
# Then on 920F:
#   ./benchdnn --conv --cfg=f32 --reset --alg=WINO --batch=shapes/conv_all.list
set -euo pipefail

SRC="${1:-shapes/conv_all.csv}"
OUT="${2:-shapes/conv_all.list}"

awk -F, '
  /^#/ { print; next }   # keep section headers as benchdnn comments
  NR == 1 { next }
  NF >= 15 {
    printf "mb%d_ic%d_ih%diw%d_oc%d_oh%dow%d_kh3kw3_sh1sw1_ph1pw1_n\"r%d\"\n",
           $1, $2, $3, $4, $5, $3, $4, ++row - 1;
  }
' "$SRC" > "$OUT"

n=$(grep -vc '^#' "$OUT" || true)
echo "wrote $OUT ($n descriptors)"
