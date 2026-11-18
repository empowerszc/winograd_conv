#!/usr/bin/env python3
"""Generate standalone HTML figures for reporting / presentations.

Output: docs/figures/*.html — self-contained (no CDN, no external assets), so
they render on an offline cluster node or when copied to another machine.

Data sources (all in-repo, parsed at generation time):
  * docs/final_benchmark_bfd6b1e.md  — 59 shapes, ours vs OpenBLAS (t16)
  * docs/sme_fmopa_gemm.md §5.3/§5.4 — 59 shapes, SME vs KML at t1 and t16

Usage:
    python3 tools/gen_report_figures.py
"""

import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")

# ---------------------------------------------------------------- data

def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().split("\n")


def parse_ob(path):
    """| # | shape | ours ms | OB ms | speedup |"""
    out = []
    for line in _rows(path):
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            idx, shape, ours, ob = m.groups()
            out.append({
                "idx": int(idx),
                "shape": shape.strip(),
                "ours": float(ours),
                "base": float(ob),
            })
    return out


def parse_sme(path):
    """Returns {'t1': [...], 't16': [...]} from the two 59-row tables."""
    cur = None
    res = {"t1": [], "t16": []}
    for line in _rows(path):
        if line.startswith("### 5.3"):
            cur = "t1"
            continue
        if line.startswith("### 5.4"):
            cur = "t16"
            continue
        if cur is None:
            continue
        m = re.match(
            r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([*\d.x]+)\s*\|",
            line,
        )
        if m:
            idx, shape, sme, kml, _ratio = m.groups()
            res[cur].append({
                "idx": int(idx),
                "shape": shape.strip(),
                "sme": float(sme),
                "kml": float(kml),
            })
    return res


def _num(cell):
    m = re.search(r"\d+(?:\.\d+)?", cell.replace("**", ""))
    return float(m.group(0)) if m else None


def parse_steps(path):
    """docs/timing_breakdown_920f.md §3.1 的 8 步拆解表。"""
    keys = ["gemm", "weight", "inx", "outx", "tileext", "inscat", "outgath", "outw"]
    out, started = [], False
    for line in _rows(path):
        if line.startswith("| row | shape | GEMM | Weight | InXform"):
            started = True
            continue
        if not started:
            continue
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 11 or not cells[0].isdigit():
            continue
        r = {"idx": int(cells[0]), "shape": cells[1].replace("**", "")}
        for k, c in zip(keys, cells[2:10]):
            r[k] = _num(c) or 0.0
        r["S"] = _num(cells[10]) or 0.0
        out.append(r)
    return out


def parse_timing(path):
    """附录 A：row/tiles/S/T1/T16/GEMM%/Weight%/eff/16T×。"""
    out, started = [], False
    for line in _rows(path):
        if line.startswith("| row | shape (mb,ic,ih,iw,oc)"):
            started = True
            continue
        if not started:
            continue
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 10 or not cells[0].isdigit():
            continue
        out.append({
            "idx": int(cells[0]),
            "shape": cells[1].replace("**", ""),
            "tiles": int(cells[2]),
            "S": float(cells[3]),
            "T1": float(cells[4]),
            "T16": float(cells[5]),
            "gemm_pct": float(cells[6]),
            "weight_pct": float(cells[7]),
            "eff": float(cells[8]),
            "sp16": float(cells[9]),
        })
    return out


def parse_shape(shape):
    """'4,192,40²,192' -> (4, 192, 40, 192); 兼容 '4,192,40,40,192'。"""
    parts = [p.strip() for p in shape.replace("²", "").split(",")]
    try:
        if len(parts) == 4:
            return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        if len(parts) == 5:
            return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[4])
    except ValueError:
        pass
    return None


def stats(rows, key_fast, key_slow):
    """speedup = slow/fast (>1 means `fast` wins)."""
    sp = [r[key_slow] / r[key_fast] for r in rows]
    tot_f = sum(r[key_fast] for r in rows)
    tot_s = sum(r[key_slow] for r in rows)
    return {
        "rows": rows,
        "n": len(rows),
        "sp": sp,
        "total_ratio": tot_f / tot_s,      # fast/slow, <1 = fast is faster
        "total_speedup": tot_s / tot_f,    # slow/fast, >1 = fast is faster
        "total_fast": tot_f,
        "total_slow": tot_s,
        "geo": math.exp(sum(math.log(v) for v in sp) / len(sp)),
        "arith": sum(sp) / len(sp),
        "median": statistics.median(sp),
        "min": min(sp),
        "max": max(sp),
        "wins": sum(1 for v in sp if v > 1.0),
    }


# ---------------------------------------------------------------- svg helpers

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def svg(w, h, body, cls="chart"):
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" role="img">{body}</svg>')


def text(x, y, s, size=11, fill="#64748b", anchor="middle", weight="400",
         family=None, extra=""):
    fam = f' font-family="{family}"' if family else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{fam} {extra}>{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke="#e2e8f0", width=1, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d} {extra}/>')


def rect(x, y, w, h, fill, rx=0, extra=""):
    attrs = (f'x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" '
             f'height="{max(h,0):.1f}" rx="{rx}" fill="{fill}"')
    e = extra.strip()
    if e.startswith("<"):
        return f'<rect {attrs}>{extra}</rect>'
    return f'<rect {attrs} {extra}/>'


def fmt(v, nd=2):
    return f"{v:.{nd}f}"


def short_shape(shape):
    """'4,192,40²,192' -> '4·192·40²·192' compact label."""
    return shape.replace(",", "·")


# ---------------------------------------------------------------- charts

C_INDIGO = "#4338ca"
C_INDIGO_L = "#818cf8"
C_TEAL = "#0f766e"
C_TEAL_L = "#5eead4"
C_ROSE = "#be123c"
C_AMBER = "#b45309"
C_SLATE = "#94a3b8"
C_GRID = "#e6eaf2"
C_AXIS = "#94a3b8"


def chart_speedup_bars(rows, st):
    """59 shapes: speedup vs OpenBLAS, baseline at 1.0."""
    W, H = 1180, 380
    L, R, T, B = 58, 26, 34, 54
    ymin, ymax = 1.0, 5.6
    iw = W - L - R
    ih = H - T - B

    def y(v):
        v = max(ymin, min(ymax, v))
        return T + (ymax - v) / (ymax - ymin) * ih

    def x(i):
        n = len(rows)
        return L + (i + 0.5) * iw / n

    bw = max(4.0, iw / len(rows) * 0.66)
    out = []
    # gridlines
    for gv in (1, 2, 3, 4, 5):
        out.append(line(L, y(gv), W - R, y(gv), C_GRID if gv != 1 else "#cbd5e1",
                        1.0 if gv != 1 else 1.4))
        out.append(text(L - 10, y(gv) + 4, f"{gv}x", 11, C_AXIS, "end"))

    for i, r in enumerate(rows):
        v = r["base"] / r["ours"]
        col = C_TEAL if v >= 3 else (C_INDIGO if v >= 1.6 else C_INDIGO_L)
        h = y(1.0) - y(v)
        out.append(rect(x(i) - bw / 2, y(v), bw, h, col, rx=2,
                        extra=f'<title>#{r["idx"]}  {esc(r["shape"])}\n'
                              f'ours {r["ours"]:.3f} ms  ·  OpenBLAS {r["base"]:.3f} ms\n'
                              f'speedup {v:.2f}x</title>'))
    # emphasise top 3
    top = sorted(rows, key=lambda r: r["base"] / r["ours"], reverse=True)[:3]
    for r in top:
        i = r["idx"]
        v = r["base"] / r["ours"]
        out.append(text(x(i), y(v) - 6, f"{v:.1f}x", 10, C_TEAL, "middle", "700"))

    # x ticks every 4
    for i in range(0, len(rows), 4):
        out.append(text(x(i), H - B + 18, str(i), 10, C_AXIS, "middle"))
    out.append(text(W - R, H - 10, "shape index", 11, C_AXIS, "end"))

    legend = (rect(L, 12, 10, 10, C_INDIGO_L, 2) + text(L + 16, 21, "1.0–1.6x", 10.5, "#475569", "start")
              + rect(L + 96, 12, 10, 10, C_INDIGO, 2) + text(L + 112, 21, "1.6–3x", 10.5, "#475569", "start")
              + rect(L + 180, 12, 10, 10, C_TEAL, 2) + text(L + 196, 21, "> 3x", 10.5, "#475569", "start")
              + text(W - R, 21, f"59/59 形状快于 OpenBLAS · 几何平均 {st['geo']:.2f}x · "
                                f"合计 {st['total_speedup']:.2f}x", 11, "#475569", "end"))
    return svg(W, H, "".join(out) + legend)


def chart_totals(st):
    """Horizontal totals comparison."""
    W, H = 560, 168
    L, T = 150, 26
    maxv = max(st["total_fast"], st["total_slow"])
    bw = W - L - 110

    def w(v):
        return v / maxv * bw

    out = []
    rows = [("OpenBLAS", st["total_slow"], "#cbd5e1", "#475569"),
            ("本项目", st["total_fast"], C_INDIGO, C_INDIGO)]
    for k, (name, v, col, tx) in enumerate(rows):
        yy = T + k * 54
        out.append(text(L - 14, yy + 20, name, 12, "#334155", "end", "600"))
        out.append(rect(L, yy + 6, w(v), 26, col, rx=4))
        out.append(text(L + w(v) + 10, yy + 24, f"{v:.1f} ms", 12, tx, "start", "700"))
    out.append(text(L, T + 128, f"合计时间比  {st['total_speedup']:.2f}x", 15, C_INDIGO, "start", "700"))
    out.append(text(L + 168, T + 128,
                    "（59 形状、16 线程、同一作业内 A/B）", 11, "#64748b", "start"))
    return svg(W, H, "".join(out))


def chart_histogram(st):
    """Speedup distribution."""
    buckets = [(1.0, 1.25, "1.0–1.25"), (1.25, 1.5, "1.25–1.5"),
               (1.5, 2.0, "1.5–2"), (2.0, 3.0, "2–3"), (3.0, 99, "> 3")]
    counts = []
    for lo, hi, _ in buckets:
        counts.append(sum(1 for v in st["sp"] if lo <= v < hi))
    W, H = 560, 244
    L, R, T, B = 46, 20, 30, 46
    ih = H - T - B
    maxc = max(counts) or 1
    bw = (W - L - R) / len(buckets) * 0.56
    out = []
    for g in range(0, maxc + 1, 5):
        yy = T + ih - g / maxc * ih
        out.append(line(L, yy, W - R, yy, C_GRID))
        out.append(text(L - 8, yy + 4, str(g), 10, C_AXIS, "end"))
    for i, ((lo, hi, lab), c) in enumerate(zip(buckets, counts)):
        cx = L + (i + 0.5) * (W - L - R) / len(buckets)
        h = c / maxc * ih
        col = C_TEAL if i >= 3 else (C_INDIGO if i >= 2 else C_INDIGO_L)
        out.append(rect(cx - bw / 2, T + ih - h, bw, h, col, rx=3,
                        extra=f'<title>{lab}x: {c} shapes</title>'))
        out.append(text(cx, T + ih - h - 7, str(c), 11, "#475569", "middle", "700"))
        out.append(text(cx, H - B + 18, lab + "x", 10.5, C_AXIS, "middle"))
    out.append(text(L, 16, "形状数", 10.5, C_AXIS, "start"))
    return svg(W, H, "".join(out))


def chart_sme(st, title):
    """SME vs KML per shape, log2 scale around 1.0 (speedup = KML/SME)."""
    rows = st["rows"]
    W, H = 1180, 300
    L, R, T, B = 54, 24, 34, 48
    iw, ih = W - L - R, H - T - B
    lo, hi = -2.0, 1.7          # log2 range (0.25x .. 3.25x)
    mid = T + (hi / (hi - lo)) * ih

    def y(logv):
        return T + (hi - logv) / (hi - lo) * ih

    def x(i):
        return L + (i + 0.5) * iw / len(rows)

    bw = max(4.0, iw / len(rows) * 0.66)
    out = []
    for gv, lab in ((0.25, "0.25x"), (0.5, "0.5x"), (1.0, "1x"), (2.0, "2x"), (3.0, "3x")):
        lv = math.log2(gv)
        if lv < lo or lv > hi:
            continue
        strong = gv == 1.0
        out.append(line(L, y(lv), W - R, y(lv),
                        "#cbd5e1" if strong else C_GRID, 1.4 if strong else 1.0))
        out.append(text(L - 8, y(lv) + 4, lab, 10.5, C_AXIS, "end"))

    for i, r in enumerate(rows):
        sp = r[f"kml"] / r["sme"]
        lv = math.log2(max(0.2, min(3.4, sp)))
        col = C_TEAL if sp >= 1.0 else C_ROSE
        y0, y1 = (y(lv), mid) if sp >= 1.0 else (mid, y(lv))
        out.append(rect(x(i) - bw / 2, y0, bw, y1 - y0, col, rx=2,
                        extra=f'<title>#{r["idx"]}  {esc(r["shape"])}\n'
                              f'SME {r["sme"]:.3f} ms  ·  KML {r["kml"]:.3f} ms\n'
                              f'speedup {sp:.2f}x</title>'))
    # labels for extremes
    for r in sorted(rows, key=lambda r: r["kml"] / r["sme"])[:3]:
        sp = r["kml"] / r["sme"]
        out.append(text(x(r["idx"]), y(min(hi - 0.05, math.log2(sp))) + 14,
                        f"{sp:.1f}x", 10, C_ROSE, "middle", "700"))
    for r in sorted(rows, key=lambda r: r["kml"] / r["sme"], reverse=True)[:3]:
        sp = r["kml"] / r["sme"]
        out.append(text(x(r["idx"]), y(min(hi - 0.05, math.log2(sp))) - 6,
                        f"{sp:.1f}x", 10, C_TEAL, "middle", "700"))

    for i in range(0, len(rows), 4):
        out.append(text(x(i), H - B + 16, str(i), 10, C_AXIS))
    out.append(text(W - R, H - 8, "shape index", 10.5, C_AXIS, "end"))

    head = text(L, 20, title, 12, "#334155", "start", "700")
    summary = (f"合计时间比 {st['total_ratio']:.3f}（SME 快 {(1-st['total_ratio'])*100:.1f}%）"
               f" · 几何 {st['geo']:.3f} · 算数 {st['arith']:.3f} · 中位 {st['median']:.3f}"
               f" · 胜 {st['wins']}/{st['n']}")
    head += text(W - R, 20, summary, 10.5, "#64748b", "end")
    return svg(W, H, "".join(out) + head)


def chart_gflops():
    """Peak efficiency bar: achieved GEMM GFLOPS vs SVE-512 peak."""
    W, H = 620, 190
    L, T = 172, 34
    bw = W - L - 120
    out = []
    rows = [("SVE-512 峰值（16 核）", 1024, "#e2e8f0", "#94a3b8"),
            ("实测（最大形状）", 960, C_INDIGO, C_INDIGO)]
    for k, (name, v, col, tx) in enumerate(rows):
        yy = T + k * 52
        out.append(text(L - 12, yy + 18, name, 11.5, "#334155", "end", "600"))
        out.append(rect(L, yy + 4, bw * v / 1024.0, 24, col, rx=4))
        out.append(text(L + bw * v / 1024.0 + 8, yy + 21, f"{v} GFLOPS", 11.5, tx, "start", "700"))
    out.append(text(L, T + 130, "折算效率 94%", 14, C_INDIGO, "start", "700"))
    out.append(text(L + 116, T + 130, "（4,384,160²,384，GEMM 部分）", 10.5, "#64748b", "start"))
    return svg(W, H, "".join(out))


# ---------------------------------------------------------------- charts (part 2)

STEP_COLORS = [
    ("gemm", "GEMM", "#4338ca"),
    ("weight", "Weight", "#818cf8"),
    ("inx", "InXform", "#0f766e"),
    ("outx", "OutXform", "#2dd4bf"),
    ("tileext", "TileExt", "#b45309"),
    ("inscat", "InScat", "#f59e0b"),
    ("outgath", "OutGath", "#be123c"),
    ("outw", "OutWrite", "#fb7185"),
]


def chart_step_composition(steps):
    """8 个代表形状的 100% 堆叠条：步耗时占比。"""
    W = 1140
    RH = 34
    L, R, T = 214, 62, 40
    H = T + RH * len(steps) + 54
    iw = W - L - R
    out = []
    out.append(text(L, 20, "占 timing 总时间 S 的比例", 11, "#64748b", "start"))
    for i, s in enumerate(steps):
        y = T + i * RH
        tot = s["S"] or 1.0
        x = L
        # 灰色底（未列入 8 步的残差）
        out.append(rect(L, y + 6, iw, 16, "#f1f5f9", rx=2))
        for key, label, col in STEP_COLORS:
            wv = s[key] / tot * iw
            if wv <= 0.4:
                continue
            out.append(rect(x, y + 6, wv, 16, col, rx=1,
                            extra=f'<title>{esc(label)} {s[key]:.2f} ms'
                                  f'（{s[key]/tot*100:.0f}%）</title>'))
            x += wv
        out.append(text(L - 12, y + 17, short_shape(s["shape"]), 11.5, "#334155", "end", "600"))
        p = parse_shape(s["shape"])
        sub = f"#{s['idx']}"
        if p:
            sub += f" · {math.ceil(p[2] / 4) ** 2} tiles · IC {p[1]}"
        out.append(text(L - 12, y + 32, sub, 9.5, "#94a3b8", "end"))
        share = (s["gemm"] + s["weight"]) / tot * 100
        col = C_INDIGO if share >= 60 else ("#475569" if share >= 35 else C_TEAL)
        out.append(text(L + iw + 8, y + 18, f"计算 {share:.0f}%", 10.5, col, "start", "700"))
    # legend
    lx = L
    for key, label, col in STEP_COLORS:
        out.append(rect(lx, H - 32, 9, 9, col, rx=2))
        out.append(text(lx + 14, H - 24, label, 10, "#64748b", "start"))
        lx += 74
    out.append(text(W - R, H - 24, "「计算」= GEMM + Weight", 10, "#94a3b8", "end"))
    return svg(W, H, "".join(out))


def chart_pareff(rows):
    """并行效率 eff = (T1/16)/T16 vs tile 数（log2 横轴）。"""
    W, H = 700, 330
    L, R, T, B = 56, 24, 30, 52
    iw, ih = W - L - R, H - T - B
    lo, hi = math.log2(8), math.log2(4096)

    def x(t):
        return L + (math.log2(max(8, min(4096, t))) - lo) / (hi - lo) * iw

    def y(v):
        return T + (1.0 - v) * ih

    out = []
    for gv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(line(L, y(gv), W - R, y(gv), "#cbd5e1" if gv == 1.0 else C_GRID,
                        1.3 if gv == 1.0 else 1.0, "4 4" if gv == 1.0 else None))
        out.append(text(L - 8, y(gv) + 4, f"{gv:.1f}", 10, C_AXIS, "end"))


    for t in (16, 49, 100, 196, 400, 784, 1600, 3136):
        out.append(line(x(t), T, x(t), T + ih, C_GRID, 0.7))
        out.append(text(x(t), H - B + 16, str(t), 9.5, C_AXIS))

    for r in rows:
        v = r["eff"]
        col = C_TEAL if v >= 0.7 else (C_INDIGO_L if v >= 0.4 else C_ROSE)
        rad = 3.4 + min(4.0, math.sqrt(r["S"]) / 6.0)
        tip = (f'#{r["idx"]} {esc(short_shape(r["shape"]))}\n'
               f'tiles={r["tiles"]}  eff={v:.2f}  16T×={r["sp16"]:.1f}\n'
               f'S={r["S"]} ms  GEMM={r["gemm_pct"]:.0f}%')
        out.append(f'<circle cx="{x(r["tiles"]):.1f}" cy="{y(v):.1f}" r="{rad:.1f}" '
                   f'fill="{col}" fill-opacity="0.62" stroke="{col}" stroke-width="1">'
                   f'<title>{tip}</title></circle>')

    out.append(text(W - R - 4, y(1.0) + 14, "理想 = 1.0（16 线程线性扩展）", 9.5, "#94a3b8", "end"))
    out.append(text(W - R, H - 8, "tile 数（对数轴）· 点径 ∝ √S", 10.5, C_AXIS, "end"))
    out.append(text(L, 18, "16 线程并行效率", 11.5, "#334155", "start", "700"))
    legend = (rect(W - R - 190, 10, 9, 9, C_TEAL, 2) + text(W - R - 177, 18, "≥0.70", 9.5, "#475569", "start")
              + rect(W - R - 128, 10, 9, 9, C_INDIGO_L, 2) + text(W - R - 115, 18, "0.40–0.70", 9.5, "#475569", "start")
              + rect(W - R - 44, 10, 9, 9, C_ROSE, 2) + text(W - R - 31, 18, "<0.40", 9.5, "#475569", "start"))
    return svg(W, H, "".join(out) + legend)


def chart_small_gemm():
    """小 GEMM 的核效率崩塌：同一 OpenBLAS 内核在不同 n 上的吞吐。"""
    rows = [("1600×384×384", 134, C_INDIGO),
            ("100×192×192", 140, C_INDIGO),
            ("4×512×512", 10.3, C_ROSE)]
    W, H = 620, 190
    L, T = 148, 34
    bw = W - L - 132
    out = []
    for k, (name, v, col) in enumerate(rows):
        yy = T + k * 44
        out.append(text(L - 12, yy + 17, name, 11.5, "#334155", "end", "600"))
        out.append(rect(L, yy + 5, bw * v / 140.0, 20, col, rx=3,
                        extra=f'<title>{name}: {v} GFLOPS</title>'))
        out.append(text(L + bw * v / 140.0 + 8, yy + 20, f"{v} GFLOPS", 11.5, col, "start", "700"))
    out.append(text(L, T + 150, "n=4 时只有 ~7% 的核效率：单次调用开销 + 分块失效", 11.5, C_ROSE, "start", "600"))
    return svg(W, H, "".join(out))


def chart_metric_bars(items, lo, hi, ticks, note):
    """四个口径的横向对比条（同一个比较，四种平均）。"""
    W, H = 560, 60 + 42 * len(items)
    L, T = 132, 34
    bw = W - L - 96

    def x(v):
        return L + (v - lo) / (hi - lo) * bw

    out = []
    for tv in ticks:
        out.append(line(x(tv), T - 6, x(tv), T + 34 * len(items), C_GRID))
        out.append(text(x(tv), T - 12, f"{tv:g}x", 9.5, C_AXIS))
    for i, (name, v, hint) in enumerate(items):
        yy = T + i * 34
        out.append(text(L - 12, yy + 16, name, 11.5, "#334155", "end", "600"))
        col = C_TEAL if v >= 1.0 else C_ROSE
        out.append(rect(min(x(v), x(1.0)), yy + 4, abs(x(v) - x(1.0)), 18, col, rx=3,
                        extra=f'<title>{name}: {v:.3f}x</title>'))
        out.append(text(max(x(v), x(1.0)) + 8, yy + 18, f"{v:.3f}x", 11, col, "start", "700"))
    out.append(text(L, H - 8, note, 10, "#94a3b8", "start"))
    return svg(W, H, "".join(out))


def chart_sections():
    """59 形状的分节构成。"""
    rows = [("Case 0-8（历史 9 case）", 9, C_INDIGO),
            ("ResNet50 3×3 s1", 12, C_INDIGO_L),
            ("VGG16 3×3 s1", 10, "#0f766e"),
            ("Lightweight / 小图", 4, "#2dd4bf"),
            ("Large batch", 4, "#b45309"),
            ("Fixed-overhead 隔离", 4, "#f59e0b"),
            ("tiles × IC 网格", 16, "#be123c")]
    W, H = 640, 60 + 30 * len(rows)
    L, T = 210, 34
    bw = W - L - 70
    out = []
    for i, (name, c, col) in enumerate(rows):
        yy = T + i * 30
        out.append(text(L - 12, yy + 16, name, 11.5, "#334155", "end", "600"))
        out.append(rect(L, yy + 5, bw * c / 16.0, 16, col, rx=3))
        out.append(text(L + bw * c / 16.0 + 8, yy + 18, f"{c}", 11.5, col, "start", "700"))
    out.append(text(L, H - 10, "合计 59 行（4 个格点与 Case 0-8 刻意重叠，用于噪声交叉验证）",
                    10, "#94a3b8", "start"))
    return svg(W, H, "".join(out))


def chart_shape_grid(ob_rows):
    """tiles × IC 网格热力图：颜色 = 对 OpenBLAS 的加速比。"""
    tiles_list = [25, 100, 400, 1600]
    ic_list = [48, 96, 192, 384]
    lut = {}
    for r in ob_rows:
        p = parse_shape(r["shape"])
        if not p:
            continue
        n, ic, ih, oc = p
        if n != 4 or oc != ic:
            continue
        t = math.ceil(ih / 4) ** 2
        lut.setdefault((t, ic), []).append(r)

    W, H = 700, 340
    L, T = 96, 58
    cw, ch = 132, 56
    out = []
    out.append(text(L, 26, "颜色 = 对 OpenBLAS 的加速比（同一作业内 A/B，t16）", 11, "#64748b", "start"))
    for j, ic in enumerate(ic_list):
        out.append(text(L + j * cw + cw / 2, T - 12, f"IC={ic}", 11, "#334155"))
    for i, t in enumerate(tiles_list):
        out.append(text(L - 12, T + i * ch + ch / 2 + 4, f"{t} tiles", 11.5, "#334155", "end", "600"))
        ih = int(round(math.sqrt(t) * 4))
        out.append(text(L - 12, T + i * ch + ch / 2 + 18, f"{ih}²", 9.5, "#94a3b8", "end"))
        for j, ic in enumerate(ic_list):
            rows = lut.get((t, ic), [])
            x0, y0 = L + j * cw + 4, T + i * ch + 4
            if not rows:
                out.append(rect(x0, y0, cw - 8, ch - 8, "#f1f5f9", rx=6))
                out.append(text(x0 + (cw - 8) / 2, y0 + ch / 2, "—", 11, "#cbd5e1"))
                continue
            r = rows[-1]
            sp = r["base"] / r["ours"]
            k = min(1.0, max(0.0, (sp - 1.0) / 2.4))
            col = f"rgb({int(224 - 155*k)},{int(231 - 175*k)},{int(255 - 60*k)})"
            tip = (f'#{r["idx"]} {esc(short_shape(r["shape"]))}\n'
                   f'ours {r["ours"]:.2f} ms · OB {r["base"]:.2f} ms · {sp:.2f}x')
            if len(rows) > 1:
                tip += "\n重复格点：" + " / ".join(f'#{q["idx"]} {q["base"]/q["ours"]:.2f}x' for q in rows)
            out.append(rect(x0, y0, cw - 8, ch - 8, col, rx=6,
                            extra=f'<title>{tip}</title>'))
            ink = "#ffffff" if k > 0.55 else "#1e1b4b"
            out.append(text(x0 + (cw - 8) / 2, y0 + ch / 2 + 2, f"{sp:.2f}x", 13, ink, "middle", "700"))
            out.append(text(x0 + (cw - 8) / 2, y0 + ch / 2 + 18, f'#{r["idx"]}', 9, ink))
    out.append(text(L, H - 14, "越靠右下「tile 多 × 通道深」，加速比越高；左上角是小空间浅通道，最接近 1.0x。",
                    10, "#64748b", "start"))
    return svg(W, H, "".join(out))


# ---------------------------------------------------------------- diagrams

def diagram_tiles():
    """6x6 input tiles -> 4x4 output, stride 4, showing overlap."""
    W, H = 760, 300
    cell = 30
    ox, oy = 70, 54          # input grid origin
    out = []
    out.append(text(ox, 26, "输入特征图（每格 = 1 像素 × IC 通道）", 11.5, "#334155", "start", "600"))

    # input grid 12x6 cells
    cols, rows = 12, 6
    for j in range(rows):
        for i in range(cols):
            x, y = ox + i * cell, oy + j * cell
            in_t0 = i < 6 and j < 6
            in_t1 = 4 <= i < 10 and j < 6
            fill = "#eef2ff"
            if in_t0 and in_t1:
                fill = "#c7d2fe"
            elif in_t0:
                fill = "#e0e7ff"
            elif in_t1:
                fill = "#ede9fe"
            out.append(rect(x, y, cell - 1, cell - 1, fill, rx=2))
    # tile outlines
    out.append(f'<rect x="{ox}" y="{oy}" width="{6*cell-1}" height="{6*cell-1}" fill="none" '
               f'stroke="{C_INDIGO}" stroke-width="2" rx="3"/>')
    out.append(f'<rect x="{ox+4*cell}" y="{oy}" width="{6*cell-1}" height="{6*cell-1}" fill="none" '
               f'stroke="#7c3aed" stroke-width="2" stroke-dasharray="5 3" rx="3"/>')
    out.append(text(ox + 3 * cell, oy + 6 * cell + 20, "tile 0：6×6 输入", 11, C_INDIGO, "middle", "700"))
    out.append(text(ox + 7 * cell, oy + 6 * cell + 20, "tile 1：6×6 输入", 11, "#7c3aed", "middle", "700"))
    out.append(text(ox + 6 * cell, oy - 10, "步长 4 → 相邻 tile 重叠 2 列", 10.5, "#64748b", "middle"))

    # arrow
    ax = ox + 12 * cell + 26
    out.append(f'<path d="M{ax} {oy+80} h46" stroke="#94a3b8" stroke-width="2" '
               f'marker-end="url(#ar)"/>')
    out.append(f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
               f'markerWidth="7" markerHeight="7" orient="auto">'
               f'<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>')

    # output tiles 4x4 each
    ox2 = ax + 60
    for t in range(2):
        bx = ox2 + t * 4 * cell + t * 8
        for j in range(4):
            for i in range(4):
                out.append(rect(bx + i * cell, oy + j * cell, cell - 1, cell - 1,
                                "#ccfbf1" if t == 0 else "#f5f3ff", rx=2))
        out.append(f'<rect x="{bx}" y="{oy}" width="{4*cell-1}" height="{4*cell-1}" fill="none" '
                   f'stroke="{C_TEAL if t==0 else "#7c3aed"}" stroke-width="2" rx="3"/>')
        out.append(text(bx + 2 * cell, oy + 4 * cell + 20, f"输出 tile {t}：4×4", 11,
                        C_TEAL if t == 0 else "#7c3aed", "middle", "700"))
    out.append(text(W - 26, 26, "输出（4×4 像素 × OC 通道）", 11.5, "#334155", "end", "600"))
    out.append(text(W / 2, H - 12,
                    "F(4,4,3,3)：一个 6×6 输入 tile 经 36 次乘法产出 16 个输出像素；"
                    "相邻 tile 重叠 2 列/行以覆盖 3×3 卷积的感受野。",
                    11, "#64748b", "middle"))
    return svg(W, H, "".join(out), cls="diagram")


def diagram_pipeline():
    """Three-phase flow with shapes."""
    W, H = 1180, 250
    boxes = [
        ("Phase 1 · 输入变换", "U = Bᵀ · d · B", "6×6×IC  →  6×6×IC", "#eef2ff", C_INDIGO,
         "每 tile 一次\n通道向量化（SVE 谓词）"),
        ("Phase 2 · Winograd 域 GEMM", "M[ts] = U[ts] · V[ts]ᵀ", "36 个切片，M×N×K = tile × OC × IC",
         "#ecfdf5", C_TEAL, "按 ts 分线程 / 每线程折叠成\n一次 batched 调用"),
        ("Phase 3 · 输出变换", "f = Aᵀ · M · A + bias", "6×6×OC  →  4×4×OC", "#fff7ed", C_AMBER,
         "每 tile 一次\n含 bias 与 clamp(ReLU)"),
    ]
    out = []
    bw, bh = 330, 150
    gap = (W - 3 * bw) / 4
    for k, (title, formula, shape, bg, accent, note) in enumerate(boxes):
        x = gap + k * (bw + gap)
        y = 42
        out.append(rect(x, y, bw, bh, bg, rx=12, extra=f'stroke="{accent}" stroke-opacity="0.25"'))
        out.append(rect(x, y, 4, bh, accent, rx=2))
        out.append(text(x + 18, y + 26, title, 12.5, accent, "start", "700"))
        out.append(text(x + 18, y + 56, formula, 12.5, "#0f172a", "start", "600",
                        family="ui-monospace, SFMono-Regular, Menlo, monospace"))
        out.append(text(x + 18, y + 80, shape, 11, "#475569", "start"))
        for li, ln in enumerate(note.split("\n")):
            out.append(text(x + 18, y + 106 + li * 16, ln, 10.5, "#64748b", "start"))
        if k < 2:
            ax = x + bw + 6
            out.append(f'<path d="M{ax} {y+bh/2} h{gap-16}" stroke="#94a3b8" '
                       f'stroke-width="2" marker-end="url(#ar2)"/>')
    out.append('<defs><marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="7" markerHeight="7" orient="auto">'
               '<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>')
    out.append(text(W / 2, 24, "一次前向的三段流水（每个 batch 重复一次）", 12.5, "#334155", "middle", "700"))
    out.append(text(W / 2, H - 16,
                    "权重变换 V = G·g·Gᵀ/576 只在首次调用做一次，之后跨调用缓存 —— "
                    "这是 Winograd 把「核窗口累加」的代价在权重侧一次付清的关键。",
                    11, "#64748b", "middle"))
    return svg(W, H, "".join(out), cls="diagram")


def diagram_multiply():
    """Multiplication reduction: direct vs Winograd."""
    W, H = 560, 210
    L, T = 128, 46
    bw = W - L - 100
    out = []
    out.append(text(L, 24, "每个 4×4 输出块（16 像素）的乘法次数", 11.5, "#334155", "start", "600"))
    rows = [("直接卷积 3×3", 144, "#cbd5e1", "#475569", "16 像素 × 9"),
            ("Winograd F(4,4)", 36, C_INDIGO, C_INDIGO, "6×6 = 36")]
    for k, (name, v, col, tx, note) in enumerate(rows):
        yy = T + k * 56
        out.append(text(L - 12, yy + 20, name, 11.5, "#334155", "end", "600"))
        out.append(rect(L, yy + 6, bw * v / 144, 26, col, rx=4))
        out.append(text(L + bw * v / 144 + 8, yy + 24, f"{v}  ({note})", 11.5, tx, "start", "700"))
    out.append(text(L, T + 132, "乘法数 −75%", 15, C_INDIGO, "start", "700"))
    out.append(text(L + 108, T + 132, "→ 每输出像素 9 次降到 2.25 次", 11.5, "#475569", "start"))
    return svg(W, H, "".join(out))


def diagram_bt_sparsity():
    """B^T zero pattern for F(4,4)."""
    bt = [[4, 0, -5, 0, 1, 0],
          [0, -4, -4, 1, 1, 0],
          [0, 4, -4, -1, 1, 0],
          [0, -2, -1, 2, 1, 0],
          [0, 2, -1, -2, 1, 0],
          [0, 4, 0, -5, 0, 1]]
    cell = 34
    ox, oy = 46, 56
    W, H = 420, 344
    out = []
    out.append(text(ox, 26, "Bᵀ（F(4,4,3,3) 输入变换矩阵）", 11.5, "#334155", "start", "600"))
    zeros = 0
    for j in range(6):
        for i in range(6):
            v = bt[j][i]
            x, y = ox + i * cell, oy + j * cell
            if v == 0:
                zeros += 1
                fill, fg = "#f1f5f9", "#cbd5e1"
            else:
                fill, fg = "#e0e7ff", C_INDIGO
            out.append(rect(x, y, cell - 2, cell - 2, fill, rx=3))
            out.append(text(x + cell / 2 - 1, y + cell / 2 + 4, "0" if v == 0 else
                            (str(v) if v > 0 else str(v)), 11, fg, "middle", "600",
                            family="ui-monospace, Menlo, monospace"))
    out.append(text(ox, oy + 6 * cell + 26, f"36 个系数中 {zeros} 个为零（{zeros/36*100:.0f}%）",
                    11, "#475569", "start"))
    out.append(text(ox, oy + 6 * cell + 46,
                    "零系数占比高 → 专用展开内核可跳过近一半运算量", 10.5, "#64748b", "start"))
    out.append(text(ox, oy + 6 * cell + 66,
                    "（当前实现用运行时判断跳过；ACL 在编译期直接消去）", 10.5, "#94a3b8", "start"))
    return svg(W, H, "".join(out), cls="diagram")


def diagram_dataflow():
    """End-to-end dataflow with layout conversions."""
    W, H = 1180, 210
    out = []
    stages = [
        ("NCHW 输入", "[N][IC][IH][IW]", "#f8fafc", "#94a3b8"),
        ("→ NHWC", "cache-blocked\n转置", "#f1f5f9", "#64748b"),
        ("Phase 1/2/3", "NHWC 计算核心", "#eef2ff", C_INDIGO),
        ("→ NCHW", "cache-blocked\n转置", "#f1f5f9", "#64748b"),
        ("NCHW 输出", "[N][OC][OH][OW]", "#f8fafc", "#94a3b8"),
    ]
    bw, bh = 178, 84
    gap = (W - 5 * bw) / 6
    for k, (name, sub, bg, accent) in enumerate(stages):
        x = gap + k * (bw + gap)
        y = 58
        strong = k == 2
        out.append(rect(x, y, bw, bh, bg, rx=10,
                        extra=f'stroke="{accent}" stroke-width="{2 if strong else 1}" '
                              f'stroke-opacity="{0.55 if strong else 0.4}"'))
        out.append(text(x + bw / 2, y + 30, name, 12, accent if strong else "#334155",
                        "middle", "700"))
        for li, ln in enumerate(sub.split("\n")):
            out.append(text(x + bw / 2, y + 52 + li * 15, ln, 10.5, "#64748b"))
        if k < 4:
            ax = x + bw + 4
            out.append(f'<path d="M{ax} {y+bh/2} h{gap-10}" stroke="#94a3b8" stroke-width="1.8" '
                       f'marker-end="url(#ar3)"/>')
    out.append('<defs><marker id="ar3" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="7" markerHeight="7" orient="auto">'
               '<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>')
    out.append(text(W / 2, 30, "端到端数据流：计算核心只认 NHWC，NCHW 由转换包装承担", 12.5,
                    "#334155", "middle", "700"))
    out.append(text(W / 2, H - 26,
                    "在无 L3、L2 仅 768 KB/核的机器上，NCHW 的跨步 tile 提取有约 16× 缓存行放大；"
                    "顺序转置的代价远低于它。", 11, "#64748b", "middle"))
    return svg(W, H, "".join(out), cls="diagram")


def diagram_openmp():
    """Single OpenMP region timeline."""
    W, H = 1180, 250
    L, R, T = 132, 40, 58
    iw = W - L - R
    lanes = ["线程 0", "线程 1", "…", "线程 N-1"]
    phases = [("权重变换", 0.10, "#c7d2fe"), ("Phase 1", 0.24, "#a5b4fc"),
              ("barrier", 0.02, "#0f172a"), ("Phase 2 (GEMM)", 0.30, "#5eead4"),
              ("barrier", 0.02, "#0f172a"), ("Phase 3", 0.22, "#fcd34d"),
              ("barrier", 0.02, "#0f172a")]
    out = []
    out.append(text(W / 2, 26, "单个 #pragma omp parallel 区域覆盖全部阶段（每 batch 一次）",
                    12.5, "#334155", "middle", "700"))
    y0 = T
    lane_h = 30
    # phase bands
    xcur = L
    for name, frac, col in phases:
        w = frac * iw
        if name == "barrier":
            out.append(line(xcur + w / 2, y0 - 6, xcur + w / 2, y0 + lane_h * len(lanes) + 4,
                            "#0f172a", 1.2, "3 3"))
        else:
            out.append(text(xcur + w / 2, y0 - 10, name, 10.5, "#475569", "middle", "600"))
        xcur += w
    for li, lane in enumerate(lanes):
        yy = y0 + li * lane_h
        out.append(text(L - 12, yy + 18, lane, 10.5, "#64748b", "end"))
        xcur = L
        for name, frac, col in phases:
            w = frac * iw
            if name != "barrier":
                out.append(rect(xcur + 1, yy + 4, w - 2, lane_h - 10, col, rx=3,
                                extra=f'opacity="0.9"'))
            xcur += w
    out.append(text(L, y0 + lane_h * len(lanes) + 26,
                    "1 次 fork/join，2 个显式 barrier（输入→GEMM、GEMM→输出），Phase 3 用 nowait",
                    11, "#475569", "start"))
    out.append(text(L, y0 + lane_h * len(lanes) + 46,
                    "对比：把权重变换放进独立 parallel 区域会多一次 fork/join，小形状反而慢约 7%",
                    10.5, "#94a3b8", "start"))
    return svg(W, H, "".join(out), cls="diagram")


def diagram_za_layout():
    """SME ZA tiles: rows = 16 OC, cols = 16 tiles (2x2 layout)."""
    W, H = 900, 470
    cell = 9.5
    blk = 16 * cell
    ox, oy = 92, 78
    gx, gy = 66, 62
    out = []
    out.append(text(24, 26, "ZA 的 4 个 tile：行 = 16 个 OC，列 = 16 个 tile", 12.5,
                    "#334155", "start", "700"))
    for t in range(4):
        r, c = divmod(t, 2)
        x = ox + c * (blk + gx)
        y = oy + r * (blk + gy)
        out.append(rect(x, y, blk, blk, "#eef2ff" if t % 2 == 0 else "#f5f3ff", rx=4))
        for i in range(0, 17):
            out.append(line(x + i * cell, y, x + i * cell, y + blk, "#c7d2fe", 0.5))
            out.append(line(x, y + i * cell, x + blk, y + i * cell, "#c7d2fe", 0.5))
        out.append(text(x - 8, y + blk / 2 + 4, f"za{t}", 11, C_INDIGO, "end", "700",
                        family="ui-monospace, Menlo, monospace"))
        out.append(text(x + blk / 2, y - 8, f"OC 块 {t}", 10, "#7c3aed", "middle", "600"))
        out.append(text(x + blk / 2, y + blk + 14, "16 tile →", 9, "#94a3b8", "middle"))
    out.append(text(ox + blk / 2, oy - 30, "16 列 = 16 个 tile", 9.5, "#94a3b8", "middle"))
    out.append(text(ox - 26, oy + blk + 22, "↑ 16 行 = 16 OC", 9.5, "#94a3b8", "start"))

    nx = ox + 2 * blk + gx + 44
    notes = [
        ("一行 = 一条 fmopa", "za[r][c] += Zm[r] · Zn[c]", C_INDIGO),
        ("一条 fmopa = 16×16 次 FMA", "SVE-512 下 256 次乘加", C_TEAL),
        ("drain 一次 = 64 B", "16 个连续 OC（M_buf 一行的一段）", C_AMBER),
    ]
    for k, (t1_, t2_, col) in enumerate(notes):
        y = oy + k * 84
        out.append(rect(nx, y, 3, 46, col, rx=2))
        out.append(text(nx + 14, y + 15, t1_, 11.5, "#334155", "start", "700"))
        out.append(text(nx + 14, y + 34, t2_, 10.5, "#64748b", "start"))
    yb = oy + 3 * 84 + 6
    out.append(text(nx, yb, "4 个 tile 并行 → 一次窗口覆盖", 10.5, "#475569", "start", "600"))
    out.append(text(nx, yb + 20, "16 tile × 64 OC", 11.5, C_INDIGO, "start", "700",
                    family="ui-monospace, Menlo, monospace"))
    return svg(W, H, "".join(out), cls="diagram")


def diagram_drain():
    """str za slice -> (tile, oc block) mapping and landing address."""
    W, H = 720, 356
    out = []
    out.append(text(24, 26, "drain：64 条 str za 切片直接落进 M_buf 的行", 12.5,
                    "#334155", "start", "700"))
    ox, oy = 60, 66
    cell = 30
    for v in range(64):
        j, i = divmod(v, 8)
        x, y = ox + i * cell, oy + j * (cell + 2)
        blk, tile = v & 3, v >> 2
        col = ["#e0e7ff", "#ccfbf1", "#fef3c7", "#fee2e2"][blk]
        out.append(rect(x, y, cell - 2, cell - 2, col, rx=3,
                        extra=f'<title>slice {v} → za{blk} 的第 {tile} 列 → '
                              f'M_buf[ts][tile={tile}][oc {blk*16}..{blk*16+15}]</title>'))
        out.append(text(x + cell / 2 - 1, y + cell / 2 + 3, str(v), 8.5, "#475569", "middle"))
    out.append(text(ox, oy + 8 * (cell + 2) + 22,
                    "slice 编号 v：oc_block = v & 3，tile = v >> 2（64 条覆盖 16 tile × 4 个 OC 块）",
                    11, "#475569", "start"))

    # address formula
    fx = ox + 8 * cell + 54
    out.append(rect(fx, 62, 300, 122, "#0f172a", rx=10))
    out.append(text(fx + 18, 88, "目标地址（无需转置）", 10.5, "#94a3b8", "start"))
    out.append(text(fx + 18, 116, "M0 + (ts*16 + tile) * OC", 11.5, "#e2e8f0", "start",
                    family="ui-monospace, Menlo, monospace"))
    out.append(text(fx + 18, 138, "     + oc_block * 16", 11.5, "#e2e8f0", "start",
                    family="ui-monospace, Menlo, monospace"))
    out.append(text(fx + 18, 166, "→ 每个 slice 都是一段 16 连续 OC", 10.5, "#5eead4", "start"))
    out.append(text(fx, 210, "若把 OC 放在 ZA 的列方向，", 10.5, "#64748b", "start"))
    out.append(text(fx, 228, "drain 出来的是跨行数据，必须再转置一次。", 10.5, "#64748b", "start"))
    out.append(text(fx, 252, "这是「行 = OC、列 = tile」的直接理由。", 10.5, C_INDIGO, "start", "600"))
    return svg(W, H, "".join(out), cls="diagram")


def diagram_ab_protocol():
    """跨作业漂移 vs 同作业 A/B。"""
    W, H = 980, 330
    out = []
    # panel 1
    out.append(rect(24, 18, 452, 128, "#fff1f2", rx=10))
    out.append(text(44, 46, "跨作业比较 · 作废", 14, C_ROSE, "start", "700"))
    out.append(text(44, 66, "node03 不同作业之间存在 3–7x 的性能态差", 11, "#9f1239", "start"))
    xs = [44, 168, 292]
    vals = [321.9, 490.7, 690.0]
    for i, (xv, v) in enumerate(zip(xs, vals)):
        bh = v / 690 * 54
        out.append(rect(xv, 108 + (54 - bh), 96, bh, "#fda4af", rx=3))
        out.append(text(xv + 48, 176, f"{v:.0f} ms", 11, C_ROSE, "middle", "700"))
        out.append(text(xv + 48, 192, f"作业 {i+1}", 10, "#9f1239"))
    out.append(line(40, 200, 452, 200, C_ROSE, 1.2, "3 3"))

    # panel 2
    out.append(rect(504, 18, 452, 250, "#ecfdf5", rx=10))
    out.append(text(524, 46, "同一作业内 A/B · 有效", 14, C_TEAL, "start", "700"))
    out.append(text(524, 66, "同一台机器、同一时段、交替测量两次", 11, "#115e59", "start"))
    rows = [("本项目", 321.9, C_INDIGO), ("OpenBLAS", 804.9, "#94a3b8")]
    for k, (name, v, col) in enumerate(rows):
        yy = 92 + k * 58
        out.append(text(592, yy + 20, name, 11.5, "#334155", "end", "600"))
        out.append(rect(604, yy + 6, 300 * v / 804.9, 24, col, rx=4))
        out.append(text(610 + 300 * v / 804.9, yy + 23, f"{v:.1f} ms", 11.5, col, "start", "700"))
    out.append(text(524, 232, "比值 2.50x 可信", 15, C_TEAL, "start", "700"))
    out.append(text(690, 232, "→ 只有这一种比较进入文档与图表", 11, "#115e59", "start"))

    out.append(text(W / 2, H - 18, "集群 920F 实测：本项目的 321.9 ms 与 690 ms 都是真实读数，差异来自机器性能态而非代码。",
                    10.5, "#64748b"))
    return svg(W, H, "".join(out), cls="diagram")


def diagram_phases():
    """三阶段演进路线。"""
    W, H = 1040, 300
    out = []
    stages = [
        ("第一阶段", "低风险 · 先做", C_TEAL, "#ecfdf5",
         ["SME 打包消掉 16×16 转置", "OC 尾块选 tile 数", "大 IC 通道分块", "V 缓存加 LRU"],
         "59 形状无回退；SME 合计比 0.896 → ≤0.85"),
        ("第二阶段", "结构性改造", C_INDIGO, "#eef2ff",
         ["小形状 hybrid 调度", "drain 与 K 循环交错", "形状感知选后端"],
         "小形状 ≥0.95x；SME 在 90% 形状不劣于 KML"),
        ("第三阶段", "探索", C_AMBER, "#fffbeb",
         ["F(4,4) 专用展开变换", "bf16 / fp16 外积"],
         "变换占比降 30%；低精度误差在 1e-4 门限内"),
    ]
    bw = 316
    for i, (name, tag, col, bg, items, crit) in enumerate(stages):
        x = 22 + i * 338
        out.append(rect(x, 26, bw, 232, bg, rx=12, extra=f'stroke="{col}" stroke-width="1.4"'))
        out.append(text(x + 20, 56, name, 14, col, "start", "700"))
        out.append(text(x + 20, 76, tag, 11, "#64748b", "start"))
        for j, it in enumerate(items):
            yy = 104 + j * 26
            out.append(f'<circle cx="{x+26}" cy="{yy-4}" r="2.6" fill="{col}"/>')
            out.append(text(x + 38, yy, it, 11.5, "#334155", "start"))
        out.append(line(x + 18, 216, x + bw - 18, 216, "#e2e8f0", 1))
        out.append(text(x + 20, 238, crit, 10.5, "#475569", "start", "600"))
        if i < 2:
            out.append(f'<path d="M{x+bw+6} 142 l10 0 m-3 -4 l4 4 -4 4" stroke="#94a3b8" '
                       f'stroke-width="1.6" fill="none"/>')
    return svg(W, H, "".join(out), cls="diagram")


def diagram_metric_choice():
    """四种平均各自回答什么问题。"""
    W, H = 860, 250
    rows = [
        ("合计时间比", "Σ 慢 / Σ 快", "整套网络跑完谁更快", "汇报主指标", C_INDIGO),
        ("几何平均", "exp(mean(ln x))", "典型一层快多少（乘性）", "算法对比常用", C_TEAL),
        ("算数平均", "mean(x)", "平均每个形状快多少", "偏袒大加速比", C_AMBER),
        ("中位数", "median(x)", "一半形状至少快多少", "抗极端值", C_ROSE),
    ]
    L, T = 176, 40
    out = []
    out.append(text(L - 12, 22, "口径", 10.5, "#94a3b8", "end", "700"))
    out.append(text(L + 6, 22, "定义", 10.5, "#94a3b8", "start", "700"))
    out.append(text(L + 250, 22, "回答的问题", 10.5, "#94a3b8", "start", "700"))
    out.append(text(W - 24, 22, "使用建议", 10.5, "#94a3b8", "end", "700"))
    for i, (name, defin, q, use, col) in enumerate(rows):
        yy = T + i * 48
        out.append(rect(L - 150, yy, 800, 40, "#f8fafc" if i % 2 else "#ffffff", rx=6))
        out.append(text(L - 138, yy + 25, name, 12, col, "start", "700"))
        out.append(f'<text x="{L+6}" y="{yy+25}" font-size="11.5" fill="#475569" '
                   f'font-family="ui-monospace, Menlo, monospace">{esc(defin)}</text>')
        out.append(text(L + 250, yy + 25, q, 11.5, "#334155", "start"))
        out.append(text(W - 34, yy + 25, use, 11, "#64748b", "end"))
    out.append(text(L - 150, H - 8, "同一个比较的四个数通常不同：本仓库 vs OpenBLAS 为 2.50 / 1.99 / 2.17 / 1.77，"
                                   "差异本身就是信息（分布右偏）。", 10.5, "#64748b", "start"))
    return svg(W, H, "".join(out), cls="diagram")


def code_block(code, lang="asm"):
    return f'<pre class="code"><code>{esc(code)}</code></pre>'


# ---------------------------------------------------------------- css

CSS = """
*, *::before, *::after { box-sizing: border-box; }
:root {
  --ink: #0b1220; --slate: #334155; --muted: #64748b; --faint: #94a3b8;
  --line: #e5e9f2; --bg: #f7f8fc; --card: #ffffff;
  --indigo: #4338ca; --indigo-2: #6366f1; --indigo-soft: #eef2ff;
  --teal: #0f766e; --teal-soft: #ecfdf5;
  --amber: #b45309; --amber-soft: #fffbeb;
  --rose: #be123c; --rose-soft: #fff1f2;
  --green: #15803d;
  --radius: 14px;
}
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--slate);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
               "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  font-size: 15px; line-height: 1.7;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
}
.page { max-width: 1220px; margin: 0 auto; padding: 40px 32px 96px; }
a { color: var(--indigo); text-decoration: none; }
a:hover { text-decoration: underline; }
.num { font-variant-numeric: tabular-nums;
       font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }

/* nav */
.nav { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 34px; }
.nav a {
  font-size: 12.5px; padding: 6px 12px; border-radius: 999px;
  background: #fff; border: 1px solid var(--line); color: var(--muted);
  transition: .15s ease;
}
.nav a:hover { border-color: #c7d2fe; color: var(--indigo); text-decoration: none; }
.nav a.on { background: var(--indigo); border-color: var(--indigo); color: #fff; }

/* header */
.eyebrow {
  font-size: 11.5px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--indigo); font-weight: 700; margin-bottom: 12px;
}
h1 { font-size: 32px; line-height: 1.28; margin: 0 0 12px; color: var(--ink); font-weight: 750; letter-spacing: -.01em; }
.lede { font-size: 15.5px; color: var(--muted); max-width: 78ch; margin: 0 0 22px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 12.5px; color: var(--faint); margin-bottom: 44px; }
.meta b { color: var(--slate); font-weight: 600; }

/* sections */
section { margin: 0 0 40px; }
h2 {
  font-size: 19px; color: var(--ink); font-weight: 700; margin: 0 0 6px;
  display: flex; align-items: baseline; gap: 10px;
}
h2 .idx {
  font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: var(--indigo);
  background: var(--indigo-soft); border-radius: 6px; padding: 2px 7px; font-weight: 700;
}
.sub { color: var(--muted); font-size: 13.5px; margin: 0 0 18px; max-width: 88ch; }
h3 { font-size: 15px; color: var(--ink); margin: 26px 0 10px; font-weight: 650; }

.card {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 24px 26px; box-shadow: 0 1px 2px rgba(15,23,42,.03), 0 18px 40px -28px rgba(15,23,42,.22);
}
.card + .card { margin-top: 16px; }
.card > h3:first-child { margin-top: 0; }

/* metric strip */
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
.metric {
  background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px;
}
.metric .k { font-size: 11.5px; color: var(--faint); letter-spacing: .04em; text-transform: uppercase; }
.metric .v { font-size: 27px; font-weight: 750; color: var(--ink); line-height: 1.25; margin-top: 6px; letter-spacing: -.02em; }
.metric .n { font-size: 12px; color: var(--muted); margin-top: 4px; }
.metric.hl { border-color: #c7d2fe; background: linear-gradient(180deg, #fbfbff, #fff); }
.metric.hl .v { color: var(--indigo); }

/* tables */
.tbl { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.tbl th, .tbl td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
.tbl th { font-size: 11.5px; letter-spacing: .05em; text-transform: uppercase; color: var(--faint); font-weight: 700; }
.tbl tr:last-child td { border-bottom: 0; }
.tbl td.c, .tbl th.c { text-align: center; }
.tbl td.r, .tbl th.r { text-align: right; }
.tbl tbody tr:hover { background: #fafbff; }
.tbl .em { color: var(--indigo); font-weight: 700; }
.tbl .warn { color: var(--rose); font-weight: 700; }

.tag {
  display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px;
  border-radius: 999px; white-space: nowrap;
}
.tag.same { background: var(--teal-soft); color: var(--teal); }
.tag.diff { background: var(--amber-soft); color: var(--amber); }
.tag.gap  { background: var(--rose-soft); color: var(--rose); }
.tag.n    { background: #f1f5f9; color: var(--muted); }

/* chart */
.chart { display: block; width: 100%; height: auto; }
.diagram { display: block; width: 100%; height: auto; }
.chart-wrap { overflow-x: auto; }

/* code */
pre.code {
  background: #0b1220; color: #dbe4ff; border-radius: 12px; padding: 18px 20px;
  overflow-x: auto; font-size: 12.5px; line-height: 1.65; margin: 14px 0 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
pre.code .c { color: #7c8db5; }

/* callout */
.callout {
  border-left: 3px solid var(--amber); background: var(--amber-soft);
  border-radius: 0 10px 10px 0; padding: 16px 20px; font-size: 13.5px; color: #78350f;
  margin: 16px 0 0;
}
.callout.rose { border-color: var(--rose); background: var(--rose-soft); color: #881337; }
.callout.indigo { border-color: var(--indigo); background: var(--indigo-soft); color: #312e81; }
.callout.teal { border-color: var(--teal); background: var(--teal-soft); color: #134e4a; }
.callout b { font-weight: 700; }

/* two-column */
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; align-items: start; }
.cols.tight { gap: 12px; }

/* footer */
.foot { margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--faint); font-size: 12px; }

/* index cards */
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
a.tile {
  display: block; background: #fff; border: 1px solid var(--line); border-radius: var(--radius);
  padding: 22px 24px; color: inherit; transition: .16s ease; height: 100%;
  box-shadow: 0 1px 2px rgba(15,23,42,.03);
}
a.tile:hover { text-decoration: none; border-color: #c7d2fe; transform: translateY(-2px);
               box-shadow: 0 10px 30px -18px rgba(67,56,202,.45); }
a.tile .n { font-size: 11.5px; color: var(--indigo); font-weight: 700; letter-spacing: .1em; }
a.tile h3 { margin: 8px 0 6px; font-size: 16.5px; color: var(--ink); }
a.tile p { margin: 0; font-size: 13px; color: var(--muted); }
a.tile .spark { margin-top: 16px; height: 42px; }

@media print {
  body { background: #fff; }
  .page { max-width: none; padding: 0; }
  .nav { display: none; }
  .card { box-shadow: none; break-inside: avoid; }
  section { break-inside: avoid; }
}
@media (max-width: 720px) {
  .page { padding: 24px 16px 64px; }
  h1 { font-size: 25px; }
  .card { padding: 18px 16px; }
}
"""

# ---------------------------------------------------------------- page shell

NAV = [
    ("index.html", "总览"),
    ("01_winograd_principles.html", "① Winograd 原理"),
    ("02_system_architecture.html", "② 系统结构"),
    ("03_sme_gemm.html", "③ SME 内核"),
    ("04_performance.html", "④ 性能结果"),
    ("05_acl_comparison.html", "⑤ 与 ACL 对比"),
    ("06_timing.html", "⑥ 步耗时"),
    ("07_methodology.html", "⑦ 测量口径"),
    ("08_roadmap.html", "⑧ 演进路线"),
    ("09_bench_shapes.html", "⑨ 用例集"),
]


def page(title, subtitle, body, active, foot_note=""):
    links = "".join(
        f'<a href="{href}" class="{"on" if href == active else ""}">{esc(label)}</a>'
        for href, label in NAV
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">
<nav class="nav">{links}</nav>
<header>
  <div class="eyebrow">Winograd 卷积复刻项目 · 图示</div>
  <h1>{esc(title)}</h1>
  <p class="lede">{subtitle}</p>
</header>
{body}
<div class="foot">
  {foot_note}
  数据来源：<code>docs/final_benchmark_bfd6b1e.md</code>（vs OpenBLAS）、
  <code>docs/sme_fmopa_gemm.md</code>（SME vs KML）。
  生成脚本：<code>tools/gen_report_figures.py</code>（本页为静态 HTML，无外部依赖）。
</div>
</div>
</body>
</html>
"""


def section(idx, title, sub, inner):
    sub_html = f'<p class="sub">{sub}</p>' if sub else ""
    return (f'<section><h2><span class="idx">{idx}</span>{title}</h2>{sub_html}'
            f'{inner}</section>')


# ---------------------------------------------------------------- pages

def page_index(ob_st, sme_st):
    body = f"""
{section("00", "一句话", "同一个算法，两种工程哲学；差异全部落在工程实现上，并且每一条都有可复现的数据。",
         f'''<div class="card">
<p style="margin:0 0 14px;font-size:15px;color:#334155">
本项目从 ACL 的 Winograd 思路出发，<b>算法与变换矩阵完全一致</b>；差异在四处工程选择：
固定 F(4,4) 主路径而不按形状降级、泛型 SVE 内核而不手写展开汇编、可插拔 GEMM 后端而不绑定单一库、
单 OpenMP 区域而不引入调度器。</p>
<div class="metrics">
  <div class="metric hl"><div class="k">vs OpenBLAS · 59 形状</div>
    <div class="v">{ob_st['total_speedup']:.2f}x</div>
    <div class="n">合计时间比 · 几何平均 {ob_st['geo']:.2f}x · 59/59 全胜</div></div>
  <div class="metric"><div class="k">vs oneDNN wino:acl · 同算法</div>
    <div class="v">1.15–9.5x</div><div class="n">e2e + benchdnn 两列互证</div></div>
  <div class="metric"><div class="k">自研 SME GEMM · t16 绑核</div>
    <div class="v">−{(1-sme_st['t16']['total_ratio'])*100:.1f}%</div>
    <div class="n">总耗时 vs KML · {sme_st['t16']['wins']}/59 形状胜</div></div>
  <div class="metric"><div class="k">正确性 · fp64 参考</div>
    <div class="v">22/22</div><div class="n">相对误差 2–6e-6（容差 1e-4）</div></div>
</div>
</div>''')}

{section("01", "图示索引", "每页都可以单独打开、投影或打印；全部为静态 HTML，无外部依赖。",
         '''<div class="grid">
  <a class="tile" href="01_winograd_principles.html">
    <div class="n">01</div><h3>Winograd 原理</h3>
    <p>tile 划分与重叠、三段流水、乘法数 144→36、Bᵀ 的零系数分布。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><rect x="0" y="8" width="140" height="10" rx="5" fill="#cbd5e1"/><rect x="0" y="26" width="35" height="10" rx="5" fill="#4338ca"/></svg></div>
  </a>
  <a class="tile" href="02_system_architecture.html">
    <div class="n">02</div><h3>系统结构</h3>
    <p>NHWC 计算核心与布局转换、三级 ISA、可插拔 GEMM 后端、单 OpenMP 区域时间线。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><rect x="0" y="16" width="46" height="12" rx="3" fill="#e0e7ff"/><rect x="54" y="16" width="46" height="12" rx="3" fill="#a5b4fc"/><rect x="108" y="16" width="46" height="12" rx="3" fill="#5eead4"/><rect x="162" y="16" width="46" height="12" rx="3" fill="#fcd34d"/></svg></div>
  </a>
  <a class="tile" href="03_sme_gemm.html">
    <div class="n">03</div><h3>SME 内核</h3>
    <p>ZA 的四个 tile 如何映射到 OC/tile、drain 的 64 条切片、微内核汇编与 ABI 陷阱。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><g fill="#e0e7ff"><rect x="0" y="4" width="30" height="8" rx="2"/><rect x="0" y="17" width="30" height="8" rx="2"/><rect x="0" y="30" width="30" height="8" rx="2"/></g><g fill="#4338ca"><rect x="36" y="4" width="8" height="8" rx="2"/><rect x="48" y="17" width="8" height="8" rx="2"/><rect x="60" y="30" width="8" height="8" rx="2"/></g></svg></div>
  </a>
  <a class="tile" href="04_performance.html">
    <div class="n">04</div><h3>性能结果</h3>
    <p>59 形状逐条加速比、分布直方图、四种平均口径、SME vs KML 两档对照。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><g fill="#4338ca"><rect x="0" y="30" width="6" height="12"/><rect x="10" y="24" width="6" height="18"/><rect x="20" y="18" width="6" height="24"/><rect x="30" y="27" width="6" height="15"/><rect x="40" y="12" width="6" height="30"/><rect x="50" y="20" width="6" height="22"/><rect x="60" y="8" width="6" height="34"/><rect x="70" y="22" width="6" height="20"/></g><g fill="#0f766e"><rect x="82" y="26" width="6" height="16"/><rect x="92" y="14" width="6" height="28"/><rect x="102" y="20" width="6" height="22"/><rect x="112" y="10" width="6" height="32"/></g></svg></div>
  </a>
  <a class="tile" href="05_acl_comparison.html">
    <div class="n">05</div><h3>与 ACL 对比</h3>
    <p>同源清单（逐元素一致）、八个维度的实现差异、优势证据与诚实的差距。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><rect x="0" y="6" width="96" height="30" rx="4" fill="#ecfdf5"/><rect x="104" y="6" width="96" height="30" rx="4" fill="#fffbeb"/><line x1="100" y1="4" x2="100" y2="38" stroke="#cbd5e1" stroke-dasharray="3 3"/></svg></div>
  </a>
  <a class="tile" href="06_timing.html">
    <div class="n">06</div><h3>步耗时拆解</h3>
    <p>8 步时间占比、三种形态、小 GEMM 效率崩塌、16 线程只拿到 7–13×。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><rect x="0" y="14" width="128" height="12" rx="3" fill="#4338ca"/><rect x="128" y="14" width="52" height="12" rx="3" fill="#0f766e"/><rect x="180" y="14" width="56" height="12" rx="3" fill="#b45309"/><rect x="0" y="30" width="72" height="8" rx="2" fill="#4338ca" opacity=".55"/><rect x="72" y="30" width="34" height="8" rx="2" fill="#0f766e" opacity=".55"/><rect x="106" y="30" width="130" height="8" rx="2" fill="#f59e0b" opacity=".55"/></svg></div>
  </a>
  <a class="tile" href="07_methodology.html">
    <div class="n">07</div><h3>测量口径</h3>
    <p>同作业 A/B、绑核、四种平均各自回答什么、方向约定与六个已知陷阱。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><circle cx="18" cy="24" r="7" fill="#0f766e"/><circle cx="52" cy="14" r="4" fill="#4338ca"/><circle cx="76" cy="30" r="5" fill="#be123c"/><line x1="0" y1="36" x2="140" y2="36" stroke="#cbd5e1" stroke-width="1.5"/><rect x="152" y="10" width="10" height="26" rx="2" fill="#818cf8"/><rect x="168" y="16" width="10" height="20" rx="2" fill="#818cf8"/><rect x="184" y="6" width="10" height="30" rx="2" fill="#818cf8"/><rect x="200" y="20" width="10" height="16" rx="2" fill="#818cf8"/></svg></div>
  </a>
  <a class="tile" href="08_roadmap.html">
    <div class="n">08</div><h3>演进路线</h3>
    <p>三阶段九项任务（做法/预期/风险）、判定标准，以及四条被数据证伪的方向。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><rect x="0" y="12" width="70" height="20" rx="4" fill="#ecfdf5" stroke="#0f766e"/><rect x="80" y="12" width="70" height="20" rx="4" fill="#eef2ff" stroke="#4338ca"/><rect x="160" y="12" width="70" height="20" rx="4" fill="#fffbeb" stroke="#b45309"/><path d="M74 22 l6 0 m-3 -3 l3 3 -3 3" stroke="#94a3b8" fill="none"/><path d="M154 22 l6 0 m-3 -3 l3 3 -3 3" stroke="#94a3b8" fill="none"/></svg></div>
  </a>
  <a class="tile" href="09_bench_shapes.html">
    <div class="n">09</div><h3>基准用例集</h3>
    <p>59 个形状的七个小节、tiles × IC 网格热力图、形状参数含义与复现命令。</p>
    <div class="spark"><svg viewBox="0 0 240 42" width="100%"><g><rect x="0" y="6" width="18" height="14" rx="2" fill="#eef2ff"/><rect x="22" y="6" width="18" height="14" rx="2" fill="#c7d2fe"/><rect x="44" y="6" width="18" height="14" rx="2" fill="#818cf8"/><rect x="66" y="6" width="18" height="14" rx="2" fill="#4338ca"/><rect x="0" y="24" width="18" height="14" rx="2" fill="#f1f5f9"/><rect x="22" y="24" width="18" height="14" rx="2" fill="#e0e7ff"/><rect x="44" y="24" width="18" height="14" rx="2" fill="#a5b4fc"/><rect x="66" y="24" width="18" height="14" rx="2" fill="#6366f1"/></g></svg></div>
  </a>
</div>''')}

{section("02", "测量口径（读图前必看）", "所有数字都是同机器、同作业、绑核口径；集群与本机是两套机器，毫秒数不可跨机比较。",
         '''<div class="card"><table class="tbl">
<thead><tr><th>结论</th><th>机器</th><th>线程 / 绑核</th><th class="r">数值</th></tr></thead>
<tbody>
<tr><td>vs OpenBLAS（鲲鹏版）</td><td>集群 920F（node03）</td><td>16T · 同一 sbatch 作业内 A/B</td><td class="r num">321.9 vs 804.9 ms</td></tr>
<tr><td>vs oneDNN <code>wino:acl</code></td><td>集群 920F</td><td>16T · 同作业同实现</td><td class="r num">1.15–9.5x</td></tr>
<tr><td>SME vs KML · t1</td><td>本机 608 核（毕昇）</td><td>1T · <code>taskset -c 3</code></td><td class="r num">3317.9 vs 3403.3 ms</td></tr>
<tr><td>SME vs KML · t16</td><td>本机 608 核</td><td>16T · <code>OMP_PROC_BIND=close</code></td><td class="r num">345.9 vs 386.3 ms</td></tr>
</tbody></table>
<div class="callout indigo" style="margin-top:16px">
<b>为什么强调绑核：</b>不绑核时 t16 的 SME 合计会在 387–558 ms 之间漂移，结论直接翻转成「SME 慢 2 倍」；
绑核后两次重复差异 &lt; 2.5%。同理，集群跨作业存在 3–7x 的性能态漂移，因此所有对比都在单作业内完成。
</div></div>''')}
"""
    return page(
        "Winograd 卷积复刻 · 图示总览",
        "九页图示覆盖「原理 → 结构 → SME 内核 → 性能 → 与 ACL 对比 → 步耗时 → 测量口径 → 路线 → 用例集」，"
        "面向汇报与答辩：每张图只讲一件事，数字都能追溯到仓库里的表。",
        body, "index.html")


def page_principles():
    body = f"""
{section("01", "tile 怎么切", "",
         f'<div class="card">{diagram_tiles()}</div>')}

{section("02", "三段流水", "Winograd 把一个卷积拆成三次小规模变换 + 一次批量小矩阵乘，乘法量集中在第二阶段。",
         f'<div class="card">{diagram_pipeline()}</div>')}

{section("03", "为什么值得做", "",
         f'''<div class="cols">
<div class="card">{diagram_multiply()}
<p style="margin:14px 0 0;font-size:13px;color:#64748b">
代价是额外的加法与两次变换：输入变换每 tile 一次、输出变换每 tile 一次。
只有当 IC×spatial 足够大、能把变换的固定开销摊薄时，净收益才为正 ——
这正好解释了「小形状该走直接卷积」的边界。</p></div>
<div class="card">{diagram_bt_sparsity()}
<p style="margin:14px 0 0;font-size:13px;color:#64748b">
Bᵀ 与 A 都有大量零系数。ACL 在编译期把它们消去（全展开汇编）；
本项目当前在运行时 <code>continue</code> 跳过 —— 这是变换阶段仍有 30–50% 优化空间的原因。</p></div>
</div>''')}

{section("04", "数值精度：为什么用相对容差", "F(4,4) 的 A 矩阵系数最大到 ±8，会把 GEMM 的累加误差放大；误差随 IC 线性增长，输出幅值也同步增长。",
         '''<div class="card"><table class="tbl">
<thead><tr><th>量</th><th class="r">典型值</th><th>说明</th></tr></thead>
<tbody>
<tr><td>绝对误差（IC=384）</td><td class="r num">~1e-2</td><td>输出幅值 ~1000，直接看绝对值会误判为 FAIL</td></tr>
<tr><td>相对误差</td><td class="r num">2–6e-6</td><td>与 IC 基本无关，属 fp32 正常水平</td></tr>
<tr><td>验证判据</td><td class="r num">1e-4 × max|ref| + 1e-5</td><td>fp64 直接卷积作参考，逐元素比较</td></tr>
</tbody></table>
<div class="callout teal" style="margin-top:16px">
早期用固定绝对容差 1e-3，导致 IC=384/768 的大用例被误报为 FAIL —— 实际相对误差只有 6e-6。
<b>判据必须随规模缩放</b>，这是验证方法学上最容易踩的坑。
</div></div>''')}
"""
    return page(
        "① Winograd 原理",
        "F(4,4,3,3) 如何把每个输出像素的 9 次乘法降到 2.25 次，以及这条路径的代价落在哪里。",
        body, "01_winograd_principles.html")


def page_architecture(ob_st):
    body = f"""
{section("01", "端到端数据流", "计算核心只认 NHWC；NCHW 由一次 cache-blocked 转置承担。",
         f'<div class="card">{diagram_dataflow()}</div>')}

{section("02", "三级 ISA：同一套矩阵，三种宽度", "变换矩阵在三种 ISA 间共用一份定义 —— 修一次矩阵，三条路径同时正确（历史上 3 次矩阵 bug 的教训）。",
         '''<div class="card"><table class="tbl">
<thead><tr><th>方面</th><th>NEON</th><th>SVE</th><th>SME</th></tr></thead>
<tbody>
<tr><td>向量宽度</td><td class="num">128 bit（4 float）</td><td class="num">VL = 512 bit（16 float）</td><td class="num">同 SVE + 64×64 ZA</td></tr>
<tr><td>tail 处理</td><td>三段降级 4→2→1</td><td>谓词 <code>whilelt</code>，一份代码</td><td>同 SVE</td></tr>
<tr><td>变换实现</td><td>intrinsics</td><td>intrinsics + 谓词</td><td>输出变换用 FMOPA 外积</td></tr>
<tr><td>通道并行度</td><td class="num">4</td><td class="num">16</td><td class="num">64（4 × VL）</td></tr>
<tr><td>代码量</td><td class="num">390 行</td><td class="num">285 行</td><td class="num">1 016 行</td></tr>
</tbody></table></div>''')}

{section("03", "GEMM 后端可插拔", "统一「U 按 tile、V 按 OC 行、K 内连续」的布局契约，后端自己决定要不要打包。",
         '''<div class="card"><table class="tbl">
<thead><tr><th>后端</th><th>实现</th><th>定位</th><th class="r">59 形状合计</th></tr></thead>
<tbody>
<tr><td><b>arm_gemm</b></td><td>ACL 53.1.0 的 SVE JIT（与 oneDNN 同源）</td><td>生产默认</td><td class="r num">321.9 ms（集群）</td></tr>
<tr><td><b>KML kblas</b></td><td>华为 KML <code>cblas_sgemm</code>（locking）</td><td>本机日常首选</td><td class="r num">233 ms（本机）</td></tr>
<tr><td><b>OpenBLAS</b></td><td>鲲鹏版 <code>cblas_sgemm</code></td><td>对照基线</td><td class="r num">804.9 ms（集群）</td></tr>
<tr><td><b>SME FMOPA</b></td><td>自研 ZA 外积内核</td><td>实验，特定形状领先</td><td class="r num">见第 ③ 页</td></tr>
<tr><td>naive</td><td>三重循环</td><td>无依赖兜底</td><td class="r num">—</td></tr>
</tbody></table>
<div class="callout indigo" style="margin-top:16px">
KML 与 arm_gemm 的排名在集群和本机<b>正好相反</b>（本机 KML 233 ms &lt; arm_gemm 328 ms）。
这正是「后端必须可插拔」的证据：单一后端会把平台相关的结论当成普适结论。
</div></div>''')}

{section("04", "并行结构：一次 fork/join", "权重变换 + 所有 batch 的输入/GEMM/输出都在同一个 parallel 区域内，靠显式 barrier 串起三段。",
         f'<div class="card">{diagram_openmp()}</div>')}

{section("05", "缓冲与缓存：热路径不做分配", "",
         '''<div class="cols">
<div class="card"><h3>线程本地缓冲</h3>
<p style="font-size:13.5px;color:#475569;margin:0">
6 个 <code>thread_local</code> 缓冲（d/U/M/f tile、权重临时）跨调用复用，
容量不足才 realloc。每个逻辑缓冲持有<b>独立</b>的 Scratch —— 早期共用一个曾导致 9 个缓冲全部别名同一块内存，
正确性测试全挂。</p></div>
<div class="card"><h3>V 跨调用缓存</h3>
<p style="font-size:13.5px;color:#475569;margin:0">
key = 权重指针 + 64 点 FNV-1a 指纹 + shape + F44 标记 + ISA，防「同址重分配」的 ABA。
命中直接跳过权重变换，KML t16 几何平均 <b>1.07x</b>、最高 <b>1.74x</b>。
容量 24 条且无淘汰是个已知短板（59 形状用例集后半段不命中）。</p></div>
</div>''')}
"""
    return page(
        "② 系统结构",
        "从张量布局到线程模型：哪些是算法决定的，哪些是为了这台无 L3 的机器做的工程取舍。",
        body, "02_system_architecture.html")


def page_sme(sme_st):
    micro = """; 一次窗口：4 个 ZA tile，K 循环每轮 4 条 fmopa
smstart
ptrue   p0.s
zero    {za0.s, za1.s, za2.s, za3.s}
1:
ld1w    {z16.s}, p0/z, [x9]          ; A: 16 个 tile 在同一个 ic 上的 U
ld1w    {z17.s}, p0/z, [x10]         ; B: OC 块 0 @ ic
ld1w    {z18.s}, p0/z, [x10, #1, MUL VL]
ld1w    {z19.s}, p0/z, [x10, #2, MUL VL]
ld1w    {z20.s}, p0/z, [x10, #3, MUL VL]
fmopa   za0.s, p0/m, p0/m, z16.s, z17.s
fmopa   za1.s, p0/m, p0/m, z16.s, z18.s
fmopa   za2.s, p0/m, p0/m, z16.s, z19.s
fmopa   za3.s, p0/m, p0/m, z16.s, z20.s
add     x9,  x9,  #64               ; A 的 k 步进 = 16 tile × 4 B
add     x10, x10, bstep             ; B 的 k 步进 = OC64 × 4 B
cmp / add / b.lt
; drain: 64 条 str za → 直接落 M_buf 的行"""

    body = f"""
{section("01", "为什么需要专用内核", "Winograd 把卷积变成 36 个「M 小、K 短、调用密」的小 GEMM，通用 BLAS 的固定开销在这里被放大。",
         f'''<div class="card"><table class="tbl">
<thead><tr><th>切片特征</th><th class="r">59 形状集范围</th><th>对通用 GEMM 的影响</th></tr></thead>
<tbody>
<tr><td>切片数 <code>n_multis</code></td><td class="r num">36（固定）</td><td>每次调用都有进入/退出 + 打包开销</td></tr>
<tr><td>M = tile 数</td><td class="r num">4 … 3136</td><td>7² 图像只有 4 个 tile，分块填不满</td></tr>
<tr><td>N = OC</td><td class="r num">16 … 512</td><td>小 OC 时不足一个向量宽度</td></tr>
<tr><td>K = IC</td><td class="r num">16 … 2048</td><td>小 IC 时流水线填不满</td></tr>
</tbody></table>
<div class="callout" style="margin-top:16px">
t1 下这类小切片的有效吞吐只有几十 GFLOPS，而单核 SME <code>fmopa</code> 探针能到约 2 TFLOPS/核。
差距不在「GEMM 写得快不快」，而在<b>执行模型是否匹配矩阵形态</b>。
</div></div>''')}

{section("02", "布局：行 = OC，列 = tile", "这个方向不是随手选的 —— 它决定 drain 出来的 64 B 能不能直接写进 M_buf。",
         f'<div class="card">{diagram_za_layout()}</div>')}

{section("03", "drain：64 条切片，零转置", "",
         f'<div class="card">{diagram_drain()}</div>')}

{section("04", "微内核", "每轮 4 条 fmopa = 4 × 256 = 1024 次 FMA，配 5 次 64 B 加载 → 约 6.4 FLOP/byte。",
         f'<div class="card">{code_block(micro)}</div>')}

{section("05", "结果与边界", "绑核口径，59 形状，同一二进制只切环境变量。",
         f'''<div class="card">
<div class="chart-wrap">{chart_sme(sme_st['t1'], "t1（taskset -c 3 固定单核）")}</div>
<div class="chart-wrap" style="margin-top:14px">{chart_sme(sme_st['t16'], "t16（OMP_PROC_BIND=close 绑 16 核）")}</div>
<p style="margin:10px 0 0;font-size:12.5px;color:#64748b">
纵轴为 log₂ 刻度，基准线 1.0；绿色 = SME 更快，红色 = KML 更快。逐形状数值与胜负归因见
<code>docs/sme_fmopa_gemm.md</code> §5。</p>
</div>
<div class="cols" style="margin-top:16px">
  <div class="card"><h3>优势区间</h3>
  <p style="font-size:13.5px;margin:0;color:#475569">
  小空间 + 深通道（7²/14²）：M 再小也不损失外积利用率，KML 在 M=4 时只剩 20–22 GFLOPS，
  SME 最高 <b>2.2x</b>；大空间（160²/224²）在 t16 下最高 <b>1.7x</b>。</p></div>
  <div class="card"><h3>劣势区间</h3>
  <p style="font-size:13.5px;margin:0;color:#475569">
  小空间 + 大 OC/IC：<code>28²</code> 的尾组只有 1 个有效 tile（ZN 利用率 6%），
  每个窗口固定 64 条 drain，浪费与 K 无关；t16 下 tile 少的形状并行度只来自 36 个切片。</p></div>
</div>''')}

{section("06", "一个代价高昂的 ABI 陷阱", "现象极具误导性：数据完全正确，但计时输出 inf。",
         '''<div class="card">
<p style="margin:0 0 12px;font-size:13.5px">
<code>smstart</code>/<code>smstop</code> 切换流式模式时，<b>所有</b> SVE z 寄存器与谓词都变成 UNPREDICTABLE；
而 AAPCS64 的 callee-saved 约定是 <code>d8–d15</code>。当时的 clobber 列表只写了实际用到的
<code>z16–z20</code> 与 <code>p0</code>，编译器以为 <code>d11</code> 里活着的常量 <code>1e6</code> 仍然有效 ——
一次调用返回后它被清零，<code>ticks / 1e6</code> 变成 <code>inf</code>。
<code>-O0</code> 不复现，独立的 <code>minwin</code> 小程序也不复现。</p>
<div class="callout rose">
<b>规则：</b>任何内含 <code>smstart</code>/<code>smstop</code> 的内联汇编，
clobber 必须覆盖 <code>d8–d15</code> 与 <code>p1–p15</code>；只写「自己用到的寄存器」是不够的。
完整定位过程见 <code>docs/sme_fmopa_gemm.md</code> §6。
</div></div>''')}
"""
    return page(
        "③ SME FMOPA GEMM",
        "把 36 个小切片直接映射到 ZA 二维累加器：行 = 16 个 OC、列 = 16 个 tile、4 个 tile = 64 个 OC。",
        body, "03_sme_gemm.html")


def page_performance(ob_st, sme_st):
    s16 = sme_st["t16"]
    s1 = sme_st["t1"]
    body = f"""
{section("01", "核心结论", "四个数字，四种口径；口径差异不是误差，而是回答了不同的问题。",
         f'''<div class="metrics">
  <div class="metric hl"><div class="k">合计时间比（OpenBLAS / 我们）</div>
    <div class="v">{ob_st['total_speedup']:.2f}x</div>
    <div class="n">{ob_st['total_fast']:.1f} ms vs {ob_st['total_slow']:.1f} ms · 59 形状 · 16T</div></div>
  <div class="metric"><div class="k">几何平均</div><div class="v">{ob_st['geo']:.2f}x</div>
    <div class="n">典型形状快多少（等权、方向对称）</div></div>
  <div class="metric"><div class="k">算数平均</div><div class="v">{ob_st['arith']:.2f}x</div>
    <div class="n">比值期望 · 被大加速比抬高（AM ≥ GM）</div></div>
  <div class="metric"><div class="k">中位数</div><div class="v">{ob_st['median']:.2f}x</div>
    <div class="n">一半形状快于该值 · 最抗离群</div></div>
</div>
<div class="callout indigo" style="margin-top:16px">
<b>该看哪个：</b>端到端/吞吐看<b>合计时间比</b>；跨实现可比性看<b>几何平均</b>；
分布形状看<b>中位数 + 逐形状图</b>。算数平均只作辅助，且必须标注方向 ——
同一份数据用算数平均总是更好看（{ob_st['arith']:.2f}x vs {ob_st['geo']:.2f}x），差异纯属口径。
</div>''')}

{section("02", "59 个形状逐条加速比（vs OpenBLAS，t16）", "基线 1.0x；越高越好。悬停任意柱可看该形状的绝对耗时。",
         f'<div class="card"><div class="chart-wrap">{chart_speedup_bars(ob_st["rows"], ob_st)}</div></div>')}

{section("03", "总体与分布", "",
         f'''<div class="cols">
<div class="card"><h3>合计时间</h3>{chart_totals(ob_st)}
<p style="margin:10px 0 0;font-size:12.5px;color:#64748b">
59 个形状总耗时之比；被大形状主导，最贴近「这套负载端到端多久」。</p></div>
<div class="card"><h3>加速比分布</h3>{chart_histogram(ob_st)}
<p style="margin:10px 0 0;font-size:12.5px;color:#64748b">
仅 4 个形状落在 1.0–1.25x，且全部是 &lt; 1.1 ms 的微负载；&gt; 3x 的 8 个是深层瘦条与大形状。</p></div>
</div>''')}

{section("04", "达到硬件上限的证据", "最大形状 4,384,160²,384 折算 GEMM 吞吐约 960 GFLOPS / 16 核。",
         f'''<div class="card">{chart_gflops()}
<p style="margin:10px 0 0;font-size:12.5px;color:#64748b">
SVE-512 峰值按 16 核 × 2 FMA 单元 × 16 float × 2 FLOP × 2.0 GHz 估算；
达到 94% 说明后端已经接近硬件上限，继续优化应转向变换与调度，而不是继续换 GEMM 库。</p></div>''')}

{section("05", "自研 SME GEMM vs KML（本机 608 核，绑核）", "speedup = KML / SME（&gt;1 表示 SME 更快）；纵轴 log₂。",
         f'''<div class="card">
<div class="chart-wrap">{chart_sme(s16, "t16（绑 16 核）：合计快 10.4%")}</div>
<div class="chart-wrap" style="margin-top:16px">{chart_sme(s1, "t1（固定单核）：合计快 2.5%")}</div>
<div class="callout teal" style="margin-top:16px">
<b>四种口径在 t16 上方向相反</b>：合计比显示 SME 快 10.4%，几何平均却是 0.901（慢约 10%）。
原因不是矛盾，而是分布双峰 —— 4 个形状赢 1.5–1.8x，同时 3 个形状输到 0.29–0.45x，
两端在 log 尺度上互相拉扯，中位数（1.022）又偏向 SME。
</div>
</div>''')}

{section("06", "口径对照表", "同一份数据换一个平均值就可能换一个结论，所以四个数一起给。",
         f'''<div class="card"><table class="tbl">
<thead><tr><th>对照</th><th class="r">合计比</th><th class="r">几何平均</th><th class="r">算数平均</th><th class="r">中位数</th><th class="r">胜出</th></tr></thead>
<tbody>
<tr><td>arm_gemm vs OpenBLAS（集群 t16）</td>
  <td class="r num em">{ob_st['total_speedup']:.2f}x</td><td class="r num">{ob_st['geo']:.2f}x</td>
  <td class="r num">{ob_st['arith']:.2f}x</td><td class="r num">{ob_st['median']:.2f}x</td>
  <td class="r num">59/59</td></tr>
<tr><td>SME vs KML（本机 t16）</td>
  <td class="r num em">{s16['total_speedup']:.2f}x</td><td class="r num">{s16['geo']:.3f}</td>
  <td class="r num">{s16['arith']:.3f}</td><td class="r num">{s16['median']:.3f}</td>
  <td class="r num">{s16['wins']}/59</td></tr>
<tr><td>SME vs KML（本机 t1）</td>
  <td class="r num em">{s1['total_speedup']:.2f}x</td><td class="r num">{s1['geo']:.3f}</td>
  <td class="r num">{s1['arith']:.3f}</td><td class="r num">{s1['median']:.3f}</td>
  <td class="r num">{s1['wins']}/59</td></tr>
</tbody></table>
<p style="margin:14px 0 0;font-size:12.5px;color:#64748b">
SME 两行的方向约定不同：合计比 = ΣSME/ΣKML（&lt;1 表示 SME 更快），
其余三列 = KML/SME（&gt;1 表示 SME 更快）。报算数平均必须写明方向，否则数字不可复现。
</p></div>''')}
"""
    return page(
        "④ 性能结果",
        "59 个形状、同一作业内 A/B、绑核测量。结论按「合计 / 几何 / 算数 / 中位」四种口径同时给出。",
        body, "04_performance.html")


def page_acl():
    rows = [
        ("算法与公式", "Y = Aᵀ[(GgGᵀ) ⊙ (BᵀdB)]A", "同", "same"),
        ("G / Bᵀ / A 矩阵", "6/-4/-4/1/1/0 …；4,0,-5,0,1,0 …；1,1,1,1,1,0 …", "逐元素相同", "same"),
        ("归一化", "recip576", "1/576", "same"),
        ("tile 划分", "6×6 输入 / 4×4 输出 / 步长 4", "同", "same"),
        ("输出变换", "融合 bias + clamp", "同", "same"),
        ("SME 输出变换", "sme_fp32_mopa_4x4_3x3", "移植同一实现", "same"),
        ("GEMM 源码", "arm_gemm（ACL 自带）", "就地编译同一子集", "same"),
        ("tile 选择", "按形状贪心选最大输出 tile；小图可退 F(2,2)", "固定 F(4,4)，不降级", "diff"),
        ("形状门禁", "oneDNN：ih/iw>112、ic/oc<64、threads>28 回退 brgconv", "无门禁", "diff"),
        ("变换内核", "手写全展开汇编（fp32 共 5 118 行）", "泛型 SVE intrinsics（1 691 行）", "diff"),
        ("布局写入", "matrix_stride 直写 GEMM 序", "U_tile → U 一次 scatter", "diff"),
        ("Winograd 域 GEMM", "一次 batched arm_gemm（nmulti=36）", "5 种后端可插拔 + 按线程折叠", "diff"),
        ("权重缓存", "prepare() 一次，primitive 生命周期", "跨调用 V 缓存（指针+指纹）", "diff"),
        ("多线程", "NEScheduler + 内核内细粒度切分", "单 OpenMP 区域 + 显式 barrier", "diff"),
        ("SME 输入变换", "有（sme_fp32_mla_6x6，363 行）", "无（保持纯 SVE）", "diff"),
        ("SME GEMM", "arm_gemm 自带 sme_*_fp32_mopa_*（需 interleave）", "自研 ZA 内核（K 主序免打包）", "diff"),
        ("覆盖面", "fp32/fp16、2D/1D、3×3/5×5", "仅 fp32 2D 3×3", "gap"),
    ]
    table = "".join(
        f'<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td>'
        f'<td class="c"><span class="tag {t}">{"一致" if t=="same" else ("差异" if t=="diff" else "收窄")}</span></td></tr>'
        for a, b, c, t in rows
    )

    adv = [
        ("A1", "算法选择更彻底", "全形状 F(4,4)，不因小图退回 F(2,2)、不因线程数回退直接卷积",
         "同算法 PK <b>1.15–9.5x</b>", "边界：极小形状 oneDNN 走 brgconv 更快（0.09–0.91x）"),
        ("A2", "GEMM 后端可插拔", "5 种后端 + 统一布局契约，可随时 A/B",
         "arm_gemm 59/59 胜 OpenBLAS（合计 2.50x）；本机 KML 反超 arm_gemm", "结论不可跨机迁移"),
        ("A3", "batched 调用摊薄开销", "每线程把连续切片折叠进一次 nmulti 调用",
         "修复后 <b>47/59 → 59/59</b>", "切片数少于线程数时退化"),
        ("A4", "单并行区域 + 线程本地缓冲", "1 次 fork/join、零热路径分配、去掉无效清零",
         "Case 0：14.2 → <b>1.94 ms</b>（7.3x）", "阶段间不能重叠"),
        ("A5", "缓存贴合裸 API + 防 ABA", "指针 + 64 点指纹 + shape + ISA 作 key",
         "KML t16 几何 <b>1.07x</b>，最高 1.74x", "容量 24 条，需 LRU"),
        ("A6", "SME 与切片形状对齐", "ZA 行 = OC、列 = tile，drain 零转置",
         "t16 合计 <b>快 10.4%</b>，最高 2.2x", "小空间大 OC/IC 落后 1.4–3.4x"),
        ("A7", "代码规模", "3 636 行核心可完整通读",
         "ACL Winograd 目录 8 056 行", "—"),
        ("A8", "独立真值验证", "fp64 直接卷积 + 相对容差 + bit-exact 布局回归",
         "22/22 通过，相对误差 2–6e-6", "—"),
    ]
    adv_html = "".join(
        f'''<tr><td class="num" style="color:#4338ca;font-weight:700">{k}</td>
<td><b>{esc(t)}</b><div style="font-size:12.5px;color:#64748b;margin-top:2px">{esc(m)}</div></td>
<td>{e}</td><td style="font-size:12.5px;color:#94a3b8">{esc(b)}</td></tr>'''
        for k, t, m, e, b in adv
    )

    gaps = [
        ("K1", "单 kernel 效率", "transform_1d_sve 约 52 cyc/向量操作（理想 ~2）；ACL 全展开、寄存器驻留",
         "F(4,4) 专用展开内核，预期变换 −30~50%"),
        ("K2", "多两趟内存往返", "输入 scatter 6% + 输出 gather 15%（Case 0 串行计时）",
         "变换直接写 GEMM 序（tile-major 已被实测证伪：−14%/−16%）"),
        ("K3", "小形状无 hybrid 退路", "batch=1、IC≤128、spatial≤56² 慢 0.09–0.91x",
         "im2col + arm_gemm 直接卷积路径"),
        ("K4", "SME GEMM 未与 ACL 的 SME 内核同机对打", "构建时用 filter=\"sve_\" 强制了 SVE 内核",
         "WINO_GEMM_FILTER=\"\" 放开，同作业 A/B"),
        ("K5", "覆盖面窄", "仅 fp32、2D、3×3、unit stride", "有意取舍"),
    ]
    gap_html = "".join(
        f'''<tr><td class="num" style="color:#be123c;font-weight:700">{k}</td>
<td><b>{esc(t)}</b></td><td style="font-size:13px">{esc(d)}</td>
<td style="font-size:12.5px;color:#64748b">{esc(f)}</td></tr>'''
        for k, t, d, f in gaps
    )

    body = f"""
{section("01", "先说同源", "汇报时先讲这一页，后面的差异才站得住：数学层面 100% 同源，差别全在工程实现与调度策略。",
         f'''<div class="card"><table class="tbl">
<thead><tr><th>维度</th><th>ACL 53.1.0</th><th>本项目</th><th class="c">判定</th></tr></thead>
<tbody>{table}</tbody></table>
<p style="margin:14px 0 0;font-size:12.5px;color:#64748b">
判定依据全部来自仓库内 <code>ComputeLibrary-53.1.0/</code> 源码逐行核对，
定位命令见 <code>docs/vs_acl.md</code> §6。</p></div>''')}

{section("02", "八条优势（机制 / 证据 / 边界）", "每条都给出它成立的边界；只讲好话的对比在答辩时站不住。",
         f'''<div class="card"><table class="tbl">
<thead><tr><th style="width:44px">#</th><th>优势</th><th>证据</th><th>适用边界</th></tr></thead>
<tbody>{adv_html}</tbody></table></div>''')}

{section("03", "六条诚实的差距", "建议汇报时主动说 K1 与 K4 —— 主动暴露短板的报告更可信，而且这两条都有明确补法。",
         f'''<div class="card"><table class="tbl">
<thead><tr><th style="width:44px">#</th><th>差距</th><th>具体</th><th>补法</th></tr></thead>
<tbody>{gap_html}</tbody></table></div>''')}

{section("04", "同算法 PK 的三种口径", "oneDNN 的三列回答三个不同问题，不能混用。",
         '''<div class="card"><table class="tbl">
<thead><tr><th>对照列</th><th>oneDNN 算法</th><th>回答的问题</th><th>结果</th></tr></thead>
<tbody>
<tr><td><code>e2e_auto</code></td><td>自动（40 wino:acl + 19 brgconv）</td><td>我们比 oneDNN 最优选择快还是慢</td>
  <td>大形状快 <b>1.1–2.8x</b>；小形状慢 0.09–0.91x</td></tr>
<tr><td><code>e2e_wino</code></td><td>强制 wino:acl</td><td>同样做 Winograd，谁更快</td>
  <td class="em">全部 1.15–9.5x</td></tr>
<tr><td><code>benchdnn_wino</code></td><td>wino:acl（driver 口径）</td><td>换一个 driver 复核</td>
  <td>与 <code>e2e_wino</code> 一致，互证</td></tr>
</tbody></table>
<div class="callout" style="margin-top:16px">
<b>口径提醒：</b>ACL 选 F(4,4) 还是 F(2,2) 取决于它的构建是否启用 SME。
集群侧 oneDNN 的实际选中项未从日志确认（历史文档记为「疑似 F(2,2)」），
因此 1.15–9.5x 应读作「我们的 F(4,4) vs 对方当时的实际选择」。
</div></div>''')}
"""
    return page(
        "⑤ 与 ACL 的差异与优势",
        "同一套数学，两种工程哲学：ACL 用「手写内核 + 形状自适应」，本项目用「泛型内核 + 系统级取舍」。",
        body, "05_acl_comparison.html")


def page_timing(steps, timing, ob_st, sme_st):
    top_weight = sorted(timing, key=lambda r: r["weight_pct"], reverse=True)[:8]
    wrows = "".join(
        f'<tr><td class="num">#{r["idx"]}</td><td>{esc(short_shape(r["shape"]))}</td>'
        f'<td class="r num">{r["S"]:.2f}</td><td class="r num">{r["T16"]:.3f}</td>'
        f'<td class="r num em">{r["weight_pct"]:.0f}%</td>'
        f'<td class="r num">{r["gemm_pct"]:.0f}%</td></tr>'
        for r in top_weight)

    small = [r for r in timing if r["tiles"] <= 4]
    body = f"""
{section("01", "8 步拆解：时间到底花在哪", "条长 = 占 timing 总时间 S 的比例；「计算」= GEMM + Weight。",
         f'''<div class="card">{chart_step_composition(steps)}
<p style="margin:14px 0 0;font-size:12.5px;color:#64748b">
数据来自 <code>bench_winograd --timing</code>（warmup 10 / repeats 50 / best-of）。
8 个代表形状覆盖三种形态：小图深通道、均衡真实 CNN、大图浅通道。</p></div>''')}

{section("02", "三种形态", "GEMM 是第一大头，但只在特定形态里垄断；多数真实 shape 是五五开。",
         '''<div class="cols">
<div class="card"><h3>① 小图深通道：计算垄断</h3>
<p style="margin:0 0 10px;font-size:13.5px">7² / 14² 且 IC、OC ≥ 512 时，GEMM + Weight 合计占 <b>97–99%</b>。
但这不是「算得多」，而是「算得低效」——见下一节。</p>
<p style="margin:0;font-size:12.5px;color:#64748b">代表：#15 #16 #20 #28 #29 #30</p></div>
<div class="card"><h3>② 均衡：真实 CNN 主体</h3>
<p style="margin:0 0 10px;font-size:13.5px">GEMM <b>33–50%</b>，变换 + 数据搬移占另一半。
优化必须同时打两侧，只换 GEMM 库最多拿到一半收益。</p>
<p style="margin:0;font-size:12.5px;color:#64748b">代表：#0 #2 #22 #24 #53 #54 #57 #58</p></div>
<div class="card"><h3>③ 大图浅通道：变换主导</h3>
<p style="margin:0 0 10px;font-size:13.5px">IC=16 的 224² 上 GEMM 只占 <b>11%</b>，
InXform / OutXform / TileExt 才是大头——这类形状拼的是内核展开与访存。</p>
<p style="margin:0;font-size:12.5px;color:#64748b">代表：#31 #33 #34</p></div>
</div>''')}

{section("03", "机制：小 GEMM 的核效率崩塌", "同一个 OpenBLAS 内核，n=4 时只剩 ~7% 的核效率。",
         f'''<div class="cols">
<div class="card">{chart_small_gemm()}</div>
<div class="card">
<p style="margin:0 0 12px;font-size:13.5px">Winograd 把一次卷积拆成 36 个独立 GEMM，每个形状是
<code>n_tiles × OC × IC</code>。当图像只有 7×7 时，<code>n_tiles = 4</code>，
GEMM 退化成「4 行」的瘦条：单次调用开销、分块与 pack 成本全部摊不薄。</p>
<p style="margin:0;font-size:13.5px">这解释了为什么深通道小图 case 里 GEMM 占比高达 74%：
<b>不是工作量集中，而是效率坍塌</b>。对策是换 JIT 内核（arm_gemm）或让矩阵扩展
（SME）直接吃掉整个小切片。</p>
<div class="callout" style="margin-top:14px">最大形状的反例：1600×384×384 达到 134 GFLOPS，
说明同一内核在大 n 上并不慢——瓶颈是形状，不是库。</div>
</div>
</div>''')}

{section("04", "Weight transform：最高性价比的固定开销", "权重与图像无关，却每次调用都重算一遍。",
         f'''<div class="card">
<table class="tbl">
<thead><tr><th>row</th><th>shape</th><th class="r">S (ms)</th><th class="r">T16 (ms)</th>
<th class="r">Weight 占比</th><th class="r">GEMM 占比</th></tr></thead>
<tbody>{wrows}</tbody></table>
<div class="callout teal" style="margin-top:16px">
<b>V 跨调用缓存已落地：</b>按「权重指针 + 64 点内容指纹 + shape + F44 + ISA」做 key，
KML t16 几何平均 <b>1.07x</b>、最高 <b>1.74x</b>；<code>WINO_V_CACHE=0</code> 可关闭。
剩余问题：<code>kMaxEntries = 24</code> 且无淘汰，59 层的网络从第 25 层起不再命中——
这就是路线图里「V 缓存加 LRU」的由来。</div>
</div>''')}

{section("05", "并行效率：16 线程拿不到 16×", "eff = (T1/16) / T16；1.0 表示完美线性扩展。",
         f'''<div class="cols">
<div class="card">{chart_pareff(timing)}</div>
<div class="card">
<p style="margin:0 0 12px;font-size:13.5px">59 个形状的 eff 落在 <b>0.13–0.83</b>：
大形状 0.77–0.83，小形状 0.46–0.58，微负载（#39，S=0.08 ms）只有 0.13。</p>
<p style="margin:0 0 12px;font-size:13.5px">净并行开销 <code>T16 − T1/16</code> 占 T16 的
<b>19–42%</b>：粗粒度 OpenMP + 2 次 barrier + 无 L3 的 NUMA 流量，都不是免费的。</p>
<div class="callout">
<b>判读提醒：</b>「我们的调度比 oneDNN 便宜」这类结论，必须用<b>同 shape 的双方线程扩展曲线</b>
作对照；只看自己这组 eff 不足以归因。</div>
</div>
</div>''')}

{section("06", "两处必须知道的测量偏差", "占比结构可信；绝对耗时以 T1 为准；内存步的重复测量差异可达 88%。",
         '''<div class="cols">
<div class="card"><h3>S（timing）vs T1（真实单线程）</h3>
<table class="tbl">
<thead><tr><th>row</th><th class="r">S</th><th class="r">T1</th><th class="r">残差</th><th class="r">残差/S</th></tr></thead>
<tbody>
<tr><td class="num">#0</td><td class="r num">22.26</td><td class="r num">17.55</td><td class="r num">4.7</td><td class="r num warn">27%</td></tr>
<tr><td class="num">#54</td><td class="r num">157.89</td><td class="r num">141.67</td><td class="r num">16.2</td><td class="r num">11%</td></tr>
<tr><td class="num">#58</td><td class="r num">526.63</td><td class="r num">498.69</td><td class="r num">27.9</td><td class="r num">5.6%</td></tr>
</tbody></table>
<p style="margin:12px 0 0;font-size:12.5px;color:#64748b">残差来自 timing 复刻的逐 tile
<code>std::fill(0)</code>（#58 约 1.2 GB 流量）、每调用 <code>std::vector</code> 分配与 V/U/M 清零。
<b>大 shape 高估 5–10%，小 shape 高估 ~27%。</b></p></div>
<div class="card"><h3>同一 shape 的重复测量（#0 vs #49）</h3>
<table class="tbl">
<thead><tr><th>步</th><th class="r">#0</th><th class="r">#49</th><th class="r">差异</th></tr></thead>
<tbody>
<tr><td>S Total</td><td class="r num">22.26</td><td class="r num">18.36</td><td class="r num warn">21%</td></tr>
<tr><td>GEMM</td><td class="r num">7.55</td><td class="r num">7.32</td><td class="r num">3%</td></tr>
<tr><td>InScat</td><td class="r num">1.51</td><td class="r num">2.50</td><td class="r num warn">65%</td></tr>
<tr><td>Weight</td><td class="r num">1.82</td><td class="r num">0.97</td><td class="r num warn">88%</td></tr>
</tbody></table>
<p style="margin:12px 0 0;font-size:12.5px;color:#64748b">两行是同一 shape（4,192,40²,192）。
<b>GEMM 是稳定列</b>，内存步与 Weight 噪声大；行 0 是进程内首个 shape，冷启动系统性偏高。</p></div>
</div>''')}
"""
    return page(
        "⑥ 步耗时拆解与并行效率",
        "哪里真的花时间：GEMM 的份额随形状从 11% 到 74%，而 16 线程只拿到 7–13×。",
        body, "06_timing.html")


def page_methodology(ob_st, sme_st):
    s1, s16 = sme_st["t1"], sme_st["t16"]
    ob_items = [("合计时间比", ob_st["total_speedup"], ""),
                ("几何平均", ob_st["geo"], ""),
                ("算数平均", ob_st["arith"], ""),
                ("中位数", ob_st["median"], "")]
    sme_items = [("合计时间比", s16["total_speedup"], ""),
                 ("几何平均", s16["geo"], ""),
                 ("算数平均", s16["arith"], ""),
                 ("中位数", s16["median"], "")]
    body = f"""
{section("01", "第一原则：比较必须在同一作业内", "跨作业的毫秒数一律作废——不是保守，是实测有 3–7x 的性能态差。",
         f'''<div class="card">{diagram_ab_protocol()}
<div class="callout rose" style="margin-top:16px">
<b>集群 920F 的实测：</b>同一份二进制、同一个 shape，在 node03 的不同 sbatch 作业里可以读出
321.9 / 490.7 / 690.0 ms。任何「上次跑的是 X，这次是 Y」的对比都不可用。
本仓库所有集群结论都产自 <code>sbatch -w node03 --exclusive --wrap="bash tools/onednn/ab_onednn.sh"</code>
这一个作业内的交替测量。</div></div>''')}

{section("02", "第二个前提：绑核（本机）", "不绑核的 t16 数字会漂到结论翻转。",
         '''<div class="card">
<p style="margin:0 0 14px;font-size:13.5px">本机 608 核 / 32 NUMA、无调度器。OpenMP 线程若被放到不同 NUMA 节点，
内存访问代价随放置变化，SME 与 KML 的对比会失真。</p>
<table class="tbl">
<thead><tr><th>档位</th><th>绑核方式</th><th>不绑核的后果</th></tr></thead>
<tbody>
<tr><td><b>t1</b></td><td><code>taskset -c 3</code></td><td>单线程被迁移 → 缓存全失效，读数抖动 ±20%</td></tr>
<tr><td><b>t16</b></td><td><code>OMP_PROC_BIND=close OMP_PLACES=cores</code></td>
<td>SME 合计在 <b>387–558 ms</b> 间漂移，结论翻转成「SME 慢 2 倍」；绑核后两次重复差异 &lt; 2.5%</td></tr>
</tbody></table>
<div class="callout" style="margin-top:16px"><b>协议：</b>SME vs KML 的每一条数字都出自绑核运行；
每个配置重复两次，取同一作业内的读数。集群与本机是两套机器，毫秒数<b>不可跨机比较</b>，
只能各自纵向解读。</div></div>''')}

{section("03", "四种平均：回答四个不同的问题", "同一个比较会给出四个不同的数——差异本身也是信息。",
         f'''<div class="card">{diagram_metric_choice()}</div>
<div class="cols" style="margin-top:16px">
<div class="card"><h3>vs OpenBLAS（集群，t16）</h3>
{chart_metric_bars(ob_items, 1.0, 3.2, [1, 1.5, 2, 2.5, 3], "基线 1.0x；全部指标 &gt; 1，方向一致")}
<p style="margin:12px 0 0;font-size:12.5px;color:#64748b">中位数 1.77 &lt; 几何 1.99 &lt; 算数 2.17 &lt; 合计 2.50：
说明少数极大加速比（大形状）拉高了合计与算数平均，分布右偏。</p></div>
<div class="card"><h3>SME vs KML（本机，t16 绑核）</h3>
{chart_metric_bars(sme_items, 0.85, 1.2, [0.85, 0.9, 1.0, 1.1, 1.2], "基线 1.0x；合计与中位 &gt;1，几何 &lt;1")}
<p style="margin:12px 0 0;font-size:12.5px;color:#64748b">合计 1.117 与几何 0.901 同时成立：
SME 赢的是少数大 shape（省下的绝对时间多），输的是小空间大 IC（数量多）。
这正是「必须同时报四个数」的典型场景。</p></div>
</div>''')}

{section("04", "方向约定：比值的分子分母写清楚", "方向写反会让结论完全相反，这是本仓库踩过的坑。",
         '''<div class="card"><table class="tbl">
<thead><tr><th>对照</th><th>定义</th><th>&gt;1 的含义</th><th>仓库字段</th></tr></thead>
<tbody>
<tr><td>vs OpenBLAS</td><td><code>OpenBLAS / 本项目</code></td><td class="em">&gt;1 = 我们更快</td><td><code>total_speedup</code></td></tr>
<tr><td>SME vs KML · 合计</td><td><code>SME / KML</code></td><td class="warn">&lt;1 = SME 更快</td><td><code>total_ratio</code></td></tr>
<tr><td>SME vs KML · 逐形状</td><td><code>KML / SME</code></td><td class="em">&gt;1 = SME 更快</td><td>图表柱高</td></tr>
</tbody></table>
<div class="callout rose" style="margin-top:16px"><b>同一个「合计」有两套写法：</b>
时间比 <code>SME/KML = 0.896</code> 与加速比 <code>KML/SME = 1.117</code> 描述的是同一件事。
生成脚本里两个字段同时存在（<code>total_ratio</code> / <code>total_speedup</code>），
就是为了避免显示方向出错。</div></div>''')}

{section("05", "已知陷阱清单", "每一条都真实踩过，写在这里是为了不再重复。",
         '''<div class="card"><table class="tbl">
<thead><tr><th>陷阱</th><th>表现</th><th>正确做法</th></tr></thead>
<tbody>
<tr><td>debug 计时</td><td><code>WINO_GEMM_DEBUG=1</code> 会放大小形状耗时，甚至把比值翻倍</td>
<td>ours 侧只信无 debug 读数（E3/E5）</td></tr>
<tr><td>timing 总时间当绝对值</td><td>S 比真实单线程高 5–27%</td>
<td>占比看 S，绝对耗时看 T1</td></tr>
<tr><td>跨机比较</td><td>集群 321.9 ms 与本机 345.9 ms 不是同一件事</td>
<td>集群一套、本机一套，各自纵向比</td></tr>
<tr><td>跨作业比较</td><td>3–7x 性能态漂移</td><td>单 sbatch 作业内 A/B</td></tr>
<tr><td>oneDNN 逐形状数据</td><td>只存在于集群 <code>build/</code>，不在仓库</td>
<td>只能引用区间（e2e_wino 1.15–9.5x），不编造逐条</td></tr>
<tr><td>V 缓存容量</td><td><code>kMaxEntries=24</code>，59 层网络第 25 层起不命中</td>
<td>层数 &gt; 24 时把 V 缓存收益按 &lt; 一半计</td></tr>
</tbody></table></div>''')}

{section("06", "复现命令", "所有图表都能从仓库里的 markdown 表重新生成，不手改 HTML。",
         f'''<div class="card">{code_block("""# 1) 正确性门（fp64 参考 + 相对容差 1e-4）
./build/bench_winograd --sve --nhwc --verify --threads 16 shapes/conv_all.csv

# 2) 集群：单作业内 A/B（ours / oneDNN e2e / benchdnn）
sbatch -w node03 --exclusive --wrap="bash tools/onednn/ab_onednn.sh"
cat build/SUMMARY.txt

# 3) 本机：SME vs KML，必须绑核
taskset -c 3 ./build_sme_kml/bench_winograd --sme --nhwc --threads 1  shapes/conv_all.csv
OMP_PROC_BIND=close OMP_PLACES=cores ./build_sme_kml/bench_winograd --sme --nhwc --threads 16 shapes/conv_all.csv

# 4) 重新生成本套 HTML（静态、无外部依赖）
python3 tools/gen_report_figures.py""", "bash")}</div>''')}
"""
    return page(
        "⑦ 测量方法与判读口径",
        "怎么证明「快」不是测量假象：同作业 A/B、绑核、四种平均、方向约定与已知陷阱。",
        body, "07_methodology.html")


def page_roadmap(sme_st):
    s16 = sme_st["t16"]
    items = [
        ("1", "小形状 hybrid 调度", "按 IC×OH×OW 与 n_tiles 走 im2col + arm_gemm 直接卷积",
         "小形状 0.09–0.91x → ~1.0x", "中", "diff"),
        ("2", "SME：消掉 Ust→Ug 转置", "输入变换按 16 tile 一组输出，寄存器内做 16×16 转置写 K 主序",
         "大 IC 场景 U 流量减半", "低", "same"),
        ("3", "SME：OC 尾块选 tile 数", "按 OC%64 选 2/3/4 个 ZA tile（如 OC=96 → 3 tile × 2 窗）",
         "尾块省最多 25% fmopa", "低", "same"),
        ("4", "SME：drain 与 K 循环交错", "上一个 window 的 drain overlap 下一组 K",
         "小 K 形状 5–15%", "中", "diff"),
        ("5", "形状感知的后端选择", "按 n_tiles%16 / OC%64 / IC 运行时选 SME / KML / arm_gemm",
         "拼接三者优势区间", "中", "diff"),
        ("6", "F(4,4) 专用展开 SVE 变换", "按 Bᵀ/Aᵀ 零系数模式全展开，去掉循环与分支",
         "变换 −30~50%", "中", "diff"),
        ("7", "大 IC 通道分块", "按 L1 32 KB 切 IC 段（IC=384 时缓冲 ~150 KB）",
         "TileExt / InXform 改善 10–25%", "低", "same"),
        ("8", "V 缓存淘汰策略", "kMaxEntries=24 → LRU + 更大上限",
         "大 OC 小空间省 2–15%", "低", "same"),
        ("9", "SME 低精度外积（探索）", "smeb16f32 / smef16f32：bf16 输入 + fp32 累加",
         "精度可接受则 GEMM 翻倍", "高", "gap"),
    ]
    rows = "".join(
        f'<tr><td class="num">{n}</td><td><b>{esc(name)}</b></td><td>{esc(how)}</td>'
        f'<td class="num">{esc(exp)}</td><td><span class="tag {tag}">{risk}</span></td></tr>'
        for n, name, how, exp, risk, tag in items)

    falsified = [
        ("tile-major U/M 布局（消灭 scatter/gather）",
         "KML −14% / arm_gemm −16%", "GEMM 的 lda 从 IC 变成 36·IC，局部性损失大于拷贝收益"),
        ("transform_1d_sve 寄存器分块",
         "IC≥96 慢 20%", "IC=48 时快 1.4x，但收益面太窄"),
        ("M=25 时强选 GEMM 内核（6x4VL/8x1VL/inter）",
         "auto 已最优，强选 ±5%", "大形状反而慢 1.2–1.45x"),
        ("展平 N 个 batch 减少 barrier",
         "小内存 −14%，大内存 +76%", "U/M 内存放大 N 倍，无 L3 直接 cache thrashing"),
    ]
    frows = "".join(
        f'<tr><td><b>{esc(name)}</b></td><td class="num warn">{esc(res)}</td><td>{esc(why)}</td></tr>'
        for name, res, why in falsified)

    body = f"""
{section("01", "三阶段路线", "先做低风险确定性收益，再做结构性改造，最后才是探索。",
         f'''<div class="card">{diagram_phases()}</div>''')}

{section("02", "九项任务：做法 / 预期 / 风险", "每一项都能落到一个可测的数字上，而不是「继续优化」。",
         f'''<div class="card"><table class="tbl">
<thead><tr><th>#</th><th>方向</th><th>具体做法</th><th>预期收益</th><th>风险</th></tr></thead>
<tbody>{rows}</tbody></table></div>''')}

{section("03", "为什么第一阶段先做这四项", "只动局部实现、不引入新路径，收益立刻可验。",
         f'''<div class="cols">
<div class="card"><h3>它们是纯实现改动</h3>
<p style="margin:0 0 10px;font-size:13.5px">#2 / #3 / #7 / #8 都不改变执行路径与调度结构：
转置在寄存器里做、尾块换 tile 数、IC 分块调循环、缓存加淘汰。
失败可以逐项回退，不会污染主路径。</p>
<p style="margin:0;font-size:13.5px">验收只有一个判据：<b>59 形状无回退</b>，
并且 SME 合计时间比从当前的 <b>{s16['total_ratio']:.3f}</b> 降到 <b>≤0.85</b>。</p></div>
<div class="card"><h3>结构改造留到第二阶段</h3>
<p style="margin:0 0 10px;font-size:13.5px">hybrid 调度会引入第二条执行路径（im2col + 直接卷积），
需要阈值 sweep 与回归集，风险和标定成本都高于第一阶段的四项。</p>
<p style="margin:0;font-size:13.5px">判据：小形状 <b>≥0.95x</b>，
且 SME 在 <b>90%</b> 的形状上不劣于 KML——先补短板，再谈全面领先。</p></div>
</div>''')}

{section("04", "已被数据证伪的方向", "这些不是「没试过」，而是试过并测出退化；重试前先读结论。",
         f'''<div class="card"><table class="tbl">
<thead><tr><th>方向</th><th>实测结果</th><th>原因</th></tr></thead>
<tbody>{frows}</tbody></table>
<div class="callout" style="margin-top:16px"><b>方法论意义：</b>四条里三条是「看起来能省一次内存遍历」的直觉优化。
本仓库的规则是：<b>任何布局/结构改动先在 59 形状上 A/B</b>，
用几何平均和合计一起看，避免被单一形状的收益误导。</div></div>''')}

{section("05", "下一步的判定标准", "每条路线都要先写清楚「什么算成功」。",
         '''<div class="card"><table class="tbl">
<thead><tr><th>动作</th><th>成功判据</th><th>失败处理</th></tr></thead>
<tbody>
<tr><td>第二阶段完成后</td><td>SME 合计比 ≤0.85；小形状 ≥0.95x</td><td>保留 KML 作为默认后端，SME 只按形状启用</td></tr>
<tr><td>第三阶段完成后</td><td>变换占比 −30%；低精度相对误差 &lt; 1e-4</td><td>精度不达标则低精度只用于非关键层</td></tr>
<tr><td>任何改动</td><td>59 形状无回退 + <code>--verify</code> 全过</td><td>回退该项，不合并</td></tr>
</tbody></table></div>''')}
"""
    return page(
        "⑧ 演进路线与已证伪方向",
        "从 0.896 到 0.85、从小形状 0.09x 到 0.95x：每条路线都带做法、判据和失败处理。",
        body, "08_roadmap.html")


def page_shapes(ob_st):
    body = f"""
{section("01", "59 个形状怎么来的", "不是随机采样：七个小节各自回答一个假设。",
         f'''<div class="cols">
<div class="card">{chart_sections()}</div>
<div class="card"><table class="tbl">
<thead><tr><th>节</th><th>验证什么</th></tr></thead>
<tbody>
<tr><td>Case 0-8</td><td>与历史 F44/F22 的 9 case 数据直接接轨</td></tr>
<tr><td>ResNet50 3×3 s1</td><td>真实 CNN：mb=1 小负载 + mb=4；含降维 bottleneck 与 2048 深通道</td></tr>
<tr><td>VGG16 3×3 s1</td><td>224² 大图小通道、14²/7² 深通道</td></tr>
<tr><td>Lightweight</td><td>轻量网络常见形状，最小负载区间</td></tr>
<tr><td>Large batch</td><td>mb=8–32：若大负载仍赢，说明是系统性优势而非固定开销摊薄</td></tr>
<tr><td>Fixed-overhead</td><td>极小负载 + 高 iteration，隔离固定开销与调度成本</td></tr>
<tr><td>tiles × IC 网格</td><td>mb=4、OC=IC、tiles∈{{25,100,400,1600}} × IC∈{{48,96,192,384}} 的全组合</td></tr>
</tbody></table></div>
</div>''')}

{section("02", "tiles × IC 网格：加速比的空间结构", "颜色 = 对 OpenBLAS 的加速比；越靠右下越高。",
         f'''<div class="card">{chart_shape_grid(ob_st['rows'])}
<div class="callout indigo" style="margin-top:16px">
<b>读图：</b>左上角（25 tiles、IC=48）只有 <b>1.19x</b>，右下角（1600 tiles、IC=384）达到 <b>3.02x</b>；
沿「tile 数↑」方向（固定 IC=48）从 1.19 → 1.59 → 1.66 → 1.71，
沿「IC↑」方向（固定 25 tiles）从 1.19 → 1.66 → 1.26 → 1.65。
两个方向都能抬升，说明收益来自<b>摊薄固定开销 + GEMM 规模变大</b>，而不是某个单一形状的偶然读数。
悬停可看到重复格点的两次测量值。</div></div>''')}

{section("03", "形状参数的含义", "所有行都是 stride=1、group=1、3×3、pad=1——这是 F(4,4,3,3) 的适用域。",
         f'''<div class="card"><table class="tbl">
<thead><tr><th>量</th><th>计算</th><th>为什么重要</th></tr></thead>
<tbody>
<tr><td>tile 数</td><td><code>ceil(ih/4)² × mb</code></td><td>每 tile 做 36 个 GEMM；tile 越多，GEMM 的 n 越大、固定开销越摊薄</td></tr>
<tr><td>IC</td><td>输入通道</td><td>GEMM 的 K 维；也是输入变换的数据量，决定 SVE 谓词利用率（IC=48 只用到 3/4 个寄存器）</td></tr>
<tr><td>OC</td><td>输出通道</td><td>GEMM 的 N 维；SME 路径按 64 一块（4 个 ZA tile），OC%64 决定尾块浪费</td></tr>
<tr><td>mb</td><td>batch</td><td>线性放大 tile 数，是「大负载」的主要来源</td></tr>
</tbody></table>
<div class="callout" style="margin-top:16px"><b>注意：</b>59 行里 4 个格点与 Case 0-8 刻意重叠
（如 4,192,40²,192 出现两次）。它们的 GEMM 读数一致、内存步可差 65%+，
是本套用例自带的噪声探针。</div></div>''')}

{section("04", "怎么跑这套用例", "同一条 CSV 同时喂给我们、oneDNN e2e 与 benchdnn。",
         f'''<div class="card">{code_block("""# 正确性门（FAIL 即中止）
./build/bench_winograd --sve --nhwc --verify --threads 16 shapes/conv_all.csv

# 全量计时：1/8/16/32/38 线程
./build/bench_winograd --sve --nhwc --threads 1,8,16,32,38 shapes/conv_all.csv

# 转成 benchdnn 描述符（mb4_ic192_ih40iw40_oc192_oh40ow40_kh3kw3_sh1sw1_ph1pw1）
bash tools/gen_benchdnn_list.sh

# oneDNN 侧两列：强制 WINO 与 auto
bash tools/onednn/run_benchdnn.sh --winograd
bash tools/onednn/run_benchdnn.sh""", "bash")}
<p style="margin:14px 0 0;font-size:12.5px;color:#64748b">
CSV 全表：<code>shapes/conv_all.csv</code>；分节说明：<code>shapes/README.md</code>；
对照工具：<code>tools/onednn/</code>。</p></div>''')}
"""
    return page(
        "⑨ 基准用例集：59 个形状",
        "七个小节、59 行；每个小节对应一个待验证的假设，而不是随机堆形状。",
        body, "09_bench_shapes.html")


# ---------------------------------------------------------------- main

def main():
    ob_rows = parse_ob(os.path.join(ROOT, "docs", "final_benchmark_bfd6b1e.md"))
    if len(ob_rows) != 59:
        print(f"warning: parsed {len(ob_rows)} rows from final_benchmark (expected 59)", file=sys.stderr)
    ob_st = stats(ob_rows, "ours", "base")
    ob_st["rows"] = ob_rows

    sme = parse_sme(os.path.join(ROOT, "docs", "sme_fmopa_gemm.md"))
    for k in ("t1", "t16"):
        if len(sme[k]) != 59:
            print(f"warning: parsed {len(sme[k])} rows for {k} (expected 59)", file=sys.stderr)
    sme_st = {"t1": stats(sme["t1"], "sme", "kml"),
              "t16": stats(sme["t16"], "sme", "kml")}

    steps = parse_steps(os.path.join(ROOT, "docs", "timing_breakdown_920f.md"))
    timing = parse_timing(os.path.join(ROOT, "docs", "timing_breakdown_920f.md"))
    for name, rows, want in (("steps", steps, 8), ("timing", timing, 59)):
        if len(rows) != want:
            print(f"warning: parsed {len(rows)} {name} rows (expected {want})", file=sys.stderr)

    os.makedirs(OUT, exist_ok=True)
    pages = {
        "index.html": page_index(ob_st, sme_st),
        "01_winograd_principles.html": page_principles(),
        "02_system_architecture.html": page_architecture(ob_st),
        "03_sme_gemm.html": page_sme(sme_st),
        "04_performance.html": page_performance(ob_st, sme_st),
        "05_acl_comparison.html": page_acl(),
        "06_timing.html": page_timing(steps, timing, ob_st, sme_st),
        "07_methodology.html": page_methodology(ob_st, sme_st),
        "08_roadmap.html": page_roadmap(sme_st),
        "09_bench_shapes.html": page_shapes(ob_st),
    }
    for name, html in pages.items():
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"wrote {os.path.relpath(path, ROOT)}  ({len(html)/1024:.1f} KB)")

    print(f"\nsummary: ob speedup={ob_st['total_speedup']:.3f} (ratio {ob_st['total_ratio']:.3f}) geo={ob_st['geo']:.3f} "
          f"arith={ob_st['arith']:.3f} median={ob_st['median']:.3f}")
    for k in ("t1", "t16"):
        s = sme_st[k]
        print(f"summary: sme {k} ratio={s['total_ratio']:.3f} speedup={s['total_speedup']:.3f} geo={s['geo']:.3f} "
              f"arith={s['arith']:.3f} median={s['median']:.3f} wins={s['wins']}")


if __name__ == "__main__":
    main()
