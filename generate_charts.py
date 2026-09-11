#!/usr/bin/env python3
"""Generate SVG charts for the README from results/sampled_reviews.csv.

Outputs charts/*.svg using presentation attributes only (GitHub-safe, no CSS
classes, no scripts): confusion_matrix.svg, class_distribution.svg,
per_class_metrics.svg, correct_split.svg.
"""
import csv
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "results" / "sampled_reviews.csv"
OUT_DIR = BASE / "charts"

CLASSES = ["positive", "neutral", "negative"]
CLASS_COLORS = {"positive": "#16a34a", "neutral": "#d97706", "negative": "#dc2626"}
INK = "#111827"
MUTED = "#6b7280"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load():
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def metrics(rows):
    n = len(rows)
    correct = sum(r["true_label"] == r["predicted_label"] for r in rows)
    per = {}
    for c in CLASSES:
        tp = sum(r["true_label"] == c and r["predicted_label"] == c for r in rows)
        fp = sum(r["true_label"] != c and r["predicted_label"] == c for r in rows)
        fn = sum(r["true_label"] == c and r["predicted_label"] != c for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per[c] = {"n": sum(r["true_label"] == c for r in rows), "p": p, "r": r, "f1": f1}
    conf = [[sum(r["true_label"] == a and r["predicted_label"] == b for r in rows)
             for b in CLASSES] for a in CLASSES]
    return n, correct, per, conf


def svg_base(w, h, title, subtitle=None):
    sub = (f'<text x="20" y="50" font-family="{FONT}" font-size="12" fill="{MUTED}">{esc(subtitle)}</text>'
           if subtitle else "")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>'
            f'<text x="20" y="28" font-size="15" font-weight="700" fill="{INK}">{esc(title)}</text>{sub}')


def confusion_svg(per, conf):
    W, H = 540, 330
    x0, y0, cw, ch = 150, 92, 120, 58
    s = [svg_base(W, H, "Confusion matrix — true vs predicted",
                  "rows: star-rating label · columns: model prediction")]
    for j, c in enumerate(CLASSES):  # column headers
        s.append(f'<circle cx="{x0 + j * cw + cw / 2 - 44}" cy="74" r="4" fill="{CLASS_COLORS[c]}"/>')
        s.append(f'<text x="{x0 + j * cw + cw / 2 - 36}" y="78" font-size="12" fill="{MUTED}" '
                 f'font-weight="600">pred {c}</text>')
    for i, c in enumerate(CLASSES):  # row labels
        s.append(f'<circle cx="46" cy="{y0 + i * ch + ch / 2 - 4}" r="4" fill="{CLASS_COLORS[c]}"/>')
        s.append(f'<text x="56" y="{y0 + i * ch + ch / 2}" font-size="12.5" fill="{INK}" '
                 f'font-weight="600">true {c}</text>')
    for i in range(3):
        for j in range(3):
            x, y = x0 + j * cw, y0 + i * ch
            diag = i == j
            fill = CLASS_COLORS[CLASSES[j]] if diag else "#f6f7f9"
            fill_op = "0.16" if diag else "1"
            txtfill = CLASS_COLORS[CLASSES[j]] if diag else INK
            s.append(f'<rect x="{x}" y="{y}" width="{cw - 6}" height="{ch - 6}" rx="6" '
                     f'fill="{fill}" fill-opacity="{fill_op}"/>')
            s.append(f'<text x="{x + (cw - 6) / 2}" y="{y + (ch - 6) / 2 + 5}" text-anchor="middle" '
                     f'font-size="16" font-weight="700" fill="{txtfill}">{conf[i][j]}</text>')
            if not diag and conf[i][j]:
                s.append(f'<text x="{x + (cw - 6) / 2}" y="{y + (ch - 6) / 2 + 21}" text-anchor="middle" '
                         f'font-size="10" fill="{MUTED}">mismatch</text>')
    s.append(f'<text x="20" y="314" font-size="11" fill="{MUTED}">diagonal = agreement with star rating</text>')
    s.append("</svg>")
    return "".join(s)


def distribution_svg(rows):
    W, H = 540, 320
    true_d = Counter(r["true_label"] for r in rows)
    pred_d = Counter(r["predicted_label"] for r in rows)
    maxv = max(max(true_d.values()), max(pred_d.values()))
    x0, xmax, y0 = 150, 510, 96
    scale = (xmax - x0) / (maxv * 1.15)
    s = [svg_base(W, H, "Label distribution — stars vs model",
                  "count of reviews per class under each labeling scheme")]
    # legend (neutral gray swatches — bars themselves are class-colored)
    s.append(f'<rect x="20" y="68" width="10" height="10" rx="2" fill="#9ca3af" fill-opacity="0.4"/>')
    s.append(f'<text x="36" y="77" font-size="11.5" fill="{MUTED}">star rating (ground truth)</text>')
    s.append(f'<rect x="190" y="68" width="10" height="10" rx="2" fill="#4b5563"/>')
    s.append(f'<text x="206" y="77" font-size="11.5" fill="{MUTED}">model prediction</text>')
    bh, gap = 17, 7
    for i, c in enumerate(CLASSES):
        y = y0 + i * (2 * bh + gap + 22)
        col = CLASS_COLORS[c]
        # true bar
        w1 = true_d[c] * scale
        s.append(f'<rect x="{x0}" y="{y}" width="{max(w1, 2)}" height="{bh}" rx="4" fill="{col}" fill-opacity="0.35"/>')
        s.append(f'<text x="{x0 + w1 + 6}" y="{y + bh - 3}" font-size="11.5" font-weight="600" fill="{INK}">{true_d[c]}</text>')
        # pred bar
        w2 = pred_d[c] * scale
        s.append(f'<rect x="{x0}" y="{y + bh + gap}" width="{max(w2, 2)}" height="{bh}" rx="4" fill="{col}"/>')
        s.append(f'<text x="{x0 + w2 + 6}" y="{y + bh + gap + bh - 3}" font-size="11.5" font-weight="600" fill="{INK}">{pred_d[c]}</text>')
        s.append(f'<text x="{x0 - 10}" y="{y + bh + 2}" text-anchor="end" font-size="12" fill="{INK}" '
                 f'font-weight="600">{c}</text>')
    # scale line
    s.append(f'<line x1="{x0}" y1="{y0 + 3 * (2*bh + gap + 22) - 10}" x2="{xmax}" y2="{y0 + 3 * (2*bh + gap + 22) - 10}" stroke="#e5e7eb"/>')
    s.append(f'<text x="150" y="314" font-size="11" fill="{MUTED}">reviews (n = {len(rows)})</text>')
    s.append("</svg>")
    return "".join(s)


def per_class_svg(per):
    W, H = 540, 330
    y0, plot_h, x0 = 96, 190, 60
    s = [svg_base(W, H, "Per-class metrics — precision, recall, F1",
                  "macro F1 = 0.596 (unweighted mean of per-class F1)")]
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y0 + plot_h - gv * plot_h
        s.append(f'<line x1="{x0}" y1="{y}" x2="520" y2="{y}" stroke="#eef1f5"/>')
        s.append(f'<text x="{x0 - 6}" y="{y + 4}" text-anchor="end" font-size="10" fill="{MUTED}">{gv:.2f}</text>')
    group_w, bar_w, gap = 145, 34, 9
    for gi, c in enumerate(CLASSES):
        gx = x0 + 16 + gi * group_w
        col = CLASS_COLORS[c]
        for bi, (key, op, label) in enumerate((("p", "0.25", "precision"), ("r", "0.55", "recall"), ("f1", "1", "F1"))):
            v = per[c][key]
            bh = max(v * plot_h, 2)
            x = gx + bi * (bar_w + gap)
            y = y0 + plot_h - bh
            s.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" rx="4" fill="{col}" fill-opacity="{op}"/>')
            s.append(f'<text x="{x + bar_w / 2}" y="{y - 5}" text-anchor="middle" font-size="10" '
                     f'fill="{MUTED}">{v:.3f}</text>')
            if gi == 0:
                s.append(f'<text x="{x + bar_w / 2}" y="{y0 + plot_h + 14}" text-anchor="middle" '
                         f'font-size="10" fill="{MUTED}">{label}</text>')
        s.append(f'<text x="{gx + 1.5 * bar_w + gap}" y="{y0 + plot_h + 14}" text-anchor="middle" '
                 f'font-size="12" font-weight="600" fill="{INK}">{c} (n={per[c]["n"]})</text>')
    s.append(f'<text x="20" y="314" font-size="11" fill="{MUTED}">F1 = harmonic mean of precision and recall</text>')
    s.append("</svg>")
    return "".join(s)


def split_svg(n, correct):
    W, H = 360, 170
    wrong = n - correct
    pct = correct / n * 100
    s = [svg_base(W, H, "Overall accuracy", "per-review agreement with star-rating label")]
    C = 2 * 3.14159 * 44
    s.append(f'<circle cx="84" cy="98" r="44" fill="none" stroke="#16a34a" stroke-width="22" '
             f'pathLength="100" stroke-dasharray="{correct / n * 100} 100" stroke-dashoffset="25" transform="rotate(-90 84 98)"/>')
    s.append(f'<circle cx="84" cy="98" r="44" fill="none" stroke="#fca5a5" stroke-width="22" '
             f'pathLength="100" stroke-dasharray="{wrong / n * 100} 100" stroke-dashoffset="{25 - correct / n * 100}" transform="rotate(-90 84 98)"/>')
    s.append(f'<text x="84" y="94" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">{pct:.1f}%</text>')
    s.append(f'<text x="84" y="112" text-anchor="middle" font-size="10.5" fill="{MUTED}">accuracy</text>')
    s.append(f'<rect x="170" y="74" width="12" height="12" rx="3" fill="#16a34a"/>')
    s.append(f'<text x="190" y="84" font-size="12" fill="{INK}">correct ({correct})</text>')
    s.append(f'<rect x="170" y="98" width="12" height="12" rx="3" fill="#fca5a5"/>')
    s.append(f'<text x="190" y="108" font-size="12" fill="{INK}">mismatch ({wrong})</text>')
    s.append(f'<text x="170" y="130" font-size="11" fill="{MUTED}">n = {n} · seed 42</text>')
    s.append("</svg>")
    return "".join(s)


def main():
    rows = load()
    n, correct, per, conf = metrics(rows)
    OUT_DIR.mkdir(exist_ok=True)
    charts = {
        "correct_split.svg": split_svg(n, correct),
        "confusion_matrix.svg": confusion_svg(per, conf),
        "class_distribution.svg": distribution_svg(rows),
        "per_class_metrics.svg": per_class_svg(per),
    }
    for name, svg in charts.items():
        (OUT_DIR / name).write_text(svg, encoding="utf-8")
        print(f"wrote charts/{name} ({len(svg) / 1024:.1f} KB)")
    print(f"\ncorrect={correct}/{n} ({correct / n:.1%})  wrong={n - correct}")
    for c in CLASSES:
        m = per[c]
        print(f"{c:<9} n={m['n']:>3}  P={m['p']:.3f}  R={m['r']:.3f}  F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
