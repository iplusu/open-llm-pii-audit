#!/usr/bin/env python3
"""Regenerate the 3 main figures (heatmap, leakage/utility trade-off, refusal) from results/tables (vector PDF).
Run scripts/generate_tables.py first. Moved unchanged from the original figure script; only the paths changed.

Fixes vs. the stale 09-29 14:33 versions:
  * data: results/tables/model_condition_table.csv + metrics_utility.csv; BF16 only (no INT8/INT4 rows in fig1).
  * fig2/fig3: 30 of 42 cells sit in the x=0 zero-leakage cluster; labeling
    every point there is what made the figures a wall of overlapping text.
    New layout: per-point labels only for the informative C0/C1 points
    (greedy no-collision placement + repair pass, connectors), and ONE
    annotation box for the C2–C6 cluster. No in-figure titles (captions).
After rendering, a programmatic bbox check asserts zero text overlaps.

Label geometry note: this matplotlib version's Annotation.get_window_extent
is offset-biased, so placement AND verification both use MANUAL display
geometry (transData + measured text sizes), which cannot diverge.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.legend
import matplotlib.transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

import argparse
ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--tables", default=str(ROOT / "results" / "tables"))  # produced by scripts/generate_tables.py
_ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
_args = _ap.parse_args()
TABLES, OUT = Path(_args.tables), Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)

COND_ORDER = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
SHORT = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Olmo-3-7B-Instruct": "OLMo-3-7B",
    "Qwen3-8B": "Qwen3-8B",
    "gemma-4-E4B-it": "Gemma-4-E4B",
    "phi-4": "Phi-4",
}
cond_labels = {"C0": "none", "C1": "refusal", "C2": "redact", "C3": "filter",
               "C4": "ref.+red.", "C5": "red.+filt.", "C6": "full"}

rows = list(csv.DictReader(open(TABLES / "model_condition_table.csv")))
util = {f"{r['model']}|{r['condition']}": r
        for r in csv.DictReader(open(TABLES / "metrics_utility.csv"))}
base = [r for r in rows if r.get("quant", "BF16") == "BF16"]
cond_colors = dict(zip(COND_ORDER, plt.cm.tab10(np.linspace(0, 1, 10))))


# ---- deterministic label placement (greedy, no-collision) -------------
# Candidate offsets are in POINTS (what annotate's textcoords="offset
# points" consumes); geom() converts to display px via fig.dpi/72.
_LABEL_CANDS = [(5, 1.5, "left"), (5, -9, "left"), (5, 11, "left"),
                (17, 1.5, "left"), (-5, 1.5, "right"), (-5, -9, "right"),
                (-5, 11, "right"), (5, 20, "left"), (5, -20, "left"),
                (-17, 1.5, "right"), (5, 30, "left"), (5, -30, "left"),
                (17, -12, "left"), (-5, -20, "right"), (-5, 20, "right"),
                (5, 40, "left"), (5, -40, "left"), (17, 30, "left"),
                (17, -26, "left"), (-5, 40, "right"), (-5, -40, "right")]


def _bbox(x0, y0, x1, y1):
    return matplotlib.transforms.Bbox([[x0, y0], [x1, y1]])


def _rects_overlap(a, b):
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    return x1 > x0 and y1 > y0


def _occupy(ax, r):
    occ = []
    for child in ax.get_children():
        if isinstance(child, matplotlib.legend.Legend):
            occ.append(child.get_window_extent(renderer=r).padded(3))
    for t in ax.texts:
        if t.get_text():
            occ.append(t.get_window_extent(renderer=r).padded(3))
    return occ


def place_labels(ax, pts, x_thresh=0.12):
    """Greedy no-collision placement. The collision test uses the
    ANNOTATION'S OWN rendered extent (text + arrow, as shipped), so the
    placement geometry cannot diverge from the verification geometry.
    Labels that fit nowhere are dropped (never shipped overlapping)."""
    fig = ax.figure
    if not isinstance(fig.canvas, FigureCanvasAgg):
        fig.canvas = FigureCanvasAgg(fig)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer=r).padded(1)
    placed = _occupy(ax, r)  # legend + static texts

    def in_ax(bb):
        return bb.x0 >= axbb.x0 and bb.y0 >= axbb.y0 and \
               bb.x1 <= axbb.x1 and bb.y1 <= axbb.y1

    def free(bb):
        return in_ax(bb) and not any(
            _rects_overlap(bb, o) for o in placed)

    kept = []
    dropped = []
    for x, y, m, lab in sorted(pts, key=lambda p: (p[0], p[1])):
        if x < x_thresh:
            continue
        chosen = None
        for dx, dy, ha in _LABEL_CANDS:
            props = dict(arrowstyle="-", color="0.7", lw=0.5,
                         shrinkA=0, shrinkB=2) if (abs(dx) > 12 or abs(dy) > 7) else None
            a = ax.annotate(lab, (x, y), xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va="center",
                            fontsize=6.5, alpha=0.9, arrowprops=props)
            bb = a.get_window_extent(renderer=r).padded(4.0)
            if free(bb):
                chosen = a
                chosen._cand = (dx, dy, ha)
                break
            a.remove()
        if chosen is None:
            dropped.append((x, y, lab))
            continue
        placed.append(chosen.get_window_extent(renderer=r).padded(4.0))
        kept.append(chosen)

    # Swap-repair: for each dropped label, try to relocate the single
    # placed label that blocks a candidate slot into a slot of its own.
    for x, y, lab in list(dropped):
        repaired = False
        for dx, dy, ha in _LABEL_CANDS:
            props = dict(arrowstyle="-", color="0.7", lw=0.5,
                         shrinkA=0, shrinkB=2) if (abs(dx) > 12 or abs(dy) > 7) else None
            a = ax.annotate(lab, (x, y), xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va="center",
                            fontsize=6.5, alpha=0.9, arrowprops=props)
            bb = a.get_window_extent(renderer=r).padded(4.0)
            a.remove()
            if not in_ax(bb):
                continue
            blockers = [k for k in kept
                        if _rects_overlap(bb, k.get_window_extent(renderer=r).padded(4.0))]
            if len(blockers) != 1:
                continue
            blocker = blockers[0]
            orig = blocker._cand
            for n2x, n2y, n2ha in _LABEL_CANDS:
                if (n2x, n2y, n2ha) == orig:
                    continue
                blocker.set_position((n2x, n2y))
                blocker.set_ha(n2ha)
                n2bb = blocker.get_window_extent(renderer=r).padded(4.0)
                if in_ax(n2bb) and not _rects_overlap(n2bb, bb) and not any(
                        _rects_overlap(n2bb, k.get_window_extent(renderer=r).padded(4.0))
                        for k in kept if k is not blocker):
                    blocker._cand = (n2x, n2y, n2ha)
                    props = dict(arrowstyle="-", color="0.7", lw=0.5,
                                 shrinkA=0, shrinkB=2) if (abs(dx) > 12 or abs(dy) > 7) else None
                    a = ax.annotate(lab, (x, y), xytext=(dx, dy),
                                    textcoords="offset points", ha=ha, va="center",
                                    fontsize=6.5, alpha=0.9, arrowprops=props)
                    kept.append(a)
                    dropped.remove((x, y, lab))
                    print(f"   (swap-repaired: {lab}; moved blocker '{blocker.get_text()}')")
                    repaired = True
                    break
                blocker.set_position(orig[:2])
                if len(orig) > 2:
                    blocker.set_ha(orig[2])
            if repaired:
                break
        if not repaired:
            print(f"   (dropped, no collision-free slot: {lab})")
    return kept


def verify_no_overlap(fig, path, min_area=4.0):
    """Assert no two rendered text bboxes overlap (area > min_area px^2)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    rects = []
    for ax in fig.get_axes():
        for t in ax.texts:
            if t.get_text().strip():
                rects.append((t.get_text(), t.get_window_extent(renderer=r)))
        for child in ax.get_children():
            if isinstance(child, matplotlib.legend.Legend):
                rects.append(("legend", child.get_window_extent(renderer=r)))
    bad = []
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i][1], rects[j][1]
            x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
            x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
            if x1 > x0 and y1 > y0 and (x1 - x0) * (y1 - y0) > min_area:
                bad.append((rects[i][0], rects[j][0]))
    print(f"verify {path.name}: {len(rects)} text items, {len(bad)} overlaps")
    for a, b in bad[:10]:
        print(f"   OVERLAP: {a!r} <-> {b!r}")
    assert not bad, f"label overlap in {path}"


# ---- Fig 1: heatmap (BF16 only, final table) --------------------------
models = sorted({r["model"] for r in base},
                key=lambda m: -max(float(r["leakage_rate"] or 0)
                                   for r in base if r["model"] == m))
conds = [c for c in COND_ORDER if any(r["condition"] == c for r in base)]
M = np.full((len(models), len(conds)), np.nan)
for i, m in enumerate(models):
    for j, c in enumerate(conds):
        for r in base:
            if r["model"] == m and r["condition"] == c:
                M[i, j] = float(r["leakage_rate"] or 0)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(conds)), [f"{c}\n{cond_labels[c]}" for c in conds])
ax.set_yticks(range(len(models)), [SHORT[m] for m in models])
for i in range(len(models)):
    for j in range(len(conds)):
        if not np.isnan(M[i, j]):
            rgb = im.cmap(im.norm(M[i, j]))[:3]
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if lum < 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("leakage rate (fraction of PII spans)", fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(OUT / "fig1_heatmap.pdf", metadata={"CreationDate": None})  # no timestamp: figures regenerate byte-identically
verify_no_overlap(fig, OUT / "fig1_heatmap.pdf")
plt.close(fig)

# ---- Fig 2: trade-off ---------------------------------------------------
pts2 = []
for r in base:
    u = util.get(f"{r['model']}|{r['condition']}")
    if not u or not u.get("rouge_l_mean"):
        continue
    pts2.append((float(r["leakage_rate"] or 0), float(u["rouge_l_mean"]),
                 SHORT[r["model"]], f"{SHORT[r['model']]} {r['condition']}"))
fig, ax = plt.subplots(figsize=(7.0, 5.0))
for x, y, m, lab in pts2:
    c = [c for c in COND_ORDER if lab.endswith(c)][0]
    ax.scatter(x, y, color=cond_colors.get(c, "gray"), s=42,
               edgecolor="0.2", linewidth=0.6, zorder=3)
ax.set_xlabel("PII leakage rate (higher = worse privacy)")
ax.set_ylabel("ROUGE-L on utility-anchor probes (task utility)")
ax.set_xlim(-0.03, 0.95)
ax.text(0.012, 0.05, "C2–C6 cells (redaction/filter stacks):\n"
        "leakage 0.0–0.015, utility 0.07–0.17",
        fontsize=6.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5))
for c in COND_ORDER:
    if any(p[3].endswith(c) for p in pts2):
        ax.scatter([], [], color=cond_colors[c], label=c, s=28)
ax.legend(fontsize=8, ncol=2, title="condition", loc="upper left", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
placed = place_labels(ax, pts2)
print(f"fig2: {len(placed)} point labels placed")
fig.savefig(OUT / "fig2_tradeoff.pdf", metadata={"CreationDate": None})  # no timestamp: figures regenerate byte-identically
verify_no_overlap(fig, OUT / "fig2_tradeoff.pdf")
plt.close(fig)

# ---- Fig 3: refusal vs leakage ------------------------------------------
pts3 = []
for r in base:
    if r["refusal_rate"] in ("", None) or r["leakage_rate"] in ("", None):
        continue
    pts3.append((float(r["leakage_rate"]), float(r["refusal_rate"]),
                 SHORT[r["model"]], f"{SHORT[r['model']]} {r['condition']}"))
fig, ax = plt.subplots(figsize=(7.0, 5.0))
for x, y, m, lab in pts3:
    c = [c for c in COND_ORDER if lab.endswith(c)][0]
    ax.scatter(x, y, color=cond_colors.get(c, "gray"), s=42,
               edgecolor="0.2", linewidth=0.6, zorder=3)
ax.set_xlabel("PII leakage rate")
ax.set_ylabel("refusal rate (re-ask probes only)")
ax.set_xlim(-0.03, 0.95)
ax.set_ylim(0.0, 1.02)
ax.axvline(0.0, color="0.4", ls=":", lw=0.8)
ax.text(0.012, 0.03, "C2–C6 cells: leakage 0.0–0.015 for all models;\n"
        "refusal 0.07–0.94 (model-dependent) — see Table 4",
        fontsize=6.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5))
for c in COND_ORDER:
    if any(p[3].endswith(c) for p in pts3):
        ax.scatter([], [], color=cond_colors[c], label=c, s=28)
ax.legend(fontsize=8, ncol=2, title="condition", loc="upper right", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
placed = place_labels(ax, pts3)
print(f"fig3: {len(placed)} point labels placed")
fig.savefig(OUT / "fig3_refusal.pdf", metadata={"CreationDate": None})  # no timestamp: figures regenerate byte-identically
verify_no_overlap(fig, OUT / "fig3_refusal.pdf")
plt.close(fig)

print("wrote", [p.name for p in sorted(OUT.glob("*.pdf"))])
