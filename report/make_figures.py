#!/usr/bin/env python3
"""Generate the report figures from the measured results.

Every number here is transcribed from a real Kaggle run on the frozen test set or the
training logs; nothing is simulated. Re-run after any new evaluation.

    python report/make_figures.py

Colours come from the project's validated categorical palette (slot 1 blue, slot 2 orange)
and a single-hue blue ramp for magnitude. One axis per chart, legend whenever two series
share a panel, recessive grid, no number printed on every point.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dddcd8"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------------------- data
CONFUSION = np.array([[298, 2, 0, 0, 0],
                      [0, 285, 15, 0, 0],
                      [0, 4, 263, 33, 0],
                      [0, 0, 8, 242, 50],
                      [0, 0, 0, 22, 278]])

SI_SDRI_PER_N = [7.91, 6.15, 5.13, 4.36, 3.47]          # section 4: all clips, true count

SEP_FROZEN = [0.62, 1.89, 2.53, 2.75, 2.73, 2.56, 2.49, 2.99, 2.93, 3.33,
              3.40, 2.58, 3.39, 3.06, 3.61, 3.53, 3.54, 3.80, 3.92, 4.08,
              4.00, 4.06, 4.25, 4.39, 4.39, 4.44, 4.48, 4.51, 4.51, 4.51]
SEP_FRESH = [0.70, 2.12, 2.62, 2.80, 2.83, 2.76, 3.10, 2.62, 3.11, 3.15,
             2.85, 3.37, 3.33, 3.39, 2.60, 3.50, 3.78, 3.50, 3.55, 4.13,
             4.17, 4.24, 4.31, 4.33, 4.48, 4.51, 4.55, 4.59, 4.59, 4.59]

# Counter, both runs. Validation and training accuracy per epoch (%).
# Frozen-data figures read from that run's train_report.json (code_version 6de9a43).
CNT_FRESH_VAL = [74.9, 61.8, 73.4, 72.4, 76.8, 80.4, 86.8, 84.1, 77.8, 84.8,
                 81.8, 88.1, 88.5, 90.3, 90.5, 89.7, 90.0, 90.5, 90.2, 90.6]
CNT_FRESH_TRAIN = [68.8, 78.6, 80.8, 82.9, 83.8, 84.9, 85.6, 86.2, 86.9, 87.4,
                   87.7, 88.3, 88.5, 88.8, 89.1, 89.4, 89.5, 89.7, 89.9, 89.8]
CNT_FROZEN_VAL = [73.33, 61.73, 74.20, 70.93, 81.40, 78.53, 79.67, 85.07, 83.60, 81.00,
                  78.33, 78.27, 80.47, 79.80, 81.60, 82.20, 83.27, 83.53, 83.73, 83.27]
CNT_FROZEN_TRAIN = [68.99, 78.73, 80.95, 82.99, 84.72, 86.36, 88.02, 89.72, 91.39, 93.13,
                    94.51, 95.82, 96.79, 97.60, 98.28, 98.88, 99.25, 99.45, 99.44, 99.54]

MASK_N = [2, 3, 4, 5]
MASK_OVERLAP = [0.151, 0.211, 0.234, 0.266]
MASK_SISDRI = [6.448, 4.942, 4.355, 3.544]
WITHIN_OVERLAP = {2: -0.60, 3: -0.55, 4: -0.49, 5: -0.63}
WITHIN_SPARSITY = {1: -0.73, 2: 0.14, 3: 0.25, 4: 0.32, 5: 0.24}


# --------------------------------------------------------------------------- fig 1
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(-5, 40); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, label, sub, colour, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    linewidth=1.6, edgecolor=colour, facecolor=fill))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=8.5, color=INK2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.4, color=INK2, shrinkA=0, shrinkB=0))

    box(1, 15, 17, 10, "3 s @ 8 kHz", "RMS-normalised", INK2, "#f0efec")
    box(31, 26, 22, 11, "CountCRNN", "0.488 M params", BLUE, "#eaf2fd")
    box(31, 3, 22, 11, "SepNet", "Conv-TasNet, 5.25 M", ORANGE, "#fdefe9")
    box(64, 15, 20, 10, "keep the N̂", "loudest slots", AQUA, "#e8f7f1")

    arrow(18.5, 20, 30.5, 31); arrow(18.5, 20, 30.5, 8.5)
    arrow(53.5, 31, 74, 25.5); arrow(53.5, 8.5, 74, 14.5)
    arrow(84.5, 20, 95, 20)

    ax.text(60, 33.5, "N̂ ∈ 1..5", fontsize=9, color=BLUE, ha="center", fontweight="bold")
    ax.text(60, -2.5, "5 speaker slots + 1 noise", fontsize=9, color=ORANGE, ha="center")
    ax.text(96, 20, "speaker_01.wav\nspeaker_02.wav\n…", fontsize=9, color=INK,
            ha="left", va="center")
    ax.text(50, 39, "Two specialists, trained apart, joined only at inference",
            fontsize=10.5, style="italic", color=INK2, ha="center")
    save(fig, "fig1_architecture.png")


# --------------------------------------------------------------------------- fig 2
def fig_confusion():
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.grid(False)
    norm = CONFUSION / CONFUSION.sum(axis=1, keepdims=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", RAMP)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)

    for i in range(5):
        for j in range(5):
            v = CONFUSION[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                    fontweight="bold" if i == j else "normal",
                    color="#ffffff" if norm[i, j] > 0.45 else INK)
    ax.set_xticks(range(5), [f"{n}" for n in range(1, 6)])
    ax.set_yticks(range(5), [f"{n}" for n in range(1, 6)])
    ax.set_xlabel("predicted speaker count"); ax.set_ylabel("true speaker count")
    ax.set_title("Counting on the frozen test set\n"
                 "91.1 % accuracy, MAE 0.089, 1500 mixtures",
                 fontsize=11, color=INK, pad=12)
    fig.text(0.5, -0.01, "every error is off by exactly one — the matrix is tridiagonal",
             ha="center", fontsize=9, style="italic", color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig2_confusion_matrix.png")


# --------------------------------------------------------------------------- fig 3
def fig_per_n():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    xs = np.arange(1, 6)
    ax.bar(xs, SI_SDRI_PER_N, width=0.62, color=BLUE, zorder=3)
    for x, v in zip(xs, SI_SDRI_PER_N):
        ax.text(x, v + 0.18, f"{v:+.2f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(xs, [f"N={n}" for n in xs])
    ax.set_ylabel("SI-SDRi (dB)")
    ax.set_ylim(0, 9.2)
    ax.set_title("Separation quality falls with the number of talkers\n"
                 "frozen test set, scored against the true count",
                 fontsize=11, color=INK, pad=10)
    ax.xaxis.grid(False)
    save(fig, "fig3_si_sdri_per_n.png")


# --------------------------------------------------------------------------- fig 4
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    labels = ["one epoch's\nmixtures, repeated", "fresh mixtures\nevery epoch"]

    ax = axes[0]
    ax.bar([0, 1], [85.1, 90.6], width=0.55, color=[GRID, BLUE], zorder=3)
    for x, v in zip([0, 1], [85.1, 90.6]):
        ax.text(x, v + 0.7, f"{v:.1f} %", ha="center", fontsize=10.5, color=INK)
    ax.set_ylim(0, 100); ax.set_ylabel("dev accuracy (%)")
    ax.set_title("Counter — +5.5 points", fontsize=11, color=BLUE, fontweight="bold")
    ax.set_xticks([0, 1], labels, fontsize=9)
    ax.xaxis.grid(False)

    ax = axes[1]
    ax.bar([0, 1], [4.51, 4.59], width=0.55, color=[GRID, ORANGE], zorder=3)
    for x, v in zip([0, 1], [4.51, 4.59]):
        ax.text(x, v + 0.05, f"{v:+.2f} dB", ha="center", fontsize=10.5, color=INK)
    ax.set_ylim(0, 5.3); ax.set_ylabel("dev SI-SDRi (dB)")
    ax.set_title("Separator — +0.08 dB (noise)", fontsize=11, color=ORANGE, fontweight="bold")
    ax.set_xticks([0, 1], labels, fontsize=9)
    ax.xaxis.grid(False)

    fig.suptitle("One dataloader bug, two models: only the counter was data-limited",
                 fontsize=11.5, color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "fig4_dataloader_ablation.png")


# --------------------------------------------------------------------------- fig 5
def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    # Training accuracy is the dashed pair. The two runs are indistinguishable for eight
    # epochs, then the frozen-data run climbs to 99.5 % while its validation stalls: that
    # divergence IS the memorisation, and it is the reason the fix was worth making.
    ep = range(1, 21)
    ax = axes[0]
    ax.plot(ep, CNT_FROZEN_TRAIN, linewidth=1.3, linestyle="--", color="#b9b8b2", zorder=2)
    ax.plot(ep, CNT_FRESH_TRAIN, linewidth=1.3, linestyle="--", color="#9fc4f0", zorder=2)
    ax.plot(ep, CNT_FROZEN_VAL, linewidth=2, color=INK2, marker="o", markersize=3.5,
            label="frozen data — dev", zorder=3)
    ax.plot(ep, CNT_FRESH_VAL, linewidth=2, color=BLUE, marker="o", markersize=3.5,
            label="fresh mixtures — dev", zorder=4)
    ax.annotate("train 99.5 %", (20, CNT_FROZEN_TRAIN[-1]), textcoords="offset points",
                xytext=(-6, 5), ha="right", fontsize=8, color="#8f8e88")
    ax.annotate("train 89.8 %", (20, CNT_FRESH_TRAIN[-1]), textcoords="offset points",
                xytext=(-6, -13), ha="right", fontsize=8, color="#7aa9e0")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
    ax.set_ylim(58, 104); ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_title("Counter  (dashed = training accuracy)",
                 fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.02),
              facecolor=SURFACE, framealpha=0.92, frameon=True, edgecolor="none")

    ax = axes[1]
    ax.plot(range(1, 31), SEP_FROZEN, linewidth=2, color=GRID, label="one epoch, repeated")
    ax.plot(range(1, 31), SEP_FRESH, linewidth=2, color=ORANGE, label="fresh mixtures")
    ax.set_xlabel("epoch"); ax.set_ylabel("dev SI-SDRi (dB)")
    ax.set_ylim(0, 5.2); ax.set_xlim(0.5, 30.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_title("Separator", fontsize=11, color=INK, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.02),
              facecolor=SURFACE, framealpha=0.92, frameon=True, edgecolor="none")

    fig.suptitle("Training curves — the counter gains from fresh data, the separator does not",
                 fontsize=11.5, color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "fig5_training_curves.png")


# --------------------------------------------------------------------------- fig 6
def fig_interpretability():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # Overlap and SI-SDRi plotted against EACH OTHER, not both against N: two measures on
    # one x-axis would have meant two y-scales, which is the one chart shape never to draw.
    ax = axes[0]
    ax.plot(MASK_OVERLAP, MASK_SISDRI, linewidth=1.6, color=GRID, zorder=2)
    ax.scatter(MASK_OVERLAP, MASK_SISDRI, s=90, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    for ov, sd, n in zip(MASK_OVERLAP, MASK_SISDRI, MASK_N):
        ax.annotate(f"N={n}", (ov, sd), textcoords="offset points", xytext=(11, 7),
                    fontsize=9.5, color=INK)
    ax.set_xlabel("pairwise mask overlap (cosine)")
    ax.set_ylabel("SI-SDRi (dB)")
    ax.set_xlim(0.135, 0.295); ax.set_ylim(3.0, 7.3)
    ax.set_title("Masks that collide separate worse\n"
                 "each point is one speaker count", fontsize=10.5, color=INK)

    ax = axes[1]
    xs = np.arange(1, 6)
    w = 0.38
    ov = [np.nan] + [WITHIN_OVERLAP[n] for n in range(2, 6)]
    sp = [WITHIN_SPARSITY[n] for n in range(1, 6)]
    ax.bar(xs - w / 2, ov, width=w, color=BLUE, label="mask overlap", zorder=3)
    ax.bar(xs + w / 2, sp, width=w, color=ORANGE, label="mask sparsity", zorder=3)
    ax.axhline(0, linewidth=1.0, color=INK2)
    ax.set_xticks(xs, [f"N={n}" for n in xs])
    ax.set_ylabel("correlation with SI-SDRi, within N")
    ax.set_ylim(-0.98, 0.78)
    ax.set_title("Only overlap survives the control\n"
                 "consistent and negative at every N", fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    ax.text(3.6, -0.90, "overlap is undefined at N=1 — no second mask to compare",
            ha="center", fontsize=7.5, color=INK2, style="italic")
    ax.xaxis.grid(False)

    fig.tight_layout()
    save(fig, "fig6_interpretability.png")


# --------------------------------------------------------------------------- fig 7
def fig_baselines():
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    names = ["chance (5 classes)", "v0 joint model, fp32", "hand-crafted features + tree",
             "this work — CountCRNN"]
    vals = [20.0, 20.0, 57.8, 91.1]
    colours = [GRID, GRID, AQUA, BLUE]
    ys = np.arange(len(vals))
    ax.barh(ys, vals, height=0.6, color=colours, zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v + 1.2, y, f"{v:.1f} %", va="center", fontsize=10, color=INK)
    ax.set_yticks(ys, names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 104); ax.set_xlabel("speaker-counting accuracy (%)")
    ax.yaxis.grid(False)
    ax.set_title("Counting accuracy against every floor that matters",
                 fontsize=11, color=INK, pad=10)
    save(fig, "fig7_baselines.png")


if __name__ == "__main__":
    print("writing figures to", OUT)
    fig_architecture()
    fig_confusion()
    fig_per_n()
    fig_ablation()
    fig_training()
    fig_interpretability()
    fig_baselines()
    print("done")
