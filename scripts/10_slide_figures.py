#!/usr/bin/env python3
"""Rebuild the figures of the final presentation from the notebooks' own outputs.

    python scripts/10_slide_figures.py                       # on Kaggle: finds every input
    python scripts/10_slide_figures.py --store store --recipes_test data/recipes_test.csv \\
        --eval report/runs/eval_report.json --out slide_figures

CPU only, no training, no GPU quota, about two minutes. It writes one PNG per slide figure
and two files that say where every number came from:

    sources.json      figure -> [input file and its code_version, or "recorded: <doc>"]
    numbers.md        every number printed on a slide, beside the file it was read from

The deck was first drawn by hand-run scripts on a laptop that has no LibriSpeech audio, so
its two pictures (slides 3 and 5) used Windows text-to-speech voices. On Kaggle the packed
store from notebook 00 is attached, so here those pictures are drawn from a **real frozen
test clip**, rendered by ``countsep.mixing.render_recipe`` -- the same call the evaluation
uses. Every chart is read from the report of the notebook that measured it:

    notebook 00  store + recipes_test.csv   slides 3, 4, 5
    notebook 01  audit_report.json          slide 5 (leakage probe)
                 tier_a_report.json         slide 11 (grid search, now with each setting's score)
    notebook 02  train_report.json          slide 10 (training curves)
                 pool_*/train_report.json   slide 11 (pooling, if COMPARE_POOLINGS was run)
    notebook 03  separator_report.json      slide 10 (separator dev SI-SDRi)
    notebook 04  eval_report.json           slides 9, 10, 11 (every test-set number)
    the code     countsep models            slides 6, 7, 8 (shapes and parameter counts)

A few numbers were measured by runs whose outputs are not notebook outputs any more -- v0,
the repeated-mix training runs, the CPU pooling proxy. Those come from ``RECORDED`` below,
each with the document that holds it, and every figure that uses one says so in
``sources.json``. Nothing is silently invented: a missing input either falls back to a
recorded value that is labelled as one, or the figure is skipped with the reason printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from _common import (_find_files, add_common_args, autodetect_store, banner, code_version,
                     find_recipes, resolve, use_agg)

# ----------------------------------------------------------------------------- recorded
# Values measured once by runs that are not a notebook output any more. Each entry says
# where the measurement is written down, so a reader can check it.
RECORDED = {
    "v0_counting_accuracy": {
        "value": 0.2000, "source": "docs/DIAGNOSIS.md (v0 checkpoint on its test set, fp32)"},
    "v0_counting_grad_share": {
        "value": 0.0053, "source": "docs/DIAGNOSIS.md (share of the shared-trunk gradient)"},
    "v0_separation_cost_db": {
        "value": 8.7, "source": "docs/DIAGNOSIS.md (single-batch overfit, 28.4 -> 19.7 dB)"},
    "counter_repeated_history": {
        "value": {"val_acc": [73.33, 61.73, 74.20, 70.93, 81.40, 78.53, 79.67, 85.07, 83.60,
                              81.00, 78.33, 78.27, 80.47, 79.80, 81.60, 82.20, 83.27, 83.53,
                              83.73, 83.27],
                  "train_acc": [68.99, 78.73, 80.95, 82.99, 84.72, 86.36, 88.02, 89.72, 91.39,
                                93.13, 94.51, 95.82, 96.79, 97.60, 98.28, 98.88, 99.25, 99.45,
                                99.44, 99.54]},
        "source": "train_report.json of the repeated-mix counter run (code 6de9a43), "
                  "as transcribed in report/make_figures.py"},
    "separator_repeated_best_db": {
        "value": 4.51, "source": "separator_report.json of the repeated-mix run, "
                                 "as transcribed in report/make_figures.py"},
    "pooling_proxy": {
        "value": {"Mean + std": 66.5, "Attentive": 67.5, "Covariance": 20.0,
                  "Eigen-spectrum": 74.4},
        "source": "docs/DESIGN.md section 2 (synthetic speech proxy, CPU, 10 epochs)"},
    "leakage_probe": {
        "value": {"before": 0.442, "after": 0.193},
        "source": "report/RESULTS.md section 6 (notebook 01 audit)"},
}
REPEATED_COUNTER_CODE = "6de9a43"

# The deck's palette, so the PNGs can sit next to the slides.
INK, INK2, MUTED, GRID = "#262626", "#595959", "#8C8C8C", "#D9D9D9"
RED, BLUE, LIGHT_BLUE, LIGHT_RED = "#C00000", "#2458A6", "#E8EEF8", "#FBE9E9"
TALK = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]      # Okabe-Ito
NOISE_COLOURS = {"none": "#BFBFBF", "white": "#8FAEDB", "pink": "#E7A1B0", "brown": "#9C6B4E"}
POOL_NAMES = {"meanstd": "Mean + std", "attentive": "Attentive", "covariance": "Covariance",
              "eigen": "Eigen-spectrum"}
ROOTS = ("/kaggle/input", "/kaggle/working")


class Ledger:
    """What each figure was drawn from. Written to sources.json and numbers.md."""

    def __init__(self) -> None:
        self.sources: dict[str, list[str]] = {}
        self.numbers: list[tuple[str, str, str, str]] = []
        self.skipped: dict[str, str] = {}

    def use(self, figure: str, source: str) -> None:
        self.sources.setdefault(figure, [])
        if source not in self.sources[figure]:
            self.sources[figure].append(source)

    def number(self, slide: str, what: str, value: str, source: str) -> None:
        self.numbers.append((slide, what, value, source))

    def skip(self, figure: str, why: str) -> None:
        self.skipped[figure] = why
        print(f"  SKIP {figure}: {why}")


# ----------------------------------------------------------------------------- inputs
def _load(path: str | None) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _recorded(key: str) -> str:
    return "recorded: " + RECORDED[key]["source"]


def _label(path: str, report: dict | None) -> str:
    version = (report or {}).get("code_version", "no code_version")
    return f"{path} ({version})"


def _candidates(filename: str, exclude_parent: tuple[str, ...] = ()) -> list[str]:
    """Every ``filename`` under the Kaggle roots, minus copies inside a git clone."""
    hits = []
    for path in _find_files(ROOTS, filename):
        parts = os.path.normpath(path).split(os.sep)
        if "speaker-count-separate-v1" in parts:           # the clone's own copies
            continue
        if os.path.basename(os.path.dirname(path)).startswith(exclude_parent):
            continue
        hits.append(path)
    return sorted(hits, key=os.path.getmtime, reverse=True)


def discover(args) -> dict:
    """Fill every input the user did not pass, and say which file each one is."""
    found = {}
    store = autodetect_store(resolve(args.store))
    found["store"] = store
    found["recipes_test"] = resolve(args.recipes_test) or find_recipes("recipes_test.csv", store)
    found["audit"] = resolve(args.audit) or next(iter(_candidates("audit_report.json")), None)
    found["tier_a"] = resolve(args.tier_a) or next(iter(_candidates("tier_a_report.json")), None)
    found["eval"] = resolve(args.eval) or next(iter(
        _candidates("eval_report.json", exclude_parent=("eval_dev",))), None)
    found["sep"] = resolve(args.sep_report) or next(iter(
        _candidates("separator_report.json", exclude_parent=("sep_n2",))), None)

    counters = ([resolve(args.counter_report)] if args.counter_report
                else _candidates("train_report.json", exclude_parent=("pool_",)))
    fresh, repeated = None, resolve(args.counter_report_repeated)
    for path in counters:
        rep = _load(path) or {}
        if "history" not in rep or "val_acc" not in (rep["history"] or [{}])[0]:
            continue                                         # not a counter report
        if str(rep.get("code_version", "")).startswith(REPEATED_COUNTER_CODE):
            repeated = repeated or path
        elif fresh is None:
            fresh = path
    found["counter"], found["counter_repeated"] = fresh, repeated
    found["pooling"] = {}
    if not args.no_pool_search:
        for path in _find_files(ROOTS, "train_report.json"):
            parent = os.path.basename(os.path.dirname(path))
            if parent.startswith("pool_") and parent[5:] in POOL_NAMES:
                found["pooling"].setdefault(parent[5:], path)
    return found


# ----------------------------------------------------------------------------- helpers
def _style(plt) -> None:
    from matplotlib import font_manager

    # The deck is set in Arial; Kaggle images usually have Liberation Sans (metric-compatible)
    # or only DejaVu. Name one font that exists, or matplotlib warns once per text element.
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Arial", "Liberation Sans") if f in have), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family, "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 200,
        "savefig.bbox": "tight"})


def _save(plt, fig, out_dir: str, name: str) -> None:
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print(f"  wrote {name}")


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _db(x: float) -> str:
    return f"{x:+.2f} dB"


def _wave(ax, t, x, colour, lim) -> None:
    ax.plot(t, x, color=colour, lw=0.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-lim, lim)
    ax.axis("off")


def pick_clip(rows: list[dict], wanted: str | None) -> dict | None:
    """The illustration clip: a named one, else 3 talkers with noise near 12 dB SNR."""
    if wanted:
        return next((r for r in rows if r["mix_id"] == wanted), None)
    pool = [r for r in rows if int(r["n_src"]) == 3 and r["noise_kind"] != "none"]
    pool = pool or [r for r in rows if int(r["n_src"]) >= 2] or rows
    return min(pool, key=lambda r: abs(float(r["snr_db"]) - 12.0)) if pool else None


# ----------------------------------------------------------------------------- slide 3
def fig_problem(plt, clip: dict, rendered: dict, out_dir: str, led: Ledger, src: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    sources, noise, mix = rendered["sources"], rendered["noise"], rendered["mix"]
    n = sources.shape[0]
    t = np.arange(mix.size) / 8000.0
    lim = float(np.abs(sources).max()) * 1.05
    fig = plt.figure(figsize=(11.0, 3.0))
    top = 0.80
    h = 0.62 / (n + 1)
    rows = [(f"Talker {k + 1}", TALK[k % 5], sources[k]) for k in range(n)]
    rows.append(("Noise", "#9E9E9E", noise))
    for k, (label, colour, sig) in enumerate(rows):
        ax = fig.add_axes([0.07, top - (k + 1) * h, 0.19, h * 0.85])
        _wave(ax, t, sig, colour, lim)
        fig.text(0.065, top - (k + 0.55) * h, label, ha="right", va="center", fontsize=10)
    fig.text(0.165, 0.93, "Hidden sources (never observed)", ha="center", style="italic",
             color=INK2)
    ax = fig.add_axes([0.33, 0.22, 0.22, 0.52])
    _wave(ax, t, mix, INK, float(np.abs(mix).max()) * 1.05)
    fig.text(0.44, 0.93, "Observed input", ha="center", style="italic", color=INK2)
    fig.text(0.44, 0.80, "one microphone, 3 s at 8 kHz", ha="center", fontsize=10)
    fig.text(0.44, 0.10, "N is unknown (1 to 5)", ha="center", fontsize=10)
    box = fig.add_axes([0.585, 0.30, 0.10, 0.36])
    box.axis("off")
    box.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                                 fc="#F2F2F2", ec=INK2, transform=box.transAxes))
    box.text(0.5, 0.66, "Count", ha="center", color=BLUE, fontsize=12, fontweight="bold")
    box.text(0.5, 0.46, "+", ha="center", color=INK2)
    box.text(0.5, 0.24, "Separate", ha="center", color=RED, fontsize=12, fontweight="bold")
    fig.text(0.64, 0.93, "Our system", ha="center", style="italic", color=INK2)
    fig.text(0.86, 0.93, "Required outputs", ha="center", style="italic", color=INK2)
    fig.text(0.73, 0.80, f"1) Count: N = {n} talkers", fontsize=11, color=BLUE,
             fontweight="bold")
    fig.text(0.73, 0.70, "2) Separate: one clean track per talker", fontsize=10)
    for k in range(n):
        ax = fig.add_axes([0.79, 0.60 - (k + 1) * 0.16, 0.19, 0.13])
        _wave(ax, t, sources[k], TALK[k % 5], lim)
        fig.text(0.785, 0.60 - (k + 0.5) * 0.16, f"Track {k + 1}", ha="right", va="center",
                 fontsize=9.5)
    fig.text(0.5, -0.02, f"Real frozen test clip {clip['mix_id']}: {n} talkers, "
             f"{clip['noise_kind']} noise at {float(clip['snr_db']):.1f} dB SNR. The output "
             f"tracks are the reference voices: the goal, not model output.",
             ha="center", fontsize=8.5, color=MUTED)
    _save(plt, fig, out_dir, "slide03_problem.png")
    led.use("slide03_problem.png", src)


# ----------------------------------------------------------------------------- slide 4
def fig_test_set(plt, rows: list[dict], out_dir: str, led: Ledger, src: str) -> None:
    ns = sorted({int(r["n_src"]) for r in rows})
    kinds = [k for k in ("none", "white", "pink", "brown")
             if any(r["noise_kind"] == k for r in rows)]
    kinds += sorted({r["noise_kind"] for r in rows} - set(kinds))
    counts = {k: [sum(1 for r in rows if int(r["n_src"]) == n and r["noise_kind"] == k)
                  for n in ns] for k in kinds}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bottom = np.zeros(len(ns))
    for k in kinds:
        vals = np.array(counts[k], dtype=float)
        ax.bar(ns, vals, bottom=bottom, width=0.62, color=NOISE_COLOURS.get(k, GRID),
               label="Clean" if k == "none" else k.capitalize(), edgecolor="white", lw=0.6)
        for x, b, v in zip(ns, bottom, vals):
            if v >= 12:
                ax.text(x, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=8,
                        color=INK)
        bottom += vals
    per_n = sorted({int(b) for b in bottom})
    ax.set_xticks(ns, [f"N = {n}" for n in ns])
    ax.set_ylabel("test mixtures")
    ax.legend(ncol=len(kinds), loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    title = f"Test set: {per_n[0]} clips per count" if len(per_n) == 1 else "Test set"
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    _save(plt, fig, out_dir, "slide04_test_set.png")
    led.use("slide04_test_set.png", src)
    led.number("4", "test clips per count", ", ".join(map(str, per_n)), src)
    for k in kinds:
        led.number("4", f"'{k}' clips per N", ", ".join(map(str, counts[k])), src)

    noisy = [float(r["snr_db"]) for r in rows if r["noise_kind"] != "none"]
    if not noisy:
        led.skip("slide04_snr.png", "no noisy clips in the recipes")
        return
    edges = [5, 8, 11, 14, 17, 20]
    hist = np.histogram(noisy, bins=edges)[0]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = [f"{a}–{b}" for a, b in zip(edges[:-1], edges[1:])]
    ax.bar(labels, hist, width=0.62, color=MUTED)
    for x, v in enumerate(hist):
        ax.text(x, v + max(hist) * 0.02, str(int(v)), ha="center", fontsize=8.5)
    ax.set_xlabel("signal-to-noise ratio (dB)")
    ax.set_ylabel("mixtures")
    ax.set_title(f"Noise level of the {len(noisy):,} noisy test clips", fontsize=11,
                 loc="left", fontweight="bold")
    _save(plt, fig, out_dir, "slide04_snr.png")
    led.use("slide04_snr.png", src)
    led.number("4", "noisy test clips per 3 dB SNR band", ", ".join(map(str, hist)), src)


# ----------------------------------------------------------------------------- slide 5
def fig_preprocessing(plt, rows: list[dict], clip: dict | None, rendered: dict | None,
                      store, audit: dict | None, out_dir: str, led: Ledger,
                      src_recipes: str, src_store: str | None, src_audit: str | None) -> None:
    fig = plt.figure(figsize=(9.0, 6.4))
    name = "slide05_preprocessing.png"

    # (a) and (b): the clip before and after, when the store is attached
    if clip is not None and rendered is not None and store is not None:
        raw = [store.get(int(i)) for i in clip["utt_idx"]]
        tmax = max(len(r) for r in raw) / 8000.0
        lim = max(float(np.abs(r).max()) for r in raw) * 1.05
        for k, r in enumerate(raw):
            ax = fig.add_axes([0.07, 0.86 - k * 0.075, 0.36, 0.065])
            tt = np.arange(len(r)) / 8000.0
            ax.plot(tt, r, color=TALK[k % 5], lw=0.4)
            ax.set_xlim(0, tmax)
            ax.set_ylim(-lim, lim)
            ax.axis("off")
            fig.text(0.065, 0.892 - k * 0.075, f"T{k + 1}", ha="right", va="center")
            fig.text(0.07 + 0.36 * len(r) / 8000.0 / tmax + 0.005, 0.892 - k * 0.075,
                     f"{len(r) / 8000.0:.1f} s", va="center", fontsize=8, color=MUTED)
        fig.text(0.25, 0.975, "(a) Before: stored utterances", ha="center",
                 fontweight="bold")
        fig.text(0.25, 0.948, "different lengths and loudness", ha="center", fontsize=9,
                 color=INK2)
        mix = rendered["mix"]
        ax = fig.add_axes([0.58, 0.63, 0.39, 0.29])
        t = np.arange(mix.size) / 8000.0
        ax.plot(t, mix, color=INK, lw=0.35)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude")
        fig.text(0.775, 0.975, "(b) After: model input", ha="center", fontweight="bold")
        fig.text(0.775, 0.948, f"3-s crop, {int(clip['n_src'])} talkers + "
                 f"{clip['noise_kind']} noise, RMS = {float(np.sqrt(np.mean(mix ** 2))):.2f}",
                 ha="center", fontsize=9, color=INK2)

        # (d) the counter's own front end on that mixture
        import torch

        from countsep.counter import build_model

        spec = build_model().to("cpu").stft(torch.from_numpy(mix[None].astype(np.float32)))
        spec = spec.detach().squeeze().numpy()
        ax = fig.add_axes([0.58, 0.10, 0.39, 0.30])
        db = 20 * np.log10(spec + 1e-6)
        ax.imshow(db, origin="lower", aspect="auto", cmap="Greys", vmin=db.max() - 70,
                  vmax=db.max(), extent=[0, t[-1], 0, 4.0])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (kHz)")
        fig.text(0.775, 0.465, "(d) Counter input: |STFT|", ha="center", fontweight="bold")
        fig.text(0.775, 0.438, f"{spec.shape[0]} frequency bins x {spec.shape[1]} frames "
                 f"(shown in dB)", ha="center", fontsize=9, color=INK2)
        led.use(name, src_store)
        led.number("5", "counter input shape", f"{spec.shape[0]} x {spec.shape[1]}",
                   "countsep.counter LinearSTFT on the clip")
    else:
        fig.text(0.5, 0.80, "(a), (b), (d) need notebook 00's store attached",
                 ha="center", color=MUTED)

    # (c) level vs N, from the recipes -- real data
    by_n: dict[int, list[float]] = {}
    for r in rows:
        by_n.setdefault(int(r["n_src"]), []).append(-20.0 * math.log10(float(r["scale"])))
    ns = sorted(by_n)
    ax = fig.add_axes([0.08, 0.10, 0.38, 0.30])
    ax.boxplot([by_n[n] for n in ns], positions=ns, widths=0.5, patch_artist=True,
               showfliers=False, medianprops=dict(color=INK, lw=1.2),
               whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
               boxprops=dict(facecolor="#E6E6E6", edgecolor=MUTED))
    ax.plot(ns, [0] * len(ns), color=RED, lw=2.0, marker="o", ms=4.5, zorder=4)
    ax.set_xlabel("true speaker count N")
    ax.set_ylabel("mixture level (dB)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.text(0.02, 0.96, "before levelling (grey)", transform=ax.transAxes, va="top",
            fontsize=8.5, color=INK2)
    ax.text(0.98, 0.04, "after: 0 dB for every N", transform=ax.transAxes, ha="right",
            fontsize=8.5, color=RED)
    fig.text(0.27, 0.465, "(c) Loudness no longer reveals N", ha="center", fontweight="bold")
    fig.text(0.27, 0.438, f"real frozen test set, {len(rows):,} mixtures", ha="center",
             fontsize=9, color=INK2)
    medians = {n: float(np.median(by_n[n])) for n in ns}
    rise = medians[ns[-1]] - medians[ns[0]]
    led.use(name, src_recipes)
    led.number("5", f"median mix level rise, N={ns[0]} to N={ns[-1]}, before levelling",
               f"{rise:+.1f} dB", src_recipes)

    # the leakage probe in the red box
    probes = (audit or {}).get("probes", {})
    before = probes.get("artefact:raw", {}).get("accuracy")
    after = probes.get("artefact_strict:mitigated", {}).get("accuracy")
    if before is not None and after is not None:
        probe_src = src_audit
    else:
        before, after = (RECORDED["leakage_probe"]["value"][k] for k in ("before", "after"))
        probe_src = "recorded: " + RECORDED["leakage_probe"]["source"]
    fig.text(0.5, 0.005, f"Leakage probe (level, peak, length only): {_pct(before)} before "
             f"levelling, {_pct(after)} after; chance is 20 %.", ha="center", fontsize=9.5,
             color=RED)
    led.use(name, probe_src)
    led.number("5", "leakage probe before / after", f"{_pct(before)} / {_pct(after)}",
               probe_src)
    _save(plt, fig, out_dir, name)


# ----------------------------------------------------------------------------- slides 6-8
def _box(ax, x, y, w, h, title, lines, params, edge, fill, param_colour) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fill, ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h * 0.80, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    for k, line in enumerate(lines):
        ax.text(x + w / 2, y + h * (0.60 - 0.14 * k), line, ha="center", va="center",
                fontsize=8.5, color=INK2)
    if params:
        ax.text(x + w / 2, y + h * 0.13, params, ha="center", va="center", fontsize=9,
                fontweight="bold", color=param_colour)


def _row(plt, stages, edge, fill, param_colour, total_note, title, out_dir, name) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 2.9))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, gap = 0.13, (1 - 0.13 * len(stages)) / (len(stages) - 1)
    for k, (stage_title, lines, params, shape) in enumerate(stages):
        x = k * (w + gap)
        first = k == 0
        _box(ax, x, 0.30, w, 0.58, stage_title, lines, params,
             MUTED if first else edge, "#F2F2F2" if first else fill, param_colour)
        ax.text(x + w / 2, 0.20, shape, ha="center", fontsize=9.5, fontweight="bold")
        if k:
            ax.annotate("", xy=(x - 0.003, 0.59), xytext=(x - gap + 0.003, 0.59),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))
    ax.text(-0.005, 0.20, "output:", ha="right", fontsize=8.5, style="italic", color=MUTED)
    ax.text(0.5, 0.04, total_note, ha="center", fontsize=9, style="italic", color=INK2)
    ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold")
    _save(plt, fig, out_dir, name)


def _shape(t) -> str:
    return " × ".join(f"{d:,}" for d in t)


def _k(n: int) -> str:
    if n == 0:
        return "0 params"
    return f"{n / 1e6:.2f} M params" if n >= 1e6 else f"{n / 1e3:.1f} k params"


def architecture(plt, out_dir: str, led: Ledger) -> None:
    """Slides 6-8: run one real forward pass and read every shape and count off it."""
    import torch

    from countsep.constants import SEG_LEN, SR
    from countsep.counter import build_model
    from countsep.separator import ModelConfig, build_separator

    counter, cfg = build_model().to("cpu").eval(), ModelConfig()   # CPU only, by design
    sep = build_separator(cfg).to("cpu").eval()
    seen: dict[str, tuple] = {}

    def keep(name):
        def hook(_module, _inp, out):
            out = out[0] if isinstance(out, (tuple, list)) else out
            seen[name] = tuple(out.shape[1:])
        return hook

    for name in ("stft", "conv", "gru", "pool", "head"):
        getattr(counter, name).register_forward_hook(keep("c_" + name))
    for name in ("encoder", "bottleneck"):
        getattr(sep, name).register_forward_hook(keep("s_" + name))
    wav = torch.randn(1, SEG_LEN)
    with torch.no_grad():
        counter(wav)
        est = sep(wav)["est"]

    def n_params(module) -> int:
        return int(sum(p.numel() for p in module.parameters()))

    c_total, s_total = n_params(counter), n_params(sep)
    spec = seen["c_stft"][-2:]
    k_slots = cfg.max_n_src + int(cfg.predict_noise)
    frames = seen["s_encoder"][-1]
    rf_frames = 1 + cfg.n_repeats * (cfg.conv_kernel - 1) * (2 ** cfg.n_blocks - 1)
    rf_s = ((rf_frames - 1) * (cfg.kernel // 2) + cfg.kernel) / SR
    tcn_share = n_params(sep.tcn) / s_total

    _row(plt, [
        ("Input", ["the recording", f"{SEG_LEN / SR:.0f} s, {SR / 1000:.0f} kHz"], "",
         _shape((1, SEG_LEN))),
        ("Spectrogram", ["linear |STFT|", "25 ms window"], _k(n_params(counter.stft)),
         _shape((1, *spec))),
        ("3 conv blocks", ["shrink frequency,", "keep time"], _k(n_params(counter.conv)),
         _shape(seen["c_conv"])),
        ("BiGRU", [f"reads all {spec[-1]}", "time steps"], _k(n_params(counter.gru)),
         _shape(seen["c_gru"])),
        ("Eigen pooling", ["counts strong", "directions"], _k(n_params(counter.pool)),
         _shape(seen["c_pool"])),
        ("Classifier", ["two small layers", f"{seen['c_head'][-1]} scores"],
         _k(n_params(counter.head)), "count, 1 to 5"),
    ], BLUE, LIGHT_BLUE, BLUE, f"{c_total:,} parameters ({c_total / 1e6:.2f} M), measured "
       f"from countsep.counter", "Counter (CountCRNN)", out_dir, "slide07_counter.png")

    _row(plt, [
        ("Input", ["the recording", f"{SEG_LEN / SR:.0f} s, {SR / 1000:.0f} kHz"], "",
         _shape((1, SEG_LEN))),
        ("Encoder", [f"{cfg.n_filters} learned", f"filters, {cfg.kernel / SR * 1000:.0f} ms"],
         _k(n_params(sep.encoder)), _shape(seen["s_encoder"])),
        ("Bottleneck", ["norm + 1×1 conv", f"{cfg.n_filters} → {cfg.bottleneck}"],
         _k(n_params(sep.pre_norm) + n_params(sep.bottleneck)), _shape(seen["s_bottleneck"])),
        ("TCN", [f"{cfg.n_blocks * cfg.n_repeats} dilated blocks", f"sees {rf_s:.2f} s"],
         _k(n_params(sep.tcn)), _shape((cfg.bottleneck, frames))),
        ("Mask head", ["one mask", "per track"],
         _k(n_params(sep.mask_prelu) + n_params(sep.mask_conv)),
         _shape((k_slots, cfg.n_filters, frames))),
        ("Decoder", ["masks back", "into audio"], _k(n_params(sep.decoder)),
         _shape(tuple(est.shape[1:]))),
    ], RED, LIGHT_RED, RED, f"{s_total:,} parameters, {tcn_share:.0%} of them in the TCN  ·  "
       f"tracks 1–{cfg.max_n_src} are voices, track {k_slots} is noise",
       "Separator (SepNet, Conv-TasNet sizes)", out_dir, "slide08_separator.png")

    # slide 6: the two networks joined
    fig, ax = plt.subplots(figsize=(11.0, 2.8))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.00, 0.30, 0.15, 0.40, "Recording", [f"{SEG_LEN / SR:.0f} s, "
         f"{SR / 1000:.0f} kHz"], "", MUTED, "#F2F2F2", INK)
    _box(ax, 0.22, 0.58, 0.26, 0.36, "Counter (CountCRNN)", ["How many people?"],
         f"{c_total / 1e6:.2f} M params", BLUE, LIGHT_BLUE, BLUE)
    _box(ax, 0.22, 0.06, 0.26, 0.36, "Separator (SepNet)",
         [f"{cfg.max_n_src} voice tracks + {int(cfg.predict_noise)} noise track"],
         f"{s_total / 1e6:.2f} M params", RED, LIGHT_RED, RED)
    _box(ax, 0.60, 0.26, 0.20, 0.48, "Pick tracks", ["keep the ones that best",
         "add back up to the", "recording"], "", MUTED, "#F2F2F2", INK)
    _box(ax, 0.86, 0.30, 0.14, 0.40, "Output", ["one track per", "person"], "", MUTED,
         "white", INK)
    arrow = dict(arrowstyle="-|>", color=INK2, lw=1.0)
    ax.annotate("", xy=(0.22, 0.76), xytext=(0.15, 0.55), arrowprops=arrow)
    ax.annotate("", xy=(0.22, 0.24), xytext=(0.15, 0.45), arrowprops=arrow)
    ax.annotate("", xy=(0.60, 0.60), xytext=(0.48, 0.76),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.annotate("", xy=(0.60, 0.40), xytext=(0.48, 0.24),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax.annotate("", xy=(0.86, 0.50), xytext=(0.80, 0.50), arrowprops=arrow)
    ax.text(0.50, 0.86, "count, 1 to 5", color=BLUE, fontweight="bold", fontsize=9.5)
    ax.text(0.50, 0.10, f"{k_slots} tracks", color=RED, fontweight="bold", fontsize=9.5)
    ax.text(0.80, 0.14, "Tracks not picked are still saved, as surplus", ha="center",
            fontsize=8.5, color=INK2)
    ax.set_title("Two networks, trained apart, joined when used", fontsize=11.5, loc="left",
                 fontweight="bold")
    _save(plt, fig, out_dir, "slide06_overview.png")

    code = "measured: countsep models, forward pass on a 3 s input"
    for name in ("slide06_overview.png", "slide07_counter.png", "slide08_separator.png"):
        led.use(name, code)
    led.number("6-7", "counter parameters", f"{c_total:,}", code)
    led.number("6, 8", "separator parameters", f"{s_total:,}", code)
    led.number("8", "separator receptive field", f"{rf_s:.2f} s ({rf_frames} frames)", code)
    led.number("8", "share of separator parameters in the TCN", f"{tcn_share:.0%}", code)


# ----------------------------------------------------------------------------- slide 9
def fig_results(plt, ev: dict, out_dir: str, led: Ledger, src: str) -> None:
    counting = ev["counting"]
    conf = np.array(counting["confusion"], dtype=float)
    labels = counting.get("labels", list(range(1, conf.shape[0] + 1)))
    recall = conf.diagonal() / np.maximum(conf.sum(1), 1)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.imshow(conf / np.maximum(conf.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            share = conf[i, j] / max(conf[i].sum(), 1)
            ax.text(j, i, f"{int(conf[i, j])}", ha="center", va="center", fontsize=10,
                    color="white" if share > 0.5 else INK)
        ax.text(conf.shape[1] - 0.3, i, f"{100 * recall[i]:.1f} %", va="center", fontsize=9.5)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), [f"N = {n}" for n in labels])
    ax.set_xlabel("guessed number")
    ax.set_ylabel("true number")
    ax.text(conf.shape[1] - 0.3, -0.75, "recall", fontsize=9, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Counting: true vs guessed number", fontsize=11, loc="left",
                 fontweight="bold", pad=18)
    _save(plt, fig, out_dir, "slide09_confusion.png")
    led.use("slide09_confusion.png", src)
    wrong_by_two = int(sum(conf[i, j] for i in range(len(conf)) for j in range(len(conf))
                           if abs(i - j) > 1))
    led.number("9", "counting accuracy", _pct(counting["accuracy"]), src)
    lo, hi = counting.get("wilson95", [float("nan")] * 2)
    led.number("9", "counting accuracy, Wilson 95 % CI", f"{_pct(lo)} to {_pct(hi)}", src)
    led.number("9", "counting MAE", f"{counting['mae']:.3f}", src)
    led.number("9", "guesses off by two or more", str(wrong_by_two), src)
    led.number("9", "recall per N", ", ".join(f"{100 * r:.1f} %" for r in recall), src)

    rules = ev.get("slot_selection_rules")
    ns = [1, 2, 3, 4, 5]
    if rules:
        best = [rules["per_n"][str(n)]["si_sdri_oracle_slots"] for n in ns]
        kept = [rules["per_n"][str(n)]["si_sdri_rebuild"] for n in ns]
    else:
        best = [ev["si_sdri_oracle_count"][str(n)] for n in ns]
        kept = None
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(ns))
    ax.bar(x - (0.19 if kept else 0), best, width=0.38 if kept else 0.6, color="#BFBFBF",
           label="if the right tracks were picked")
    if kept:
        ax.bar(x + 0.19, kept, width=0.38, color=RED, label="tracks the system keeps")
        for xi, v in zip(x, kept):
            ax.text(xi + 0.19, v + 0.12, f"{v:+.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, [f"N = {n}" for n in ns])
    ax.set_ylabel("SI-SDRi (dB)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Separation gain by number of people (dB)", fontsize=11, loc="left",
                 fontweight="bold")
    _save(plt, fig, out_dir, "slide09_separation.png")
    led.use("slide09_separation.png", src)
    led.number("9", "SI-SDRi per N, right tracks picked",
               ", ".join(f"{v:+.2f}" for v in best), src)
    if rules:
        led.number("9", "SI-SDRi per N, tracks kept (rebuild)",
                   ", ".join(f"{v:+.2f}" for v in kept), src)
        led.number("9", "SI-SDRi pooled, kept / if picked perfectly",
                   f"{_db(rules['pooled']['rebuild'])} / {_db(rules['pooled']['oracle'])}", src)
        right2 = rules["per_n"]["2"].get("right_slots_rebuild")
        if right2 is not None:
            led.number("9", "N = 2 clips where the right pair is kept", _pct(right2), src)
        if "p_si_snr" in rules:
            led.number("9", "P-SI-SNR end to end, rebuild / loudest",
                       f"{_db(rules['p_si_snr']['rebuild'])} / "
                       f"{_db(rules['p_si_snr']['loudest'])}", src)
        paired = rules.get("paired_rebuild_minus_loudest")
        if paired:
            led.number("9", "rebuild - loudest per clip, test (95 % CI)",
                       f"{_db(paired['mean'])} [{paired['ci95'][0]:+.2f}, "
                       f"{paired['ci95'][1]:+.2f}]", src)
    else:
        led.number("9", "P-SI-SNR end to end (loudest rule)", _db(ev["p_si_snr"]), src)


# ----------------------------------------------------------------------------- slide 10
def fig_ablation(plt, ev: dict | None, counter: dict | None, counter_rep: dict | None,
                 sep: dict | None, out_dir: str, led: Ledger, srcs: dict) -> None:
    v0 = RECORDED["v0_counting_accuracy"]
    if ev:
        acc = ev["counting"]["accuracy"]
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        ax.bar(["One shared network (v0)", "Two networks (ours)"],
               [100 * v0["value"], 100 * acc], width=0.5, color=["#BFBFBF", BLUE])
        for xi, v in enumerate([100 * v0["value"], 100 * acc]):
            ax.text(xi, v + 2, f"{v:.1f} %", ha="center", fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel("counting accuracy, test (%)")
        ax.set_title("A. One shared network vs two separate ones", fontsize=11, loc="left",
                     fontweight="bold")
        _save(plt, fig, out_dir, "slide10_shared_vs_two.png")
        led.use("slide10_shared_vs_two.png", srcs["eval"])
        led.use("slide10_shared_vs_two.png", _recorded("v0_counting_accuracy"))
    else:
        led.skip("slide10_shared_vs_two.png", "no eval_report.json (notebook 04)")
    led.number("10", "v0 counting accuracy", _pct(v0["value"]), _recorded("v0_counting_accuracy"))
    led.number("10", "v0 counter's share of the learning signal",
               f"{100 * RECORDED['v0_counting_grad_share']['value']:.2f} %",
               _recorded("v0_counting_grad_share"))
    led.number("10", "v0 separation lost to the counting terms",
               f"-{RECORDED['v0_separation_cost_db']['value']:.1f} dB",
               _recorded("v0_separation_cost_db"))

    if not counter:
        led.skip("slide10_fresh_vs_repeated.png", "no counter train_report.json (notebook 02)")
        return
    fresh_val = [100 * r["val_acc"] for r in counter["history"]]
    fresh_train = [100 * r["train_acc"] for r in counter["history"]]
    if counter_rep:
        rep_val = [100 * r["val_acc"] for r in counter_rep["history"]]
        rep_train = [100 * r["train_acc"] for r in counter_rep["history"]]
        rep_src = srcs["counter_repeated"]
    else:
        rep_val = RECORDED["counter_repeated_history"]["value"]["val_acc"]
        rep_train = RECORDED["counter_repeated_history"]["value"]["train_acc"]
        rep_src = _recorded("counter_repeated_history")
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for ys, colour, dash, label in ((fresh_val, BLUE, "-", "fresh, dev"),
                                    (fresh_train, "#8FAEDB", "--", "fresh, train"),
                                    (rep_val, INK2, "-", "repeated, dev"),
                                    (rep_train, "#A6A6A6", "--", "repeated, train")):
        ax.plot(range(1, len(ys) + 1), ys, dash, color=colour, lw=2 if dash == "-" else 1.4,
                marker="o" if dash == "-" else None, ms=3, label=label)
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("counter training epoch")
    ax.set_ylabel("accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    ax.set_title("B. Fresh vs repeated training mixes", fontsize=11, loc="left",
                 fontweight="bold")
    _save(plt, fig, out_dir, "slide10_fresh_vs_repeated.png")
    led.use("slide10_fresh_vs_repeated.png", srcs["counter"])
    led.use("slide10_fresh_vs_repeated.png", rep_src)
    led.number("10", "counter dev accuracy, repeated / fresh",
               f"{max(rep_val):.1f} % / {100 * counter['best_val_accuracy']:.1f} %",
               f"{rep_src}; {srcs['counter']}")
    led.number("10", "counter train - dev gap at the last epoch, repeated / fresh",
               f"{rep_train[-1] - rep_val[-1]:+.1f} / {fresh_train[-1] - fresh_val[-1]:+.1f} pts",
               f"{rep_src}; {srcs['counter']}")
    rep_sep = RECORDED["separator_repeated_best_db"]
    if sep:
        led.number("10", "separator dev SI-SDRi, repeated / fresh",
                   f"{rep_sep['value']:+.2f} / {sep['best_val_si_sdri']:+.2f} dB",
                   f"recorded: {rep_sep['source']}; {srcs['sep']}")


# ----------------------------------------------------------------------------- slide 11
def fig_tuning(plt, tier_a: dict | None, ev: dict | None, pooling: dict[str, dict],
               out_dir: str, led: Ledger, srcs: dict) -> None:
    if tier_a and tier_a.get("search"):
        search = tier_a["search"]
        keys = list(search[0]["params"])
        names = {"max_iter": "Trees", "learning_rate": "Learning rate",
                 "max_leaf_nodes": "Max leaves"}
        header = ["Setting", *[names.get(k, k.replace("_", " ")) for k in keys], "CV accuracy"]
        n_folds = len(search[0].get("folds", [])) or 5
        cells = [[str(i + 1), *[f"{r['params'][k]:g}" for k in keys],
                  f"{100 * r['accuracy']:.1f} % ± {100 * r.get('accuracy_std', 0):.1f}"]
                 for i, r in enumerate(search)]
        fig, ax = plt.subplots(figsize=(6.0, 0.36 * (len(cells) + 1)))
        ax.axis("off")
        table = ax.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="center",
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        best_i = max(range(len(search)), key=lambda i: search[i]["accuracy"])
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            if row == 0:
                cell.set_facecolor("#404040")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            elif row - 1 == best_i:
                cell.set_facecolor(LIGHT_RED)
        ax.set_title(f"A. Grid search on the baseline counter ({n_folds}-fold CV)",
                     fontsize=11, loc="left", fontweight="bold")
        _save(plt, fig, out_dir, "slide11_grid.png")
        led.use("slide11_grid.png", srcs["tier_a"])
        accs = [r["accuracy"] for r in search]
        led.number("11", "best CV accuracy of the tree", _pct(max(accs)), srcs["tier_a"])
        led.number("11", "spread of the 6 settings", f"{100 * (max(accs) - min(accs)):.1f} "
                   f"points", srcs["tier_a"])
        led.number("11", "CV accuracy of every setting", ", ".join(_pct(a) for a in accs),
                   srcs["tier_a"])
    else:
        led.skip("slide11_grid.png", "no tier_a_report.json (notebook 01)")
    if ev and ev.get("tier_a_test"):
        led.number("11", "tree / counter on the same test clips",
                   f"{_pct(ev['tier_a_test']['accuracy'])} / {_pct(ev['counting']['accuracy'])}",
                   srcs["eval"])

    if len(pooling) == len(POOL_NAMES):
        values = {POOL_NAMES[k]: 100 * pooling[k]["best_val_accuracy"] for k in POOL_NAMES}
        pool_src = "; ".join(srcs["pooling"][k] for k in POOL_NAMES)
        note = "Real LibriSpeech dev set, notebook 02's pooling comparison."
    else:
        values = RECORDED["pooling_proxy"]["value"]
        pool_src = "recorded: " + RECORDED["pooling_proxy"]["source"]
        note = "Quick CPU test on synthetic speech: trust the order, not the numbers."
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    names = list(values)
    best = max(values, key=values.get)
    ax.bar(names, [values[n] for n in names], width=0.6,
           color=[BLUE if n == best else "#BFBFBF" for n in names])
    for xi, n in enumerate(names):
        ax.text(xi, values[n] + 2, f"{values[n]:.1f} %", ha="center", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("dev accuracy (%)")
    ax.set_title("B. Which pooling for the counter?", fontsize=11, loc="left",
                 fontweight="bold")
    ax.text(0, -0.2, note, transform=ax.transAxes, fontsize=8.5, style="italic", color=MUTED)
    _save(plt, fig, out_dir, "slide11_pooling.png")
    led.use("slide11_pooling.png", pool_src)
    led.number("11", "pooling accuracy", ", ".join(f"{n} {values[n]:.1f} %" for n in names),
               pool_src)


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--recipes_test", default=None)
    ap.add_argument("--split", default="test", help="store split the recipes belong to")
    ap.add_argument("--audit", default=None, help="notebook 01's audit_report.json")
    ap.add_argument("--tier_a", default=None, help="notebook 01's tier_a_report.json")
    ap.add_argument("--counter_report", default=None, help="notebook 02's train_report.json")
    ap.add_argument("--counter_report_repeated", default=None,
                    help="the repeated-mix counter run's train_report.json, if attached")
    ap.add_argument("--sep_report", default=None, help="notebook 03's separator_report.json")
    ap.add_argument("--eval", default=None, help="notebook 04's eval_report.json")
    ap.add_argument("--clip", default=None, help="mix_id of the illustration clip")
    ap.add_argument("--no_pool_search", action="store_true",
                    help="use the recorded pooling proxy even if pool_* runs are attached")
    ap.add_argument("--out", default="/kaggle/working/slide_figures")
    args = ap.parse_args()

    from countsep.utils import format_table, json_dump_atomic

    banner("10 - slide figures from the notebooks' outputs   (CPU, no training)")
    print(f"  code {code_version()}")
    out_dir = resolve(args.out)
    os.makedirs(out_dir, exist_ok=True)
    found = discover(args)
    reports = {k: _load(found[k]) for k in ("audit", "tier_a", "counter", "counter_repeated",
                                            "sep", "eval")}
    pooling = {k: _load(p) for k, p in found["pooling"].items()}
    srcs = {k: _label(found[k], reports[k]) for k in reports if found[k]}
    srcs["pooling"] = {k: _label(p, pooling[k]) for k, p in found["pooling"].items()}

    rows_in = [[k, found[k] or "NOT FOUND"] for k in
               ("store", "recipes_test", "audit", "tier_a", "counter", "counter_repeated",
                "sep", "eval")]
    rows_in.append(["pooling runs", ", ".join(sorted(found["pooling"])) or "none"])
    print(format_table(rows_in, ["input", "file"]))

    plt = use_agg()
    _style(plt)
    led = Ledger()

    banner("slides 3-5: the data")
    rows = None
    if found["recipes_test"]:
        from countsep.mixing import read_recipes

        rows = read_recipes(found["recipes_test"])
    store = rendered = clip = None
    if rows and found["store"]:
        try:
            from _common import build_store_and_bank
            from countsep.mixing import render_recipe

            store, bank = build_store_and_bank(found["store"], args.split)
            clip = pick_clip(rows, args.clip)
            rendered = render_recipe(clip, store, bank) if clip else None
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  store could not render a clip ({exc}); slides 3 and 5a/b/d skipped")
            store = rendered = clip = None
    src_store = (f"{found['store']} + {found['recipes_test']} (clip {clip['mix_id']})"
                 if rendered is not None else None)
    if rendered is not None:
        fig_problem(plt, clip, rendered, out_dir, led, src_store)
    else:
        led.skip("slide03_problem.png", "needs notebook 00's store and recipes_test.csv")
    if rows:
        fig_test_set(plt, rows, out_dir, led, found["recipes_test"])
        fig_preprocessing(plt, rows, clip, rendered, store, reports["audit"], out_dir, led,
                          found["recipes_test"], src_store, srcs.get("audit"))
    else:
        for name in ("slide04_test_set.png", "slide04_snr.png", "slide05_preprocessing.png"):
            led.skip(name, "no recipes_test.csv (notebook 00)")

    banner("slides 6-8: the architecture, measured from the code")
    architecture(plt, out_dir, led)

    banner("slides 9-11: the results")
    if reports["eval"]:
        fig_results(plt, reports["eval"], out_dir, led, srcs["eval"])
    else:
        for name in ("slide09_confusion.png", "slide09_separation.png"):
            led.skip(name, "no eval_report.json (notebook 04)")
    fig_ablation(plt, reports["eval"], reports["counter"], reports["counter_repeated"],
                 reports["sep"], out_dir, led, srcs)
    fig_tuning(plt, reports["tier_a"], reports["eval"], pooling, out_dir, led, srcs)

    banner("where every number came from")
    recorded = sorted({s for used in led.sources.values() for s in used
                       if s.startswith("recorded:")})
    print(format_table([[s, w[:58], v] for s, w, v, _src in led.numbers],
                       ["slide", "what", "value"]))
    if recorded:
        print("\n  Recorded values (not read from an attached notebook output):")
        for s in recorded:
            print(f"    {s}")
    lines = ["# Numbers on the slides, and where each was read from", "",
             f"Written by `scripts/10_slide_figures.py` at code {code_version()}.", "",
             "| slide | what | value | source |", "|---|---|---|---|"]
    lines += [f"| {s} | {w} | {v} | `{src}` |" for s, w, v, src in led.numbers]
    if led.skipped:
        lines += ["", "## Skipped figures", ""]
        lines += [f"- `{name}`: {why}" for name, why in led.skipped.items()]
    with open(os.path.join(out_dir, "numbers.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    json_dump_atomic({"code_version": code_version(), "inputs": {k: v for k, v in found.items()
                                                                 if k != "pooling"},
                      "pooling_runs": found["pooling"], "figures": led.sources,
                      "skipped": led.skipped}, os.path.join(out_dir, "sources.json"))
    written = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    print(f"\n  {len(written)} figures, sources.json and numbers.md -> {out_dir}")
    if led.skipped:
        print(f"  {len(led.skipped)} skipped: {', '.join(sorted(led.skipped))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
