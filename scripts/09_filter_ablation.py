#!/usr/bin/env python3
"""Channel ablation: which learned filters does the separator need, and for which job?

    python scripts/09_filter_ablation.py --store /kaggle/input/.../store \\
        --recipes_test data/recipes_test.csv \\
        --separator /kaggle/working/sep/ckpt/best.pt --out /kaggle/working/ablation

**No training.** Forward passes on the checkpoint you already have: the test clips are
rendered once, then scored ~19 times with different encoder channels switched off.

THE QUESTION (the proposal's)
-----------------------------
Which filters suppress noise, and which separate speakers? The test set already contains the
two jobs in isolation, so the question can be asked directly:

* **denoise only** -- N = 1 with noise: one talker, and the only work is removing the noise;
* **separate only** -- N >= 2 with no noise: pure speaker separation;
* **both** -- N >= 2 with noise.

The 512 encoder filters are sorted by peak frequency and cut into equal-count bands (64
filters each by default). Each band is switched off in turn, and every clip's SI-SDRi is
compared with the same clip unablated -- a paired difference, so clip difficulty cancels.

TWO ABLATIONS, BECAUSE THE OBVIOUS ONE IS CONFOUNDED
----------------------------------------------------
The encoder output feeds the mask estimator AND is the representation the masks multiply
before decoding. Zeroing a band of encoder channels (``full``) therefore deletes that band
from the OUTPUT too, and the drop mostly measures how much speech energy lives there. So the
primary ablation (``analysis``) hides the band from the mask estimator only, leaving
reconstruction intact: what drops is what the network needed to SEE in order to decide.
A random set of the same size is the control for both.

Report the ``analysis`` table as the finding and the ``full`` table as context. The raw dB
columns cannot be compared across jobs (denoising is more sensitive to removing ANY filters),
so each band's cost is divided by its own job's random-control cost before the jobs are
compared -- see ``specialisation``.

RE-SCORING AN OLD REPORT
------------------------
Reports written before that fix carry a raw-dB verdict that flags every band. The verdict
needs only the saved means and standard errors, so it can be recomputed on a CPU:

    python scripts/09_filter_ablation.py --rescore report/runs/ablation_report.json

The superseded verdict is kept in the file under ``specialisation_raw_db_superseded``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from _common import (add_common_args, banner, build_store_and_bank, code_version,
                     require_store, resolve)

TASKS: tuple[str, ...] = ("denoise only", "separate only", "both")


def band_groups(peaks: np.ndarray, n_bands: int) -> list[dict]:
    """Equal-COUNT bands of filters ordered by peak frequency, with each band's Hz range."""
    order = np.argsort(peaks, kind="stable")
    bands = []
    for k, idx in enumerate(np.array_split(order, int(n_bands)), start=1):
        bands.append({"name": f"B{k}", "indices": [int(i) for i in idx],
                      "lo_hz": float(peaks[idx].min()), "hi_hz": float(peaks[idx].max())})
    return bands


def cache_batches(loader) -> list[dict]:
    """Render every test clip once; the ~19 scoring passes then reuse the same tensors."""
    keep = ("mix", "refs", "n_src", "is_noisy")
    return [{k: batch[k] for k in keep} for batch in loader]


def clip_scores(model, batches: list[dict], device, max_n_src: int) -> np.ndarray:
    """Mean oracle-slot SI-SDRi per clip (true count), NaN where no reference is usable."""
    import torch

    from countsep.metrics import usable_si_sdri

    out = []
    with torch.no_grad():
        for batch in batches:
            est = model(batch["mix"].to(device))["est"].float().cpu().numpy()[:, :max_n_src]
            refs, mixes = batch["refs"].numpy(), batch["mix"].numpy()
            for i, n in enumerate(batch["n_src"].tolist()):
                vals, _ = usable_si_sdri(est[i], refs[i], mixes[i], int(n))
                vals = [float(v) for v in np.asarray(vals).ravel() if np.isfinite(v)]
                out.append(float(np.mean(vals)) if vals else float("nan"))
    return np.asarray(out, dtype=np.float64)


def task_masks(batches: list[dict], valid: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks over clips for the three jobs, restricted to scorable clips."""
    n = np.concatenate([b["n_src"].numpy() for b in batches])
    noisy = np.concatenate([b["is_noisy"].numpy() for b in batches]).astype(bool)
    return {"denoise only": (n == 1) & noisy & valid,
            "separate only": (n >= 2) & ~noisy & valid,
            "both": (n >= 2) & noisy & valid}


def summarise(delta: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict]:
    """Mean paired delta, its standard error and n, per job."""
    out = {}
    for task, mask in masks.items():
        d = delta[mask]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        out[task] = {"mean": float(d.mean()) if len(d) else float("nan"), "se": se,
                     "n": int(len(d))}
    return out


def fmt_cell(stat: dict) -> str:
    if stat["mean"] != stat["mean"]:
        return "--"
    return f"{stat['mean']:+.2f} +- {1.96 * stat['se']:.2f}"


def run_ablation(model, batches, device, max_n_src, baseline, masks, groups, ablate) -> dict:
    """Score every group under one ablation kind; returns ``{group name: summary}``."""
    results = {}
    for group in groups:
        with ablate(model, group["indices"]):
            scores = clip_scores(model, batches, device, max_n_src)
        results[group["name"]] = summarise(scores - baseline, masks)
        print(f"    {group['name']:<7} done", flush=True)
    return results


def print_table(title: str, groups: list[dict], results: dict) -> None:
    from countsep.utils import format_table

    print(f"\n  {title}   (paired delta SI-SDRi, dB, +- 95 % CI; negative = the band mattered)")
    rows = []
    for g in groups:
        span = (f"{g['lo_hz']:.0f}-{g['hi_hz']:.0f} Hz" if g["name"] != "random"
                else "random filters")
        rows.append([g["name"], span] + [fmt_cell(results[g["name"]][t]) for t in TASKS])
    print(format_table(rows, ["band", "filter peaks", *TASKS]))


def relative_to_control(stat: dict, control: dict) -> dict:
    """A band's cost divided by the random-filter control's cost, with an approximate 95 % CI.

    The CI is delta-method on the log ratio, treating the two means as independent -- they are
    paired on the same clips, so this is conservative. NaN when the control did not hurt.
    """
    a, b = stat["mean"], control["mean"]
    if not (a < 0 and b < 0):
        return {"ratio": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rel_se = float(np.sqrt((stat["se"] / a) ** 2 + (control["se"] / b) ** 2))
    ratio = a / b
    return {"ratio": float(ratio), "lo": float(ratio * np.exp(-1.96 * rel_se)),
            "hi": float(ratio * np.exp(1.96 * rel_se))}


def fmt_ratio(rel: dict) -> str:
    if rel["ratio"] != rel["ratio"]:
        return f"{'n/a':>20}"
    return f"{rel['ratio']:5.1f}x [{rel['lo']:.1f}, {rel['hi']:.1f}]".rjust(20)


def specialisation(groups: list[dict], results: dict) -> list[dict]:
    """Per band: its cost relative to the random control, for each job, and whether they differ.

    Comparing the raw dB deltas of the two jobs is NOT a specialisation test. Denoising-only
    clips are far more sensitive to removing ANY filters -- on the reported run the random
    control cost denoising 0.39 dB and separation 0.06 dB -- so every band looks "more
    important for denoising" in raw dB. Dividing each job's band cost by that job's control
    cost removes the job's overall sensitivity; a band specialises only where the two ratios'
    CIs do not overlap.
    """
    out = []
    for g in groups:
        rel = {t: relative_to_control(results[g["name"]][t], results["random"][t])
               for t in ("denoise only", "separate only")}
        d, s = rel["denoise only"], rel["separate only"]
        clear = bool(d["ratio"] == d["ratio"] and s["ratio"] == s["ratio"]
                     and (d["lo"] > s["hi"] or s["lo"] > d["hi"]))
        out.append({"band": g["name"], "denoise_vs_control": d, "separate_vs_control": s,
                    "clear": clear})
    return out


def print_verdict(spec: list[dict]) -> None:
    """The control-normalised comparison of the two jobs, band by band, and what it allows."""
    clear = [s for s in spec if s["clear"]]
    print("\n  Raw dB cannot compare the two jobs: denoising is more sensitive to removing ANY")
    print("  filters (the random control above). So each band's cost is divided by its own")
    print("  job's control cost -- 1.0x means 'no more than random filters', 10x means ten")
    print("  times as much -- and a band specialises only where the two jobs' CIs separate.")
    for s in spec:
        cells = [fmt_ratio(s[k]) for k in ("denoise_vs_control", "separate_vs_control")]
        flag = "  <- the jobs differ beyond the CI" if s["clear"] else ""
        print(f"    {s['band']}: denoise {cells[0]}   separate {cells[1]}{flag}")
    undefined = any(s[k]["ratio"] != s[k]["ratio"] for s in spec
                    for k in ("denoise_vs_control", "separate_vs_control"))
    if undefined:
        print("\n  n/a = the random control did not lower that job's score, so there is no")
        print("  sensitivity to divide by and no comparison can be made for that job.")
    if clear:
        print(f"\n  {len(clear)} band(s) matter differently to the two jobs beyond their CI.")
        print(f"  With {len(spec)} bands tested at 95 %, about {0.05 * len(spec):.1f} would be")
        print("  flagged by chance alone, so quote a flagged band with its CI.")
    elif undefined:
        print("\n  With a job's control undefined, this run cannot say whether the bands")
        print("  specialise. Use more clips, or a checkpoint that actually separates.")
    else:
        print("\n  No band matters differently to the two jobs beyond its CI, once each job's")
        print("  overall sensitivity is divided out. The honest reading is a shared reliance --")
        print("  both jobs lean on the same bands -- not a noise band and a speaker band.")


def rescore_report(path: str) -> int:
    """Recompute the verdict of a saved report from its means and SEs, in place. CPU only."""
    from countsep.utils import json_dump_atomic, json_load

    if not os.path.exists(path):
        print(f"FAILED: no report at {path}", file=sys.stderr)
        return 2
    report = json_load(path)
    analysis = report["analysis_ablation"]
    if "random" not in analysis:
        print(f"FAILED: {path} has no random-filter control to divide by", file=sys.stderr)
        return 2
    bands_only = [b for b in report["bands"] if b["name"] != "random"]
    old = report.get("specialisation_analysis", [])
    if old and "denoise_vs_control" not in old[0]:
        report["specialisation_raw_db_superseded"] = old
    report["specialisation_analysis"] = specialisation(bands_only, analysis)
    report["rescored"] = {
        "code_version": code_version(),
        "note": ("specialisation_analysis recomputed from the saved analysis_ablation means "
                 "and SEs: each band's cost divided by its job's random-control cost. The "
                 "measurements themselves are unchanged.")}
    banner("09 - re-scoring a saved ablation report   (no model, no GPU)")
    print(f"  report: {path}   ({report.get('code_version', '?')}, {report.get('n_clips', '?')} clips)")
    print_verdict(report["specialisation_analysis"])
    json_dump_atomic(report, path)
    print(f"\nreport -> {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--recipes_test", default="data/recipes_test.csv")
    ap.add_argument("--separator", help="separator checkpoint (required unless --rescore)")
    ap.add_argument("--rescore", metavar="REPORT",
                    help="recompute the verdict of a saved ablation_report.json and exit")
    ap.add_argument("--limit", type=int, default=600,
                    help="test clips, stratified over N (600 -> 120 per N)")
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="/kaggle/working/ablation")
    args = ap.parse_args()
    if args.rescore:
        return rescore_report(resolve(args.rescore) or args.rescore)
    if not args.separator:
        ap.error("--separator is required unless --rescore is given")

    import torch

    from countsep.checkpoint import load_checkpoint
    from countsep.constants import MAX_N_SRC, SR
    from countsep.datasets import DEFAULT_MIXING, FrozenMixDataset, build_loader
    from countsep.interpret import (ablate_encoder_filters, ablate_mask_input,
                                    extract_filterbank, peak_frequencies)
    from countsep.separator import ModelConfig, build_separator
    from countsep.utils import json_dump_atomic, pick_device

    banner("09 - channel ablation   (forward passes only, no training)")
    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(None)

    sep_path = resolve(args.separator) or args.separator
    state = torch.load(sep_path, map_location="cpu", weights_only=False)
    cfg = state.get("cfg", {}).get("model", {})
    separator = build_separator(ModelConfig(**cfg) if cfg else ModelConfig()).to(device).eval()
    load_checkpoint(sep_path, model=separator)
    print(f"  separator: {separator.describe()}")

    store, bank = build_store_and_bank(store_root, args.test_split,
                                       noise_kinds=DEFAULT_MIXING["noise_kinds"])
    recipes = resolve(args.recipes_test) or args.recipes_test
    dataset = FrozenMixDataset(store, bank, recipes, want="separate", limit=args.limit)
    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, persistent=False)
    print(f"  {len(dataset)} test mixtures {dataset.counts_per_n()}   device: {device}")
    batches = cache_batches(loader)

    banner("baseline   (nothing ablated)")
    baseline = clip_scores(separator, batches, device, MAX_N_SRC)
    valid = np.isfinite(baseline)
    masks = task_masks(batches, valid)
    for task in TASKS:
        print(f"  {task:<14} {int(masks[task].sum()):>4} clips   SI-SDRi "
              f"{float(np.mean(baseline[masks[task]])) if masks[task].any() else float('nan'):+.2f} dB")
    print(f"  ({int((~valid).sum())} clean single-speaker clips have no improvement to "
          f"measure and are left out)")

    filters = extract_filterbank(separator)
    peaks = peak_frequencies(filters, sr=SR)
    groups = band_groups(peaks, args.bands)
    size = len(groups[0]["indices"])
    rng = np.random.default_rng(args.seed)
    groups.append({"name": "random", "lo_hz": float("nan"), "hi_hz": float("nan"),
                   "indices": [int(i) for i in rng.choice(len(peaks), size=size, replace=False)]})
    print(f"\n  {len(peaks)} filters -> {args.bands} bands of ~{size} by peak frequency, "
          f"plus {size} random filters as the control")

    banner("1. ANALYSIS ablation   (hidden from the mask estimator; reconstruction intact)")
    analysis = run_ablation(separator, batches, device, MAX_N_SRC, baseline, masks, groups,
                            ablate_mask_input)
    banner("2. FULL ablation   (removed from the representation too -- confounded by energy)")
    full = run_ablation(separator, batches, device, MAX_N_SRC, baseline, masks, groups,
                        ablate_encoder_filters)

    banner("results")
    print_table("ANALYSIS ablation -- the finding", groups, analysis)
    print_table("FULL ablation -- context only", groups, full)

    bands_only = groups[:-1]
    spec = specialisation(bands_only, analysis)
    worst_d = min(bands_only, key=lambda g: analysis[g["name"]]["denoise only"]["mean"])
    worst_s = min(bands_only, key=lambda g: analysis[g["name"]]["separate only"]["mean"])
    control = analysis["random"]
    print("\n  Reading the ANALYSIS table:")
    print(f"    most needed for denoising ....... {worst_d['name']} "
          f"({worst_d['lo_hz']:.0f}-{worst_d['hi_hz']:.0f} Hz): "
          f"{analysis[worst_d['name']]['denoise only']['mean']:+.2f} dB")
    print(f"    most needed for separating ...... {worst_s['name']} "
          f"({worst_s['lo_hz']:.0f}-{worst_s['hi_hz']:.0f} Hz): "
          f"{analysis[worst_s['name']]['separate only']['mean']:+.2f} dB")
    print(f"    random {size} filters (control) .. denoise "
          f"{control['denoise only']['mean']:+.2f}, separate "
          f"{control['separate only']['mean']:+.2f} dB")
    print_verdict(spec)
    print("\n  Caveats for the write-up: one checkpoint; bands are equal-COUNT, not")
    print("  equal-width; the ratio CIs ignore the pairing of band and control on the same")
    print("  clips, so they are conservative; and synthetic white/pink/brown noise is")
    print("  spectrally smooth, so 'denoising' here means removing stationary broadband")
    print("  noise, nothing else.")

    report = {"code_version": code_version(), "separator": sep_path, "n_clips": len(dataset),
              "tasks": {t: int(masks[t].sum()) for t in TASKS},
              "baseline_si_sdri": {t: float(np.mean(baseline[masks[t]])) if masks[t].any()
                                   else None for t in TASKS},
              "bands": [{k: v for k, v in g.items() if k != "indices"} for g in groups],
              "analysis_ablation": analysis, "full_ablation": full,
              "specialisation_analysis": spec}
    path = os.path.join(out_dir, "ablation_report.json")
    json_dump_atomic(report, path)
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        print(f"\nFAILED: report missing or truncated at {path}", file=sys.stderr)
        return 2
    print(f"\nreport -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
