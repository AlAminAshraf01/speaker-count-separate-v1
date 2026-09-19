#!/usr/bin/env python3
"""Open the box: the learned filterbank, mask geometry, and what the two models agree on.

    python scripts/07_interpret.py --store /kaggle/input/.../store \\
        --recipes_test data/recipes_test.csv \\
        --separator /kaggle/working/sep/ckpt/best.pt \\
        --counter /kaggle/working/counter/ckpt/best.pt --out /kaggle/working/interpret

**No training.** Forward passes on checkpoints you already have, so this is the part of the
project to protect when the schedule slips: it costs minutes and it is the difference between
reporting that performance degrades with N and *explaining* why.

THE ARGUMENT
------------
As N grows the separator must partition the **same** 512-filter encoder basis among more
sources. If pairwise mask overlap rises and mask sparsity falls with N, the degradation curve
has a mechanistic explanation computed from the network's own internals rather than an asserted
one.

WHY v1 CAN ASK A QUESTION v0 COULD NOT
--------------------------------------
In v0 the counting head hung off the separator's trunk, so any correlation between counting
confidence and mask geometry was partly an artefact of the shared representation -- the two
quantities were computed from the same tensor. In v1 the counter is a separate model with its
own parameters and its own gradient. A correlation between its confidence and the separator's
mask overlap is therefore a statement about the **audio**: two independently trained models
agreeing that a particular mixture is hard.

That is a stronger claim than v0 could make, and it costs nothing extra to measure.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from _common import (add_common_args, banner, build_store_and_bank, code_version,
                     require_store, resolve)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--recipes_test", default="data/recipes_test.csv")
    ap.add_argument("--separator", required=True)
    ap.add_argument("--counter", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="/kaggle/working/interpret")
    args = ap.parse_args()

    import torch

    from countsep.checkpoint import load_checkpoint
    from countsep.constants import N_LIST, SR
    from countsep.datasets import DEFAULT_MIXING, FrozenMixDataset, build_loader
    from countsep.interpret import (collect_mask_records, correlate, extract_filterbank,
                                    filter_centre_frequencies, group_by_n, hoyer_sparsity,
                                    mel_reference, peak_frequencies)
    from countsep.separator import ModelConfig, build_separator
    from countsep.utils import format_table, json_dump_atomic, pick_device

    banner("07 - interpretability   (forward passes only, no training)")
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

    report: dict = {"code_version": code_version(), "separator": sep_path}

    # ---------------------------------------------------------------- 1. the filterbank
    banner("1. what the encoder learned")
    print("  Conv-TasNet replaces the STFT with a learned 1-D convolutional encoder. That IS")
    print("  the 'custom feature extraction' deliverable -- so the question is what it chose.\n")
    filters = extract_filterbank(separator)
    centres = filter_centre_frequencies(filters, sr=SR)
    peaks = peak_frequencies(filters, sr=SR)
    mel = mel_reference(len(filters), sr=SR)

    order = np.argsort(peaks)
    quartiles = np.percentile(peaks, [0, 25, 50, 75, 100])
    print(f"  {len(filters)} filters, length {filters.shape[1]} samples "
          f"({1000 * filters.shape[1] / SR:.1f} ms)")
    print(f"  peak frequency quartiles (Hz): "
          + "  ".join(f"{v:.0f}" for v in quartiles))
    below_1k = float((peaks < 1000).mean())
    print(f"  {below_1k:.0%} of filters peak below 1 kHz "
          f"(a mel scale would put {float((mel < 1000).mean()):.0%} there)")
    print(f"  mean Hoyer sparsity of a filter's spectrum: "
          f"{float(np.mean([hoyer_sparsity(np.abs(np.fft.rfft(f))) for f in filters])):.3f}")
    print("\n  A basis concentrated at low frequencies is the expected result: that is where")
    print("  voiced speech has its harmonic structure, and pitch is what tells two talkers")
    print("  apart. If it came out flat, the encoder had not learned anything speech-specific.")
    report["filterbank"] = {
        "n_filters": int(len(filters)), "length": int(filters.shape[1]),
        "peak_hz_quartiles": [float(v) for v in quartiles],
        "fraction_below_1khz": below_1k,
        "mel_fraction_below_1khz": float((mel < 1000).mean()),
    }

    # ---------------------------------------------------------------- 2. mask geometry
    banner("2. mask geometry against N   (the mechanism behind the degradation curve)")
    store, bank = build_store_and_bank(store_root, args.test_split,
                                       noise_kinds=DEFAULT_MIXING["noise_kinds"])
    recipes = resolve(args.recipes_test) or args.recipes_test
    dataset = FrozenMixDataset(store, bank, recipes, want="separate", limit=args.limit)
    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, persistent=False)
    print(f"  {len(dataset)} mixtures {dataset.counts_per_n()}\n")

    records = collect_mask_records(separator, loader, device)
    keys = ("overlap_cosine", "sparsity_hoyer", "entropy", "si_sdri")
    by_n = group_by_n(records, [k for k in keys])
    rows = []
    for n in N_LIST:
        vals = by_n.get(n, {})
        rows.append([f"N={n}"] + [f"{vals.get(k, float('nan')):.3f}" for k in keys])
    print(format_table(rows, ["true N", "mask overlap", "sparsity", "entropy", "SI-SDRi"]))
    print("\n  The claim to check: overlap RISES and sparsity FALLS as N grows, because the")
    print("  same basis is being split among more sources. If both hold, the degradation")
    print("  curve is explained by the network's own internals rather than asserted.")
    report["mask_geometry"] = {str(n): v for n, v in sorted(by_n.items())}

    # ---------------------------------------------------------------- 3. does geometry predict quality
    banner("3. does mask geometry predict separation quality?")
    for x_key in ("overlap_cosine", "sparsity_hoyer"):
        stat = correlate(records, x_key, "si_sdri")
        print(f"  {x_key:<15} vs SI-SDRi:  pearson r = {stat['pearson_r']:+.3f} "
              f"(p = {stat['pearson_p']:.3g})   spearman = {stat['spearman_r']:+.3f}  "
              f"(n = {stat['n']})")
    print("\n  A strong negative correlation for overlap is the mechanistic statement: mixtures")
    print("  whose masks fight each other are the mixtures the model separates badly.")
    report["correlations"] = {k: correlate(records, k, "si_sdri")
                              for k in ("overlap_cosine", "sparsity_hoyer", "entropy")}

    # ---------------------------------------------------------------- 4. two models agreeing
    counter_path = resolve(args.counter) if args.counter else None
    if counter_path and os.path.exists(counter_path):
        banner("4. do the two INDEPENDENT models agree on which mixtures are hard?")
        from countsep.counter import ModelConfig as CounterConfig
        from countsep.counter import build_model as build_counter

        c_state = torch.load(counter_path, map_location="cpu", weights_only=False)
        pooling = str(c_state.get("cfg", {}).get("pooling", "eigen"))
        counter = build_counter(CounterConfig(pooling=pooling)).to(device).eval()
        load_checkpoint(counter_path, model=counter)

        confidences = []
        with torch.no_grad():
            for batch in loader:
                probs = torch.softmax(counter(batch["mix"].to(device))["logits"].float(), -1)
                confidences.extend(probs.max(-1).values.cpu().tolist())
        for rec, conf in zip(records, confidences):
            rec["count_confidence"] = float(conf)

        stat = correlate(records, "count_confidence", "si_sdri")
        overlap_stat = correlate(records, "count_confidence", "overlap_cosine")
        print(f"  counter confidence vs separator SI-SDRi ....... r = "
              f"{stat['pearson_r']:+.3f}  (p = {stat['pearson_p']:.3g}, n = {stat['n']})")
        print(f"  counter confidence vs separator mask overlap .. r = "
              f"{overlap_stat['pearson_r']:+.3f}  (p = {overlap_stat['pearson_p']:.3g}, "
              f"n = {overlap_stat['n']})")
        print("\n  These two models share NO parameters and were trained on different")
        print("  objectives. A correlation here is therefore a statement about the audio --")
        print("  some mixtures are simply hard -- and not an artefact of a shared trunk.")
        print("  v0 could not make this claim: its count head read the very features whose")
        print("  geometry it was being correlated against.")
        report["cross_model"] = {"confidence_vs_si_sdri": stat,
                                 "confidence_vs_overlap": overlap_stat}

    path = os.path.join(out_dir, "interpret_report.json")
    json_dump_atomic(report, path)
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        print(f"\nFAILED: report missing or truncated at {path}", file=sys.stderr)
        return 2
    print(f"\nreport -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
