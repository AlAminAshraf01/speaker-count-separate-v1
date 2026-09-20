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


def trend_across_n(by_n: dict, key: str, n_list) -> float:
    """Net movement of one statistic from the lowest defined N to the highest.

    Lives at module level because a ``def`` inside ``main()`` is the shape of the
    predecessor's worst bug, and ``tests/test_scripts_are_reachable.py`` fails the build
    on it. It caught this function when it was written inline.
    """
    vals = [by_n.get(n, {}).get(key, float("nan")) for n in n_list]
    vals = [v for v in vals if v == v]              # nan cells: N=1 has no slot pair
    return (vals[-1] - vals[0]) if len(vals) > 1 else float("nan")


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
    from countsep.interpret import (collect_mask_records, correlate, correlate_within,
                                    extract_filterbank,
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
    # Two references, so the reader compares against numbers rather than against a claim.
    # A uniform basis on this grid puts 1000/(SR/2) of its peaks below 1 kHz.
    mel_below_1k = float((mel < 1000).mean())
    flat_below_1k = 1000.0 / (SR / 2)
    print("\n  Against two references rather than against an expectation:")
    print(f"    mel would put ....... {mel_below_1k:.0%} below 1 kHz")
    print(f"    uniform would put ... {flat_below_1k:.0%}")
    print(f"    measured ............ {below_1k:.0%}")
    if abs(below_1k - flat_below_1k) < abs(below_1k - mel_below_1k):
        print("\n  So the basis tiles the band roughly UNIFORMLY. It is not mel-like and it is")
        print("  not concentrated at low frequencies. Report that as the finding.")
        print("  It does not license the opposite conclusion either: a histogram of peak")
        print("  frequencies cannot settle whether the encoder learned anything")
        print("  speech-specific, and nothing in this script tests that. Leave it open.")
    else:
        print("\n  The basis is concentrated at low frequencies, closer to mel than to uniform.")
        print("  That is where voiced speech carries its harmonic structure, and pitch is")
        print("  what separates two talkers.")
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
    d_overlap = trend_across_n(by_n, "overlap_cosine", N_LIST)
    d_sparsity = trend_across_n(by_n, "sparsity_hoyer", N_LIST)
    print("\n  The prediction was that overlap RISES and sparsity FALLS as N grows, because")
    print("  the same basis is split among more sources. Whether it held:")
    verdict_o = "RISES, as predicted" if d_overlap > 0 else "does NOT rise -- claim FAILS"
    verdict_s = "falls, as predicted" if d_sparsity < 0 else "does NOT fall -- claim FAILS"
    print(f"    overlap   {d_overlap:+.3f} across N   -> {verdict_o}")
    print(f"    sparsity  {d_sparsity:+.3f} across N   -> {verdict_s}")
    if not (d_overlap > 0 and d_sparsity < 0):
        print("\n  Only part of the mechanism is supported. Report the half that held and the")
        print("  half that did not. A degradation curve half-explained is still a result;")
        print("  claiming both halves when the table above says otherwise is not.")
    report["mask_geometry"] = {str(n): v for n, v in sorted(by_n.items())}

    # ---------------------------------------------------------------- 3. does geometry predict quality
    banner("3. does mask geometry predict separation quality?")
    print("  A POOLED correlation mixes two things: that N moves both variables, and that")
    print("  geometry may predict quality WITHIN a fixed N. Only the second is the")
    print("  mechanistic claim. Across the group means alone N makes overlap and SI-SDRi")
    print("  nearly collinear, so quote the within-N row.\n")
    for x_key in ("overlap_cosine", "sparsity_hoyer"):
        stat = correlate(records, x_key, "si_sdri")
        within = correlate_within(records, x_key, "si_sdri")
        per = "  ".join(
            f"N{n}:{v['pearson_r']:+.2f}"
            for n, v in sorted(within["per_group"].items())
            if v["pearson_r"] == v["pearson_r"])
        print(f"  {x_key}")
        print(f"    pooled    r = {stat['pearson_r']:+.3f}  (p = {stat['pearson_p']:.3g})   "
              f"spearman = {stat['spearman_r']:+.3f}   n = {stat['n']}")
        print(f"    within-N  r = {within['pooled_r']:+.3f}   {per}")
    overlap_within = correlate_within(records, "overlap_cosine", "si_sdri")["pooled_r"]
    print("")
    if overlap_within < -0.2:
        print(f"  Within a fixed N, overlap still predicts quality (r = {overlap_within:+.3f}).")
        print("  That is the mechanistic statement: among mixtures with the SAME number of")
        print("  talkers, the ones whose masks fight each other are the ones separated badly.")
    elif overlap_within > 0.2:
        print(f"  Within a fixed N the relationship REVERSES SIGN (r = {overlap_within:+.3f}).")
        print("  Higher overlap going with BETTER separation is the opposite of the mechanism,")
        print("  so the pooled negative number is N acting on both variables and nothing more.")
        print("  Report the reversal; it is more interesting than the pooled figure and it")
        print("  rules the mechanism out rather than leaving it open.")
    else:
        print(f"  Within a fixed N the relationship is {overlap_within:+.3f} -- weak or absent.")
        print("  So the pooled number is largely N acting on both variables, not geometry")
        print("  predicting quality mixture by mixture. The honest claim is the weaker one:")
        print("  overlap rises with N and quality falls with N. Do not say more than that.")
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
        within_stat = correlate_within(records, "count_confidence", "si_sdri")
        print(f"  counter confidence vs separator SI-SDRi ....... r = "
              f"{stat['pearson_r']:+.3f}  (p = {stat['pearson_p']:.3g}, n = {stat['n']})")
        print(f"  counter confidence vs separator mask overlap .. r = "
              f"{overlap_stat['pearson_r']:+.3f}  (p = {overlap_stat['pearson_p']:.3g}, "
              f"n = {overlap_stat['n']})")
        within_per_n = "  ".join(
            f"N{n}:{v['pearson_r']:+.2f}"
            for n, v in sorted(within_stat["per_group"].items())
            if v["pearson_r"] == v["pearson_r"])
        print(f"  within N, confidence vs SI-SDRi ............... r = "
              f"{within_stat['pooled_r']:+.3f}   {within_per_n}")
        print("")
        print("  The two models share NO parameters and optimise different objectives, so a")
        print("  correlation here cannot be a shared-trunk artefact the way v0's would have")
        print("  been -- its count head read the very features whose geometry it was")
        print("  correlated against. But architectural independence does not rule out the")
        print("  OTHER confound: both quantities move with N on their own. Quote the")
        print("  within-N row, which is the claim; the pooled row mostly restates that N")
        print("  exists.")
        if abs(within_stat["pooled_r"]) < 0.15:
            print("")
            print("  As measured, the within-N effect is weak. The honest report is that the")
            print("  independence is real and the agreement is not established.")
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
