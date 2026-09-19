#!/usr/bin/env python3
"""Milestone 1: the leakage audit and the baselines, on CPU, before any GPU hour is spent.

This runs first and it runs every time. It answers three questions, and until it has answered
them no neural number from this project means anything:

1. **Is the evaluation sound?**  Fit a shallow tree to the *artefact* family alone -- mixture
   level, peak level, length. These describe how the mixture was assembled, not who is talking.
   After the mitigation (fixed-length crop divided by its RMS) this must land at chance. If it
   does not, the counter can score without modelling speech and every downstream number is void.

2. **What is the bar?**  Fit gradient boosting to the *acoustic* family -- crest factor,
   kurtosis, sparsity, spectral flatness and entropy. This is legitimate evidence and it is
   surprisingly strong: measured 69.3 % on a speaker-disjoint split at production settings,
   against 20 % chance. The neural counter has to beat this or it has bought nothing. The
   model this project replaces scored 20.00 %.

3. **Are the splits actually disjoint?**  Pairwise speaker overlap between train, dev and test,
   asserted to be zero on the *real* store rather than on a synthetic corpus whose generator
   makes it true by construction.

    python scripts/01_audit_and_baselines.py --store /kaggle/input/.../store \\
        --splits train-100 dev test --per_class 300 --out /kaggle/working/audit

CPU only, no torch, a few minutes. Writes ``audit_report.json``.

WHY THIS SCRIPT IS SHAPED THE WAY IT IS
---------------------------------------
Its predecessor, ``scripts/03_count_leak_probe.py`` in the sibling repo, was structurally dead
for months: a ``def`` landed inside ``main()``, every probe became unreachable code after an
unconditional ``return``, and the script exited 0 while writing nothing. Two headline numbers
were quoted from it that it had never produced, and nobody noticed, because the exit code was
0 and the header still printed.

So this script is built to fail loudly:

* every stage appends to ``report`` and the stages are a flat list, not nested in a helper;
* :func:`main` ends by verifying the JSON file exists on disk and is non-empty, and returns a
  non-zero exit code if it does not;
* ``--strict`` (default on) turns a detected leak or a speaker-overlap into a non-zero exit;
* ``tests/test_audit.py`` asserts that running this end to end produces a report with all four
  sections populated.

A green tick from this script is a claim. It should be expensive to fake.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from _common import add_common_args, banner, build_store_and_bank, code_version, require_store, resolve


def collect(store, bank, n_list, per_class, seg_len, rng, *, mitigated: bool,
            p_clean: float, gain_db_range, snr_db_range) -> tuple[list, np.ndarray, list]:
    """Render ``per_class`` mixtures per speaker count and featurise each one.

    ``mitigated=False`` reproduces what the audio looks like *before* the fixed-length
    RMS-normalised crop, which is the only way the artefact probe can show the cue it is
    supposed to have removed. ``mitigated=True`` is what the model actually sees.
    """
    from countsep.features import all_features
    from countsep.mixing import render_recipe, sample_recipe

    rows: list[dict] = []
    labels: list[int] = []
    speakers: list[list[str]] = []

    for n_src in n_list:
        for _ in range(per_class):
            recipe = sample_recipe(store, bank, n_src, rng, seg_len=seg_len,
                                   gain_db_range=gain_db_range,
                                   snr_db_range=snr_db_range, p_clean=p_clean)
            rendered = render_recipe(recipe, store, bank, seg_len=seg_len)
            mix = np.asarray(rendered["mix"], dtype=np.float64)

            if not mitigated:
                # Undo the anti-leak normalisation: multiply the stored scale back out, so the
                # mixture carries the level it had before the crop-and-divide. That level is
                # the artefact the probe is hunting for.
                mix = mix / (float(recipe["scale"]) + 1e-12)

            rows.append(all_features(mix))
            labels.append(int(n_src))
            speakers.append([str(store.speaker_ids[int(i)]) for i in recipe["utt_idx"]])

    return rows, np.array(labels, dtype=int), speakers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--splits", nargs="+", default=["train-100", "dev", "test"],
                    help="splits to audit for speaker disjointness")
    ap.add_argument("--probe_split", default="train-100",
                    help="split the probes draw their mixtures from")
    ap.add_argument("--n_list", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--per_class", type=int, default=300)
    ap.add_argument("--seg_seconds", type=float, default=3.0)
    ap.add_argument("--p_clean", type=float, default=0.2)
    ap.add_argument("--gain_db", nargs=2, type=float, default=[-5.0, 5.0])
    ap.add_argument("--snr_db", nargs=2, type=float, default=[0.0, 20.0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--out", default="/kaggle/working/audit")
    ap.add_argument("--strict", action="store_true", default=True,
                    help="exit non-zero on a detected leak or a speaker overlap (default)")
    ap.add_argument("--no_strict", dest="strict", action="store_false")
    args = ap.parse_args()

    from countsep.baselines import (interpret_probe, naive_scores, probe,
                                    speaker_disjointness, wilson_interval)
    from countsep.pack import SourceStore
    from countsep.utils import json_dump_atomic

    banner("01 - leakage audit + naive baselines   (CPU, no GPU quota)")
    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    report: dict = {"code_version": code_version(), "store": store_root, "sections": []}
    problems: list[str] = []

    sr_seg = int(round(args.seg_seconds * 8000))
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- 1. disjointness
    banner("1. speaker disjointness across splits   (every count must be 0)")
    present: dict[str, list[str]] = {}
    for split in args.splits:
        try:
            st = SourceStore(store_root, split, mmap=True)
        except FileNotFoundError:
            print(f"  {split:<12} not packed in this store -- skipped")
            continue
        present[split] = [str(s) for s in st.speaker_ids[st.target_idx]]
    if len(present) < 2:
        problems.append(f"only {len(present)} split(s) found, cannot audit disjointness")
        disjoint = {"sizes": {k: len(v) for k, v in present.items()}, "pairs": [], "ok": False}
    else:
        disjoint = speaker_disjointness(present)
        for name, n in disjoint["sizes"].items():
            print(f"  {name:<12} {len(set(present[name])):>5} distinct speakers "
                  f"({n:>6} utterances)")
        for pair in disjoint["pairs"]:
            mark = "OK" if pair["n_shared"] == 0 else "*** OVERLAP ***"
            print(f"  {pair['a']:>12} & {pair['b']:<12} shared: {pair['n_shared']:>4}  {mark}")
            if pair["n_shared"]:
                problems.append(f"{pair['a']} and {pair['b']} share {pair['n_shared']} speakers")
    report["sections"].append({"name": "disjointness", **disjoint})

    # ---------------------------------------------------------------- 2. render + featurise
    banner(f"2. rendering mixtures from {args.probe_split!r}")
    store, bank = build_store_and_bank(store_root, args.probe_split,
                                       noise_store=resolve(args.noise_store))
    total = len(args.n_list) * args.per_class
    print(f"  {total} mixtures, N in {args.n_list}, {args.seg_seconds:g} s at 8 kHz, "
          f"p_clean={args.p_clean}, gain {args.gain_db} dB, SNR {args.snr_db} dB")
    print(f"  noise kinds available: {bank.kinds}")

    raw_rows, y, speakers_per_mix = collect(
        store, bank, args.n_list, args.per_class, sr_seg, np.random.default_rng(args.seed),
        mitigated=False, p_clean=args.p_clean,
        gain_db_range=tuple(args.gain_db), snr_db_range=tuple(args.snr_db))
    mit_rows, y2, _ = collect(
        store, bank, args.n_list, args.per_class, sr_seg, np.random.default_rng(args.seed),
        mitigated=True, p_clean=args.p_clean,
        gain_db_range=tuple(args.gain_db), snr_db_range=tuple(args.snr_db))
    assert np.array_equal(y, y2), "the two renderings disagree on labels"
    print(f"  done: {len(raw_rows)} unmitigated + {len(mit_rows)} mitigated feature rows")

    # a speaker appearing in a great many mixtures is its own hazard
    from collections import Counter
    reuse = Counter(s for mix in speakers_per_mix for s in mix)
    print(f"  speaker reuse across mixtures: min {min(reuse.values())}, "
          f"median {int(np.median(list(reuse.values())))}, max {max(reuse.values())}")
    report["speaker_reuse"] = {"min": min(reuse.values()), "max": max(reuse.values()),
                               "median": float(np.median(list(reuse.values()))),
                               "n_distinct": len(reuse)}

    # ---------------------------------------------------------------- 3. naive baselines
    banner("3. naive predictors   (these need no learning at all)")
    classes = sorted(set(int(v) for v in y))
    naive = naive_scores(np.array([classes.index(int(v)) for v in y]), len(classes))
    for name, res in naive.items():
        lo, hi = wilson_interval(res["accuracy"], len(y))
        print(f"  {name:<20} accuracy {res['accuracy']:>6.1%}  "
              f"[{lo:.1%}, {hi:.1%}]   MAE {res['mae']:.3f}")
    report["naive"] = naive

    # ---------------------------------------------------------------- 4. the probes
    banner("4. feature probes")
    probes: dict[str, dict] = {}
    plan = [
        ("artefact", raw_rows, "tree", "BEFORE mitigation (level/duration present)"),
        ("artefact_strict", mit_rows, "tree", "AFTER mitigation (level and length are constant)"),
        ("acoustic", mit_rows, "gbm", "AFTER mitigation -- this is the bar"),
        ("everything", mit_rows, "gbm", "AFTER mitigation, both families"),
    ]
    for group, rows, model, note in plan:
        key = f"{group}:{'raw' if rows is raw_rows else 'mitigated'}"
        res = probe(rows, y, group, n_folds=args.folds, max_depth=args.max_depth,
                    seed=args.seed, model=model)
        verdict, why = interpret_probe(group, res["accuracy"], res["chance"])
        lo, hi = wilson_interval(res["accuracy"], res["n"])
        res.update({"verdict": verdict, "explanation": why, "note": note,
                    "ci95": [lo, hi]})
        probes[key] = res
        print(f"\n  {group:<11} {note}")
        print(f"    {model:<4} {res['accuracy']:>6.1%}  [{lo:.1%}, {hi:.1%}]   "
              f"chance {res['chance']:.1%}   -> {verdict}")
        print(f"    {why}")
        if group == "artefact_strict" and verdict == "BROKEN":
            problems.append("the strict artefact probe is above chance AFTER mitigation: "
                            "the fixed-length RMS-normalised crop did not take effect")
    report["probes"] = probes

    # ---------------------------------------------------------------- 5. the bar
    banner("5. the bar for the neural counter")
    bar = probes["acoustic:mitigated"]["accuracy"]
    chance = probes["acoustic:mitigated"]["chance"]
    print(f"  chance ................................. {chance:>6.1%}")
    print(f"  best naive predictor ................... "
          f"{max(r['accuracy'] for r in naive.values()):>6.1%}")
    print(f"  hand-crafted acoustic features + GBM ... {bar:>6.1%}   <-- BEAT THIS")
    print()
    print("  For reference, the joint count-and-separate model this project replaces scored")
    print("  20.00 % in fp32 on a balanced 5-class set -- below every row above.")
    report["bar"] = {"chance": chance, "acoustic_gbm": bar}

    # ---------------------------------------------------------------- write, then VERIFY
    path = os.path.join(out_dir, "audit_report.json")
    report["problems"] = problems
    report["ok"] = not problems
    json_dump_atomic(report, path)

    # The predecessor exited 0 having written nothing. Prove we did not.
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        print(f"\nFAILED: expected a report at {path} and it is missing or truncated.",
              file=sys.stderr)
        return 2
    for required in ("naive", "probes", "bar"):
        if not report.get(required):
            print(f"\nFAILED: report section {required!r} is empty.", file=sys.stderr)
            return 2

    print(f"\nreport -> {path}  ({os.path.getsize(path)} bytes, "
          f"{len(report['sections'])} audit sections, {len(probes)} probes)")

    if problems:
        print("\nPROBLEMS FOUND:", file=sys.stderr)
        for p in problems:
            print(f"  * {p}", file=sys.stderr)
        if args.strict:
            print("\nExiting non-zero because --strict is on. Fix these before training.",
                  file=sys.stderr)
            return 1
    else:
        print("\nAudit clean: splits disjoint, artefact cue removed, bar recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
