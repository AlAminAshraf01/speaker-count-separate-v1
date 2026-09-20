#!/usr/bin/env python3
"""Final evaluation of the whole system on the frozen test set.

    python scripts/06_evaluate.py --store /kaggle/input/.../store \\
        --recipes_test data/recipes_test.csv \\
        --counter /kaggle/working/counter/ckpt/best.pt \\
        --separator /kaggle/working/sep/ckpt/best.pt \\
        --tier_a /kaggle/working/tier_a/tier_a_model.joblib --out /kaggle/working/eval

REPORT FOUR NUMBERS, NEVER ONE
------------------------------
A single score cannot describe a system that does two jobs, and averaging them hides which half
is broken. This prints, and the report JSON stores:

1. **Counting accuracy plus the full confusion matrix.** The classes are ordinal: 3 -> 4 is not
   the failure 3 -> 5 is, and accuracy alone cannot tell them apart. MAE and off-by-one come
   with it.
2. **P-SI-SNR over the whole test set** -- the metric that stays defined when the predicted
   count is wrong, so it cannot be gamed by a system that quietly refuses to commit.
3. **SI-SDRi per N on the count-correct subset only.** This is the number comparable to the
   fixed-N separation literature, because that literature is always told N.
4. **SI-SDRi with the TRUE count forced.** Separation quality with a perfect counter. Without
   it, a poor end-to-end score cannot be attributed to either half, and the two halves are
   trained separately precisely so they can be debugged separately.

Plus the naive floors, because "we beat a trivial baseline" means nothing until the baseline
has a number beside it.

ONE MEASUREMENT RULE THAT COST v0 A HEADLINE
--------------------------------------------
SI-SDRi is **undefined for a clean single-speaker mixture**: the mixture already *is* the
target, so the "improvement" being measured is EPS. ``usable_si_sdri`` leaves those references
out and returns how many it dropped; this script reports the count rather than hiding it. In v0
that single bug turned a +1.2 dB result into a reported -8.68 dB.
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
    ap.add_argument("--counter", default=None, help="best.pt from 04_train_counter.py")
    ap.add_argument("--separator", default=None, help="best.pt from 05_train_separator.py")
    ap.add_argument("--tier_a", default=None, help="tier_a_model.joblib from 03_tier_a.py")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="/kaggle/working/eval")
    args = ap.parse_args()

    import torch

    from countsep.baselines import naive_scores, wilson_interval
    from countsep.constants import MAX_N_SRC, N_CLASSES, N_LIST, class_to_n, n_to_class
    from countsep.datasets import DEFAULT_MIXING, FrozenMixDataset, build_loader
    from countsep.metrics import count_report, format_confusion, p_si_snr, usable_si_sdri
    from countsep.mixing import recipe_speakers
    from countsep.utils import format_table, json_dump_atomic, pick_device

    banner("06 - final evaluation of the whole system   (FROZEN test set)")
    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    recipes = resolve(args.recipes_test) or args.recipes_test
    if not os.path.exists(recipes):
        raise SystemExit(f"no frozen test recipes at {recipes!r}. Run 01_make_frozen_sets.py.")

    store, bank = build_store_and_bank(store_root, args.test_split,
                                       noise_kinds=DEFAULT_MIXING["noise_kinds"])
    dataset = FrozenMixDataset(store, bank, recipes, want="separate", limit=args.limit)
    speakers = [recipe_speakers(r, store) for r in dataset.recipes]
    distinct = sorted({s for row in speakers for s in row})
    device = pick_device(None)
    print(f"  {len(dataset)} mixtures {dataset.counts_per_n()} from "
          f"{len(distinct)} distinct speakers")
    print(f"  device: {device}")

    report: dict = {"code_version": code_version(), "n": len(dataset),
                    "n_speakers": len(distinct), "recipes": recipes}

    y_true = np.array([n_to_class(int(r["n_src"])) for r in dataset.recipes])
    banner("naive floors")
    naive = naive_scores(y_true, N_CLASSES)
    for name, res in naive.items():
        print(f"  {name:<20} {res['accuracy']:>6.1%}   MAE {res['mae']:.3f}")
    report["naive"] = naive

    # ---------------------------------------------------------------- counting
    counter = None
    counter_ckpt = resolve(args.counter) if args.counter else None
    if counter_ckpt and os.path.exists(counter_ckpt):
        from countsep.checkpoint import load_checkpoint
        from countsep.counter import ModelConfig as CounterConfig
        from countsep.counter import build_model as build_counter

        state = torch.load(counter_ckpt, map_location="cpu", weights_only=False)
        pooling = str(state.get("cfg", {}).get("pooling", "eigen"))
        counter = build_counter(CounterConfig(pooling=pooling)).to(device).eval()
        load_checkpoint(counter_ckpt, model=counter)
        print(f"\n  counter: {counter.describe()} (pooling {pooling}, scored in fp32)")

    separator = None
    sep_ckpt = resolve(args.separator) if args.separator else None
    if sep_ckpt and os.path.exists(sep_ckpt):
        from countsep.checkpoint import load_checkpoint
        from countsep.separator import ModelConfig as SepConfig
        from countsep.separator import build_separator

        state = torch.load(sep_ckpt, map_location="cpu", weights_only=False)
        cfg = state.get("cfg", {}).get("model", {})
        separator = build_separator(SepConfig(**cfg) if cfg else SepConfig()).to(device).eval()
        load_checkpoint(sep_ckpt, model=separator)
        print(f"  separator: {separator.describe()}")

    if counter is None and separator is None:
        print("\nFAILED: pass --counter and/or --separator; nothing to evaluate.",
              file=sys.stderr)
        return 2

    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, persistent=False)

    pred_cls: list[int] = []
    psi: list[float] = []
    per_n_correct: dict[int, list[float]] = {}
    per_n_oracle: dict[int, list[float]] = {}
    dropped = 0

    banner("scoring")
    with torch.no_grad():
        for batch in loader:
            mix = batch["mix"].to(device)
            n_true = batch["n_src"].numpy()

            if counter is not None:
                logits = counter(mix)["logits"].float().cpu()
                n_hat = np.array([class_to_n(int(c)) for c in logits.argmax(-1)])
                pred_cls.extend(int(c) for c in logits.argmax(-1))
            else:
                n_hat = n_true.copy()

            if separator is None:
                continue
            est = separator(mix)["est"].float().cpu().numpy()[:, :MAX_N_SRC]
            refs = batch["refs"].numpy()
            mixes = batch["mix"].numpy()
            for i in range(len(n_true)):
                psi.append(p_si_snr(est[i], refs[i], int(n_true[i]), int(n_hat[i]),
                                    max_n_src=MAX_N_SRC))
                scores, n_drop = usable_si_sdri(est[i], refs[i], mixes[i], int(n_true[i]))
                dropped += int(n_drop)
                finite = [float(v) for v in np.asarray(scores).ravel() if np.isfinite(v)]
                if finite:
                    per_n_oracle.setdefault(int(n_true[i]), []).extend(finite)
                    if int(n_hat[i]) == int(n_true[i]):
                        per_n_correct.setdefault(int(n_true[i]), []).extend(finite)

    # ---------------------------------------------------------------- 1. counting
    if counter is not None:
        banner("1. counting")
        pred_n = [class_to_n(c) for c in pred_cls]
        true_n = [int(r["n_src"]) for r in dataset.recipes]
        rep = count_report(true_n, pred_n)
        lo, hi = wilson_interval(rep["accuracy"], len(true_n))
        correct = np.array(pred_cls) == y_true

        by_speaker: dict[str, list[int]] = {}
        for i, spk in enumerate(speakers):
            by_speaker.setdefault(str(spk[0]), []).append(i)
        keys = sorted(by_speaker)
        rng = np.random.default_rng(args.seed)
        boots = []
        for _ in range(args.n_boot):
            drawn = rng.choice(len(keys), size=len(keys), replace=True)
            picked = np.concatenate([by_speaker[keys[k]] for k in drawn])
            boots.append(float(correct[picked].mean()))
        blo, bhi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

        print(f"  accuracy ........ {rep['accuracy']:.1%}")
        print(f"    Wilson 95 % ... [{lo:.1%}, {hi:.1%}]   (treats mixtures as independent)")
        print(f"    speaker 95 % .. [{blo:.1%}, {bhi:.1%}]   <-- the honest one "
              f"({len(distinct)} clusters, not {len(dataset)} mixtures)")
        print(f"  MAE ............. {rep['mae']:.3f}")
        print("\n" + format_confusion(np.array(rep["confusion"])))
        rep.update({"wilson95": [lo, hi], "speaker_bootstrap95": [blo, bhi]})
        report["counting"] = rep

    # ---------------------------------------------------------------- 2-4. separation
    if separator is not None:
        banner("2. P-SI-SNR over the whole test set   (the ONLY counter-sensitive number)")
        print("  Defined even when the count is wrong, so it cannot be gamed by abstaining,")
        print("  and it is the one score here that reads the PREDICTED count: it keeps the")
        print("  n_hat loudest slots, exactly as inference does. Sections 3 and 4 both score")
        print("  against the true count, so neither of them responds to the counter at all.")
        print("  NOTE: this is an ABSOLUTE SI-SNR, not an improvement. Do not compare it")
        print("  with the SI-SDRi values below -- they measure different things in the same unit.")
        print(f"  P-SI-SNR ........ {float(np.mean(psi)):+.2f} dB   (n={len(psi)})")
        report["p_si_snr"] = float(np.mean(psi))

        banner("3. SI-SDRi per N, true-count scoring, COUNT-CORRECT clips only")
        print("  Scored against the true sources, restricted to the clips the counter got")
        print("  right. This is the row comparable to the fixed-N literature, which is always")
        print("  told N -- but note the restriction makes it a slightly easier subset.")
        rows = []
        for n in N_LIST:
            vals = per_n_correct.get(n, [])
            rows.append([f"N={n}", str(len(vals)),
                         f"{np.mean(vals):+.2f}" if vals else "--"])
        print(format_table(rows, ["true N", "sources", "SI-SDRi dB"]))
        pooled = [v for vals in per_n_correct.values() for v in vals]
        if pooled:
            print(f"  pooled .......... {float(np.mean(pooled)):+.2f} dB")
        report["si_sdri_count_correct"] = {str(n): float(np.mean(v))
                                           for n, v in sorted(per_n_correct.items()) if v}

        banner("4. SI-SDRi per N, true-count scoring, ALL clips")
        print("  Separation quality given a perfect counter, over every clip.")
        print("")
        print("  Section 3 minus section 4 is NOT the cost of miscounting, though an earlier")
        print("  version of this script and of docs/DESIGN.md both said it was. Both sections")
        print("  score with the TRUE count -- section 3 is literally a subset of the same")
        print("  numbers -- so neither reads the counter. Their difference is composition:")
        print("  section 3 drops the clips the counter missed, which shifts its mix of N.")
        print("  For what miscounting actually costs, read P-SI-SNR in section 2.")
        rows = []
        for n in N_LIST:
            vals = per_n_oracle.get(n, [])
            rows.append([f"N={n}", str(len(vals)),
                         f"{np.mean(vals):+.2f}" if vals else "--"])
        print(format_table(rows, ["true N", "sources", "SI-SDRi dB"]))
        pooled_o = [v for vals in per_n_oracle.values() for v in vals]
        if pooled_o:
            print(f"  pooled .......... {float(np.mean(pooled_o)):+.2f} dB")
        report["si_sdri_oracle_count"] = {str(n): float(np.mean(v))
                                          for n, v in sorted(per_n_oracle.items()) if v}
        report["n_degenerate_refs_dropped"] = dropped
        if dropped:
            print(f"\n  {dropped} reference(s) dropped as degenerate (the mixture already WAS")
            print("  the target, so 'improvement' there measures EPS, not separation).")

        print("\n  published Conv-TasNet, LibriMix 8k min, N=2: +14.76 dB "
              "(200 epochs, train-360)")

    path = os.path.join(out_dir, "eval_report.json")
    json_dump_atomic(report, path)
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        print(f"\nFAILED: report missing or truncated at {path}", file=sys.stderr)
        return 2
    print(f"\nreport -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
