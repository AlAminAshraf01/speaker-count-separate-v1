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

Sections 3 and 4 let the REFERENCES pick which of the 5 slots to score (Hungarian matching over
all of them), which inference cannot do. So the script also reports:

5. **SI-SDRi of the N loudest slots** -- the rule the system was designed with -- beside the
   oracle-slot number, plus how often the loudest slots are the right ones and by what margin
   in dB. Section 10 scores the level-blind ``rebuild`` rule inference uses since the dev check.
6. **Macro averages over N.** The pooled rows weight each SOURCE, so one N=5 clip counts five
   times; the macro row weights each N equally.
7. **Every number by noise type and SNR band**, from the SNR and noise kind stored in each
   recipe.
8. **The Tier A tree on the same test clips** (``--tier_a``), with an exact McNemar test against
   the counter. Its 57.8 % in ``tier_a_report.json`` is cross-validation on TRAINING speakers,
   so it is not the same experiment as the counter's test accuracy; this row is.
9. **Oracle spectral masks (IRM / IBM)**, computed from the true sources and noise -- an upper
   bound on masking, not a competitor -- and the unprocessed mixture, which is 0 dB SI-SDRi by
   definition.
10. **Choosing slots without loudness.** Two level-blind rules that need no retraining: keep
   the N slots whose least-squares combination (plus the noise slot) best rebuilds the mixture
   -- the proposed rule, fixed before the run -- and, as a second opinion, the N slots that
   each explain the most mixture energy. Scored like section 5.

Every clip's numbers are also written to ``eval_records.json`` so figures can be redrawn
without re-running the models.

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

#: SNR bands for the condition breakdown. Test SNRs are drawn from [5, 20] dB.
SNR_BANDS: tuple[tuple[float, float], ...] = ((5.0, 10.0), (10.0, 15.0), (15.0, 20.01))


def _mean(values) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _fmt(value: float, spec: str = "+.2f") -> str:
    return format(value, spec) if value == value else "--"


def per_n_means(records: list[dict], key: str, n_list) -> dict[int, float]:
    """Mean over every SOURCE of the per-clip list ``key``, grouped by true N."""
    out = {}
    for n in n_list:
        vals = [v for r in records if r["n_true"] == n for v in (r.get(key) or [])]
        out[int(n)] = _mean(vals)
    return out


def macro(per_n: dict[int, float], n_list) -> float:
    """Unweighted mean over the N values that have a number -- each N counts once."""
    return _mean([per_n.get(int(n)) for n in n_list])


def condition_of(rec: dict) -> tuple[str, str]:
    """(noise kind, SNR band) of one clip, for the breakdown table."""
    kind = str(rec["noise_kind"])
    if kind == "none":
        return kind, "clean"
    for lo, hi in SNR_BANDS:
        if lo <= rec["snr_db"] < hi:
            return kind, f"{lo:.0f}-{min(hi, 20.0):.0f} dB"
    return kind, "other"


def condition_rows(records: list[dict], groups: list[tuple[str, list[dict]]],
                   has_counter: bool, has_sep: bool) -> tuple[list[list[str]], dict]:
    """One table row per condition group: counting accuracy and N=2..5 macro SI-SDRi.

    Separation is macro-averaged over N = 2..5 only. Clean N=1 references are dropped as
    degenerate, so the "clean" group has no N=1 sources while the noisy groups do; averaging
    over the same set of N in every group keeps the comparison about the noise, not about
    which N happened to survive.
    """
    rows, stored = [], {}
    for label, group in groups:
        acc = (_mean([float(r["n_hat"] == r["n_true"]) for r in group])
               if has_counter else float("nan"))
        oracle = macro(per_n_means(group, "oracle", (2, 3, 4, 5)), (2, 3, 4, 5)) \
            if has_sep else float("nan")
        loud = macro(per_n_means(group, "loud", (2, 3, 4, 5)), (2, 3, 4, 5)) \
            if has_sep else float("nan")
        psi = _mean([r.get("psi") for r in group]) if has_sep else float("nan")
        rows.append([label, str(len(group)), _fmt(100 * acc, ".1f"), _fmt(oracle),
                     _fmt(loud), _fmt(psi)])
        stored[label] = {"n": len(group), "count_accuracy": acc,
                         "si_sdri_oracle_slots_n2to5": oracle,
                         "si_sdri_loudest_slots_n2to5": loud, "p_si_snr": psi}
    return rows, stored


def load_tier_a(path: str | None):
    """The Tier A tree from 03_tier_a.py, or None when it is absent or cannot be read."""
    if not path or not os.path.exists(path):
        return None
    try:
        import joblib

        bundle = joblib.load(path)
        return bundle if "model" in bundle and "features" in bundle else None
    except Exception as exc:                       # noqa: BLE001 - optional input
        print(f"  could not load Tier A from {path} ({exc}); skipping it")
        return None


def speaker_bootstrap(correct: np.ndarray, speakers: list[list[str]], n_boot: int,
                      seed: int) -> tuple[float, float]:
    """Percentile interval from resampling FIRST speakers with their clips intact.

    Grouping by the first speaker only is a known simplification: an N=5 clip's other four
    talkers can land in several clusters. See docs/DESIGN.md section 5.
    """
    by_speaker: dict[str, list[int]] = {}
    for i, spk in enumerate(speakers):
        by_speaker.setdefault(str(spk[0]), []).append(i)
    keys = sorted(by_speaker)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        drawn = rng.choice(len(keys), size=len(keys), replace=True)
        picked = np.concatenate([by_speaker[keys[k]] for k in drawn])
        boots.append(float(correct[picked].mean()))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def score_separation(rec: dict, est: np.ndarray, refs: np.ndarray, mix: np.ndarray,
                     noise: np.ndarray, *, max_n_src: int, oracle_masks: bool,
                     noise_est: np.ndarray | None = None) -> int:
    """Fill one clip's separation fields in ``rec``; returns degenerate references dropped."""
    from countsep.baselines import ideal_mask_estimates
    from countsep.metrics import (best_permutation, consistent_slots, loudest_slots, p_si_snr,
                                  projection_slots, slot_selection, usable_si_sdri)

    n_true = rec["n_true"]
    n_hat = rec["n_hat"] if rec["n_hat"] is not None else n_true
    rec["psi"] = float(p_si_snr(est, refs, n_true, n_hat, max_n_src=max_n_src))
    oracle, n_drop = usable_si_sdri(est, refs, mix, n_true)
    loud, _ = usable_si_sdri(est[loudest_slots(est, n_true)], refs, mix, n_true)
    rec["oracle"] = [float(v) for v in np.asarray(oracle).ravel() if np.isfinite(v)]
    rec["loud"] = [float(v) for v in np.asarray(loud).ravel() if np.isfinite(v)]
    sel = slot_selection(est, refs, mix, n_true)
    rec.update({"sel_correct": bool(sel["correct"]), "gap_db": sel["gap_db"],
                "kept_db": sel["kept_db"], "spare_db": sel["spare_db"]})
    matched = set(int(c) for c in best_permutation(est, refs, n_true)[1])
    for rule, picked in (("consistent", consistent_slots(est, mix, n_true, noise_est)),
                         ("projection", projection_slots(est, mix, n_true))):
        vals, _ = usable_si_sdri(est[picked], refs, mix, n_true)
        rec[rule] = [float(v) for v in np.asarray(vals).ravel() if np.isfinite(v)]
        rec[f"sel_{rule}_correct"] = set(int(k) for k in picked) == matched
    # End to end with the rebuild rule: the PREDICTED count, and the slots rebuild keeps.
    rec["psi_rebuild"] = float(p_si_snr(est, refs, n_true, n_hat, max_n_src=max_n_src,
                                        chosen=consistent_slots(est, mix, n_hat, noise_est)))
    if oracle_masks:
        for mode in ("irm", "ibm"):
            masked = ideal_mask_estimates(mix, refs[:n_true], noise, mode=mode)
            vals, _ = usable_si_sdri(masked, refs, mix, n_true)
            rec[mode] = [float(v) for v in np.asarray(vals).ravel() if np.isfinite(v)]
    return int(n_drop)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--recipes_test", default="data/recipes_test.csv")
    ap.add_argument("--counter", default=None, help="best.pt from 04_train_counter.py")
    ap.add_argument("--separator", default=None, help="best.pt from 05_train_separator.py")
    ap.add_argument("--tier_a", default=None, help="tier_a_model.joblib from 03_tier_a.py")
    ap.add_argument("--no_oracle_masks", action="store_true",
                    help="skip the IRM/IBM rows (they are CPU-only and take ~1 min)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="/kaggle/working/eval")
    args = ap.parse_args()

    import torch

    from countsep.baselines import mcnemar_exact, naive_scores, wilson_interval
    from countsep.constants import MAX_N_SRC, N_CLASSES, N_LIST, class_to_n, n_to_class
    from countsep.datasets import DEFAULT_MIXING, FrozenMixDataset, build_loader
    from countsep.features import acoustic_features, features_to_matrix
    from countsep.metrics import count_report, format_confusion
    from countsep.mixing import recipe_speakers
    from countsep.utils import format_table, json_dump_atomic, pick_device

    banner(f"06 - final evaluation of the whole system   (FROZEN {args.test_split} split)")
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

    # ---------------------------------------------------------------- load the models
    counter = None
    counter_ckpt = resolve(args.counter) if args.counter else None
    if counter_ckpt and os.path.exists(counter_ckpt):
        from countsep.checkpoint import load_checkpoint
        from countsep.counter import build_model as build_counter
        from countsep.counter import config_from_checkpoint

        state = torch.load(counter_ckpt, map_location="cpu", weights_only=False)
        counter = build_counter(config_from_checkpoint(state)).to(device).eval()
        load_checkpoint(counter_ckpt, model=counter)
        print(f"\n  counter: {counter.describe()} (scored in fp32)")

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

    tier_a = load_tier_a(resolve(args.tier_a) if args.tier_a else None)
    if tier_a is not None:
        print(f"  Tier A: {type(tier_a['model']).__name__} over {len(tier_a['features'])} "
              f"features")

    if counter is None and separator is None:
        print("\nFAILED: pass --counter and/or --separator; nothing to evaluate.",
              file=sys.stderr)
        return 2

    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, persistent=False)

    # ---------------------------------------------------------------- one pass, every clip
    records: list[dict] = []
    tier_rows: list[dict] = []
    dropped = 0
    banner("scoring")
    with torch.no_grad():
        for batch in loader:
            mix = batch["mix"].to(device)
            n_true = batch["n_src"].numpy()
            mixes = batch["mix"].numpy()
            n_hat = None
            if counter is not None:
                logits = counter(mix)["logits"].float().cpu()
                n_hat = [class_to_n(int(c)) for c in logits.argmax(-1)]
            est_all = (separator(mix)["est"].float().cpu().numpy()
                       if separator is not None else None)
            est = est_all[:, :MAX_N_SRC] if est_all is not None else None
            noise_est = (est_all[:, MAX_N_SRC] if est_all is not None
                         and est_all.shape[1] > MAX_N_SRC else None)
            for i in range(len(n_true)):
                recipe = dataset.recipes[len(records)]
                rec = {"mix_id": str(recipe["mix_id"]), "n_true": int(n_true[i]),
                       "n_hat": int(n_hat[i]) if n_hat is not None else None,
                       "snr_db": float(recipe["snr_db"]),
                       "noise_kind": str(recipe["noise_kind"])}
                if est is not None:
                    dropped += score_separation(
                        rec, est[i], batch["refs"][i].numpy(), mixes[i],
                        batch["noise"][i].numpy(), max_n_src=MAX_N_SRC,
                        oracle_masks=not args.no_oracle_masks,
                        noise_est=noise_est[i] if noise_est is not None else None)
                if tier_a is not None:
                    tier_rows.append(acoustic_features(mixes[i]))
                records.append(rec)
            print(f"    {len(records)}/{len(dataset)}", end="\r", flush=True)
    print()
    has_counter, has_sep = counter is not None, separator is not None

    # ---------------------------------------------------------------- 1. counting
    correct = None
    if has_counter:
        banner("1. counting")
        pred_n = [r["n_hat"] for r in records]
        true_n = [r["n_true"] for r in records]
        rep = count_report(true_n, pred_n)
        lo, hi = wilson_interval(rep["accuracy"], len(true_n))
        correct = np.array(pred_n) == np.array(true_n)
        blo, bhi = speaker_bootstrap(correct, speakers, args.n_boot, args.seed)

        print(f"  accuracy ........ {rep['accuracy']:.1%}")
        print(f"    Wilson 95 % ... [{lo:.1%}, {hi:.1%}]   (treats mixtures as independent)")
        # Which of the two is "honest" depends on how they came out, so decide here
        # rather than hard-coding it. A cluster bootstrap resamples whole speakers and
        # keeps each drawn speaker's clips intact, so it never resamples WITHIN a
        # cluster: it measures between-speaker variance and misses the binomial
        # variation Wilson covers. When speakers are homogeneous it therefore comes out
        # NARROWER than Wilson and under-covers -- measured here, that happens in 62 %
        # of simulated runs with no speaker effect at all. Calling the narrower one
        # "honest" would understate the uncertainty in the headline number.
        print(f"    speaker 95 % .. [{blo:.1%}, {bhi:.1%}]   "
              f"({len(distinct)} clusters, not {len(dataset)} mixtures)")
        if (bhi - blo) > (hi - lo):
            print(f"    -> QUOTE THE SPEAKER INTERVAL. It is wider, so clip-level")
            print(f"       independence was the optimistic assumption: these {len(dataset)}")
            print(f"       mixtures really do come from only {len(distinct)} people.")
        else:
            print(f"    -> QUOTE WILSON. The speaker interval came out NARROWER, which means")
            print(f"       between-speaker variation is smaller than ordinary binomial noise")
            print(f"       -- a real finding about the data, but the narrower of two intervals")
            print(f"       is never the conservative one. Report Wilson and say the speaker")
            print(f"       bootstrap agreed, rather than quoting the tighter number.")
        print(f"  MAE ............. {rep['mae']:.3f}")
        print("\n" + format_confusion(np.array(rep["confusion"])))
        rep.update({"wilson95": [lo, hi], "speaker_bootstrap95": [blo, bhi]})
        report["counting"] = rep

    # ---------------------------------------------------------------- 2-4. separation
    if has_sep:
        psi = [r["psi"] for r in records]
        oracle_n = per_n_means(records, "oracle", N_LIST)
        correct_recs = [r for r in records if r["n_hat"] is None or r["n_hat"] == r["n_true"]]
        banner("2. P-SI-SNR over the whole test set   (the ONLY counter-sensitive number)")
        print("  Defined even when the count is wrong, so it cannot be gamed by abstaining,")
        print("  and it is the one score here that reads the PREDICTED count: it keeps the")
        print("  n_hat loudest slots (section 10 repeats it with rebuild, the default rule at")
        print("  inference since the dev check). Sections 3 and 4 both score")
        print("  against the true count, so neither of them responds to the counter at all.")
        print("  NOTE: this is an ABSOLUTE SI-SNR, not an improvement. Do not compare it")
        print("  with the SI-SDRi values below -- they measure different things in the same unit.")
        print(f"  P-SI-SNR ........ {float(np.mean(psi)):+.2f} dB   (n={len(psi)})")
        psi_n = {n: _mean([r["psi"] for r in records if r["n_true"] == n]) for n in N_LIST}
        print("  per N ........... " + "  ".join(f"N{n}:{_fmt(v)}" for n, v in psi_n.items()))
        report["p_si_snr"] = float(np.mean(psi))
        report["p_si_snr_per_n"] = {str(n): v for n, v in psi_n.items()}

        banner("3. SI-SDRi per N, true-count scoring, COUNT-CORRECT clips only")
        print("  Scored against the true sources, restricted to the clips the counter got")
        print("  right. This is the row comparable to the fixed-N literature, which is always")
        print("  told N -- but note the restriction makes it a slightly easier subset.")
        correct_n = per_n_means(correct_recs, "oracle", N_LIST)
        rows = [[f"N={n}", str(sum(len(r["oracle"]) for r in correct_recs if r["n_true"] == n)),
                 _fmt(correct_n[n])] for n in N_LIST]
        print(format_table(rows, ["true N", "sources", "SI-SDRi dB"]))
        pooled = [v for r in correct_recs for v in r["oracle"]]
        if pooled:
            print(f"  pooled .......... {float(np.mean(pooled)):+.2f} dB")
        report["si_sdri_count_correct"] = {str(n): v for n, v in correct_n.items() if v == v}

        banner("4. SI-SDRi per N, true-count scoring, ALL clips")
        print("  Separation quality given a perfect counter, over every clip.")
        print("")
        print("  Section 3 minus section 4 is NOT the cost of miscounting, though an earlier")
        print("  version of this script and of docs/DESIGN.md both said it was. Both sections")
        print("  score with the TRUE count -- section 3 is literally a subset of the same")
        print("  numbers -- so neither reads the counter. Their difference is composition:")
        print("  section 3 drops the clips the counter missed, which shifts its mix of N.")
        print("  For what miscounting actually costs, read P-SI-SNR in section 2.")
        rows = [[f"N={n}", str(sum(len(r["oracle"]) for r in records if r["n_true"] == n)),
                 _fmt(oracle_n[n])] for n in N_LIST]
        print(format_table(rows, ["true N", "sources", "SI-SDRi dB"]))
        pooled_o = [v for r in records for v in r["oracle"]]
        if pooled_o:
            print(f"  pooled .......... {float(np.mean(pooled_o)):+.2f} dB")
        report["si_sdri_oracle_count"] = {str(n): v for n, v in oracle_n.items() if v == v}
        report["n_degenerate_refs_dropped"] = dropped
        if dropped:
            print(f"\n  {dropped} reference(s) dropped as degenerate (the mixture already WAS")
            print("  the target, so 'improvement' there measures EPS, not separation).")

        # ------------------------------------------------------------ 5. the slots it keeps
        banner("5. the N LOUDEST slots   (the rule the system was designed with)")
        print("  Sections 3 and 4 let the references choose which of the 5 slots to score.")
        print("  Inference cannot. The design kept the N loudest; this scores THOSE slots (true N")
        print("  forced), and asks how often loudness picked the right ones, and by how much.")
        print("  gap = quietest real-speaker slot minus loudest spare slot, in dB; a negative")
        print("  gap is a clip where loudness keeps a spare instead of a speaker.\n")
        loud_n = per_n_means(records, "loud", N_LIST)
        rows, sel_store = [], {}
        for n in N_LIST:
            group = [r for r in records if r["n_true"] == n]
            gaps = [r["gap_db"] for r in group if r["gap_db"] is not None]
            sel = _mean([float(r["sel_correct"]) for r in group]) if n < MAX_N_SRC else 1.0
            med = float(np.median(gaps)) if gaps else float("nan")
            p05 = float(np.percentile(gaps, 5)) if gaps else float("nan")
            under3 = _mean([float(g < 3.0) for g in gaps])
            kept = float(np.median([r["kept_db"] for r in group])) if group else float("nan")
            rows.append([f"N={n}", _fmt(oracle_n[n]), _fmt(loud_n[n]),
                         _fmt(loud_n[n] - oracle_n[n]),
                         (f"{sel:.1%}" if n < MAX_N_SRC else "(all kept)"),
                         _fmt(med), _fmt(p05), _fmt(100 * under3, ".0f"), _fmt(kept, "+.1f")])
            sel_store[str(n)] = {"si_sdri_oracle_slots": oracle_n[n],
                                 "si_sdri_loudest_slots": loud_n[n],
                                 "selection_correct": sel if n < MAX_N_SRC else None,
                                 "gap_db_median": med, "gap_db_p05": p05,
                                 "gap_under_3db": under3, "kept_db_median": kept}
        print(format_table(rows, ["true N", "oracle", "loudest", "cost", "right slots",
                                  "gap med", "gap p5", "% gap<3", "kept dB"]))
        all_gaps = [r["gap_db"] for r in records if r["gap_db"] is not None]
        loud_correct = per_n_means(correct_recs, "loud", N_LIST)
        pooled_l = [v for r in records for v in r["loud"]]
        print(f"\n  pooled, loudest slots .. {_mean(pooled_l):+.2f} dB   "
              f"(oracle slots {_mean(pooled_o):+.2f} dB)")
        print(f"  count-correct clips, loudest slots (the true end-to-end number): "
              f"{_mean([v for r in correct_recs for v in r['loud']]):+.2f} dB")
        if all_gaps:
            print(f"  gap over N=1..4 ........ median {np.median(all_gaps):+.2f} dB, "
                  f"{100 * _mean([float(g < 0) for g in all_gaps]):.1f} % negative")
        print("  'kept dB' is the level of the real-speaker slots relative to the mixture.")
        print("  SI-SDR cannot see scale, so the separation loss never asked for a level.")
        report["slot_selection"] = {
            "per_n": sel_store, "pooled_loudest_slots": _mean(pooled_l),
            "pooled_oracle_slots": _mean(pooled_o),
            "si_sdri_count_correct_loudest": {str(n): v for n, v in loud_correct.items()
                                              if v == v},
            "gap_db_median_n1to4": float(np.median(all_gaps)) if all_gaps else None,
            "gap_negative_fraction_n1to4": _mean([float(g < 0) for g in all_gaps])}

        # ------------------------------------------------------------ 6. macro averages
        banner("6. macro averages   (each N counts once, not each source)")
        print("  The pooled rows weight SOURCES: one N=5 clip contributes five values and an")
        print("  N=1 clip one, so N=4-5 dominate them. The macro row weights each N equally.\n")
        macro_store = {}
        for label, per_n in (("oracle slots", oracle_n), ("loudest slots", loud_n)):
            m15, m25 = macro(per_n, N_LIST), macro(per_n, (2, 3, 4, 5))
            print(f"  {label:<14} N=1..5 {m15:+.2f} dB    N=2..5 {m25:+.2f} dB")
            macro_store[label.replace(" ", "_")] = {"n1to5": m15, "n2to5": m25}
        report["macro"] = macro_store

    # ---------------------------------------------------------------- 7. by condition
    banner("7. by noise type and SNR band")
    print("  Counting accuracy over every N; separation macro-averaged over N=2..5 so every")
    print("  group averages the same N (clean N=1 references are dropped as degenerate).\n")
    kinds = sorted({condition_of(r)[0] for r in records})
    bands = ["clean"] + [f"{lo:.0f}-{min(hi, 20.0):.0f} dB" for lo, hi in SNR_BANDS]
    groups = [(f"noise: {k}", [r for r in records if condition_of(r)[0] == k]) for k in kinds]
    groups += [(f"SNR: {b}", [r for r in records if condition_of(r)[1] == b]) for b in bands]
    groups = [(label, g) for label, g in groups if g]
    rows, cond_store = condition_rows(records, groups, has_counter, has_sep)
    print(format_table(rows, ["condition", "clips", "count %", "SI-SDRi oracle",
                              "SI-SDRi loudest", "P-SI-SNR"]))
    print("\n  These are test SNRs of 5-20 dB, the training range. Nothing here measures the")
    print("  proposal's -6 to +3 dB, or recorded (non-synthetic) noise.")
    report["by_condition"] = cond_store

    # ---------------------------------------------------------------- 8. Tier A, same clips
    if tier_a is not None:
        banner("8. Tier A on the SAME test clips   (16 features + gradient-boosted tree)")
        X, _ = features_to_matrix(tier_rows, tier_a["features"])
        tier_pred = [class_to_n(int(c)) for c in tier_a["model"].predict(X)]
        true_n = [r["n_true"] for r in records]
        t_rep = count_report(true_n, tier_pred)
        t_lo, t_hi = wilson_interval(t_rep["accuracy"], len(true_n))
        print(f"  accuracy ........ {t_rep['accuracy']:.1%}   Wilson 95 % [{t_lo:.1%}, "
              f"{t_hi:.1%}]   MAE {t_rep['mae']:.3f}   (n={len(tier_rows)})")
        print("\n" + format_confusion(np.array(t_rep["confusion"])))
        t_rep.update({"wilson95": [t_lo, t_hi]})
        if correct is not None:
            tier_correct = np.array(tier_pred) == np.array(true_n)
            mc = mcnemar_exact(correct, tier_correct)
            print(f"\n  paired against the counter (exact McNemar): counter right & tree wrong "
                  f"{mc['only_a_correct']}, tree right & counter wrong {mc['only_b_correct']}, "
                  f"p = {mc['p_value']:.3g}")
            t_rep["mcnemar_vs_counter"] = mc
        print("\n  tier_a_report.json's accuracy is 5-fold CV on TRAINING speakers; THIS row is")
        print("  the one to put beside the counter's test accuracy.")
        report["tier_a_test"] = t_rep

    # ---------------------------------------------------------------- 9. oracle masks
    if has_sep and not args.no_oracle_masks:
        banner("9. oracle spectral masks   (an UPPER bound on masking, not a competitor)")
        print("  IRM and IBM are computed from the TRUE sources and noise (STFT 256/64), so no")
        print("  system that has to estimate them can match them in general. The unprocessed")
        print("  mixture is the floor: 0 dB SI-SDRi by definition. Conv-TasNet beat both oracles")
        print("  on WSJ0-2mix with 200 epochs; how far this separator is from them is budget.\n")
        irm_n, ibm_n = per_n_means(records, "irm", N_LIST), per_n_means(records, "ibm", N_LIST)
        rows = [[f"N={n}", "+0.00", _fmt(oracle_n[n]), _fmt(loud_n[n]), _fmt(irm_n[n]),
                 _fmt(ibm_n[n])] for n in N_LIST]
        rows.append(["macro 2-5", "+0.00", _fmt(macro(oracle_n, (2, 3, 4, 5))),
                     _fmt(macro(loud_n, (2, 3, 4, 5))), _fmt(macro(irm_n, (2, 3, 4, 5))),
                     _fmt(macro(ibm_n, (2, 3, 4, 5)))])
        print(format_table(rows, ["true N", "mixture", "ours (oracle slots)",
                                  "ours (loudest)", "IRM", "IBM"]))
        report["oracle_masks"] = {"irm": {str(n): v for n, v in irm_n.items()},
                                  "ibm": {str(n): v for n, v in ibm_n.items()},
                                  "stft": {"n_fft": 256, "hop": 64}}

    # ---------------------------------------------------------------- 10. level-blind selection
    if has_sep:
        banner("10. choosing slots WITHOUT loudness   (no retraining; inference-side only)")
        print("  Section 5 shows loudness no longer ranks talkers above spares. Two level-blind")
        print("  rules that use only the model's own outputs and the mixture:")
        print("    rebuild    -- the N slots (+ the noise slot) whose least-squares mix best")
        print("                  rebuilds the input. The system's rule: proposed after the")
        print("                  first test run, adopted only after a dev check fixed in")
        print("                  advance (paired CI above zero on dev).")
        print("    projection -- the N slots that each explain the most mixture energy alone.")
        print("                  A second opinion; it was never a candidate for adoption.")
        print(f"  Scored here on the {args.test_split} split.\n")
        rule_n = {rule: per_n_means(records, rule, N_LIST)
                  for rule in ("consistent", "projection")}
        rows, rule_store = [], {}
        for n in N_LIST:
            group = [r for r in records if r["n_true"] == n]
            right = {rule: (_mean([float(r[f"sel_{rule}_correct"]) for r in group])
                            if n < MAX_N_SRC else 1.0) for rule in rule_n}
            loud_right = (_mean([float(r["sel_correct"]) for r in group])
                          if n < MAX_N_SRC else 1.0)
            rates = [f"{v:.1%}" if n < MAX_N_SRC else "(all)"
                     for v in (loud_right, right["consistent"], right["projection"])]
            rows.append([f"N={n}", _fmt(oracle_n[n]), _fmt(loud_n[n]),
                         _fmt(rule_n["consistent"][n]), _fmt(rule_n["projection"][n]), *rates])
            rule_store[str(n)] = {"si_sdri_oracle_slots": oracle_n[n],
                                  "si_sdri_loudest": loud_n[n],
                                  "si_sdri_rebuild": rule_n["consistent"][n],
                                  "si_sdri_projection": rule_n["projection"][n],
                                  "right_slots_loudest": loud_right if n < MAX_N_SRC else None,
                                  "right_slots_rebuild":
                                      right["consistent"] if n < MAX_N_SRC else None,
                                  "right_slots_projection":
                                      right["projection"] if n < MAX_N_SRC else None}
        print(format_table(rows, ["true N", "oracle", "loudest", "rebuild", "projection",
                                  "right: loud", "right: rebuild", "right: proj"]))
        pooled = {rule: _mean([v for r in records for v in r[rule]]) for rule in rule_n}
        pooled_cc = {rule: _mean([v for r in correct_recs for v in r[rule]]) for rule in rule_n}
        macro_r = {rule: macro(rule_n[rule], (2, 3, 4, 5)) for rule in rule_n}
        print(f"\n  pooled ........... oracle {_mean(pooled_o):+.2f}   loudest {_mean(pooled_l):+.2f}"
              f"   rebuild {pooled['consistent']:+.2f}   projection {pooled['projection']:+.2f} dB")
        print(f"  macro N=2..5 ..... oracle {macro(oracle_n, (2, 3, 4, 5)):+.2f}   loudest "
              f"{macro(loud_n, (2, 3, 4, 5)):+.2f}   rebuild {macro_r['consistent']:+.2f}   "
              f"projection {macro_r['projection']:+.2f} dB")
        print(f"  count-correct .... rebuild {pooled_cc['consistent']:+.2f}   projection "
              f"{pooled_cc['projection']:+.2f} dB   (the end-to-end number for each rule)")
        diffs = np.array([np.mean(r["consistent"]) - np.mean(r["loud"]) for r in records
                          if r["consistent"] and r["loud"] and r["n_true"] < MAX_N_SRC])
        d_mean = float(diffs.mean()) if len(diffs) else float("nan")
        d_half = float(1.96 * diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 \
            else float("nan")
        print(f"  paired, rebuild - loudest ... {d_mean:+.2f} dB, 95 % CI "
              f"[{d_mean - d_half:+.2f}, {d_mean + d_half:+.2f}]   (per clip, N=1..4, "
              f"n={len(diffs)})")
        print("    Adoption rule, fixed in advance: rebuild is the default only because this")
        print("    CI was above zero on the DEV split (notebook 04's dev check), not just here.")
        psi_rb = _mean([r["psi_rebuild"] for r in records])
        psi_rb_n = {n: _mean([r["psi_rebuild"] for r in records if r["n_true"] == n])
                    for n in N_LIST}
        print(f"  P-SI-SNR, predicted count ... loudest {float(np.mean(psi)):+.2f}   "
              f"rebuild {psi_rb:+.2f} dB   (end to end: the counter's N and the rule's slots)")
        print("    per N, rebuild ........... " + "  ".join(
            f"N{n}:{_fmt(v)}" for n, v in psi_rb_n.items()))
        print(f"  noise slot used by rebuild: {'yes' if noise_est is not None else 'no'}")
        report["slot_selection_rules"] = {
            "proposed_rule": "rebuild", "per_n": rule_store,
            "pooled": {"oracle": _mean(pooled_o), "loudest": _mean(pooled_l),
                       "rebuild": pooled["consistent"], "projection": pooled["projection"]},
            "macro_n2to5": {"rebuild": macro_r["consistent"],
                            "projection": macro_r["projection"]},
            "count_correct_pooled": {"rebuild": pooled_cc["consistent"],
                                     "projection": pooled_cc["projection"]},
            "paired_rebuild_minus_loudest": {"mean": d_mean, "ci95": [d_mean - d_half,
                                                                       d_mean + d_half],
                                             "n_clips": int(len(diffs))},
            "p_si_snr": {"loudest": float(np.mean(psi)), "rebuild": psi_rb},
            "p_si_snr_rebuild_per_n": {str(n): v for n, v in psi_rb_n.items()},
            "noise_slot_used": noise_est is not None}

    if has_sep:
        print("\n  published Conv-TasNet, LibriMix 8k min, N=2: +14.76 dB "
              "(200 epochs, train-360)")

    records_path = os.path.join(out_dir, "eval_records.json")
    json_dump_atomic(records, records_path)
    path = os.path.join(out_dir, "eval_report.json")
    json_dump_atomic(report, path)
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        print(f"\nFAILED: report missing or truncated at {path}", file=sys.stderr)
        return 2
    print(f"\nreport  -> {path}\nrecords -> {records_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
