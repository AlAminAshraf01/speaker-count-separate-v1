# %% [markdown]
# # 04 — The final numbers
#
# **Accelerator: GPU T4 ×2.** **Runtime: ~11 minutes.** Costs roughly **0.18 GPU-hours**
# (11m 2s measured for evaluation, interpretability and the channel ablation on 99d75fe; the
# dev slot-selection check adds ~4 min if you switch it back on).
#
# ## Open the test set twice, not twenty times
#
# Every choice you have made so far — which pooling, which preset, how many epochs — was made
# on the **dev** set. This notebook is the only thing that touches the **test** set. Run it when
# you are finished, not while you are still deciding. A test set you tune against is just a
# second dev set with a misleading name.
#
# ## Before you press Run
#
# 1. **+ Add Input → Notebook Output →** notebook 00 (the store and the recipes)
# 2. **+ Add Input → Notebook Output →** notebook 01 (the Tier A counting model)
# 3. **+ Add Input → Notebook Output →** notebook 02 (the counter checkpoint)
# 4. **+ Add Input → Notebook Output →** notebook 03 (the separator checkpoint)
# 5. **Settings → Accelerator → GPU T4 ×2**
#
# Missing pieces are skipped rather than fatal — you can evaluate the counter alone, or the
# separator alone, and add the other later.

# %include _bootstrap.py

# %%
import glob
import sys
import time
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store, find_recipes

STORE = autodetect_store()
RECIPES_TEST = find_recipes("recipes_test.csv", STORE)

# Pin these to a specific path if you want to force a particular run; otherwise the
# newest attached checkpoint wins and every candidate is printed.
COUNTER_OVERRIDE = None
SEPARATOR_OVERRIDE = None


def _find(pattern, override=None):
    """The NEWEST matching checkpoint, with every candidate shown.

    This used to be `sorted(glob(...))[0]` -- alphabetical, silent. That is fine while one
    run is attached and quietly wrong the moment two are, which is exactly what happens
    when you re-run a training notebook and attach both versions to compare. Alphabetical
    order has nothing to do with which model you meant, and the final evaluation is the
    worst possible place to load the wrong weights without being told.
    """
    if override:
        print(f"    using the pinned path: {override}")
        return override
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    hits.sort(key=os.path.getmtime, reverse=True)
    if len(hits) > 1:
        print(f"    {len(hits)} candidates matched {pattern} -- taking the newest:")
        for i, h in enumerate(hits):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(h)))
            print(f"      {'->' if i == 0 else '  '} {when}  {h}")
        print("      (set COUNTER_OVERRIDE / SEPARATOR_OVERRIDE above to pin one instead)")
    return hits[0]

COUNTER   = (_find("/kaggle/input/**/counter/ckpt/best.pt", COUNTER_OVERRIDE)
             or _find("/kaggle/input/**/ckpt/best.pt"))
SEPARATOR = _find("/kaggle/input/**/sep/ckpt/best.pt", SEPARATOR_OVERRIDE)
TIER_A    = _find("/kaggle/input/**/tier_a_model.joblib")

print("store       :", STORE)
print("recipes test:", RECIPES_TEST)
print("counter     :", COUNTER or "not attached -- counting will be skipped")
print("separator   :", SEPARATOR or "not attached -- separation will be skipped")
print("Tier A      :", TIER_A or "not attached")
if STORE is None or RECIPES_TEST is None:
    raise SystemExit("Attach notebook 00's output: '+ Add Input' -> 'Notebook Output'.")
if COUNTER is None and SEPARATOR is None:
    raise SystemExit("Attach notebook 02's and/or notebook 03's output -- nothing to evaluate.")

run(f"python scripts/preflight.py --for eval --store {STORE}"
    f" --recipes_test {RECIPES_TEST}"
    + (f" --ckpt {COUNTER}" if COUNTER else "")
    + f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

# %% [markdown]
# ## Dev check: which slot-selection rule should the system use?  (~4 min)
#
# The first test run showed that "keep the N loudest slots" usually keeps a spare once two or
# more people talk, and that a level-blind rule — keep the N slots that best **rebuild** the
# mixture — scored higher on the test set. A rule chosen by looking at the test set has to be
# confirmed somewhere else before it becomes the system. This cell scores the same comparison
# on the **dev** split (different speakers, never used for any decision about slot selection).
#
# **The adoption rule, fixed before this cell was first run:** rebuild becomes the default only
# if the paired per-clip difference, rebuild − loudest, has a 95 % CI above zero here on dev.
#
# **Outcome (bc59305):** +0.61 dB, 95 % CI [+0.50, +0.72], n = 1,130 clips — **ADOPT**. Rebuild
# is now the default in `pipeline.py` and `08_infer.py`. The check is off by default now that it
# has been decided; set `RUN_DEV_SELECTION_CHECK = False` to repeat it (~4 min of GPU).
#
# (The dev split did pick which separator checkpoint was saved as `best.pt`, but that choice
# was about separation quality, not about how slots are chosen.)

# %%
RUN_DEV_SELECTION_CHECK = False
RECIPES_DEV = find_recipes("recipes_dev.csv", STORE)
if RUN_DEV_SELECTION_CHECK and SEPARATOR and RECIPES_DEV:
    cmd = (f"python scripts/06_evaluate.py"
           f" --store {STORE}"
           f" --test_split dev"
           f" --recipes_test {RECIPES_DEV}"
           f" --no_oracle_masks --n_boot 200 --batch_size 16"
           f" --out /kaggle/working/eval_dev"
           f" --separator {SEPARATOR}")
    if COUNTER:
        cmd += f" --counter {COUNTER}"
    run(cmd)
    import json
    with open("/kaggle/working/eval_dev/eval_report.json") as fh:
        dev_rules = json.load(fh)["slot_selection_rules"]
    paired = dev_rules["paired_rebuild_minus_loudest"]
    lo, hi = paired["ci95"]
    print(f"\nDEV: rebuild - loudest = {paired['mean']:+.2f} dB, 95 % CI [{lo:+.2f}, {hi:+.2f}]"
          f"   (n={paired['n_clips']} clips)")
    print("DEV verdict:", "ADOPT rebuild" if lo > 0 else "keep loudest -- rebuild not confirmed")
else:
    print("dev check skipped:",
          "disabled" if not RUN_DEV_SELECTION_CHECK else
          "no separator attached" if not SEPARATOR else "recipes_dev.csv not found")

# %% [markdown]
# ## Score the whole system  (~15 min)
#
# ### Four numbers, never one
#
# A single score cannot describe a system that does two jobs, and averaging them hides which
# half is broken. This prints all four:
#
# **1. Counting accuracy and the full confusion matrix.** The classes are ordered, so guessing
# 4 when the answer is 3 is a much smaller mistake than guessing 1 — and plain accuracy treats
# them identically. MAE and the matrix come with it.
#
# **2. P-SI-SNR over the whole test set.** Ordinary SI-SDR is undefined when the predicted
# count is wrong — there is no sensible pairing between 3 estimates and 4 true sources.
# P-SI-SNR pads the mismatch with a −30 dB floor so the number stays defined. That matters
# because it means a system cannot score well by quietly refusing to commit.
#
# **3. SI-SDRi per N, count-correct clips only.** The fixed-N separation literature is always
# *told* how many speakers there are, so this is the only row of yours that is comparable to it.
#
# **4. SI-SDRi per N with the true count forced.** Separation quality given a perfect counter.
# The gap between 3 and 4 is **not** what miscounting costs: both score against the true count
# and 3 is a subset of 4, so their difference is only which clips are included. P-SI-SNR (2) is
# the one number that reacts to the counter.
#
# ### And five more, because 3 and 4 let the answer key pick the slots
#
# Sections 3 and 4 match each true speaker to whichever of the 5 output slots fits it best.
# The real system cannot do that. It was designed to keep the **N loudest** slots, so the script
# also prints:
#
# **5.** SI-SDRi of the loudest slots, how often they are the right ones, and the loudness gap
# between real speakers and spare slots, in dB. **10.** The same for **rebuild** — the N slots
# that best rebuild the mixture — which is the rule the system uses since the dev check above. **6.** Averages where each N counts once (the
# pooled rows count each *speaker*, so N = 5 dominates). **7.** Everything by noise type and SNR.
# **8.** The Tier A tree scored on these **same** test clips, with a paired McNemar test against
# the counter — `tier_a_report.json`'s 57.8 % is cross-validation on training speakers, a
# different experiment. **9.** Oracle IRM/IBM masks: an upper bound on masking, not a rival.
#
# ### Which confidence interval to quote
#
# Two are printed. The **Wilson** interval assumes the test clips are independent trials — they
# are not quite, because 1,500 clips come from about **32 people**. The **speaker-level
# bootstrap** resamples people instead of clips. The script tells you which to quote, because it
# depends on how they come out: on the reported run the bootstrap came out *narrower*, so quote
# **Wilson** and say the bootstrap agreed.

# %%
cmd = (f"python scripts/06_evaluate.py"
       f" --store {STORE}"
       f" --recipes_test {RECIPES_TEST}"
       f" --n_boot 2000"
       f" --batch_size 16"
       f" --out /kaggle/working/eval")
if COUNTER:
    cmd += f" --counter {COUNTER}"
if SEPARATOR:
    cmd += f" --separator {SEPARATOR}"
if TIER_A:
    cmd += f" --tier_a {TIER_A}"
run(cmd)

# %% [markdown]
# ## The table for your report

# %%
import json

with open("/kaggle/working/eval/eval_report.json") as fh:
    report = json.load(fh)

print(f"frozen test set: {report['n']} mixtures from {report['n_speakers']} speakers\n")

if "counting" in report:
    c = report["counting"]
    lo, hi = c["wilson95"]
    blo, bhi = c["speaker_bootstrap95"]
    print("COUNTING")
    print(f"  accuracy .............. {c['accuracy']:.1%}  [Wilson 95 %: {lo:.1%}, {hi:.1%};"
          f" speaker bootstrap {blo:.1%}, {bhi:.1%}]")
    print(f"  MAE ................... {c['mae']:.3f}")
    print(f"  majority-class naive .. {report['naive']['majority_class']['accuracy']:.1%}"
          f"   MAE {report['naive']['majority_class']['mae']:.3f}")
    if "tier_a_test" in report:
        t = report["tier_a_test"]
        tlo, thi = t["wilson95"]
        print(f"  Tier A tree, SAME clips {t['accuracy']:.1%}  [Wilson 95 %: {tlo:.1%}, {thi:.1%}]"
              f"   MAE {t['mae']:.3f}")
        if "mcnemar_vs_counter" in t:
            m = t["mcnemar_vs_counter"]
            print(f"  McNemar, counter vs tree: {m['only_a_correct']} vs {m['only_b_correct']} "
                  f"discordant clips, p = {m['p_value']:.2g}")

if "p_si_snr" in report:
    print("\nSEPARATION")
    print(f"  P-SI-SNR (all clips) .......... {report['p_si_snr']:+.2f} dB")
    for label, key in (("count-correct clips", "si_sdri_count_correct"),
                       ("true count forced  ", "si_sdri_oracle_count")):
        vals = report.get(key, {})
        if vals:
            row = "  ".join(f"N{n}:{v:+.1f}" for n, v in sorted(vals.items()))
            print(f"  SI-SDRi, {label} .. {row}")
    sel = report.get("slot_selection", {})
    if sel:
        row = "  ".join(f"N{n}:{v['si_sdri_loudest_slots']:+.1f}"
                        for n, v in sorted(sel["per_n"].items()))
        print(f"  SI-SDRi, LOUDEST slots kept . {row}   <- the original rule")
        print(f"  pooled: oracle slots {sel['pooled_oracle_slots']:+.2f} dB, "
              f"loudest slots {sel['pooled_loudest_slots']:+.2f} dB")
        print(f"  speaker-vs-spare loudness gap, N=1..4: median "
              f"{sel['gap_db_median_n1to4']:+.2f} dB, "
              f"{sel['gap_negative_fraction_n1to4']:.1%} of clips negative")
    rules = report.get("slot_selection_rules", {})
    if rules:
        for rule in ("rebuild", "projection"):
            row = "  ".join(f"N{n}:{v[f'si_sdri_{rule}']:+.1f}"
                            for n, v in sorted(rules["per_n"].items()))
            tag = "   <- what the system outputs" if rule == rules["proposed_rule"] else ""
            print(f"  SI-SDRi, {rule.upper():<10} slots . {row}{tag}")
        p = rules["pooled"]
        print(f"  pooled: loudest {p['loudest']:+.2f}  rebuild {p['rebuild']:+.2f}  "
              f"projection {p['projection']:+.2f} dB   (count-correct rebuild "
              f"{rules['count_correct_pooled']['rebuild']:+.2f} dB)")
        right = "  ".join(f"N{n}: {v['right_slots_loudest']:.0%} -> {v['right_slots_rebuild']:.0%}"
                          for n, v in sorted(rules["per_n"].items())
                          if v["right_slots_rebuild"] is not None)
        print(f"  right slots, loudest -> rebuild: {right}")
        if "paired_rebuild_minus_loudest" in rules:
            pr = rules["paired_rebuild_minus_loudest"]
            print(f"  paired, rebuild - loudest: {pr['mean']:+.2f} dB, 95 % CI "
                  f"[{pr['ci95'][0]:+.2f}, {pr['ci95'][1]:+.2f}]")
            print(f"  P-SI-SNR end to end: loudest {rules['p_si_snr']['loudest']:+.2f}, "
                  f"rebuild {rules['p_si_snr']['rebuild']:+.2f} dB")
    for label, key in (("oracle slots ", "oracle_slots"), ("loudest slots", "loudest_slots")):
        m = report.get("macro", {}).get(key)
        if m:
            print(f"  macro over N, {label} .. N=1..5 {m['n1to5']:+.2f}  N=2..5 {m['n2to5']:+.2f}")
    masks = report.get("oracle_masks")
    if masks:
        for mode in ("irm", "ibm"):
            row = "  ".join(f"N{n}:{v:+.1f}" for n, v in sorted(masks[mode].items()))
            print(f"  oracle {mode.upper()} (upper bound) .... {row}")
    print(f"  published Conv-TasNet N=2 ..... +14.76 dB (200 epochs, train-360)")

# %% [markdown]
# ## Opening the box  (~3 min, no training)
#
# This is the course's interpretability deliverable, and it costs almost nothing: forward
# passes on the checkpoints you already have. Protect this part when the schedule slips.
#
# **The argument.** As N grows, the separator has to split the *same* learned filterbank among
# more voices. If mask overlap rises and sparsity falls as N goes up, then the fact that
# separation gets worse with more speakers has a *mechanical* explanation computed from the
# network's own internals — rather than being a curve you point at and assert.
#
# **And a question v0 could not ask.** In the old version the counting head sat on the
# separator's own trunk, so any correlation between counting confidence and mask geometry was
# partly just the two things being computed from the same tensor. Here the counter is a
# completely separate model. If its confidence still correlates with the separator's mask
# overlap, that says something about the **audio** — two independently trained models agreeing
# that a particular mixture is hard — and not about a shared representation.

# %%
if SEPARATOR:
    run(f"python scripts/07_interpret.py"
        f" --store {STORE}"
        f" --recipes_test {RECIPES_TEST}"
        f" --separator {SEPARATOR}"
        + (f" --counter {COUNTER}" if COUNTER else "")
        + f" --limit 300 --batch_size 8"
        f" --out /kaggle/working/interpret")
else:
    print("no separator attached -- skipping the mask-geometry analysis")

# %% [markdown]
# ## Which filters does it need, and for which job?  (~5–10 min, no training)
#
# The proposal asked which parts of the network **suppress noise** and which **separate
# speakers**. The test set already contains both jobs on their own: one talker plus noise is
# pure denoising, and two or more talkers with no noise is pure separation.
#
# The 512 learned encoder filters are sorted by the frequency they respond to and cut into 8
# bands. Each band is switched off in turn, and every clip is scored again against itself with
# nothing switched off. **Two versions:** *analysis* hides the band only from the part of the
# network that decides the masks (the finding); *full* also deletes it from the output, which
# mostly measures how much speech energy lives there (context). A random set of filters of the
# same size is the control.

# %%
if SEPARATOR:
    run(f"python scripts/09_filter_ablation.py"
        f" --store {STORE}"
        f" --recipes_test {RECIPES_TEST}"
        f" --separator {SEPARATOR}"
        f" --limit 600 --bands 8 --batch_size 16"
        f" --out /kaggle/working/ablation")
else:
    print("no separator attached -- skipping the channel ablation")

# %% [markdown]
# ## Writing it up honestly
#
# Four sentences worth putting in the report, because an examiner will ask and it is better to
# have answered first.
#
# 1. **Quote the interval the script tells you to, and say which one it is.** On the reported
#    run the speaker bootstrap came out narrower than Wilson, so Wilson is the conservative one.
#
# 2. **Say what "fully overlapped" means.** Every clip here has all N people talking at once for
#    the full 3 seconds. Real conversation is mostly people taking turns, which is a different
#    and harder problem (diarisation). A good number here does not mean speaker counting is
#    solved.
#
# 3. **Report the gap to published separation as budget.** 14.76 dB comes from 200 epochs on
#    train-clean-360. Give your epoch count and training set next to your number and the gap
#    explains itself.
#
# 4. **The interesting finding is the comparison, not the absolute.** The old project put both
#    jobs in one network; counting received 0.53 % of the gradient and scored at chance, and
#    separation lost about 8.7 dB to the auxiliary objectives. Splitting them into two
#    specialists and joining them at inference is the result — say what it cost (about 5 % more
#    compute) and what it bought.
#
# **Save Version → Save & Run All (Commit)** to keep the report.
