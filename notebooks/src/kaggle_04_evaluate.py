# %% [markdown]
# # 04 — The final numbers
#
# **Accelerator: GPU T4 ×2.** **Runtime: ~15 minutes.** Costs roughly **0.3 GPU-hours**.
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
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store, find_recipes

STORE = autodetect_store()
RECIPES_TEST = find_recipes("recipes_test.csv", STORE)

def _find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None

COUNTER   = _find("/kaggle/input/**/counter/ckpt/best.pt") or _find("/kaggle/input/**/ckpt/best.pt")
SEPARATOR = _find("/kaggle/input/**/sep/ckpt/best.pt")
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
# **The gap between 3 and 4 is exactly what miscounting costs you**, and it is the number that
# tells you which half to work on next. Being able to ask that question is the practical payoff
# of training the two models separately.
#
# ### And the confidence interval that is actually honest
#
# Two are printed. The **Wilson** interval assumes the test clips are independent trials — they
# are not, because the packer reserves 20 % of speakers for babble, so 1,500 clips come from
# about **32 people**. The **speaker-level bootstrap** resamples people instead of clips. It is
# wider, and it is the one to quote.

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
    lo, hi = c["speaker_bootstrap95"]
    print("COUNTING")
    print(f"  accuracy .............. {c['accuracy']:.1%}  [speaker 95 %: {lo:.1%}, {hi:.1%}]")
    print(f"  MAE ................... {c['mae']:.3f}")
    print(f"  majority-class naive .. {report['naive']['majority_class']['accuracy']:.1%}"
          f"   MAE {report['naive']['majority_class']['mae']:.3f}")

if "p_si_snr" in report:
    print("\nSEPARATION")
    print(f"  P-SI-SNR (all clips) .......... {report['p_si_snr']:+.2f} dB")
    for label, key in (("count-correct clips", "si_sdri_count_correct"),
                       ("true count forced  ", "si_sdri_oracle_count")):
        vals = report.get(key, {})
        if vals:
            row = "  ".join(f"N{n}:{v:+.1f}" for n, v in sorted(vals.items()))
            print(f"  SI-SDRi, {label} .. {row}")
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
# ## Writing it up honestly
#
# Four sentences worth putting in the report, because an examiner will ask and it is better to
# have answered first.
#
# 1. **Quote the speaker-level interval and say which one it is.** A Wilson interval over 1,500
#    clips drawn from 32 people understates the uncertainty.
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
