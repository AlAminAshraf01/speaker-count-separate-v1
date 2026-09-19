# %% [markdown]
# # 01 — The audit, and the counting half on CPU
#
# **Accelerator: None (CPU).** No GPU quota. **Runtime: about 25–40 minutes.**
#
# ## Read this bit
#
# **This notebook is the deliverable.** Not a warm-up, not a baseline to beat later — if you
# run only one notebook in this project, run this one. When it finishes you have every number
# the course brief asks for, and you have spent zero of your 30 GPU-hours.
#
# The reason is a measurement. In the sibling project, a 5.3-million-parameter neural network
# scored **20.00 %** at counting speakers — exactly chance on a 5-class problem. Sixteen
# hand-crafted numbers fed to a gradient-boosted tree scored **69.3 %** on the same data. A
# decision tree beat the network by 49 points. `docs/DIAGNOSIS.md` explains why.
#
# So we do the cheap, certain thing first and properly, and treat the neural network
# (notebook 02) as optional upside.
#
# ## What runs here
#
# | step | what it answers |
# |---|---|
# | **the audit** | Is the evaluation honest? Can a model cheat by reading loudness instead of listening? |
# | **naive baselines** | What do you get for free, with no learning at all? |
# | **Tier A** | Feature extraction, a speaker-disjoint K-fold search, the final model |
# | **feature importance** | *Why* does it know? Every answer is a named acoustic quantity |
#
# ## Before you press Run
#
# 1. **Settings → Accelerator → None**
# 2. **+ Add Input → Notebook Output →** the output of notebook 00
# 3. **Settings → Internet → On** (only if the code comes from GitHub)

# %include _bootstrap.py

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store, find_recipes

STORE = autodetect_store()
RECIPES_DEV = find_recipes("recipes_dev.csv", STORE)
RECIPES_TEST = find_recipes("recipes_test.csv", STORE)
OUT = "/kaggle/working/tier_a"

print("store       :", STORE)
print("recipes dev :", RECIPES_DEV)
print("recipes test:", RECIPES_TEST)
if STORE is None:
    raise SystemExit(
        "No packed store found. In the right-hand panel: '+ Add Input' -> 'Notebook Output'\n"
        "-> pick your run of notebook 00. Then re-run this cell.")

run(f"python scripts/preflight.py --for cpu --store {STORE}"
    f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

# %% [markdown]
# ## Step 1 — the leakage audit  (~5 min)
#
# ### What "leakage" means here, in plain words
#
# When you build a mixture by adding up N voices, the result gets **louder** as N grows. If you
# leave it that way, a model can score well by measuring loudness and never listening to
# speech at all. That is not a speaker counter; it is a volume meter, and it will collapse the
# moment it meets real audio.
#
# The fix is structural: every clip is cut to exactly 3 seconds and divided by its own loudness.
# After that, loudness and length are **identical for every clip at every N**, so there is
# nothing left to cheat on.
#
# ### How this script proves it rather than claiming it
#
# It runs the same probe twice:
#
# * on audio **before** the fix — this *must* score well above chance, or the probe is broken
#   and its clean verdict would be worthless;
# * on audio **after** the fix — this must sit at chance.
#
# > ### Why this script is built to fail loudly
# >
# > The equivalent script in the sibling project was **dead code for months**. Somebody pasted
# > a function into the middle of `main()`, which quietly cut the rest of it off; the script
# > printed its heading, wrote nothing, and exited with code 0 — *success*. Two headline numbers
# > were quoted from it in project documents that it had never actually produced.
# >
# > So this one checks that its own report file exists before it exits, and the test suite
# > fails the build on that exact bug shape.

# %%
run(f"python scripts/02_audit_and_baselines.py"
    f" --store {STORE}"
    f" --splits train-100 dev test"
    f" --probe_split train-100"
    f" --per_class 300"
    f" --out /kaggle/working/audit")

# %% [markdown]
# ## Step 2 — Tier A: the model  (~20–30 min)
#
# Sixteen acoustic numbers per clip, then a gradient-boosted tree.
#
# ### The idea in one sentence
#
# When you add up more and more voices, the mixture **fills in its own silences** and starts to
# look like plain noise — so statistics that measure "peakiness" fall in a predictable way as
# the number of talkers goes up. Measured here: kurtosis spreads **5×** across 1 to 5 speakers.
#
# ### The K-fold is grouped by speaker, and that matters
#
# An ordinary K-fold would put the same person in both the training and the validation half.
# LibriSpeech speakers can be identified from their vocabulary alone, so the tree would be
# partly recognising *voices* rather than counting them, and the score would be flattering and
# wrong. Here the folds are cut so **no speaker crosses the line**. The number this prints is
# usually a few points lower than the naive version. It is the honest one.

# %%
run(f"python scripts/03_tier_a.py"
    f" --store {STORE}"
    f" --train_split train-100"
    f" --per_class 1500"
    f" --folds 5"
    f" --out {OUT}")

# %% [markdown]
# ## Step 3 — read the result
#
# The last table of the previous cell is your **interpretability deliverable**: it ranks the
# acoustic quantities by how much the model actually relies on them, and every row has a name
# you can say out loud — "envelope sparsity", "kurtosis" — instead of "filter 287".
#
# One honest caveat to write in your report: permutation importance **splits the credit between
# correlated features**. Kurtosis of the signal and kurtosis of the envelope measure nearly the
# same thing, so one of them can show near-zero importance while the pair is jointly essential.
# Do not read a zero as "this feature is useless".

# %%
import json

with open(os.path.join(OUT, "tier_a_report.json")) as fh:
    report = json.load(fh)

bar = report["bar"]
print(f"chance                                 20.0%")
print(f"best naive predictor                   "
      f"{max(r['accuracy'] for r in report['naive'].values()):.1%}")
print(f"Tier A, speaker-disjoint K-fold        {bar:.1%}   <-- YOUR RESULT")
print()
print(f"For comparison, the 5.30 M joint model in the sibling repo scored 20.00 %.")
print()
print(f"Pass this to notebook 02 as --bar {bar:.3f}")

# %% [markdown]
# ## You are done — this is a complete project
#
# **Save Version → Save & Run All (Commit)** so the model and the reports become a dataset.
#
# What you have now, mapped to the five things the course asks for:
#
# | phase | where it is |
# |---|---|
# | 1 · EDA, correlations, outliers | feature distributions per N + the importance table |
# | 2 · custom feature extraction | the 16 acoustic features — a hand-designed transform |
# | 3 · hyperparameter search + K-fold | the grouped 5-fold search in step 2 |
# | 4 · data-leakage audit | step 1, which actually runs and proves its own probe works |
# | 5 · baselines + interpretability | three naive predictors + permutation importance |
#
# notebooks 02 and 03 (the neural models) is **optional**. If your GPU quota is gone, or the week is
# gone, stop here and write it up. That is not settling — that is the plan working.
