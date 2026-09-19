# %% [markdown]
# # 03 — Train the separator
#
# **Accelerator: GPU T4 ×2.** **Runtime: up to 11 hours.** Costs roughly **10 GPU-hours** of
# your 30 per week. This is the expensive notebook; everything else together is under 4.
#
# ## What this trains, and what it deliberately does not
#
# A Conv-TasNet separator with 5 speaker slots plus 1 noise slot. **It has no counting head.**
# Counting is notebook 02's job.
#
# That split is the main change in v1, and it comes from measurements on the old project:
#
# | measured in v0 | number |
# |---|---|
# | share of the shared trunk's gradient going to counting | **0.53 %** |
# | share going to separation | 86.14 % |
# | ratio at the encoder | **633 : 1** |
# | cost of the auxiliary objectives to separation | **−8.7 dB** |
# | pooled model, official Libri2Mix N=2 | 0.08 dB |
# | separation-only control, 8 epochs instead of 38 | **5.49 dB** |
#
# The two jobs were taking gradient from each other, and the sharing was buying almost
# nothing: removing the separator from that model saved only **8.4 %** of the forward FLOPs,
# because the TCN is the expense. So v1 trains them apart and joins them at inference.
#
# ## Two loss bugs that are fixed here
#
# **1. The clamp was backwards.** It was supposed to stop one easy example dominating a batch.
# Measured with autograd, it did the opposite — the gradient *peaked* exactly at the clamp
# point, so an already-solved 1-speaker mixture carried about **16×** the gradient of a hard
# 5-speaker one. v1 uses a hard clamp, which is flat above the threshold: once an example is
# separated well enough it stops competing for the optimiser's attention.
#
# **2. The silence term was 1-speaker-weighted.** It divided by the number of *spare output
# slots*, and spare slots run 4, 3, 2, 1, 0 as N goes 1 to 5. So in a balanced batch the single
# 1-speaker clip supplied **40 %** of that term and the 5-speaker clip supplied none. It is now
# a plain per-clip average.
#
# Both bugs pushed the model toward "everything is one speaker". That is exactly what the old
# model ended up doing.
#
# ## Before you press Run
#
# 1. **Settings → Accelerator → GPU T4 ×2**
# 2. **Settings → Persistence → Variables and Files** ← important, see below
# 3. **+ Add Input → Notebook Output →** notebook 00's output
# 4. **Settings → Internet → On** (only if the code comes from GitHub)
#
# **About the 12-hour limit.** This notebook is budgeted to stop itself at 10.5 hours so Kaggle
# does not kill it mid-write. If it does get killed, just **Run All** again — it finds the last
# checkpoint and carries on. Nothing is lost, but only if Persistence is on.

# %include _bootstrap.py

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_store, find_recipes

STORE = autodetect_store()
RECIPES_DEV = find_recipes("recipes_dev.csv", STORE)
OUT = "/kaggle/working/sep"
PRESET = "paper"        # paper (5.25 M) | small | tiny

print("store      :", STORE)
print("recipes dev:", RECIPES_DEV)
if STORE is None:
    raise SystemExit("No packed store. '+ Add Input' -> 'Notebook Output' -> notebook 00.")

run(f"python scripts/preflight.py --for train --store {STORE}"
    + (f" --recipes_dev {RECIPES_DEV}" if RECIPES_DEV else "")
    + f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

# %% [markdown]
# ## Decide the budget before you spend it
#
# The single most useful thing you can do before a long training run is work out how many
# epochs actually fit. Do the arithmetic now, not at hour eleven.

# ### Why batch 12 fits, and what to do if it does not
#
# This notebook OOMed on a T4 the first time it ran. Not marginally: one forward+backward
# retained **18.05 GiB** of activations against a **14.56 GiB** card, so batch 12 was never
# going to fit. 55 % of that was `GlobalLayerNorm`, which saved three full-size tensors per
# call across 49 instances. Folding its affine into `(B, C, 1)` coefficients cut the model to
# **11.33 GiB** — batch 12 now fits with about **1.77 GiB** to spare, and it is slightly
# faster besides.
#
# If you still hit `torch.OutOfMemoryError`, your clone is older than that fix — re-run so the
# bootstrap pulls it. Only if it persists, drop to `BATCH_SIZE = 8` **and** raise
# `STEPS_PER_EPOCH` to `1500`. Take both: this script sizes an epoch as
# `steps_per_epoch * batch_size` mixtures, so changing the batch alone quietly cuts your
# training data by a third. Together they hold the epoch, the data and the GPU budget fixed.
#
# Do **not** reach for `--amp`. It looks like a free 2x and measures 1.39x, because under
# autocast `x.float()` inside gLN becomes a real widening copy — so 50 % of the peak stays
# fp32 no matter what. It would not reach batch 12 anyway, and it reopens the precision
# failure this version was rebuilt to close.

# %%
EPOCHS = 30
STEPS_PER_EPOCH = 1000
BATCH_SIZE = 12

print(f"mixtures seen per epoch : {STEPS_PER_EPOCH * BATCH_SIZE:,}")
print(f"target epochs           : {EPOCHS}")
print(f"total mixtures          : {EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE:,}")
print()
print("For scale: published Conv-TasNet uses 200 epochs on train-clean-360 to reach 14.76 dB")
print("SI-SDRi at 2 speakers. You have 30 GPU-hours a week. You will not reach 14.76 dB, and")
print("that is fine -- report your epoch count and training set beside your number, and the")
print("gap reads as budget rather than as a mystery.")

# %% [markdown]
# ## Train  (up to ~11 h)
#
# Watch the `SI-SDRi` column and the per-N breakdown beside it. Two things to look for:
#
# * **N=1 should be high and boring.** A one-speaker mixture is solved by copying the input.
#   If N=1 is bad, something is wrong with the plumbing, not the model.
# * **N=5 is the hard one.** If it stays near 0 dB while N=2 climbs, the model is learning to
#   separate the easy cases and giving up on the crowded ones — which is what the two loss bugs
#   above used to cause.
#
# `(k degenerate refs dropped)` is normal and honest. SI-SDRi is undefined for a clean
# one-speaker mixture, because the mixture already *is* the target and the "improvement" being
# measured is floating-point noise. Those references are left out and counted. In v0 that one
# bug turned a +1.2 dB result into a reported −8.68 dB.

# %%
run(f"python scripts/05_train_separator.py"
    f" --store {STORE}"
    + (f" --recipes_dev {RECIPES_DEV}" if RECIPES_DEV else "")
    + f" --preset {PRESET}"
    f" --n_list 1 2 3 4 5"
    f" --epochs {EPOCHS}"
    f" --steps_per_epoch {STEPS_PER_EPOCH}"
    f" --batch_size {BATCH_SIZE}"
    f" --lr 1e-3"
    f" --num_workers 2"
    f" --clamp_si_sdr 30.0"
    f" --time_budget_h 10.5"
    f" --out {OUT}")

# %% [markdown]
# ## Optional — the fixed-N=2 control
#
# About **2.5 GPU-hours**, and it is worth every one of them if you can spare them.
#
# Train the same model on 2-speaker mixtures only. That is the configuration the published
# 14.76 dB refers to, so it is the only number you have that is directly comparable to the
# literature — and it tells you whether a weak pooled result is a *separation* problem or a
# *variable-N* problem. In v0 this control scored 5.49 dB while the pooled model scored 0.08 dB,
# which is how it became clear the pipeline could separate and the pooled task was the cost.

# %%
RUN_FIXED_N2 = False        # set True if you have the quota

if RUN_FIXED_N2:
    run(f"python scripts/05_train_separator.py"
        f" --store {STORE}"
        + (f" --recipes_dev {RECIPES_DEV}" if RECIPES_DEV else "")
        + f" --preset {PRESET}"
        f" --n_list 2"
        f" --epochs 10 --steps_per_epoch 1000 --batch_size 12"
        f" --time_budget_h 3.0"
        f" --out /kaggle/working/sep_n2")

# %% [markdown]
# ## Read the result
#
# **Save Version → Save & Run All (Commit)** so the checkpoint becomes a dataset for notebook 04.

# %%
import json

with open(os.path.join(OUT, "separator_report.json")) as fh:
    report = json.load(fh)

print(f"preset .................. {report['preset']}")
print(f"parameters .............. {report['params'] / 1e6:.2f} M")
print(f"best dev SI-SDRi ........ {report['best_val_si_sdri']:+.2f} dB")
print(f"epochs completed ........ {len(report['history'])}")
print(f"loss ..................... {report['loss']}")
print()
print("published (N=2, 200 epochs, train-360): +14.76 dB")
print()
print("These are DEV numbers. The test set is opened once, in notebook 04.")
