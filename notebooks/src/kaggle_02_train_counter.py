# %% [markdown]
# # 02 — Train the counter
#
# **Accelerator: GPU T4 ×2.** **Runtime: 2–4 hours.** Costs roughly **3 GPU-hours** of your
# 30 per week.
#
# ## This notebook has to earn its place
#
# Notebook 01 already counted speakers on CPU. This one only matters if it **beats** that
# number. If it does not, that is a result worth reporting, not a failure to hide — and the
# training log says so on every line.
#
# It trains the counter **alone**, which is the central change in v1. In the old project a
# counting head hung off the separator's trunk and received **0.53 %** of the gradient while
# separation took 86.14 % — 633:1 at the encoder. It never learned, and settled on answering
# "1 speaker" for 1445 of 1500 test mixtures. Here counting gets 100 % of its own gradient.
#
# ## The one thing that actually matters here
#
# It is not the architecture, the learning rate, or the amount of data. It is **how the model
# summarises time**. Measured on identical data, changing *only* that one layer:
#
# | how it summarises | accuracy |
# |---|---|
# | mean + standard deviation (what the old project used) | 66.5 % |
# | attention-weighted mean + std | 67.5 % |
# | the full covariance matrix | **20.0 % — collapsed to "1 speaker"** |
# | **the covariance's eigenvalue spectrum** (the default here) | **74.4 %** |
#
# An 8-point swing from one layer. The winner is also the *smallest* and *fastest* of the four.
#
# The intuition: if N people are talking, the sound occupies roughly **N independent
# directions**. Eigen*values* tell you **how many** directions carry energy — that is the count.
# Eigen*vectors* tell you **which** directions — and a counter should not care which. Throwing
# the eigenvectors away is not a shortcut; it is the whole idea.
#
# ## Before you press Run
#
# 1. **Settings → Accelerator → GPU T4 ×2**
# 2. **+ Add Input → Notebook Output →** notebook 00's output (the store)
# 3. **+ Add Input → Notebook Output →** notebook 01's output (for the bar)
# 4. **Settings → Internet → On** (only if the code comes from GitHub)
# 5. **Settings → Persistence → Variables and Files** — so a 12-hour timeout does not lose
#    your checkpoints

# %include _bootstrap.py

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
from _common import autodetect_ckpt, autodetect_store, find_recipes

STORE = autodetect_store()
RECIPES_DEV = find_recipes("recipes_dev.csv", STORE)
OUT = "/kaggle/working/counter"
POOLING = "eigen"        # meanstd | attentive | covariance | eigen

print("store      :", STORE)
print("recipes dev:", RECIPES_DEV)
if STORE is None:
    raise SystemExit("No packed store. '+ Add Input' -> 'Notebook Output' -> notebook 00.")

# %% [markdown]
# ## Find the bar from notebook 01
#
# If notebook 01's output is attached, its number is used automatically. Otherwise the default
# is the measured 69.3 %, which is close enough to be a fair target.

# %%
import glob
import json

BAR = 0.693
for path in glob.glob("/kaggle/input/**/tier_a_report.json", recursive=True):
    with open(path) as fh:
        BAR = float(json.load(fh)["bar"])
    print(f"found Tier A report: {path}")
    break
else:
    print("no Tier A report attached; using the measured default")
print(f"the bar to beat: {BAR:.1%}")

# %% [markdown]
# ## Check before you spend  (~30 s)
#
# The row to watch is **precision**.
#
# The old counter scored **44.9 %** under fp16 and **20.00 %** in fp32 *from the same saved
# model* — they agreed on only 16 % of their answers. It had been trained in one kind of
# arithmetic and tested in the other, so the thing that was measured was never the thing that
# was trained. Nobody noticed for weeks, because both numbers look like plausible accuracies.
#
# Here that is a thirty-second check that stops the notebook, instead of a discovery you make
# after eleven hours of GPU time.

# %%
run(f"python scripts/preflight.py --for train --store {STORE}"
    + (f" --recipes_dev {RECIPES_DEV}" if RECIPES_DEV else "")
    + f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

# %% [markdown]
# ## Train  (~2–3 h)
#
# Notes on the settings, since you may want to change them:
#
# * **`--amp` is off.** Mixed precision would make this maybe 30 % faster and reopens exactly
#   the failure described above. The model is small; the trade is not worth it. If you do turn
#   it on, the preflight check and the end-of-training check both still run.
# * **It resumes.** If the 12-hour limit kills the session, re-run this same cell — it picks up
#   from the last checkpoint rather than starting over.
# * **`--time_budget_h 10.5`** stops it cleanly before Kaggle's hard cut-off.
# * Every epoch prints `beats the bar` or `BELOW THE BAR`. Watch that column, not the loss.

# %%
run(f"python scripts/04_train_counter.py"
    f" --store {STORE}"
    + (f" --recipes_dev {RECIPES_DEV}" if RECIPES_DEV else "")
    + f" --pooling {POOLING}"
    f" --epochs 20"
    f" --steps_per_epoch 2000"
    f" --batch_size 64"
    f" --lr 2e-3"
    f" --num_workers 3"
    f" --bar {BAR:.4f}"
    f" --time_budget_h 10.5"
    f" --out {OUT}")

# %% [markdown]
# ## Optional — compare the four pooling layers yourself
#
# This is the measurement that produced the table at the top. Four short runs, about
# **1.5 GPU-hours** in total. Worth doing if you want the comparison in your own report with
# your own data; skip it if quota is tight.
#
# Expect `covariance` to collapse to 20 %. That is the expected result, not a bug — and it is
# the most interesting row in the table, because it is the same idea as `eigen` implemented
# the obvious way.

# %%
COMPARE_POOLINGS = False        # set True to run it

if COMPARE_POOLINGS:
    for pooling in ("meanstd", "attentive", "covariance", "eigen"):
        run(f"python scripts/04_train_counter.py"
            f" --store {STORE}"
            + (f" --recipes_dev {RECIPES_DEV}" if RECIPES_DEV else "")
            + f" --pooling {pooling}"
            f" --epochs 6 --steps_per_epoch 1000 --batch_size 64"
            f" --bar {BAR:.4f} --time_budget_h 2.0"
            f" --out /kaggle/working/pool_{pooling}")

# %% [markdown]
# ## Read the verdict
#
# The final block of the training log says either `worth keeping` or `BOUGHT NOTHING`. Both are
# legitimate outcomes and both belong in your report.
#
# **Save Version → Save & Run All (Commit)** so the checkpoint becomes a dataset for notebook 03.

# %%
with open(os.path.join(OUT, "train_report.json")) as fh:
    report = json.load(fh)

print(f"pooling ................. {report['pooling']}")
print(f"parameters .............. {report['params'] / 1e6:.3f} M")
print(f"best dev accuracy ....... {report['best_val_accuracy']:.1%}")
print(f"the bar (Tier A) ........ {report['bar']:.1%}")
print(f"beats the bar ........... {report['beats_bar']}")
print(f"fp32/fp16 agreement ..... {report['precision_agreement']:.1%}")

# The per-N breakdown, not just the average. 85 % overall could be 95 % at N=1 and
# 55 % at N=5, and those are different projects -- the shape is what tells you which
# half to work on next, and it is 9,000 log lines up otherwise.
best = report.get("best_checkpoint", {})
recall = best.get("per_class_recall")
if recall:
    print(f"MAE ..................... {best['mae']:.3f}")
    print(f"off-by-one .............. {best['off_by_one']:.1%}   "
          f"(100 % means every error is a neighbour)")
    print("per-N recall ............ "
          + "  ".join(f"N={n}:{r:.0%}" for n, r in zip(range(1, 6), recall)))
    worst = min(range(len(recall)), key=lambda i: recall[i]) + 1
    print(f"weakest class ........... N={worst} at {min(recall):.0%}")
print()
print("Nothing here is a final number. These are DEV numbers, used to choose things.")
print("The test set is opened once, in notebook 04, and not before.")
