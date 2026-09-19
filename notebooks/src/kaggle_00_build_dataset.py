# %% [markdown]
# # 00 — Build the dataset
#
# **Accelerator: None (CPU).** This notebook spends **no GPU quota**. Do not turn a GPU on —
# you would burn your 30 hours a week on `soundfile`.
#
# **Runtime: about 10–15 minutes.** You run this notebook **once, ever.**
#
# ## What it does, in plain words
#
# Libri2Mix ships the *separate* voices (`s1/`, `s2/`), not only the mixed audio. This notebook
# copies every one of those voices into **one big flat file per split**, and then writes the
# dev and test sets as **recipe CSVs** — short text files that say "take utterance 412, crop it
# at sample 8000, add white noise at 12 dB" and can rebuild the exact audio later.
#
# | why | what you get |
# |---|---|
# | **Disk** | ~4 GB instead of the ~32 GB it would take to save every mixture as a WAV |
# | **Speed** | one long sequential read instead of 100,000 tiny file opens |
# | **Any N** | mixtures for 1–5 speakers are built on the fly, so nothing is baked in |
# | **Frozen tests** | the test set is a few hundred kB of text you can put in git |
#
# ## Before you press Run
#
# 1. **Settings → Accelerator → None**
# 2. **Settings → Internet → On** (only needed if the code comes from GitHub)
# 3. **+ Add Input → Datasets → search `libri2mix-8khz-min`** (by `unconscious`, 9.96 GB)
#
# ## After it finishes
#
# **Save Version → Save & Run All (Commit).** The output becomes a dataset you attach to every
# later notebook. See `docs/KAGGLE_RUNBOOK.md` step 3.

# %include _bootstrap.py

# %% [markdown]
# ## Check before you spend
#
# Thirty seconds, writes nothing. It also tells you if these notebook cells are older than the
# code they just downloaded — which is the most expensive kind of confusion, because the log
# looks completely normal.

# %%
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))

run(f"python scripts/preflight.py --for data"
    f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

# %% [markdown]
# ## Find the Libri2Mix input
#
# `autodetect_libri2mix` searches the usual Kaggle paths by looking at folder *contents*, not
# at a fixed path, because Kaggle mounts datasets differently depending on how you added them.
# Normally you change nothing here.

# %%
from _common import autodetect_libri2mix

LIBRI2MIX_DIR = autodetect_libri2mix()      # or paste the path yourself
STORE = "/kaggle/working/store"
DATA = "/kaggle/working/data"

print("Libri2Mix :", LIBRI2MIX_DIR)
if LIBRI2MIX_DIR is None:
    raise SystemExit(
        "Not found. In the right-hand panel: '+ Add Input' -> Datasets -> search for\n"
        "'libri2mix-8khz-min' -> Add. Then re-run this cell.")
for split in ("train-100", "dev", "test"):
    path = os.path.join(LIBRI2MIX_DIR, split)
    print(f"  {split:<10} {'ok' if os.path.isdir(path) else 'MISSING'}  {path}")

# %% [markdown]
# ## Step 1 — pack the voices  (~10 min)
#
# Each utterance is stored as its **loudest 8-second window**. A random 3-second crop of a
# LibriSpeech recording often lands in a pause, and a mixture built from three silent crops is
# not really a three-speaker mixture.
#
# 20 % of the speakers in each split are set aside with `role = babble` and are **never** used
# as counting targets. (We do not use babble noise at all in this project — see step 2 — but
# the split is kept so the store stays compatible with the sibling repo.)
#
# This step is restartable: a split that is already packed is skipped.

# %%
run(f"python scripts/00_pack_sources.py"
    f" --libri2mix_dir {LIBRI2MIX_DIR}"
    f" --out {STORE}"
    f" --splits train-100 dev test"
    f" --babble_frac 0.2"
    f" --cap_seconds 8.0"
    f" --seed 72")

# %% [markdown]
# ## Step 2 — freeze the dev and test sets  (~2 min)
#
# These two CSV files **are** the evaluation protocol. Generate them once, commit them, and
# never regenerate them — different recipes mean a different test set, and every number you
# have already written down becomes incomparable.
#
# ### The mixing settings, and why they are not the sibling repo's
#
# | setting | old project | here | why |
# |---|---|---|---|
# | gain jitter | ±5 dB | **±2.5 dB** | ±5 dB was measured to cost 9 accuracy points |
# | SNR | 0–20 dB | **5–20 dB** | below 5 dB the noise drowns the cue we are counting |
# | babble noise | 20 % of clips | **removed** | babble *is* 4–8 real talkers, so a "2 speaker" clip with babble has six voices in it and a label that says two |
#
# The script **refuses to run** if you ask for babble. That is deliberate: it is not a tuning
# knob, it is a wrong label.

# %%
run(f"python scripts/01_make_frozen_sets.py"
    f" --store {STORE}"
    f" --out {DATA}"
    f" --splits dev test"
    f" --n_list 1 2 3 4 5"
    f" --n_per_class 300"
    f" --noise_kinds white pink brown"
    f" --seed 72")

# %% [markdown]
# ## Step 3 — copy the recipes where you can find them
#
# Download these two files from the notebook output and commit them to the repo, so the
# evaluation protocol lives in git rather than only in a Kaggle output.

# %%
run(f"ls -la {DATA}")
print("\nDownload recipes_dev.csv and recipes_test.csv from the Output tab on the right,")
print("then put them in the repo's data/ folder and commit them.")

# %% [markdown]
# ## Step 4 — check what you built
#
# The last two lines are the ones that matter. If speaker disjointness is not 0 everywhere,
# stop: the same person is in both your training and your test data, and every number you
# measure afterwards is meaningless.

# %%
run(f"python scripts/preflight.py --for cpu --store {STORE}")

# %% [markdown]
# ## Now: Save Version → Save & Run All (Commit)
#
# When it finishes, the whole `/kaggle/working` folder becomes a dataset. In every later
# notebook you attach it with **+ Add Input → Notebook Output → this notebook**.
#
# You never run this notebook again.
