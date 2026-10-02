# %% [markdown]
# # 06 — The slide figures, rebuilt from the notebooks' own outputs
#
# **Accelerator: None (CPU).** No GPU quota. **Runtime: about 2 minutes.** Nothing is trained.
#
# The final presentation's figures were first drawn on a laptop, with the numbers typed in from
# the reports. This notebook redraws every one of them **straight from the outputs of notebooks
# 00–04**, and writes down which file each number came from. Use it to check the deck, to
# answer "where did this figure come from?", or to redraw a figure after a re-run.
#
# | slide | figure | read from |
# |---|---|---|
# | 3 | the problem: hidden voices → mixture → count + separate | a **real frozen test clip**, rendered from notebook 00's store |
# | 4 | test set per count and noise type; SNR histogram | notebook 00's `recipes_test.csv` |
# | 5 | before/after preprocessing, level vs N, the counter's spectrogram, the leakage probe | notebook 00's store + recipes, notebook 01's `audit_report.json` |
# | 6–8 | overview, counter and separator diagrams | the model code itself: one forward pass, every shape and parameter count measured |
# | 9 | confusion matrix; separation per N | notebook 04's `eval_report.json` |
# | 10 | shared vs two networks; fresh vs repeated training curves | notebook 04, notebook 02's `train_report.json`, notebook 03's `separator_report.json` |
# | 11 | grid search (**now with each setting's CV score**); pooling comparison | notebook 01's `tier_a_report.json`; notebook 02's `pool_*` runs if you ran them |
#
# **One change from the deck.** The deck's two pictures (slides 3 and 5) used Windows
# text-to-speech voices, because the laptop had no LibriSpeech audio. Here the store is attached,
# so they show a **real test clip** from the frozen test set — the same audio the evaluation
# scored.
#
# ## Before you press Run
#
# 1. **Settings → Accelerator → None**
# 2. **+ Add Input → Notebook Output →** notebooks **00, 01, 02, 03 and 04**. Attach whichever
#    you have: every input is optional. A figure whose input is missing is skipped, or drawn from
#    a recorded value that is labelled as one — never silently.
# 3. **Settings → Internet → On** (only if the code comes from GitHub)

# %include _bootstrap.py

# %%
sys.path.insert(0, os.path.join(REPO, "scripts"))
run(f"python scripts/preflight.py --for data"
    f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

# %% [markdown]
# ## Draw them
#
# Every input is found by itself, and the table at the top of the log says which file was used
# for each. To pin one instead, put its path in `PIN` — for example the `train_report.json` of
# the earlier repeated-mix counter run, if you still have that notebook version, as
# `counter_report_repeated`. `CLIP` picks the illustration clip by its `mix_id`; by default it is
# the three-talker noisy test clip closest to 12 dB SNR.

# %%
OUT = "/kaggle/working/slide_figures"
CLIP = None                 # e.g. "test_n3_00005"
PIN = {                     # e.g. {"eval": "/kaggle/input/.../eval/eval_report.json"}
}

cmd = f"python scripts/10_slide_figures.py --out {OUT}"
if CLIP:
    cmd += f" --clip {CLIP}"
for flag, path in PIN.items():
    cmd += f" --{flag} {path}"
run(cmd)

# %% [markdown]
# ## Look at them
#
# `numbers.md` lists every number on the slides beside the file it was read from. A row whose
# source starts with `recorded:` came from a document in the repo, not from an attached output:
# v0's numbers, the repeated-mix training runs and the CPU pooling proxy are measurements from
# runs that are no longer notebook outputs.

# %%
import glob

from IPython.display import Image, Markdown, display

for path in sorted(glob.glob(os.path.join(OUT, "*.png"))):
    print("\n" + os.path.basename(path))
    display(Image(filename=path, width=900))

with open(os.path.join(OUT, "numbers.md"), encoding="utf-8") as fh:
    display(Markdown(fh.read()))

shutil.make_archive("/kaggle/working/slide_figures", "zip", OUT)
print("\nall figures in one file: /kaggle/working/slide_figures.zip")

# %% [markdown]
# ## Putting them in the deck
#
# Download `slide_figures.zip` from the **Output** panel. The PNGs use the deck's colours and are
# saved at 200 dpi, so they can replace a slide's chart or picture directly. The deck's own
# charts are editable PowerPoint charts; if you keep those, compare their numbers with
# `numbers.md` instead.
#
# **Save Version → Save & Run All (Commit)** to keep the figures in this notebook's output.
