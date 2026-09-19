# %% [markdown]
# # 05 — The demo: audio in, one track per person out
#
# **Accelerator: None (CPU) is enough.** **Runtime: ~2 minutes.** Costs **no GPU quota**.
#
# Every other notebook reports an average over 1,500 mixtures. This one runs a single file and
# lets you *listen* to it, because an average cannot be played out loud and a viva can.
#
# It is also the only notebook that uses the system as a system. Notebook 02 trains the counter,
# notebook 03 trains the separator, notebook 04 scores them — this is the one that joins them:
#
# ```
#   3 s of audio ──┬──►  counter    ──►  N̂ = "3 people"
#                  └──►  separator  ──►  5 slots + noise
#                                              │
#                            keep the N̂ loudest┘ ──►  speaker_01.wav, speaker_02.wav, ...
# ```
#
# ## Why the predicted count is printed here, and was not in the old version
#
# The old project's demo **deliberately deleted** its "N speakers detected" line. Its counting
# head answered **1** for 1445 of 1500 test mixtures, so that line printed the same number
# whoever was talking — a headline that looks like a result and carries no information.
#
# In this version the counter is a separate model with its own gradient, so the number means
# something and is printed. It comes with the full probability distribution, so a clip the model
# is unsure about *looks* unsure instead of confidently wrong.
#
# ## Before you press Run
#
# 1. **+ Add Input → Notebook Output →** notebook 00 (the store and the recipes)
# 2. **+ Add Input → Notebook Output →** notebook 02 (the counter checkpoint)
# 3. **+ Add Input → Notebook Output →** notebook 03 (the separator checkpoint)
# 4. **Settings → Accelerator → None** — this is forward passes on one file; a GPU here is
#    30 hours a week spent on nothing.

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

print("store     :", STORE)
print("recipes   :", RECIPES_TEST)
print("counter   :", COUNTER or "MISSING")
print("separator :", SEPARATOR or "MISSING")
if COUNTER is None or SEPARATOR is None:
    raise SystemExit(
        "This notebook needs BOTH models -- it is the one that joins them.\n"
        "'+ Add Input' -> 'Notebook Output' -> notebook 02, then again for notebook 03.")
if STORE is None or RECIPES_TEST is None:
    raise SystemExit("Attach notebook 00's output: '+ Add Input' -> 'Notebook Output'.")

run(f"python scripts/preflight.py --for eval --store {STORE}"
    f" --recipes_test {RECIPES_TEST} --ckpt {COUNTER}"
    f" --cells_src {CELLS_SRC} --cells_sha {CELLS_SHA}")

# %% [markdown]
# ## Run it on mixtures whose true count you already know
#
# These are drawn from the **frozen test set**, so the notebook can print *predicted vs true*
# rather than asking you to take the number on faith. Two clips: an easy one and a hard one.
#
# **What to look at in the output, in order:**
#
# 1. **The probability bars.** A model that is right for the right reason puts most of its mass
#    on one class. Mass spread over 4 and 5 on a 5-speaker clip is a near-miss; mass parked on
#    1 regardless of the input is the old project's failure, and you would see it immediately.
#
# 2. **The slot power table.** The separator always emits 5 slots. Training pushes the unused
#    ones toward −30 dB, so on a 3-speaker clip you want a clear cliff after slot 3. If the
#    powers slope gently instead, the two models disagree about the count — and you can hear
#    which one is right.

# %%
DEMOS = [(2, "two talkers -- the easy case"), (4, "four talkers -- the hard case")]

for n, label in DEMOS:
    print("\n" + "=" * 74 + f"\n{label}\n" + "=" * 74)
    run(f"python scripts/08_infer.py"
        f" --counter {COUNTER} --separator {SEPARATOR}"
        f" --store {STORE} --from_recipes {RECIPES_TEST}"
        f" --n {n} --index 0 --all_slots"
        f" --out /kaggle/working/demo_n{n}")

# %% [markdown]
# ## Listen
#
# The mixture first, then one track per person the system decided was there. The surplus slots
# are included so you can hear what "rejected" sounds like — they should be near-silence or a
# faint smear, not a fourth voice.
#
# All files share **one** gain, so the tracks keep their relative loudness. Normalising each
# track separately would make a whisper and a shout come out the same, which would hide exactly
# the thing the slot-power table is showing you.

# %%
import json
from IPython.display import Audio, display

for n, label in DEMOS:
    out_dir = f"/kaggle/working/demo_n{n}"
    with open(os.path.join(out_dir, "infer_report.json")) as fh:
        rep = json.load(fh)
    print("\n" + "=" * 74)
    print(f"{label}   true {rep['true_n']}  ->  predicted {rep['n_hat']}"
          f"   ({'correct' if rep['true_n'] == rep['n_hat'] else 'WRONG'})")
    print("=" * 74)
    for name in rep["files"]:
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            continue
        print(f"\n{name}")
        display(Audio(filename=path))

# %% [markdown]
# ## Your own recording
#
# **+ Add Input → Upload → New Dataset**, upload any audio file, then put its path below and
# run the cell. Any format and any sample rate — it is converted to 8 kHz mono internally.
#
# Two honest warnings before you read too much into the result:
#
# **The models were trained on fully-overlapped speech.** Every training clip has all N people
# talking at once for the full three seconds. A normal conversation is mostly people *taking
# turns*, and during a single-speaker stretch the honest answer is "1" even though three people
# are in the room. That is not a bug, it is a different problem — diarisation — and it is the
# first thing an examiner will poke at.
#
# **Long files are processed in 3-second windows**, because both models were trained at that
# length; feeding one long block instead took the old counter from a working score to **0
# correct out of 300**. Windows are permutation-aligned to their neighbour before being
# cross-faded together — without that step a speaker changes track partway through the file,
# which was measured here at 0.908 correlation with speaker A over the first half of a clip and
# 1.000 with speaker B over the second.

# %%
MY_FILE = None      # e.g. "/kaggle/input/my-audio/meeting.m4a"

if MY_FILE:
    run(f"python scripts/08_infer.py --counter {COUNTER} --separator {SEPARATOR}"
        f" --input {MY_FILE} --all_slots --max_seconds 60"
        f" --out /kaggle/working/demo_mine")
    for path in sorted(glob.glob("/kaggle/working/demo_mine/*.wav")):
        print("\n" + os.path.basename(path))
        display(Audio(filename=path))
else:
    print("set MY_FILE above to run this on your own recording")

# %% [markdown]
# ## What this notebook is worth in the report
#
# One figure and one sentence, not a page.
#
# **The figure:** the slot-power table for a clip the system got right. It shows the cliff after
# slot N̂ — the separator's own output agreeing with the counter's answer, which is two
# independently trained models arriving at the same number. That is a stronger statement than
# either accuracy figure alone, and the old version could not make it at all: its count head
# read the separator's own features, so agreement between them was guaranteed by construction
# rather than earned.
#
# **The sentence:** quote a clip the system got *wrong*, and say what the probability bars
# looked like. A project that shows a failure case and explains it reads as one where somebody
# checked; a project with only successes reads as one where somebody stopped looking.
#
# **Save Version → Save & Run All (Commit)** to keep the audio in the output.
