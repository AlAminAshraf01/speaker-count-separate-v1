# How to run this on Kaggle — step by step

Written for someone who has never run this project before. Follow it in order. Nothing here
assumes you remember anything from the other documents.

**Total time: about 15 hours, most of it waiting.** You can stop after Step 5 — about an hour —
and already have a complete, defensible project that cost **zero GPU-hours**.

---

## What you are about to build

A system that listens to 3 seconds of audio, says **how many people are talking** (1 to 5), and
then **separates their voices** into individual tracks.

It is two models, trained separately and joined at the end:

- **the counter** — a small network (0.49 M parameters) that only counts;
- **the separator** — a Conv-TasNet (5.25 M parameters) that only separates.

The old version put both jobs in one network and did neither well. The counting half received
**0.53 %** of the training signal while separation took 86 %, so it never learned and ended up
answering "1 speaker" to almost everything. Splitting them costs about 5 % more compute and
gives each job its own full training signal. `docs/DIAGNOSIS.md` has the measurements.

There is also a **Tier A** counter that uses no GPU at all: sixteen measurements of the sound
(how "spiky" it is, how much silence it contains) fed to a decision tree. It scores 57.8 %,
which is 38 points better than the old 5.3-million-parameter network managed. It is the bar
everything else has to clear, and on its own it is already a complete project.

---

## What "Kaggle free tier" gives you

| | |
|---|---|
| GPU time | **30 hours per week**, reset every Saturday |
| Session length | **12 hours** maximum, then it is killed |
| Working disk | **20 GB** in `/kaggle/working` |
| CPU | 4 cores |
| CPU-only notebooks | **unlimited** — they do not touch the GPU quota |

This project uses about **13.3 GPU-hours**, and Steps 1–5 use **zero**.

---

## Step 1 — Put the code somewhere Kaggle can reach it

Pick **one** of these two. Route A is better if you can use it.

### Route A — GitHub (recommended)

The repository already exists at
**https://github.com/AlAminAshraf01/speaker-count-separate-v1**, and `REPO_URL` in
`notebooks/src/_bootstrap.py` already points at it. So there is nothing to do for Route A — just
make sure the repo is **public** (Settings → General → Danger Zone → Change visibility) so
Kaggle can clone it without a token.

If you fork it or rename it, change `REPO_URL`, then rebuild and push:

```bash
python tools/build_notebooks.py
git add -A && git commit -m "point at my repo" && git push
```

> **Route A needs Internet switched on inside the notebook**, and Kaggle only allows that once
> you have verified your phone number (Settings → Phone Verification). If you cannot or do not
> want to do that, use Route B.

### Route B — Upload the folder as a dataset

1. Go to **kaggle.com → Datasets → New Dataset**.
2. Drag the whole `speaker-count-separate-v1` folder in.
3. Title it exactly **`speaker-count-separate-v1`**.
4. Create.

That is all. The notebooks look for `/kaggle/input/speaker-count-separate-v1` automatically if
they cannot clone from GitHub.

---

## Step 2 — Import the first notebook

1. **kaggle.com → Code → New Notebook**
2. **File → Import Notebook**
3. Upload `notebooks/kaggle_00_build_dataset.ipynb` from this folder.

> Always upload the file from `notebooks/`, **not** from `notebooks/src/`. The `src` files are
> the editable originals; the `.ipynb` files are the ones Kaggle understands.

---

## Step 3 — Build the dataset  *(CPU, ~15 minutes, no GPU quota)*

In the notebook you just imported, set up the right-hand panel:

| Setting | Value |
|---|---|
| **Accelerator** | **None** |
| **Internet** | **On** (Route A only) |
| **Persistence** | Variables and Files |

Then add the audio:

1. Click **+ Add Input**
2. Choose **Datasets**
3. Search for **`libri2mix-8khz-min`** (by `unconscious`, about 10 GB)
4. Click **Add**

Now click **Run All** at the top.

**What it does:** copies every individual voice recording into one big flat file (much faster to
read than 100,000 small files), and writes the dev and test sets as small CSV "recipes" that can
rebuild the exact audio later.

**What you should see at the end:**

```
train-100  27800  201  50   ~57 h
dev         2250   32   8   ~3.7 h
test        2073   32   8   ~3.3 h

speaker disjointness across splits (must all be 0):
   train-100 & dev        shared speakers:    0  OK
   train-100 & test       shared speakers:    0  OK
         dev & test       shared speakers:    0  OK
```

> **If any of those is not 0, stop.** It means the same person is in both your practice data and
> your exam data, and every score you measure afterwards is fake.

The `speakers` column counts only the voices used as counting targets — 20 % of each split is
held back — so LibriMix's advertised 251/40/40 shows up here as **201/32/32**. That is correct.

**Finally:** click **Save Version → Save & Run All (Commit)**, and wait for it to finish.

You never run this notebook again.

---

## Step 4 — Download the two recipe files

1. Open the finished version of notebook 00.
2. Go to the **Output** tab on the right.
3. Download `data/recipes_dev.csv` and `data/recipes_test.csv`.
4. Put both in the `data/` folder of your local copy and commit them.

**Why bother:** these two text files *are* your exam paper. Keeping them in git means the exam
never changes. If you regenerate them later, every score you already wrote down becomes
meaningless, because it was measured on a different test.

---

## Step 5 — The audit and the CPU counter  *(CPU, ~40 minutes, no GPU quota)*

1. **New Notebook → File → Import Notebook →** `notebooks/kaggle_01_audit_and_tier_a.ipynb`
2. Settings: **Accelerator = None**, Internet On (Route A only)
3. **+ Add Input → Notebook Output →** pick your saved run of notebook 00
4. **Run All**

### What the two halves are doing

**First, the honesty check.** When you add several voices together, the result gets louder. If
you leave it that way, a model can score well just by measuring volume and never listening to a
single word. So every clip is cut to exactly 3 seconds and divided by its own loudness, which
makes volume and length identical everywhere.

The script proves this rather than claiming it. It runs the same test twice — once on audio
*before* the fix and once *after*:

```
artefact        BEFORE mitigation ............ 44.2 %   -> LEAK
artefact_strict AFTER  mitigation ............ 20.0 %   -> OK
acoustic        AFTER  mitigation ............ 46.3 %   -> SIGNAL
```

The top line has to be high, or the test is broken and its clean verdict means nothing. The
middle line has to be 20 % (pure chance). The bottom line is real speech evidence surviving.

**Then, the model.** Sixteen acoustic numbers per clip, then a gradient-boosted tree, with a
5-fold search that keeps **each speaker entirely on one side of the split**. An ordinary split
would let the model recognise voices instead of counting them, which inflates the score.

Those are the real figures from this project's own LibriSpeech run, not illustrations.

The `acoustic` probe's 46.3 % and Tier A's 57.8 % are both honest and they differ because the
probe is a quick 1,500-mixture depth-3 tree and Tier A is a tuned 7,500-mixture grid search.
**Tier A's number is the bar**, and notebook 02 reads it out of `tier_a_report.json` by itself.

**Write down the last number it prints.** It looks like:

```
Tier A, speaker-disjoint K-fold .......  57.8%  <-- BEAT THIS
```

**Save Version → Save & Run All (Commit).**

### You can stop here

At this point you have a complete project that covers everything the course asks for, for
**zero GPU-hours**:

| The course wants | You have |
|---|---|
| EDA, correlations, outliers | per-speaker-count feature distributions and the importance table |
| Custom feature extraction | the 16 acoustic features — your own hand-designed transform |
| Hyperparameter search + K-fold | the speaker-grouped 5-fold search |
| Data-leakage audit | the before/after probe above |
| Baselines + interpretability | three naive predictors + the feature importance ranking |

Steps 6, 7 and 8 add the neural counter and the separator. They are where the GPU quota goes.

---

## Step 6 — Train the counter  *(GPU, ~3 hours, ~3 GPU-hours)*

1. **New Notebook → File → Import Notebook →** `notebooks/kaggle_02_train_counter.ipynb`
2. Settings: **Accelerator = GPU T4 ×2**, **Persistence = Variables and Files**
3. **+ Add Input → Notebook Output →** notebook 00's run (the audio)
4. **+ Add Input → Notebook Output →** notebook 01's run (so it knows the bar)
5. **Run All**

### The one thing worth understanding

It is not the architecture or the learning rate. It is **how the model summarises three seconds
into one answer**. Changing only that one layer, on identical data:

| summary method | score |
|---|---|
| average + spread | 66.5 % |
| attention-weighted average | 67.5 % |
| the full covariance matrix | **20.0 % — collapsed, answers "1 speaker" to everything** |
| **the covariance's eigenvalue spectrum** ← default | **74.4 %** |

An 8-point swing from one layer, and the winner is the smallest and fastest.

> **Those four are a synthetic speech proxy, not LibriSpeech** — they were run to rank the
> operators, and the ranking is the point. Do not compare them to your Tier A number, which
> is real audio. This notebook is what replaces them.

The intuition: if N people talk at once, the sound fills roughly **N independent directions**.
Eigenvalues say *how many* directions carry energy — that is your count. Eigenvectors say *which*
directions — and a counter should not care which. Throwing the eigenvectors away is the whole
idea, not a shortcut.

### What to watch in the log

Every epoch ends in `[beats the bar]` or `[BELOW THE BAR]`. Watch that, not the loss.

### If the session is killed at 12 hours

Just **Run All** again. It finds the last checkpoint and carries on. Nothing is lost.

**Save Version → Save & Run All (Commit)** when it finishes.

---

## Step 7 — Train the separator  *(GPU, ~11 hours, ~10 GPU-hours)*

This is the expensive one. Everything else together is under 4 GPU-hours.

1. **New Notebook → File → Import Notebook →** `notebooks/kaggle_03_train_separator.ipynb`
2. Settings: **Accelerator = GPU T4 ×2**, **Persistence = Variables and Files**
3. **+ Add Input → Notebook Output →** notebook 00's run
4. **Run All**

### What to watch

The `SI-SDRi` column with the per-N breakdown beside it. Two things:

- **N=1 should be high and boring.** A one-speaker mixture is solved by copying the input. If
  N=1 is bad, that is a plumbing bug, not a model problem.
- **N=5 is the hard one.** If it sits near 0 dB while N=2 climbs, the model is learning the easy
  cases and giving up on the crowded ones — which is exactly what the two loss bugs in the old
  version used to cause, and both are fixed here.

`(k degenerate refs dropped)` in the log is normal and honest. SI-SDRi is undefined for a clean
one-speaker mixture, because the mixture already *is* the target and the "improvement" being
measured is floating-point noise. Those are left out and counted. In the old version that one
bug turned a +1.2 dB result into a reported −8.68 dB.

### It will stop itself at 10.5 hours

That is deliberate, so Kaggle's 12-hour limit does not kill it mid-write. If it does get killed,
**Run All** again — it resumes from the last checkpoint. That only works with **Persistence**
turned on.

**Save Version → Save & Run All (Commit)** when it finishes.

---

## Step 8 — Get your final numbers  *(~15 minutes, ~0.3 GPU-hours)*

1. **New Notebook → File → Import Notebook →** `notebooks/kaggle_04_evaluate.ipynb`
2. **+ Add Input → Notebook Output →** notebooks 00, 01, 02 and 03 (missing ones are skipped)
3. **Accelerator = GPU T4 ×2**
4. **Run All**

**Do this last, once.** This is the only notebook that touches the test set. Every choice you
made before now was made on the dev set. A test set you tune against is just a second dev set
with a misleading name.

### Four numbers, never one

A single score cannot describe a system that does two jobs, so it prints all four:

1. **Counting accuracy** plus the confusion matrix and MAE.
2. **P-SI-SNR** over every clip — stays defined even when the count is wrong, so a system
   cannot score well by refusing to commit.
3. **SI-SDRi on count-correct clips only** — the only row comparable to the separation
   literature, which is always *told* how many speakers there are.
4. **SI-SDRi with the true count forced** — separation quality given a perfect counter. **The
   gap between 3 and 4 is exactly what miscounting costs you**, and it tells you which half to
   improve next.

### Reading the output

Two confidence intervals are printed and **they are not interchangeable**:

- The **Wilson** interval assumes your 1,500 test clips are 1,500 independent tries. They are
  not — they were made from about **32 people**.
- The **speaker-level bootstrap** resamples *people* instead of clips. It is wider, and it is
  the honest one. **Quote that one**, and say in your report which is which.

When two counting models are attached it also runs **McNemar's test** to compare them. Because
both scored the exact same clips, that is the correct comparison — checking whether two error
bars overlap is too cautious here and will call a real difference a tie.

---

## If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `No packed store found` | notebook 00's output is not attached | **+ Add Input → Notebook Output →** notebook 00 |
| `Not found. Add the dataset 'libri2mix-8khz-min'` | the audio is not attached | **+ Add Input → Datasets →** search for it |
| `FAIL gpu   no GPU` | GPU notebook without a GPU | **Settings → Accelerator → GPU T4 ×2** |
| `FAIL cells  STALE` | your `.ipynb` is older than your code | re-import the notebook from `notebooks/` |
| `FAIL precision` | fp32 and fp16 disagree | do not train. Report it — this is the exact bug that broke the old project |
| Separator SI-SDRi is negative early | normal for the first few epochs | let it run; if N=1 stays negative after ~5 epochs, that is a plumbing bug |
| `0 sources` for N=3,4,5 in the eval table | you passed `--limit` | it is stratified now, but a very small limit still starves the rare classes |
| `shared speakers: 7` (not 0) | train and test overlap | rebuild the store; do not train on it |
| Session killed at 12 h | normal Kaggle limit | **Run All** again; training resumes |
| `torch.OutOfMemoryError` in notebook 03 | the separator needs more than the card has | your clone is stale — the fix is in `GlobalLayerNorm`. Re-run so the bootstrap pulls it. If it still OOMs, set `BATCH_SIZE = 8` **and** `STEPS_PER_EPOCH = 1500`, which keeps the mixtures-per-epoch and the GPU budget identical |
| `refusing to freeze ... BABBLE` | you asked for babble noise | do not. Babble is 4–8 real talkers, so the count label would be wrong |

### Changing anything in the code

Always rebuild and push the notebooks, or Kaggle keeps running the old cells:

```bash
python tools/build_notebooks.py
python tools/run_all_tests.py
git add -A && git commit -m "..." && git push
```

Then **re-import the `.ipynb`** into Kaggle. Pulling new code does *not* update cells you already
imported — that is why the `cells` check exists.

---

## Quick reference

| Notebook | Accelerator | Time | GPU cost | Needs |
|---|---|---|---|---|
| `kaggle_00_build_dataset` | None | ~15 min | 0 | `libri2mix-8khz-min` |
| `kaggle_01_audit_and_tier_a` | None | ~40 min | **0** | notebook 00 output |
| `kaggle_02_train_counter` | GPU T4 ×2 | ~3 h | ~3 h | notebooks 00 + 01 |
| `kaggle_03_train_separator` | GPU T4 ×2 | ~11 h | ~10 h | notebook 00 |
| `kaggle_04_evaluate` | GPU T4 ×2 | ~15 min | ~0.3 h | notebooks 00–03 |
| `kaggle_05_demo` | None | ~2 min | **0** | notebooks 00 + 02 + 03 |

**Total: about 13.3 GPU-hours out of 30 per week.** Notebooks 00 and 01 cost nothing.

### Suggested order across a week

| when | do | cost |
|---|---|---|
| day 1 | notebooks 00 and 01 | 0 |
| day 1 | notebook 02 (counter) | ~3 h |
| day 2 | notebook 03 (separator) — start it and leave it | ~10 h |
| day 2 | notebook 04 | ~0.3 h |
| day 2 | notebook 05 (the demo — listen to it) | 0 |
| spare quota | the fixed-N=2 control in notebook 03 | ~2.5 h |

---

## Step 9 — Listen to it  *(CPU, ~2 minutes, no GPU quota)*

1. **New Notebook → File → Import Notebook →** `notebooks/kaggle_05_demo.ipynb`
2. Settings: **Accelerator = None**
3. **+ Add Input → Notebook Output →** notebooks 00, 02 and 03
4. **Run All**

Every other notebook reports an average over 1,500 mixtures. This one runs a single file and
plays it back, because an average cannot be listened to and a viva can. It is also the only
notebook that uses the system *as a system* — 02 trains the counter, 03 trains the separator,
04 scores them, and this is the one that joins them into "audio in, one track per person out".

It runs on clips from the frozen test set, so it prints **predicted vs true** rather than
asking you to trust the number. Two things to look at:

- **The probability bars.** Most of the mass on one class is a model that is right for a
  reason. Mass spread across 4 and 5 on a 5-speaker clip is a near miss. Mass parked on 1
  whatever the input is the old project's failure, and you would see it instantly.
- **The slot power table.** The separator always emits 5 slots and training pushes the unused
  ones toward −30 dB, so a 3-speaker clip should show a clear cliff after slot 3. A gentle
  slope instead means the counter and the separator disagree — and you can hear who is right.

To run it on **your own recording**: **+ Add Input → Upload → New Dataset**, then set
`MY_FILE` in the last cell. Any format, any sample rate.

> Temper your expectations on real audio. The models are trained on **fully overlapped**
> speech — all N people talking at once for the full three seconds. In a real conversation
> people take turns, and during a single-speaker stretch "1" is the honest answer even with
> three people in the room. That is a different problem (diarisation), not a bug.

---

## Two things to be honest about in your report

**Every clip is fully overlapped.** All N people talk at once for the full three seconds. Real
conversation is mostly turn-taking, and counting speakers there is a different and harder
problem called diarisation. A good score here is a real result, but it does not mean speaker
counting is solved. Say that before an examiner says it to you.

**Report the separation gap as budget.** Published Conv-TasNet reaches 14.76 dB on this data
using 200 epochs on a training set three times larger than yours. You will not reach it on 30
GPU-hours a week. Put your epoch count and training set next to your number and the gap explains
itself — that is a much stronger position than a number with no context.
