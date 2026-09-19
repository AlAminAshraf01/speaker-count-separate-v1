# speaker-count-separate-v1

Count how many people are talking in a short, noisy, fully-overlapped recording (N = 1..5), then
separate their voices. EEE 402 Group 10, version 1.

**One job per model.** A dedicated counter and a dedicated separator, trained apart and joined
at inference. That is the whole difference from v0, and it is not a stylistic preference — it
comes out of measuring why v0's single multi-task network did both jobs badly.

---

## The three numbers that produced this design

| measured in v0 | |
|---|---|
| share of the shared trunk's gradient reaching the **counting** objective | **0.53 %** |
| share reaching separation | 86.14 % |
| ratio at the encoder | **633 : 1** |

The counting head was reading features that nothing had ever asked to encode a speaker count.
It settled on the class marginal and answered **"1 speaker" for 1445 of 1500** test mixtures —
20.00 % on a 5-class problem, which is exactly chance.

Separation did not get away with it either. v0's own ablation priced the auxiliary objectives at
**−8.7 dB**, and the pooled model scored **0.08 dB** SI-SDRi on the official Libri2Mix test set
where a separation-only control reached **5.49 dB** in a fifth of the epochs.

And the sharing was buying almost nothing to offset that: removing the separator from v0's model
saved only **8.4 %** of the forward FLOPs, because the TCN is the expense. A dedicated
0.49 M-parameter counter adds roughly **5 %** on top of a 5.25 M separator — and gets 100 % of
its own gradient in exchange.

One more number, for scale. Sixteen hand-crafted acoustic scalars fed to a gradient-boosted tree
count speakers at **69.3 %** on the same data. A decision tree beat the 5.3 M network by
**49 points**. That system is in here as `countsep.features` + `countsep.baselines`, it runs on
CPU in minutes, and nothing neural is allowed to claim victory without beating it.

---

## Start here

| document | what it answers |
|---|---|
| **[docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md)** | **How to run it.** Step by step, plain words, from an empty Kaggle account to final numbers. |
| **[docs/DIAGNOSIS.md](docs/DIAGNOSIS.md)** | What went wrong in v0. Nine findings, every one measured or verified against v0's own source. |
| **[docs/DESIGN.md](docs/DESIGN.md)** | The v1 architecture, the milestone ladder with kill conditions, and the GPU budget. |
| **[docs/DATA.md](docs/DATA.md)** | *"Would a different dataset help?"* No — and the mixing **settings** matter far more than the corpus. |

---

## The notebooks

| notebook | accelerator | time | GPU cost |
|---|---|---|---|
| `kaggle_00_build_dataset` | None | ~15 min | 0 |
| `kaggle_01_audit_and_tier_a` | None | ~40 min | **0** — a complete counting project on its own |
| `kaggle_02_train_counter` | GPU T4 ×2 | ~3 h | ~3 h |
| `kaggle_03_train_separator` | GPU T4 ×2 | ~11 h | ~10 h |
| `kaggle_04_evaluate` | GPU T4 ×2 | ~15 min | ~0.3 h |

**About 13.3 GPU-hours of a 30 h/week free-tier quota.** Notebooks 00 and 01 cost nothing and
already produce a defensible result, so the expensive half is optional in the literal sense.

---

## What is fixed, and how you can tell

Four defects from v0, each with a test that fails if it comes back.

**1. The tasks starved each other.** → Two models, trained separately, joined by
`countsep.pipeline`. `tests/test_separator.py::test_the_separator_has_no_count_head`.

**2. Two loss terms were structurally biased toward N=1.**
- `soft_clamp` multiplied by the SI-SDR gradient *peaks* at the clamp point instead of vanishing
  there (measured: 0.999 / 9.091 / **15.811** / 9.091 at 0 / 20 / 30 / 40 dB), so a trivially
  solved 1-speaker clip carried ~16× the gradient of a hard 5-speaker one. → `hard_clamp`, flat
  above τ.
- The silence term divided by the number of leftover **slots**, and leftover slots run 4, 3, 2,
  1, 0 as N goes 1..5 — so a balanced batch gave the single N=1 clip 40 % of that term and the
  N=5 clip none. → a per-item average.

**3. The model trained as one function and was measured as another.** `autocast` promotes
`nn.LayerNorm` to fp32 but cannot protect a hand-written one, and v0 had **49** hand-written
norms with `eps = 1e-8` — *exactly 0.0 in fp16*. Same checkpoint: 44.9 % in fp16, 20.00 % in
fp32, agreeing on 16.1 % of predictions. → statistics computed in fp32 regardless of autocast,
`MODEL_EPS` representable in fp16, AMP off by default, and a preflight check that **refuses to
train** when the two precisions disagree.

**4. The leakage audit had never run.** A `def` landed inside `main()`, every probe became
unreachable code after a return, and the script exited **0** while writing nothing — for months,
with two headline numbers quoted from it that it had never produced. → the audit verifies its own
report before exiting, and `tests/test_scripts_are_reachable.py` fails the build on that exact
bug shape (validated against the original).

```bash
python tools/run_all_tests.py     # 6 files, ~45 s, no pytest, no dataset needed
```

---

## Layout

```
docs/                      runbook, diagnosis, design, data
src/countsep/
  counter.py               CountCRNN, 0.49 M params, 4 pooling operators (eigen by default)
  separator.py             SepNet: Conv-TasNet, 5 speaker slots + 1 noise slot, NO count head
  pipeline.py              count -> separate -> keep the N loudest slots
  losses.py                rectangular PIT; hard clamp; per-item silence; no counting term
  features.py baselines.py the 16-scalar CPU counter and the probes that gate every number
  metrics.py               SI-SDR, P-SI-SNR, usable_si_sdri, confusion reporting
  interpret.py             filterbank, mask geometry, cross-model correlations
  datasets.py              dynamic mixing + frozen recipes; want="count" | "separate"
  mixing.py pack.py noise.py audio.py utils.py checkpoint.py constants.py
scripts/
  00_pack_sources  01_make_frozen_sets  02_audit_and_baselines  03_tier_a
  04_train_counter  05_train_separator  06_evaluate  07_interpret  preflight
notebooks/src/*.py         editable originals (percent format, '# %include')
notebooks/*.ipynb          built by tools/build_notebooks.py -- upload THESE to Kaggle
tests/                     6 files; structural, numerical and end-to-end
```

Never hand-edit a `.ipynb`. Edit `notebooks/src/*.py`, run `python tools/build_notebooks.py`,
and commit both — `tests/test_notebooks.py` fails the build on a stale notebook, because Kaggle
imports the `.ipynb` and a stale one silently runs last week's cells against this week's code.

---

## Two things to be honest about in the report

**Every clip is fully overlapped.** All N people talk at once for the full three seconds. Real
conversation is mostly turn-taking, and counting speakers there is a different and harder problem
(diarisation). A good number here does not mean speaker counting is solved — say so before an
examiner says it to you.

**Report the gap to published separation as budget.** Conv-TasNet's 14.76 dB on LibriMix 8 kHz
min comes from 200 epochs on train-clean-360. Put your epoch count and training set next to your
number and the gap explains itself.
