# What went wrong in v0, and what v1 does about it

This document is the reason this repository exists. It is a post-mortem of
`speaker-count-separate` — the joint count-and-separate model this is version 1 of — written
before any code here was designed, so that every choice in `docs/DESIGN.md` points back to a
named, measured failure rather than to a preference.

**It is not only about counting.** The same multi-task arrangement that starved the counting
head also cost separation roughly 8.7 dB, and §1–§3 below apply to both halves.

Every claim below is either **measured** (a number produced by running something) or
**verified** (a claim from the old repo re-checked against its own source). Claims that are
reasoning only are marked *inference*.

---

## 0. The headline

| system | 5-class accuracy, balanced test set |
|---|---|
| chance | 20.0 % |
| **CountSepNet count head, fp32 (5.30 M params, 10.6 GFLOP/s of audio)** | **20.00 %** |
| CountSepNet count head, fp16 autocast, same checkpoint | 44.9 % |
| 16 hand-crafted scalars + a gradient-boosted tree (measured here, §5) | **57.8 %** |

A 5.3-million-parameter network scored *exactly chance* while a decision-tree ensemble over
sixteen numbers you can compute in numpy scored 58 %. The task is not the problem. Nothing
about the counting failure is evidence that counting is hard.

Both rows are real LibriSpeech: the 20.00 % is v0's own trained checkpoint on its test set,
the 57.8 % is a speaker-disjoint 5-fold over 7,500 mixtures from 201 talkers (MAE 0.480,
fold spread ±1.5 points).

The fp32 model answered **"1 speaker" for 1445 of 1500 mixtures**.

---

## 1. The count head was starved of gradient — by a factor of ~160

Measured by back-propagating each weighted loss term alone, at initialisation, `small`
preset, one RMS-normalised mixture per N=1..5:

| term | weight | ‖g‖ into shared trunk | share | ‖dL/d feat‖ | share |
|---|---|---|---|---|---|
| separation | 1.0 | 1.741e+02 | **86.14 %** | 5.134e-01 | **96.82 %** |
| silence | 1.0 | 2.199e+01 | 10.88 % | 1.178e-02 | 2.22 % |
| noise slot | 0.2 | 4.952e+00 | 2.45 % | 3.036e-03 | 0.57 % |
| **counting** | **0.5** | **1.073e+00** | **0.53 %** | **2.054e-03** | **0.39 %** |

Into the encoder specifically the ratio separation:counting is **633 : 1**.

The count head reads the TCN skip-sum. That tensor's update direction was set 99.5 % by
objectives that do not care about N. Over 38 epochs the trunk was never meaningfully asked to
make its pooled statistics vary with speaker count, so the head became a linear read-out of
features that do not encode the label, and its only reachable optimum is the class marginal —
cross-entropy ln 5 = 1.609, accuracy at chance.

To reach parity, `w_count` would have to be **O(100)**. The old repo's own troubleshooting
table prescribes *"raise `loss.w_count` to 1.0"* — a 2× change against a 160× deficit.

**There is no loss balancing anywhere in that repo.** Zero occurrences of GradNorm, PCGrad,
uncertainty weighting or per-term gradient normalisation. The only normaliser is a single
**global** `clip_grad_norm_(model.parameters(), 5.0)`; with a measured total norm of 201 at
initialisation, that is a uniform 0.0248 rescale which couples the count head's realised step
size to how large the separator's gradient happened to be that step.

> **What follows for this project:** the counter trains alone. Cross-entropy becomes 100 % of
> the trunk gradient instead of 0.53 %. This one change is worth more than any architecture
> decision, and it is free.

---

## 2. Two of the three competing terms were structurally biased toward N=1

### 2a. The soft clamp does the opposite of what its docstring claims — verified

`soft_clamp(s, τ) = −10·log₁₀(10^(−s/10) + 10^(−τ/10))` with τ = 30. Its own docstring, and
`DESIGN.md`, and `CONTRACT.md:328`, all say it "prevents a single easy example dominating".

Measured here by autograd, multiplied by the way the SI-SDR gradient itself scales (≈10^(s/20)):

| achieved SI-SDR | d(clamp)/ds | ×\|dSISDR/d est\| | **product** |
|---|---|---|---|
| 0 dB | 0.9990 | 1.00 | 0.999 |
| 10 dB | 0.9901 | 3.16 | 3.131 |
| 20 dB | 0.9091 | 10.00 | 9.091 |
| **30 dB (= τ)** | 0.5000 | 31.62 | **15.811 ← peak** |
| 35 dB | 0.2403 | 56.23 | 13.510 |
| 40 dB | 0.0909 | 100.00 | 9.091 |

The gradient is **maximised exactly at the clamp point**. N=1 is trivially solvable — copy the
mixture — so N=1 items live in the 20–35 dB band and carry ~16× the gradient of an N=5 item
sitting near 0 dB. The clamp upweights the easiest speaker counts.

A hard `.clamp_max(τ)` would behave as the docstring describes. This soft one does not.

### 2b. The silence term's denominator counts slots, not items

`sil = sil_sum / sil_count` where `sil_count` accumulates **leftover slots**, not batch size.
Leftover slots per item: N=1 → 4, N=2 → 3, N=3 → 2, N=4 → 1, **N=5 → 0**. In a balanced batch
the single N=1 item supplies 40 % of the second-largest gradient term and the N=5 item supplies
nothing.

So two of the three non-count terms pull the shared trunk toward whatever representation solves
N=1 — which is precisely what the degenerate counter later reports.

---

## 3. The precision story: the model was trained as one function and measured as another

Four claims, all verified here on torch 2.14:

| claim | verified result |
|---|---|
| `torch.argmax` on all-NaN logits returns index 0 | `[0, 0, 0]` — **index 0 is class N=1** |
| `eps = 1e-8` is representable in fp16 | `float16(1e-8) == 0.0`, and `v + 1e-8 == v` |
| autocast promotes `nn.LayerNorm` to fp32 | yes → `torch.float32` |
| autocast protects a hand-rolled mean/var/sqrt norm | **no** → stays `bfloat16`/`float16` |

`GlobalLayerNorm` in that repo is hand-written from `mean`/`var`/`sqrt`/`div`. There are **49
instances** of it between the waveform and the classifier. The project re-implemented the one
operator autocast specifically protects, and thereby opted out of that protection 49 times, each
dividing by an unguarded `sqrt(var + 0.0)`.

The model was **trained** under fp16 autocast and **validated and evaluated** in fp32. The
function that was fitted and the function that was measured are not the same function. The fp32
signature — "1 speaker" for 1445 of 1500 — is exactly what `argmax` returns on NaN logits.

### What the 44.9 % does and does not mean

- It **does** prove the fp16 perturbation carries information about N. On a balanced 5-class
  set any label-independent perturbation yields exactly 20.0 %; 44.9 % cannot come from noise.
- It **does not** demonstrate ordinal competence. 44.9 % at MAE 0.773 is statistically
  indistinguishable from the strategy *"detect the single-talker case, then always answer 3"*,
  which scores 40.15 % at MAE 0.798.

### One popular explanation is refuted

*"fp16 rounding error is itself a level cue, because louder activations round worse."*
**Measured false.** Relative error of an fp16 mean over 3000 values, at internal gains
1, 4, 16, 64 and 256:

```
gain    1.0 : 1.8370e-04      gain   64.0 : 1.8370e-04
gain    4.0 : 1.8370e-04      gain  256.0 : 1.8370e-04
gain   16.0 : 1.8370e-04
```

Identical to five significant figures — floating point is scale-invariant by construction. A
*threshold/overflow cliff* mechanism remains viable; a *proportional* one does not.

Related measurement, same checkpoint architecture at near-initialisation: fp16-vs-fp32 pooled
statistics differ by 0.2 % relative, giving a logit delta of **2.8e-4**.

> **Correction.** An earlier draft of this document set that 2.8e-4 against an "across-batch
> logit spread of 5.3e-3" and concluded that numerical noise was worth ~5 % of the entire
> between-input signal. That comparison was wrong and the conclusion does not follow: the
> inputs in that measurement were `torch.randn` waveforms, which are *statistically identical
> to one another*, so a small spread was the correct answer rather than evidence of a
> degenerate head. Re-measured on the same untrained architecture with acoustically distinct
> inputs — real mixtures at N = 1..5 — the spread is **2.25e-2, a factor of 7.4 larger** than
> the white-noise figure. The fp16 delta is therefore ~1 % of the between-input signal at
> initialisation, not ~5 %, and nothing about head degeneracy can be inferred from it.
>
> The rest of §3 is unaffected: it rests on the four verified claims in the table above and on
> the arithmetic that 44.9 % on a balanced 5-class set cannot come from a label-independent
> perturbation. This correction is recorded rather than quietly edited because the mistake —
> comparing against a baseline that could not have been large — is the same species as the
> ones this document catalogues.

> **What follows for this project:** one precision everywhere. Train, validate and evaluate in
> the same arithmetic; use `nn.LayerNorm`/`nn.BatchNorm` rather than hand-rolled norms; set
> `eps` to a value representable in the working precision; and keep a test that asserts fp32 and
> fp16 predictions agree on >95 % of a dev batch.

---

## 4. The leak probe has never run. Not once.

`scripts/03_count_leak_probe.py` is **structurally dead code**, and it exits 0.

Commit `3e3f13a "Stop labelling legitimate acoustic evidence as a leak"` inserted
`def probe_flag(...)` at line 155 — *inside the body of* `main()`. Parsed with `ast`:

```
top-level def main:       lines  91-152     <- ends here
top-level def probe_flag: lines 155-263     <- everything else got captured in here
    nested def run:       lines 174-184
```

`probe_flag` returns unconditionally at line 170. Every probe, the naive-predictor baseline, the
speaker-disjointness audit and the `leak_report.json` write live at lines 171–263 — *after* that
return, *inside* the wrong function. `main()` falls off its end at line 152 returning `None`, and
`sys.exit(None)` is exit code 0.

Run to completion against a real store, it prints two lines and writes nothing:

```
$ python scripts/03_count_leak_probe.py --store .../store --recipes .../recipes_dev.csv --out probe_out
==========================================================================
03 - count-label leakage audit + naive predictor
==========================================================================
store   : ...\store
recipes : ...\recipes_dev.csv (200 mixtures)
>>> EXIT CODE: 0
>>> leak_report.json written?          (nothing)
```

**Consequence:** the project's quoted "35.0 % naive baseline" and "22.6 % mitigated artefact
probe" were never produced by this repository. They survive from a legacy standalone script.
Two numbers that gate the entire interpretation of the counting result have been unreproducible
for months, and nothing noticed, because the script exits 0.

> **What follows for this project:** every audit script must fail loudly. The audit is milestone
> 1, it runs on CPU before any GPU hour is spent, and a test asserts that it wrote its report.

---

## 5. The task is fine. The anti-leak mitigation works. Measured.

Using the old repo's **own** mixing code (`csnet.mixing.sample_recipe` / `render_recipe`) at
production settings, over a synthetic speech proxy from its own
`tools/make_fake_libri2mix.py` (harmonic stacks, per-speaker formants, syllabic silences).
16 hand-crafted features — crest factor, kurtosis of signal and of envelope, envelope sparsity,
spectral flatness mean/std/p90, spectral entropy mean/std, TF-sparsity at top 1 %/5 %,
modulation depth, silence fraction, Hoyer ratio — into `HistGradientBoostingClassifier`,
5-fold CV, 260 mixtures per class:

| condition | level-only probe | HOS + spectral |
|---|---|---|
| production (3 s, RMS-norm, gain ±5 dB, 80 % noisy) | 21.9 % | **61.4 %** |
| clean only (no additive noise) | 22.2 % | 78.2 % |
| clean, no gain jitter | 29.6 % | 87.2 % |
| production but noise ≥ 10 dB SNR | 20.4 % | 69.2 % |
| production, no babble in the noise bank | 21.0 % | 64.0 % |
| 6 s crop instead of 3 s | 19.2 % | 54.5 % |
| clean + 6 s | 23.7 % | 70.8 % |

Two conclusions:

1. **The anti-leak mitigation is sound.** A level-only probe sits at chance (20 %) in every
   condition. Fixed 3 s crop ÷ RMS genuinely removes the level and duration cues. The old
   project got this part right; it just never re-measured it (§4).
2. **Real signal survives, and the pipeline's own settings destroy most of it.** Additive noise
   costs ~17 points, per-source gain jitter ±5 dB costs ~9 points, babble specifically ~3 points.

*Caveat, stated plainly:* the corpus is a synthetic proxy, so absolute numbers are not
LibriSpeech numbers — the **contrasts** are the result. The 6 s row is almost certainly an
artefact of the proxy's silence structure and should **not** be read as evidence against longer
windows; published work finds accuracy rises with duration.

### Which cue is actually carrying the signal

Same pipeline, 120 mixtures per N, mean of each feature. This is the measurement that decides
what the neural front end has to preserve:

| cue | N=1 | N=2 | N=3 | N=4 | N=5 | clean spread | at production settings |
|---|---|---|---|---|---|---|---|
| **kurtosis (signal)** | 9.74 | 4.79 | 3.26 | 2.40 | **1.95** | **5.0×** | 7.50 → 2.06, still 3.6× |
| kurtosis (envelope) | 17.64 | 9.95 | 7.85 | 6.47 | 6.03 | 2.9× | 17.79 → 7.62 |
| silence fraction | 0.462 | 0.241 | 0.136 | 0.083 | 0.054 | 8.6× | 0.121 → 0.025 |
| envelope sparsity | 0.483 | 0.614 | 0.681 | 0.727 | 0.756 | rises 1.6× | 0.605 → 0.771 |
| crest factor | 6.75 | 6.67 | 6.41 | 6.07 | 5.85 | 1.15× | 6.55 → 5.87 |
| spectral flatness | 0.230 | 0.308 | 0.349 | 0.368 | 0.386 | rises 1.7× | **0.437, 0.473, 0.455, 0.465, 0.451** |

Three things follow, and all three are design decisions:

1. **Kurtosis is the workhorse**, not crest factor. The central-limit argument is real and it
   is strongest in the fourth moment of the signal itself.
2. **Crest factor is the weakest member of the family** — a 1.15× spread against kurtosis's
   5×. The old project's docs name crest first; the measurement does not support that ranking.
3. **Spectral flatness is close to useless once noise is added.** Clean, it rises monotonically;
   at production settings it is non-monotonic and pinned near 0.45, because flatness is
   measuring the *noise colour* rather than the speaker count. Anything that leans on flatness
   is really leaning on which noise kind was drawn.

A design note this makes unavoidable: the fourth moment of the waveform is the single strongest
cue, and a `mean`+`std` pooling layer is a *second*-moment read-out. §6 is not a theoretical
objection.

### Headroom, on a speaker-disjoint split

Train on `train-100` speakers, test on `dev` speakers, 3500 / 1100 mixtures, production settings:

| system | accuracy |
|---|---|
| 16 hand-crafted features + gradient boosting | **69.3 %** |
| learned linear-STFT CRNN, 0.476 M params, 14 CPU epochs | 67.1 % |
| chance | 20.0 % |

> **Proxy numbers.** This comparison was run on a synthetic speech proxy, not LibriSpeech.
> On real audio the same feature set and the same speaker-disjoint protocol give **57.8 %**,
> so read the two rows against each other, not as absolutes.

The CRNN is data- and compute-starved here and uses exactly the mean+std pooling that §6 says is
wrong, so **67.1 % is a floor for the learned route, not its ceiling.** The operative number is
the other one: **any design that does not clearly beat ~69 % has bought nothing over a decision
tree**, and that is the bar this project sets for itself.

---

## 6. The head read the wrong statistic

The count head's entire evidence is `mean` and `std` over time of a 128-channel 1×1 projection
of the TCN skip-sum — that is, the mean vector plus `sqrt(diag(Cov))` of the channel covariance.
Every off-diagonal covariance term, every temporal autocorrelation and every higher moment is
discarded.

But "how many concurrent sources" is a **cardinality / rank** property: at any frame, N talkers
means N dominant components in the channel covariance. That lives in the off-diagonal terms.
And the old project's own record says the cue that survives its normalisation is *fourth*-moment
— crest factor and kurtosis. A second-moment, diagonal-only read-out was pointed at a
fourth-moment, off-diagonal cue. *(inference, but the measurement in `docs/POOLING.md` tests it
directly.)*

Compounding it: 49 gLN + 2 LayerNorm sit between waveform and classifier, and the head's own
`nn.LayerNorm` over the pooled vector subtracts that vector's mean — deleting the common-mode
direction, which is the natural encoding for a scalar count.

---

## 7. Label noise nobody accounted for

`p_clean = 0.2` with noise kinds drawn uniformly from (white, pink, brown, babble, real) means
roughly **20 % of training mixtures carry babble noise that is itself 4–8 real talkers**, and
about 10 % of all mixtures carry it below 10 dB SNR — above the threshold where those extra
talkers are audible and countable. Those mixtures' count labels are factually wrong.

---

## 8. What this project inherits

**Adopt:** the packed-store format (`audio.i16` + `index.csv`, self-describing); `SourceStore`;
`checkpoint.py` and `find_resume()` (the load-bearing Kaggle asset — note it ranks by recorded
global step, not mtime, and the old `CONTRACT.md` documents the old wrong rule); the
percent-format `notebooks/src/*.py → .ipynb` build system; `_bootstrap.py`; `run_all_tests.py`;
`NoiseBank`; the frozen-recipe-CSV evaluation protocol; `12_preflight.py`; `audio.py`,
`utils.py`.

**Refuse:** the model; `datasets.py`'s batch dict (it allocates a `(5, 24000)` refs tensor and a
`(24000,)` noise tensor per item that a counter never reads); `GlobalLayerNorm`; the PIT loss;
the 5-slot mask head; `soft_clamp`; the silence term; and the belief that any of the old repo's
docstrings describe what its code does.

---

## 9. The five things this project does differently

1. **The counter trains alone.** 0.53 % of the gradient becomes 100 % of it. (§1)
2. **One precision, everywhere**, with `nn.LayerNorm` and a test that fp32 and fp16 agree. (§3)
3. **A pooling operator that can express cardinality**, not `mean`+`std`. (§6)
4. **Baselines and the leak audit run first, on CPU, and fail loudly** — before a GPU hour is
   spent, with a test asserting the report was written. The bar is 57.8 %, not 20 %. (§4, §5)
5. **Mixing settings chosen against measured costs**, and a label-noise policy that never puts
   speech in the noise. (§5, §7)
