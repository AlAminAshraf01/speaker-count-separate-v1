# The v1 design

## The decision, in one paragraph

v0 put counting and separation in one network and both jobs suffered. v1 splits them into two
specialists, trains each with 100 % of its own gradient, and joins them at inference. The cost
is about **5 % more compute**; the thing it buys is that each half can now be measured,
debugged and improved independently — which v0 structurally could not do, because a bad
end-to-end number there could not be attributed to either half. Everything else in this
document follows from that one choice, or from a loss bug that the choice made visible.

---

## 1. Why two models and not one

| measured in v0 | value |
|---|---|
| counting's share of the gradient into the shared trunk | **0.53 %** |
| separation's share | 86.14 % |
| ratio at the encoder | **633 : 1** |
| cost of the auxiliary objectives to separation (single-batch ablation) | **−8.7 dB** |
| pooled model, official Libri2Mix N=2 | 0.08 dB |
| separation-only control, 8 epochs vs the pooled model's 38 | **5.49 dB** |
| FLOPs saved by removing the separator from v0's model | only **8.4 %** |

Read the last row together with the first. Sharing a trunk was saving 8.4 % of the compute while
costing counting 99.5 % of its gradient and separation ~8.7 dB. That is not a trade-off worth
making, and `w_count` could not have fixed it: reaching parity would have needed **O(100)**,
against the 0.5 it had and the 1.0 the v0 troubleshooting notes recommended.

**The architecture:**

```
                     ┌─────────────────────────────┐
 3 s @ 8 kHz  ──┬──► │ CountCRNN      0.49 M params│ ──► N̂ ∈ 1..5
 RMS-normalised │    └─────────────────────────────┘
                │    ┌─────────────────────────────┐
                └──► │ SepNet         5.25 M params│ ──► 5 speaker slots + 1 noise slot
                     └─────────────────────────────┘
                                                          │
                                       keep the N̂ loudest ┘ ──► estimated sources
```

**Slot selection is by loudness**, which reads exactly what the loss optimised: the rectangular
PIT loss pushes surplus slots toward −30 dB relative to the mixture, so real speakers are the
loud slots. It also degrades gracefully — predicting 3 when the truth is 4 returns the three
loudest real speakers rather than a scrambled set.

---

## 2. The counter

`countsep.counter.CountCRNN`, 0.49 M parameters. Linear-magnitude STFT (25 ms / 10 ms) → 2-D
convs → BiGRU → **pooling** → 5-way softmax.

**The pooling operator is the single most important choice**, worth 8 accuracy points — more
than the architecture, the front end and the schedule combined. Changing *only* that layer,
10 CPU epochs on identical speaker-disjoint data:

> **These four rows are a synthetic speech proxy, not LibriSpeech.** They rank the operators
> against each other, which is what they were run for, and the ranking is the claim. Their
> absolute values are not comparable to the 57.8 % below, which is real audio. Notebook 02
> replaces them.

| pooling | params | accuracy | MAE |
|---|---|---|---|
| `meanstd` (what v0 used) | 0.508 M | 66.5 % | 0.396 |
| `attentive` (Okabe et al.) | 0.525 M | 67.5 % | 0.422 |
| `covariance` (log-Euclidean, 560-d) | 0.551 M | **20.0 % — collapsed** | 2.000 |
| **`eigen`** (spectrum + effective rank, 65-d) — default | **0.488 M** | **74.4 %** | **0.305** |
| *hand-crafted features + GBM, same proxy* | — | *69.3 %* | — |

**The bar the counter actually has to clear is 57.8 %**, measured on real LibriSpeech by
`03_tier_a.py` — speaker-disjoint 5-fold, 7,500 mixtures, 201 talkers, MAE 0.480, fold spread
±1.5 points, six of six grid configurations inside one point of each other. Notebook 02 reads
that value out of `tier_a_report.json`; nothing is typed by hand.

**The rank hypothesis was right; the obvious implementation of it was wrong.** Both
`covariance` and `eigen` read the same channel covariance. The full vectorisation hands the
classifier 560 correlated dimensions and collapses to 20.0 % with MAE exactly 2.000 — the
signature of answering "1 speaker" to everything, which is v0's failure reached by a different
route. The eigenvalue spectrum hands it 65 and reaches 74.4 %.

The difference is which part is nuisance. Eigen**vectors** say *which* directions the talkers
occupy, and a counter should be invariant to that. Eigen**values** say *how many* directions
carry energy — that is the count. Discarding the eigenvectors is the inductive bias, not a
shortcut. `eigen` is also the smallest and fastest of the four.

**How much to claim: nothing yet.** On the proxy, `eigen`'s 74.4 % and the tree's 69.3 % had
Wilson intervals of [71.7, 76.9] and [66.5, 72.0] — overlapping by 0.3 points. Both numbers
have since been superseded on one side: the tree scores **57.8 %** on real audio, and `eigen`
has not been run there at all. So the honest statement is that the operator ranking held on a
proxy and the real comparison is pending, not that `eigen` beats the tree.

When both numbers do exist, compare them with **McNemar** rather than by checking whether the
intervals overlap. Both models score the identical clips, so the test is paired; overlapping
intervals are too conservative for paired data and will call a real difference a tie.

---

## 3. The separator

`countsep.separator.SepNet`, 5.25 M parameters. Conv-TasNet (Luo & Mesgarani 2019, Table I) with
`max_n_src = 5` speaker slots plus one noise slot, and **no count head**.

Three repairs, all of which pushed v0 toward N=1-shaped solutions:

**The clamp was backwards.** `soft_clamp` multiplied by the SI-SDR gradient (~10^(s/20)) peaks
at the clamp point:

| achieved SI-SDR | soft product | hard product |
|---|---|---|
| 0 dB | 0.999 | 1.000 |
| 20 dB | 9.091 | 10.000 |
| **30 dB (= τ)** | **15.811 ← peak** | **0.000** |
| 40 dB | 9.091 | 0.000 |

A 1-speaker mixture is solved by copying the input, so it lives in that band and carried ~16×
the gradient of a hard 5-speaker one. `hard_clamp` is flat above τ: a solved example stops
competing for the optimiser.

**The silence term counted slots, not items.** Leftover slots run 4, 3, 2, 1, 0 for N = 1..5, so
a balanced batch gave the N=1 item 40 % of the second-largest gradient term and the N=5 item
nothing. Now a per-item average.

**The normalisations were also 55 % of the memory.** The readable form of gLN --- subtract,
divide, scale, shift --- retains *three* full-size tensors per call, and there are 49 of them.
Measured at the paper preset and batch 12: **18.05 GiB** of saved activations against a
**14.56 GiB** T4, of which gLN was **10.02 GiB**. The first real run OOMed in TCN block 19 of
24 on the first batch. Because `mean`/`var` reduce over channels *and* time they are
`(B, 1, 1)`, while `gamma`/`beta` are `(1, C, 1)`, so the whole affine folds into `(B, C, 1)`
coefficients applied with one `addcmul` --- exact algebra, one full-size tensor instead of
three. Activations fall to **11.33 GiB** (batch 12 fits with ~1.77 GiB spare, ceiling 13) and
the forward gets *faster*, 30.6 -> 25.1 ms. Gradient checkpointing was measured as the
alternative --- 11.7x less memory for +30 % compute --- and is not needed at this batch.

**The normalisations defeated autocast.** `GlobalLayerNorm` is hand-written because Conv-TasNet
normalises over channels *and* time, which `nn.LayerNorm` does not do — that part is legitimate.
What was not is that autocast's fp32 promotion list covers `layer_norm` and not a hand-rolled
equivalent, so v0's 49 instances opted out 49 times, with `eps = 1e-8` which is *exactly 0.0* in
fp16. v1 computes the statistics in fp32 regardless and uses `MODEL_EPS = 1e-5`.

---

## 4. Data

Full reasoning in [DATA.md](DATA.md). No corpus change: LibriSpeech plus dynamic mixing. The
**settings** matter far more than the corpus, and they are chosen against measured costs:

| setting | v0 | v1 | why |
|---|---|---|---|
| gain jitter | ±5 dB | **±2.5 dB** | ±5 dB cost 9 accuracy points |
| SNR | 0–20 dB | **5–20 dB** | below 5 dB the noise drowns the cue |
| babble noise | 20 % of clips | **removed** | babble *is* 4–8 talkers, so those labels were wrong |

`01_make_frozen_sets.py` **refuses** to build an evaluation set that can draw babble. That is a
correctness constraint, not a tuning knob.

There are also **two epsilons**, and collapsing them is a mistake this project made and caught.
`MODEL_EPS = 1e-5` guards torch divisions and must be fp16-representable; `EPS = 1e-12` guards
float64 DSP. Using 1e-5 in the mixer leaves a residual level of −8.686e-05 dB at N=1 and
−3.884e-05 dB at N=5 — monotone in N, above float32 resolution, and a depth-3 tree reads it at
**39.2 %** against 20 % chance. The audit caught it within a minute. An epsilon is not a free
parameter: too small for the working precision and it is zero, too large and it is signal.

---

## 5. Evaluation — four numbers, never one

`06_evaluate.py` reports all of these, because a single score cannot describe a system that does
two jobs and averaging them hides which half is broken:

1. **Counting accuracy + the full confusion matrix + MAE.** The classes are ordinal; 3→4 is not
   the failure 3→5 is.
2. **P-SI-SNR over the whole test set** — stays defined when the count is wrong, so it cannot be
   gamed by a system that refuses to commit.
3. **SI-SDRi per N, count-correct subset only** — the row comparable to the fixed-N literature,
   which is always told N.
4. **SI-SDRi per N with the true count forced** — separation given a perfect counter. **The gap
   between 3 and 4 is what miscounting costs**, and it says which half to work on next. Being
   able to ask that question is the practical payoff of the two-specialist design.

**Intervals.** `pack.assign_roles` reserves 20 % of each split for babble, so LibriMix's
251/40/40 become **201/32/32** usable target speakers. The 1,500 test mixtures are 1,500 draws
from **32 people**. Report the Wilson interval *and* a speaker-level bootstrap over those 32
clusters, say which is which, and quote the bootstrap.

**One measurement rule.** SI-SDRi is undefined for a clean single-speaker mixture: the mixture
already *is* the target, so "improvement" measures EPS. `usable_si_sdri` drops those references
and returns how many. In v0 that bug turned +1.2 dB into a reported −8.68 dB.

---

## 6. Budget and milestones

| # | milestone | notebook | GPU cost | kill condition |
|---|---|---|---|---|
| M0 | pack the store, freeze dev/test | 00 | **0** | speaker disjointness ≠ 0 → stop |
| M1 | audit + Tier A counting | 01 | **0** | artefact-strict probe above chance → the mitigation is broken |
| M2 | train the counter | 02 | ~3 h | below the Tier A bar → report that, ship M1 |
| M3 | train the separator | 03 | ~10 h | N=1 SI-SDRi poor → plumbing bug, not a model problem |
| M4 | final evaluation + interpretability | 04 | ~0.3 h | — |
| M5 | fixed-N=2 control *(optional)* | 03 | ~2.5 h | the only literature-comparable number |
| M6 | the demo: one file in, one track per talker out | 05 | **0** | — |

**~13.3 GPU-hours of 30/week.** M0 and M1 cost nothing and already produce a complete,
defensible counting project — so if the quota runs out, the project does not.

Cut order: M5, M3, M2. The project stays coherent at every step; it just narrows to counting.

---

## 7. The top risks

| risk | mitigation |
|---|---|
| **Separation lands far below 14.76 dB** | It will — that number is 200 epochs on train-360. Report epochs and training set beside it. Run M5 for a comparable number. |
| **The counter does not beat the 57.8 % tree** | That is a *result*, not a failure, and an interesting one. M1 already shipped. |
| **fp16/fp32 divergence returns** | Preflight refuses to train on it; `tests/test_counter.py` and `tests/test_separator.py` assert it. |
| **An audit that silently does nothing** | Scripts verify their own report file; a structural test fails the build on unreachable code. |
| **4 vCPU cannot feed a T4** with on-the-fly mixing | The counter's batch is ~7× smaller than the separator's (`want="count"`). If the loader still starves the GPU, pre-render one epoch. |

---

## 8. What is deliberately *not* here

- **No shared trunk.** §1 is the reason.
- **No `w_count` on the separator.** It defaults to 0.0 and a separator has no count head.
- **No `soft_clamp` in the training path.** Kept only so v0's behaviour stays reproducible.
- **No babble noise anywhere.** Refused, not discouraged.
- **No AMP by default.** The models are small; the ~30 % speed-up does not justify reopening the
  failure mode that produced 44.9 % and 20.00 % from one checkpoint.
