# Changes from the proposal, the course standards, and limitations

Written to be lifted into the final report. Every number is either measured on the real
LibriSpeech run (`RESULTS.md`) or computed from the code; nothing is estimated.

> Numbers from the notebook 04 re-run at code_version `02d5a91` (2026-09-28, same checkpoints,
> no training) are filled in from [`runs/eval_report.json`](runs/eval_report.json) and
> [`runs/ablation_report.json`](runs/ablation_report.json); tables in
> [RESULTS.md](RESULTS.md) §3, §4b, §4c and §7.

---

## 1. Changes from the proposal

The group submitted two proposal documents: the **written proposal** and the **proposal
presentation**. They disagree about the dataset, so both are listed.

| | Written proposal | Proposal presentation | What was done | Why |
|---|---|---|---|---|
| **Speech data** | Kaggle Libri2Mix 8 kHz | WHAM! (WSJ0 speech + recorded noise) | LibriSpeech via Libri2Mix 8 kHz *min*, repacked into single-speaker sources and remixed on the fly | WHAM!'s public release contains only the noise recordings. Its speech comes from WSJ0, which the LDC licenses for a fee the group does not hold. Libri2Mix was created as the open-licence alternative to wsj0-2mix, and it is what the written proposal already named. |
| **Task** | separate overlapping speech ("crowded channels") | two-speaker separation | **count** the talkers (1–5) **and separate** them | Extension beyond the proposal, made after the first design — one network doing both jobs — counted at chance (20.00 %). The two-specialist design is the result of diagnosing that failure. *Needs the course teacher's approval.* |
| **Noise** | "dynamic background noise profiles" | recorded café/restaurant/bar/park noise at −6 to +3 dB SNR | synthetic white, pink and brown noise at 5–20 dB SNR; 25 % of clips clean | Recorded ambient noise in cafés and restaurants contains background talkers, which makes the speaker-count label ambiguous — the same reason babble noise is refused everywhere in the code. Below 5 dB the counting cue is drowned out. **Consequence: noise cancellation in real environments was not evaluated** (limitation L1). |
| **Feature extraction** | learnable convolutional encoder instead of the STFT | same | kept for the separator (512 learned filters); the counter uses a linear-magnitude STFT, and the Tier A baseline 16 hand-crafted features | — |
| **Hyperparameter tuning** | grid/random search over TCN depth and bottleneck, K-fold | same | grid search × speaker-grouped stratified 5-fold for the Tier A tree only; both networks use fixed configurations (the separator is Conv-TasNet's Table I) | One separator run costs 8.85 GPU-hours of the 30 per week. A 4-configuration, 3-fold search at that size is 12 runs ≈ 106 GPU-hours, three and a half weeks of quota. Limitation L9. |
| **Leakage testing** | speakers and noise profiles isolated across splits | same | speaker-disjoint splits (0 shared speakers on all three pairs); level/duration artefact probe before (44.2 %) and after (19.3 %, chance 20 %) mitigation | Noise is synthesised, so there are no shared recordings to leak. |
| **Benchmarking** | SI-SDR vs "baseline linear filters or basic spectral masking" | same | naive count predictors; the Tier A tree; the unprocessed mixture (0 dB SI-SDRi floor); **oracle IRM/IBM masks** as an upper bound; published Conv-TasNet numbers | No *non-oracle* classical separator (Wiener filter, spectral subtraction) was run. Limitation L15. |
| **Interpretability** | visualise the learned filterbank | filterbank vs gammatone, mask heatmaps, sparsity, **TCN attribution (ablation, Integrated Gradients, receptive field)**, PCA/t-SNE | filterbank frequency analysis (vs mel and uniform); mask overlap, sparsity and entropy against N with within-N controls; **channel ablation**; **receptive field** (analytic, §3); permutation importance for the tree; counter–separator agreement | Integrated Gradients, the gammatone comparison and PCA/t-SNE were not done. Limitation L15. |

Suggested paragraph for the report:

> **Changes from the proposal.** The written proposal specified Libri2Mix (8 kHz); the proposal
> presentation described WHAM!. WHAM!'s public release contains only the noise recordings — its
> speech is generated from WSJ0, which is licensed through the LDC — so we kept LibriSpeech
> speech via Libri2Mix, the open-licence counterpart of wsj0-2mix. We extended the task from
> separation to counting and separating one to five speakers, after a single network trained on
> both jobs counted at chance (20.00 %). Background noise is synthetic (white, pink, brown) at
> 5–20 dB SNR rather than recorded ambient noise at −6 to +3 dB, because recorded café and
> restaurant noise contains background talkers that would corrupt the speaker-count label; as a
> result, noise cancellation in real environments was not evaluated.

---

## 2. The five mandatory standards — where each is met

| # | Course standard | Where it is met | Gap |
|---|---|---|---|
| 1 | Data interpretation & statistical analysis (EDA, correlations, class balance, variance, outliers) | `03_tier_a.py` §2: per-N feature distributions, overlap between classes, correlations between the 16 features; `02_audit_and_baselines.py`. Classes are balanced by construction (300 test clips per N). | no explicit outlier analysis; the spread of every feature is reported, not its extremes |
| 2 | Feature extraction & domain transformation | the learned 512-filter encoder (separator); linear STFT (counter); 16 hand-crafted acoustic features (Tier A) | — |
| 3 | Hyperparameter tuning with a structured search and stratified K-fold | Tier A: 6-configuration grid × stratified 5-fold, grouped by speaker | Only on the tree. Neither network was searched (L9). Folds are grouped by each mixture's first speaker only (L11). |
| 4 | Data-leakage testing | speaker-disjoint splits verified before training; the artefact probe shows the leak (44.2 %) and its removal (19.3 %); every model input is normalised by its own RMS, and no scaler or other statistic is fitted across clips, so nothing computed on one split reaches another | — |
| 5 | Benchmarking & interpretability | naive floors, the Tier A tree (also on the same test clips, with an exact McNemar test), oracle IRM/IBM, the unprocessed mixture, literature; permutation importance (tree), filterbank analysis, mask geometry, channel ablation, receptive field | no SHAP/LIME (permutation importance is the feature-importance metric used); no non-oracle classical separator |

---

## 3. Receptive field of the separator (analytic)

The separator is Conv-TasNet with L = 16 (stride 8 samples), P = 3, X = 8 blocks per repeat with
dilations 1, 2, …, 128, and R = 3 repeats. Each block widens the view by (P − 1) × dilation
encoder frames, so one repeat adds 2 × (1 + 2 + … + 128) = 510 frames.

| after | encoder frames | samples | seconds | share of the 3 s window |
|---|---|---|---|---|
| repeat 1 (blocks 1–8) | 511 | 4,096 | 0.51 s | 17 % |
| repeat 2 (blocks 9–16) | 1,021 | 8,176 | 1.02 s | 34 % |
| repeat 3 (blocks 17–24) | 1,531 | 12,256 | **1.53 s** | **51 %** |

(samples = (frames − 1) × 8 + 16.) The network is non-causal, so each output sample sees about
**±0.77 s** around itself — the 1.53 s the Conv-TasNet paper reports for this configuration.
Every mask decision is therefore made from about half the clip. Strictly, the global layer norm
computes its statistics over the whole clip, so every output depends weakly on all of it, but the
convolutional path, which carries all the local structure, spans 1.53 s. For recordings longer
than 3 s, keeping each talker on the same track across windows is done by an explicit
permutation alignment in `pipeline.separate_long`, not by the network.

---

## 4. Limitations

Each is stated with its evidence and what would remove it. **None of the fixes involving
retraining were carried out.**

### Data and task realism

**L1. Noise is synthetic and the test set is in-distribution.** Training and test noise are
stationary white, pink and brown noise at 5–20 dB SNR; the test mixtures come from the same
generator as training, differing only in the speakers. So 91.1 % and +4.63 dB are in-distribution
numbers. Recorded noise, the proposal's −6 to +3 dB range, reverberation and other corpora were
never evaluated. *Within* the training range the system is flat: counting 89.2–92.7 % and
separation +4.58 to +5.01 dB (matched slots, N = 2..5) across clean, white, pink and brown noise
and the three SNR bands — which is evidence of nothing outside that range.
*Fix:* an out-of-distribution test set — recorded noise (WHAM!/ESC-50 without talkers), low SNR,
room impulse responses, or LibriCount.

**L2. Every talker speaks for the whole clip at nearly the same level.** Each source is
normalised to unit RMS with ±2.5 dB jitter, and the crop picker rejects near-silent crops, so all
N talkers are active across all 3 s (only 0.42 % of test sources are zero-padded). Real
conversation is mostly turn-taking, with partial overlap and level differences of 10 dB or more.
*Fix:* random onsets and durations inside the window, wider gains, results by overlap ratio.

**L3. The count range is closed at 1–5.** There is no "nobody talking" class and no "more than
five" class, so silence or a six-person mixture is forced into 1–5. On long recordings the count
is the average of per-window counts, which estimates how many people talk *at once*, not how many
distinct people speak.

**L4. The label ignores activity.** A talker whose crop is nearly silent still counts. The crop
picker avoids this in almost every clip but falls back to the loudest attempt when no crop passes
its threshold.

### Separation

**L5. Output levels collapse, and choosing slots by loudness fails for N ≥ 2.** SI-SDR ignores
scale, and the silence term pushes spare slots down with nothing pulling real ones up, so every
slot drifts 30–35 dB below the mixture. Inference keeps the N loudest slots, but the headline
SI-SDRi lets the references choose the slots, so **+4.63 dB is an upper bound**. Scoring the
slots the system actually keeps: **+3.65 dB pooled** (+3.40 dB macro over N = 2..5), and at
N = 2 **+3.06 dB instead of +6.15 dB**. The loudest N slots are exactly the talkers in 96 % of
N = 1 clips but only **13 % / 18 % / 28 %** at N = 2 / 3 / 4; the median speaker-vs-spare gap is
−1.16 dB, and in 61 % of N = 1..4 clips a spare is louder than a talker. The demo clip is one of
those (RESULTS §8, corrected). The counter is not the problem: on count-correct clips the
kept-slot score is the same +3.65 dB.
*Fix (retraining):* a mixture-consistency constraint or a level-matching loss term, so that a
slot's level carries information; or select slots by a learned activity score instead of power.
*Partial fix (no retraining, adopted at `bc59305` after a dev check):* keep the N slots whose
least-squares combination, plus the noise slot, best rebuilds the mixture
(`countsep.metrics.consistent_slots`), which ignores output level. It was built after the test set
exposed the loudness failure, so it was adopted only under a rule fixed in advance: a paired
rebuild − loudest CI above zero on the **dev** split. Dev gave **+0.61 dB, 95 % CI [+0.50, +0.72]**
over 1,130 clips, so `rebuild` is now the default in `pipeline.py` and `08_infer.py`
(`--select loudest` keeps the old rule). On test (`99d75fe`) it lifts pooled SI-SDRi from +3.65 to
**+4.06 dB** (+3.40 → +3.90 macro over N = 2..5; count-correct clips +4.06 dB) and the right-slot
rate from 13 / 18 / 28 % to **22 / 37 / 53 %** at N = 2 / 3 / 4, recovering 42 % of the gap to
matched slots; paired on test +0.59 dB [+0.49, +0.68], and end-to-end P-SI-SNR +3.81 → +4.23
dB. Dev shows the same pattern (RESULTS §4d). It is not a full fix: at N = 2 it
reaches +3.63 dB against +6.15 dB with matched slots and still keeps a wrong slot in about three
clips out of four — the demo clip among them (RESULTS §8). The per-slot projection rule,
reported as a second opinion, is worse than loudness (+2.76 dB).

**L6. The training budget is small.** 30 epochs × 1,000 steps (30,000 optimiser steps) on
train-clean-100 (57 h, 201 speakers), against Conv-TasNet's 200 epochs on train-clean-360. At N = 2
this work reaches +6.15 dB against the published +14.76 dB, which is also told N = 2 exactly. The
dataloader ablation shows the separator is step-limited, not data-limited.

**L7. The noise slot was trained with the old soft clamp.** The speaker slots used the corrected
hard clamp; the noise-slot term kept the soft one until after the reported run. It carries weight
0.2 and no reported metric scores the noise slot, but its gradient does reach the shared network.
Fixed in the code; the reported checkpoint is unchanged.

**L8. Long recordings.** `separate_long` aligns each 3 s window to the previous one by
correlation. If a talker is silent throughout an overlap, the alignment has nothing to match and a
track swap can carry through the rest of the file; output levels can also step between windows.

### Methodology

**L9. No neural hyperparameter search, one seed each.** Both networks use fixed configurations
and seed 72, so run-to-run variance is unmeasured.

**L10. The counter's pooling was chosen on a synthetic proxy.** `eigen` pooling beat `meanstd`,
`attentive` and `covariance` on a synthetic speech proxy (74.4 % vs 66.5–67.5 %). On real audio
only `eigen` was trained. It works (91.1 %), but it was never shown to be the best pooling there.

**L11. The Tier A comparison.** The tree's 57.8 % is 5-fold cross-validation on *training*
speakers, and its folds are grouped by each mixture's *first* speaker only, so the other talkers
in a multi-speaker mixture can appear on both sides of a fold. It is therefore not the same
experiment as the counter's test accuracy. Scored on the **same 1,500 test clips** the tree
reaches **59.7 %** [57.2, 62.1] (MAE 0.457) against the counter's 91.1 %; exact McNemar, 519 vs
48 discordant clips, p = 7 × 10⁻¹⁰¹. The cross-validation figure came out 1.9 points *below* the
held-out one, so it was not optimistic. What remains is that the tree saw 7,500 training
mixtures and the counter 2.56 M, so the comparison is between a small model and a large one, not
between two models given equal data.

**L12. A small, reused test pool.** The 1,500 test mixtures come from 32 speakers, and 1,540
distinct utterances fill their 4,500 source slots (one is used 15 times), so clips are not
independent and the Wilson interval [89.5 %, 92.4 %] is somewhat optimistic. The speaker bootstrap
([89.9 %, 92.2 %]) resamples by first speaker only. A 20 % share of speakers was reserved for
babble noise that is never used, which cost 50 training and 8 test speakers.

**L13. Pooled averages weight sources.** An N = 5 clip contributes five values and an N = 1 clip
one, so the pooled +4.63 dB is dominated by N = 4–5. Weighting each N equally gives +5.40 dB
over N = 1..5 and +4.78 dB over N = 2..5 (matched slots); +4.26 / +3.40 dB for the slots the
system keeps. Report the macro row beside the pooled one.

**L14. Confidence is not calibrated.** The counter's softmax is used as a confidence score
(including in the counter–separator agreement analysis) without a calibration check (ECE,
temperature scaling).

### Interpretability and baselines

**L15. What was promised and not done.** Integrated Gradients, the gammatone filterbank comparison
and PCA/t-SNE feature clustering were not run, and there is no non-oracle classical separation
baseline — only oracle IRM/IBM ceilings (+12.46 / +13.13 dB macro over N = 2..5; the separator
reaches 38 % of the IRM with matched slots, 27 % with the slots it keeps). The channel ablation
that was run is a zero-ablation of equal-count frequency bands of one checkpoint; it measures what
the network depends on, not a unique causal role, and "denoising" in it means removing stationary
synthetic noise. Its finding (RESULTS §7): the mask estimator relies overwhelmingly on filters
peaking at 0–500 Hz — ~12× a random-filter control for both denoising and separating — and no
band specialises in one job once each job's overall sensitivity is divided out. The first
printed verdict of `09_filter_ablation.py` compared raw dB and wrongly reported all eight bands
as specialised; corrected in the script, and the saved report re-scored from its own means and
SEs (`09_filter_ablation.py --rescore`), so its verdict now matches.

### Engineering

**L16. Two measurement-code defects, found after the reported runs and fixed.** (a) The
checkpoint logic could overwrite the best checkpoint when a run was resumed. Both reported runs
finished inside one Kaggle session (2 h 30 min and 8 h 51 min of a 12 h limit), and their saved
training histories begin at epoch 1, which a run resumed under the old code could not produce
(it discarded the history on resume). So they were never resumed and their numbers are
unaffected. (b) The fp32/fp16 agreement check compared fp32 with fp32 while AMP was off, so
the "100 % agreement" it reported measured nothing; both models trained and were evaluated in
fp32, so no precision mismatch exists to detect.

---

## 5. Future work, in order of value per GPU-hour

1. Mixture consistency (or a level-matching term) in the separator, then retrain (~9 GPU-h):
   removes L5 at the root.
2. An out-of-distribution test set — recorded noise without talkers, low SNR, reverberation
   (no training): measures L1.
3. Partial-overlap, level-varied training mixtures with activity-based labels, plus N = 0 and
   "> 5" classes (~3 GPU-h for the counter): L2–L4.
4. Three seeds per model and a small hyperparameter search on the reduced preset: L9–L10.
5. Truly speaker-disjoint Tier A folds and repacking without the babble reservation (CPU): L11–L12.
