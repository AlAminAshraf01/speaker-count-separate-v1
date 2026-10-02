# Would a different dataset be more effective?

You asked this explicitly, so here is the direct answer first and the reasoning after.

> **No — do not switch corpora. Keep LibriSpeech and keep mixing on the fly.** The single
> highest-value data change is *more speakers from the same corpus* (add `train-clean-360`:
> **201** → ~940 usable target speakers for ~21 GB of read-only Kaggle input). The second is **one honest
> cross-corpus test set** built from VCTK, which is the number the report currently lacks
> entirely. Everything else — a different training corpus, LibriMix, a real meeting corpus —
> is either impossible on a free-tier budget, licence-blocked, or structurally unable to
> supply the labels this task needs.

The instinct behind the question is right, though, and worth naming: the counter failed, so
maybe the data was wrong. It wasn't. `docs/DIAGNOSIS.md` shows a decision tree over sixteen
hand-crafted numbers reaching 57.8 % on exactly this data while the 5.3 M-parameter network
reached 20.0 %. **The information was in the data the whole time.** Changing corpora would have
changed nothing and cost a week.

---

## 1. Why the corpus is not the bottleneck

| what changed | measured 5-class accuracy |
|---|---|
| hand-crafted features, production mixing settings | 61.4 % |
| …with no additive noise | 78.2 % |
| …and no per-source gain jitter | 87.2 % |
| the 5.3 M joint network, same data, fp32 | 20.0 % |

The gap between 87.2 % and 61.4 % is entirely **mixing settings** — noise and gain jitter,
both of which we choose. The gap between 61.4 % and 20.0 % is entirely **the model and its
training**. Neither gap is a corpus problem, and no new corpus closes either one.

---

## 2. What a counting corpus actually has to supply

Four requirements, and they eliminate almost everything:

1. **A ground-truth count of *concurrent* talkers.** Not a diarisation RTTM you could
   post-process, not a speaker inventory — the number of people talking *at once*, per clip.
2. **Enough distinct speakers that the model cannot memorise timbres.** With 201 usable target
   speakers and mixtures of 5, a counter can partly solve the task by recognising voices.
3. **Speaker-disjoint train/dev/test splits**, enforceable and checkable.
4. **A download that fits a free-tier budget** — 30 GPU-h/week, 12 h sessions, 20 GB working
   disk, 4 vCPU.

Requirement 1 is the killer. **Real conversation is sparse**: people mostly take turns. A
meeting corpus with five participants almost never contains five *simultaneous* talkers, so it
cannot supply N=4 or N=5 labels at all — the labels it can supply are 0, 1 and occasionally 2.
This is why the synthetic-mixing approach is not a compromise here; it is the only way to get a
balanced 1–5 dataset.

---

## 3. The options, assessed

### Keep: LibriSpeech + dynamic mixing — **the recommendation**

Already packed as 8 kHz int16 memmaps with speaker-disjoint splits, ~4 GB. Mixing on the fly
gives unlimited unique mixtures at zero disk cost and full control over N, SNR, gain and
duration. Nothing else on this list offers that control, and control is what
`docs/DIAGNOSIS.md` §5 shows is worth 26 accuracy points.

### Add: `train-clean-360` — **highest value per GB**

**201** → ~940 usable *target* speakers for ~21 GB attached as read-only Kaggle input (never
touching the 20 GB working disk). If upload bandwidth binds, take 300–400 speakers for ~8 GB:
for counting, **timbre diversity matters far more than hours per speaker**. This is the one
change that addresses a real weakness — with 201 speakers a counter can partially memorise
voices.

> **Mind the speaker arithmetic — it is not what the corpus advertises.** `pack.assign_roles`
> reserves `babble_frac = 0.2` of every split's speakers for babble noise only, so they never
> appear as targets. LibriMix's 251/40/40 become **201 / 32 / 32** usable target speakers for
> train-100 / dev / test. The test set's 1,500 mixtures are therefore drawn from **32
> speakers**, and that — not 1,500 — is the number of independent units. A Wilson interval on
> 1,500 mixtures (±2.5 points near 50 %) is an *underestimate* of the real uncertainty; a
> speaker-level bootstrap over 32 clusters is the honest interval and it is considerably wider.
> Report both, and say which one is which.

### Add: VCTK as a cross-corpus **test set only** — **the missing result**

110 speakers, 44 h, CC BY 4.0, already on Kaggle (`destynova/vctk-corpus-092`). Different
microphone chain, different accents, different speaking style. Build a few thousand 3 s
mixtures with this project's own mixing code: ~1–2 GB, zero upload, minutes of compute.

This is the number the report currently has no version of: *does a LibriSpeech-trained counter
work on anything that is not LibriSpeech?* Expect it to drop. **Report the drop** — an honest
cross-corpus number is worth more than a flattering in-corpus one, and every examiner knows it.

### Add: MUSAN noise + simulated reverb — realism, with one hard rule

MUSAN (CC BY 4.0, Kaggle `nhattruongdev/musan-noise`) for real noise, and `pyroomacoustics`
for RIRs at zero disk cost, holding out OpenSLR 28's *measured* RIRs (1.3 GB) as the reverb
**eval** condition so train and test reverb do not come from the same generator.

> **The hard rule: never put speech in the noise.** Use MUSAN's `noise` and `music` partitions
> and never its `speech` partition. This is not hypothetical — the predecessor drew ~20 % of
> its training mixtures with **babble** noise, which is 4–8 real talkers, at SNRs down to 0 dB.
> Those mixtures' count labels are factually wrong: a "2 speaker" clip with six audible talkers
> in the background is teaching the model that six people sounds like two.

### Report but never train on: LibriCount

832 MB, CC BY 4.0, Zenodo DOI `10.5281/zenodo.1216072`, classes 0–10 (2,860 clips for 1–5),
labels from WebRTC VAD, clean/anechoic/English/0 dB.

**LibriCount is built from LibriSpeech `test-clean`** — the same 40 speakers our own test split
is drawn from (32 of them as targets, 8 held out for babble). So:

- It is **not** independent evidence of generalisation. It is the same speakers, the same
  LibriVox read-speech style, the same recording chain.
- Training or tuning on it would silently contaminate our own test split.
- CountNet's headline MAE 0.27 is over **11 classes including a 0-speaker class** and is **not
  comparable** to a 1–5 MAE. Quoting it side by side with ours would be misleading.

It still earns its place as a *reported external benchmark* with those caveats written down,
because several published systems report on it. Enforce the rule in code — a manifest check
that no LibriCount path can enter a training or validation loader — not in a comment.

### Rejected, with reasons

| corpus | why not |
|---|---|
| **LibriMix / Libri3Mix** | Generation needs ~430 GB (Libri2Mix) / ~332 GB (Libri3Mix) plus 30 GB LibriSpeech and 50 GB WHAM. Our dynamic mixing already gives arbitrary N at ~0 GB, which is strictly better. |
| **AMI, CHiME-5/6, AliMeeting, VoxConverse, DIHARD** | All fail requirement 1: real conversation is sparse, so no N=4–5 concurrent labels exist. Also 108–120 GB, and DIHARD is licence-blocked. |
| **WSJ0-2mix/3mix** | LDC-licensed (paid). ~100 speakers and a 5 k vocabulary against LibriMix's ~1,000 and 60 k. Paying for a worse corpus. |
| **TIMIT** | Licence-blocked. |
| **VoxCeleb1/2** | Recorded in the wild with existing background speech — the count label would be unknowable. |
| **AISHELL-1** | Viable as a *cross-language* stretch (400 speakers, 15 GB, Apache 2.0) but strictly a stretch goal after VCTK. |

---

## 4. The mixing settings matter more than the corpus

This is the part worth internalising, because it is where the measured evidence is strongest.

| setting | measured cost | decision |
|---|---|---|
| additive noise at 0–20 dB SNR, 80 % of clips | **−17 points** | keep noise, but **raise the floor to 5 dB** and report the clean number separately |
| per-source gain jitter ±5 dB | **−9 points** | **narrow to ±2.5 dB** |
| babble noise (speech in the noise) | −3 points, **and corrupts the label** | **remove entirely** |

A counter evaluated only on clean, level-matched mixtures is a counter with an inflated number.
A counter trained at 0 dB SNR against babble is a counter trained on wrong labels. The settings
above are the compromise, and — this is the point — **every one of them is reported as a
separate condition** rather than folded into one headline figure. The accuracy-vs-SNR curve is
a result, not a caveat.

---

## 5. What to do, in order

1. Run `scripts/02_audit_and_baselines.py` on the existing store. CPU, minutes, zero GPU quota.
   It prints the bar the neural counter must beat and proves the splits are disjoint.
2. Retire babble from the noise bank; narrow gain jitter to ±2.5 dB; raise the SNR floor to 5 dB.
3. Train on what already exists. Do not add a single GB until a model beats the bar.
4. Then add `train-clean-360` and measure whether more speakers helps. That is an ablation, and
   an ablation is a report section.
5. Build the VCTK cross-corpus test set and report the drop.
6. Report LibriCount with its caveats, if time allows. It is the first thing to cut.

Step 3 is the one people skip. The predecessor added speakers, added noise kinds and added
speaker counts before it had ever measured a working counter — and then could not tell which of
those decisions had cost it anything.
