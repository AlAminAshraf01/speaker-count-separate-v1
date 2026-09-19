# `data/` — the frozen evaluation protocol

This folder holds `recipes_dev.csv` and `recipes_test.csv`. They are **committed to git on
purpose**, and they are the only generated files in this repo that are.

## Why a CSV and not audio

A recipe is one line of text naming which packed utterances to crop, at which sample offsets, at
what gain, with which noise at which SNR, and the final normalisation factor.
`spkcount.mixing.render_recipe()` turns that back into audio **bit for bit**.

So the frozen test set is a few hundred kB of text rather than several GB of WAV — small enough
to version-control, which is what "freeze the test set" actually has to mean. A test set that
lives only in a Kaggle output is not frozen; it is merely inconvenient to regenerate.

## How they get here

`scripts/01_make_frozen_sets.py` writes them once, from a packed store. On Kaggle that happens
in `notebooks/kaggle_00_build_dataset.ipynb`. Download them from the notebook's Output tab and
commit them.

```bash
python scripts/01_make_frozen_sets.py --store /kaggle/working/store --out data \
    --splits dev test --n_list 1 2 3 4 5 --n_per_class 300 \
    --noise_kinds white pink brown --seed 72
```

## Four rules

1. **Generate once.** The script refuses to overwrite an existing file without `--force`.
2. **Never regenerate after you have reported a number.** Different recipes mean a different
   test set, and every result you have already written down becomes incomparable.
3. **Commit both files.** Without them nobody — including you, next month — can reproduce your
   evaluation.
4. **A recipe only means something together with the store it indexes.** `utt_idx` is a row
   number in `<store>/<split>/index.csv`. Rebuilding the store with a different seed, a
   different `--limit`, or a different `--babble_frac` invalidates every recipe. That is the
   main reason to build the store exactly once.

## The mixing settings baked into these files

| setting | value | why |
|---|---|---|
| segment | 3.0 s, RMS-normalised | the anti-leak mitigation — see `docs/DIAGNOSIS.md` §5 |
| gain jitter | ±2.5 dB | ±5 dB was measured to cost 9 accuracy points |
| SNR | 5–20 dB | below 5 dB the noise drowns the cue being counted |
| `p_clean` | 0.25 | a clean condition, reported as its own row |
| noise kinds | white, pink, brown | **never babble** |
| per class | 300 | 1,500 mixtures across N = 1..5, balanced |

> **Babble is refused, not discouraged.** The script exits with an error if you ask for it.
> Babble noise *is* 4–8 real talkers, so a "2 speaker" mixture carrying babble at 5 dB SNR has
> six audible voices and a label that says two. The predecessor project drew roughly 20 % of its
> training mixtures that way. That is not noise augmentation; it is label corruption.

**These settings must match `spkcount.datasets.DEFAULT_MIXING`**, or the frozen set measures a
different task from the one the model trained on. `01_make_frozen_sets.py` prints both so a
mismatch is visible rather than silent.

## Columns

| column | meaning |
|---|---|
| `mix_id` | stable identifier, e.g. `test_n3_00042` |
| `n_src` | number of speakers — the label |
| `utt_idx` | `\|`-joined row indices into that split's `index.csv` |
| `crop_start` | `\|`-joined sample offsets, one per source |
| `gain_db` | `\|`-joined per-source gains, applied after RMS normalisation |
| `noise_kind` | `white` / `pink` / `brown` / `none` |
| `noise_id` | seed that deterministically regenerates that exact noise |
| `snr_db` | speech-to-noise ratio |
| `scale` | the final RMS normalisation factor, stored rather than recomputed |

> **`scale` is a near-perfect predictor of `n_src`, and must never be used as a feature.** It is
> stored so a re-render is exact, nothing more. The mixture's loudness before normalisation
> grows with the number of voices, so `scale` encodes the answer. Reading it would be the
> textbook version of the leak the whole pipeline is built to prevent.

## How many independent things are in here

1,500 test mixtures, but they are drawn from **32 speakers** — `pack.assign_roles` reserves 20 %
of each split for babble, so LibriMix's 40 test speakers become 32 usable targets.

That is why `scripts/05_evaluate.py` reports a **speaker-level bootstrap** interval alongside the
Wilson interval. The Wilson interval treats 1,500 clips as 1,500 independent trials and is too
narrow. Quote the bootstrap one.
