"""Hand-crafted acoustic features for speaker counting, and the level-only artefact probe.

Two feature families, kept deliberately separate, because they answer different questions.

``ARTEFACT`` -- mixture level, peak level and length. These are cues that exist only because
of how the mixture was *assembled*, not because of who is talking in it. If a classifier can
read N off these, the evaluation is void: the model would be scoring the recipe, not the
audio. The mitigation is structural (fixed-length crop, divided by its RMS), and the point of
this family is to *prove* the mitigation works rather than to assert it.

``ACOUSTIC`` -- crest factor, kurtosis, sparsity, spectral flatness and entropy, modulation
depth. These are legitimate evidence. Summing N sparse speech signals moves the mixture toward
Gaussian (central limit), so the higher-order statistics fall monotonically with N, and the
time-frequency representation becomes less sparse as W-disjoint orthogonality degrades
(Rickard: ~0.96 -> 0.92 -> 0.88 for N = 2, 3, 4). A classifier reading *these* is the task
working, not a leak.

Measured on the old project's own mixing code at production settings (see docs/DIAGNOSIS.md
section 5): ARTEFACT scores 21.9 % against 20 % chance -- at chance, so the mitigation holds --
while ACOUSTIC scores 61.4 %, rising to 87.2 % on clean mixtures with no gain jitter. On a
speaker-disjoint split ACOUSTIC + gradient boosting reaches 69.3 %.

That 69.3 % is not a footnote. It is the bar. The joint model this project replaces scored
20.00 % in fp32 -- 49 points *below* a decision-tree ensemble over sixteen numbers. Any neural
counter here that does not clearly beat the ACOUSTIC baseline has bought nothing, and
``scripts/01_baselines.py`` exists to make that comparison unavoidable rather than optional.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal as sps
from scipy import stats

from .constants import EPS, SR

ARTEFACT: tuple[str, ...] = ("level_db", "peak_db", "n_samples")
"""Assembly artefacts, for probing audio **before** mitigation.

``peak_db`` belongs here only while the mixture still carries its original level. Once the
mixture has been divided by its RMS, ``peak_db`` is ``20·log10(max|x| / rms)`` -- which *is*
the crest factor, a legitimate acoustic cue. Probing this family on mitigated audio therefore
scores above chance for an honest reason and would be misread as a surviving leak. Use
:data:`ARTEFACT_STRICT` for that direction.
"""

ARTEFACT_STRICT: tuple[str, ...] = ("level_db", "n_samples")
"""The two cues that are **provably constant** after a fixed-length crop divided by its RMS.

``level_db`` is 0 and ``n_samples`` is ``SEG_LEN``, for every mixture at every N, by
construction. A probe over this family must therefore land at chance -- not "near" chance, at
it -- and any reading above it means the mitigation was not applied, the crop length varies,
or the features were computed on the wrong signal. That makes it a genuine assertion about the
pipeline rather than a statistic.
"""

ACOUSTIC: tuple[str, ...] = (
    "crest", "kurtosis_x", "kurtosis_env", "skew_absx",
    "env_sparsity", "env_p10_p90",
    "flatness_mean", "flatness_std", "flatness_p90",
    "entropy_mean", "entropy_std",
    "tf_top1pct", "tf_top5pct",
    "modulation_depth", "silence_frac", "hoyer",
)
"""Legitimate acoustic evidence: higher-order statistics, sparsity and spectral shape."""

ALL_FEATURES: tuple[str, ...] = ARTEFACT + ACOUSTIC


def _stft_mag(x: np.ndarray, sr: int, nperseg: int, noverlap: int) -> np.ndarray:
    """Linear-magnitude STFT, normalised to unit mean.

    Linear, not log or mel: Stoter et al. measured that linear magnitude beats MEL40,
    LOGSTFT and MFCC20 for counting, because telling a one-source bin from a multi-source bin
    needs frequency resolution rather than perceptual warping.

    The unit-mean rescale is not cosmetic. Every spectral feature below is a *ratio* and so is
    scale-invariant in exact arithmetic -- but they are all floored by a constant epsilon, and
    a constant floor under a signal that has been scaled down is relatively larger, which
    quietly makes the feature track level. Measured before this fix: ``flatness_mean`` moved
    0.25 % under a gain of 0.01, enough to fail the scale-invariance test in
    ``tests/test_features.py``. Normalising the spectrum first pins the epsilon to a fixed
    place relative to the data, and the features become invariant to machine precision.

    This is the same class of mistake as the predecessor's ``eps=1e-8`` inside a hand-written
    norm running in fp16, where the epsilon was not small -- it was exactly zero.
    """
    _f, _t, spec = sps.stft(x, sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(spec)
    return mag / (float(mag.mean()) + 1e-20) + 1e-10


def artefact_features(x: np.ndarray) -> dict[str, float]:
    """The three cues the mitigation is supposed to have destroyed.

    After a fixed-length crop divided by its RMS, ``level_db`` is 0 and ``n_samples`` is
    constant *by construction* -- so a probe over this family lands at chance, and that is the
    result we want. Feeding it unmitigated audio is what makes the probe informative.
    """
    x = np.asarray(x, dtype=np.float64)
    # max(), not +EPS. An epsilon is meant to keep log10 away from zero, not to shift every
    # normal value: with EPS = 1e-5, `rms + EPS` makes a 10x gain read as 19.99992 dB instead
    # of 20, which is small until something asserts exactness. max() is exact everywhere the
    # signal is above the floor and still safe at it.
    level = max(float(np.sqrt(np.mean(np.square(x)))), EPS)
    peak = max(float(np.max(np.abs(x))), EPS)
    return {
        "level_db": float(20.0 * np.log10(level)),
        "peak_db": float(20.0 * np.log10(peak)),
        "n_samples": float(x.size),
    }


def acoustic_features(x: np.ndarray, sr: int = SR, *, nperseg: int = 256,
                      noverlap: int = 192) -> dict[str, float]:
    """Higher-order statistics, envelope sparsity and spectral shape.

    Every one of these is invariant to the RMS normalisation -- they are ratios, standardised
    moments or normalised distributions -- which is exactly why they survive the mitigation
    that kills the artefact family.
    """
    x = np.asarray(x, dtype=np.float64)
    level = max(float(np.sqrt(np.mean(np.square(x)))), EPS)

    env = np.abs(sps.hilbert(x))
    env_mean = float(np.mean(env)) + EPS
    env_rms = float(np.sqrt(np.mean(np.square(env)))) + EPS

    spec = _stft_mag(x, sr, nperseg, noverlap)
    flat = np.exp(np.mean(np.log(spec), axis=0)) / (np.mean(spec, axis=0) + EPS)
    prob = spec / (spec.sum(axis=0, keepdims=True) + EPS)
    entropy = -(prob * np.log(prob + EPS)).sum(axis=0)

    power = np.square(spec)
    power = power / (power.sum() + EPS)
    ranked = np.sort(power.ravel())[::-1]

    return {
        # fourth-moment family: falls monotonically with N as the sum tends to Gaussian
        "crest": float(np.max(np.abs(x)) / level),
        "kurtosis_x": float(stats.kurtosis(x)),
        "kurtosis_env": float(stats.kurtosis(env)),
        "skew_absx": float(stats.skew(np.abs(x))),
        # envelope sparsity: more talkers means fewer shared silences
        "env_sparsity": float(env_mean / env_rms),
        "env_p10_p90": float(np.percentile(env, 10) / (np.percentile(env, 90) + EPS)),
        # spectral shape
        "flatness_mean": float(flat.mean()),
        "flatness_std": float(flat.std()),
        "flatness_p90": float(np.percentile(flat, 90)),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        # time-frequency sparsity: the W-disjoint orthogonality proxy
        "tf_top1pct": float(ranked[: max(1, int(0.01 * ranked.size))].sum()),
        "tf_top5pct": float(ranked[: max(1, int(0.05 * ranked.size))].sum()),
        "modulation_depth": float(np.mean(np.abs(np.diff(env))) / env_mean),
        "silence_frac": float((env < 0.1 * env_mean).mean()),
        "hoyer": float(np.log((np.linalg.norm(spec, 1) + EPS)
                              / (np.linalg.norm(spec, 2) + EPS))),
    }


def all_features(x: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Both families in one dict, keyed by the names in :data:`ALL_FEATURES`."""
    out = artefact_features(x)
    out.update(acoustic_features(x, sr))
    return out


def features_to_matrix(rows: Sequence[dict[str, float]],
                       keys: Sequence[str] = ALL_FEATURES) -> tuple[np.ndarray, list[str]]:
    """Stack feature dicts into ``(n_rows, n_keys)``, with non-finite values zeroed.

    Kurtosis of a digital-silence crop is NaN, and one NaN row would otherwise take a whole
    fold down. Zeroing is honest here because the matrix is standardised downstream anyway.
    """
    keys = list(keys)
    mat = np.zeros((len(rows), len(keys)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, key in enumerate(keys):
            mat[i, j] = float(row.get(key, 0.0))
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0), keys


def feature_group(name: str) -> tuple[str, ...]:
    """Look up a named feature family. Raises rather than returning an empty tuple."""
    groups = {"artefact": ARTEFACT, "artefact_strict": ARTEFACT_STRICT,
              "acoustic": ACOUSTIC, "everything": ALL_FEATURES}
    if name not in groups:
        raise KeyError(f"unknown feature group {name!r}; have {sorted(groups)}")
    return groups[name]
