"""Feature invariants: what must survive the anti-leak normalisation, and what must not.

The whole evaluation rests on a claim about these two families: that the artefact features
carry the count *only* because of how the mixture was assembled, and the acoustic features
carry it because of what is in the audio. These tests check that claim mechanically rather
than trusting the docstrings -- the predecessor's docstrings described a clamp that did the
opposite of what it said.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def _talker(rng: np.random.Generator, length: int) -> np.ndarray:
    """One synthetic talker: a harmonic stack gated by syllables with REAL silences.

    The silences are the point, and getting them wrong is a mistake worth recording. An
    earlier version of this helper gated with ``sin(...) > -0.2``, which is never actually
    zero for long and leaves the "talker" sounding continuously. Under that proxy the crest
    factor *rose* with N instead of falling, because summing several dense signals produces
    occasional constructive peaks while there are no silences left to fill in. The monotone
    trend the acoustic baseline depends on only appears when each source is genuinely sparse
    in time -- which real speech is, at roughly 4 syllables per second with gaps between them.
    """
    t = np.arange(length) / 8000.0
    f0 = float(rng.uniform(85.0, 230.0))
    contour = f0 * (1.0 + 0.05 * np.sin(2 * np.pi * 0.8 * t + rng.uniform(0, 6.3)))
    phase = 2 * np.pi * np.cumsum(contour) / 8000.0
    wave = sum(np.sin(h * phase) / h for h in range(1, 22))

    envelope = np.zeros(length)
    pos = 0
    while pos < length:
        on = int(rng.uniform(0.12, 0.45) * 8000)
        off = int(rng.uniform(0.08, 0.35) * 8000)      # a real gap, not a dip
        end = min(length, pos + on)
        if end > pos:
            envelope[pos:end] = np.hanning(max(2, end - pos))[: end - pos] * rng.uniform(0.5, 1.0)
        pos = end + off
    return wave * envelope


def _speechlike(n_src: int, length: int = 24000, seed: int = 0) -> np.ndarray:
    """N summed talkers, each RMS-normalised first, then the sum normalised.

    Mirrors the construction order in :mod:`countsep.mixing`, so the statistics these tests
    assert on are the ones the audit will actually see.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(length)
    for _ in range(n_src):
        src = _talker(rng, length)
        total += src / (np.sqrt(np.mean(src ** 2)) + 1e-12)
    return (total / (np.sqrt(np.mean(total ** 2)) + 1e-12)).astype(np.float64)


def test_acoustic_features_are_scale_invariant() -> None:
    """Every acoustic feature must be unchanged by a gain.

    This is why they survive dividing the mixture by its RMS while the artefact family does
    not -- and it is the property that makes the mitigation safe rather than destructive.
    """
    from countsep.features import acoustic_features

    x = _speechlike(3)
    base = acoustic_features(x)
    for gain in (0.01, 0.5, 4.0, 100.0):
        scaled = acoustic_features(x * gain)
        for key, value in base.items():
            got = scaled[key]
            denom = max(abs(value), 1e-6)
            assert abs(got - value) / denom < 1e-3, (
                f"{key} moved from {value:.6g} to {got:.6g} under a gain of {gain}; "
                f"an acoustic feature that tracks level is a level cue in disguise")


def test_artefact_level_is_not_scale_invariant() -> None:
    """The artefact probe only works because level_db *does* move with gain."""
    from countsep.features import artefact_features

    x = _speechlike(3)
    assert abs(artefact_features(x * 10.0)["level_db"]
               - artefact_features(x)["level_db"] - 20.0) < 1e-6


def test_strict_artefact_features_are_constant_after_mitigation() -> None:
    """After a fixed-length crop divided by its RMS these are the same for every mixture.

    That is what licenses the audit's strongest assertion: its probe over this family has
    nothing to learn and must land at chance, so any reading above it is a broken pipeline
    rather than a subtle leak.
    """
    from countsep.features import ARTEFACT_STRICT, artefact_features

    values = []
    for n_src in (1, 2, 3, 4, 5):
        x = _speechlike(n_src, seed=n_src)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)      # the mitigation
        feats = artefact_features(x)
        values.append([feats[k] for k in ARTEFACT_STRICT])
    arr = np.array(values)
    assert np.allclose(arr, arr[0], atol=1e-6), (
        f"strict artefact features vary across N after mitigation:\n{arr}\n"
        f"They are supposed to be constant (level 0 dB, length SEG_LEN).")


def test_higher_order_statistics_fall_with_speaker_count() -> None:
    """The central-limit argument, checked rather than asserted.

    Summing more independent sparse sources moves the mixture toward Gaussian, so crest
    factor and kurtosis fall. This is the cue that survives the mitigation, and if it did not
    hold, the acoustic baseline would have nothing to read.
    """
    from countsep.features import acoustic_features

    crest, kurt = [], []
    for n_src in (1, 2, 3, 4, 5):
        # average several draws: a single draw is noisy, the trend is what matters
        rows = [acoustic_features(_speechlike(n_src, seed=100 * n_src + s)) for s in range(6)]
        crest.append(float(np.mean([r["crest"] for r in rows])))
        kurt.append(float(np.mean([r["kurtosis_x"] for r in rows])))
    assert crest[0] > crest[-1], f"crest factor did not fall with N: {crest}"
    assert kurt[0] > kurt[-1], f"kurtosis did not fall with N: {kurt}"


def test_features_never_emit_nan_into_the_matrix() -> None:
    """Kurtosis of digital silence is NaN, and one NaN row takes down a whole CV fold."""
    from countsep.features import ALL_FEATURES, all_features, features_to_matrix

    rows = [all_features(np.zeros(8000)),
            all_features(_speechlike(2)),
            all_features(np.full(8000, 1e-30))]
    matrix, keys = features_to_matrix(rows, ALL_FEATURES)
    assert np.isfinite(matrix).all(), "features_to_matrix let a non-finite value through"
    assert matrix.shape == (3, len(ALL_FEATURES))
    assert list(keys) == list(ALL_FEATURES)


def test_feature_groups_are_disjoint_and_named() -> None:
    from countsep.features import ACOUSTIC, ALL_FEATURES, ARTEFACT, ARTEFACT_STRICT, feature_group

    assert not (set(ARTEFACT) & set(ACOUSTIC)), "a feature is in both families"
    assert set(ARTEFACT_STRICT) <= set(ARTEFACT), "strict artefact must be a subset"
    assert set(ALL_FEATURES) == set(ARTEFACT) | set(ACOUSTIC)
    assert feature_group("acoustic") == ACOUSTIC
    try:
        feature_group("nonsense")
    except KeyError:
        return
    raise AssertionError("feature_group accepted an unknown group instead of raising")


if __name__ == "__main__":
    sys.exit(run_checks({
        "acoustic features are scale-invariant": test_acoustic_features_are_scale_invariant,
        "artefact level tracks gain": test_artefact_level_is_not_scale_invariant,
        "strict artefact is constant after mitigation": test_strict_artefact_features_are_constant_after_mitigation,
        "higher-order statistics fall with N": test_higher_order_statistics_fall_with_speaker_count,
        "no NaN reaches the feature matrix": test_features_never_emit_nan_into_the_matrix,
        "feature groups are disjoint and named": test_feature_groups_are_disjoint_and_named,
    }))
