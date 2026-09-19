"""Naive predictors and the feature probes that gate every neural number in this project.

Three things live here, and they are all *floors* rather than models.

1. :func:`naive_scores` -- what you get for free. Majority class and uniform random are both
   exactly 1/K on a balanced K-class set, but the MAE differs sharply between them and the
   literature precedent (Stoter et al.) is to quote a constant mid-range predictor, so all
   three are reported.

2. :func:`probe` -- fit a small model to one named feature family and cross-validate it. Run
   over ``ARTEFACT`` this is the leakage audit; run over ``ACOUSTIC`` it is the benchmark the
   neural counter has to beat.

3. :func:`interpret_probe` -- turn a probe's accuracy into a verdict, because "above chance"
   means opposite things for the two families. An artefact probe above chance is a *leak* and
   invalidates the evaluation. An acoustic probe above chance is the *task working*: five
   overlapping talkers really do produce a flatter, less peaky signal than one, and calling
   that contamination would invite the report to describe legitimate evidence as a bug.

Deliberately dependency-light: scikit-learn only, no torch, so the audit runs on a CPU
notebook before a single GPU hour is spent.

**The failure this module exists to prevent.** The predecessor project's equivalent script
was structurally dead for months -- a function definition landed inside ``main()``, every
probe became unreachable code after an unconditional ``return``, and the script exited 0 while
writing nothing. Two headline numbers were quoted from it that it had never produced. So
everything here returns data that a caller must consume, :func:`probe` raises on a degenerate
input rather than returning a number that looks fine, and the calling script verifies its own
report file exists before exiting. See ``docs/DIAGNOSIS.md`` section 4.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .features import ALL_FEATURES, feature_group, features_to_matrix


def naive_scores(y: np.ndarray, n_classes: int) -> dict[str, dict[str, float]]:
    """Accuracy and MAE of the three predictors that require no learning at all."""
    y = np.asarray(y).astype(int)
    if y.size == 0:
        raise ValueError("naive_scores got an empty label array")
    classes = np.arange(n_classes)

    counts = np.bincount(y, minlength=n_classes)
    majority = int(np.argmax(counts))
    mid = int(round(float(classes.mean())))

    out = {
        "majority_class": {
            "prediction": float(majority),
            "accuracy": float((y == majority).mean()),
            "mae": float(np.abs(y - majority).mean()),
        },
        "constant_midrange": {
            "prediction": float(mid),
            "accuracy": float((y == mid).mean()),
            "mae": float(np.abs(y - mid).mean()),
        },
        "uniform_random": {
            "prediction": float("nan"),
            # expected values, computed rather than sampled, so the number is stable
            "accuracy": float(1.0 / n_classes),
            "mae": float(np.mean([np.abs(y - c).mean() for c in classes])),
        },
    }
    return out


def probe(rows: Sequence[dict[str, float]], y: np.ndarray, group: str = "everything", *,
          n_folds: int = 5, max_depth: int | None = 3, seed: int = 72,
          model: str = "tree") -> dict:
    """Cross-validate a classifier over one feature family.

    ``model='tree'`` is a depth-limited decision tree -- deliberately weak, because the
    artefact probe should answer "is the cue *there at all*", and a shallow tree that finds it
    proves the cue is trivially available. ``model='gbm'`` is gradient boosting, used for the
    acoustic family where the question is instead "how high is the bar for the neural model".

    Raises on a degenerate input rather than quietly returning a plausible-looking number.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    y = np.asarray(y).astype(int)
    if len(rows) != y.size:
        raise ValueError(f"{len(rows)} feature rows but {y.size} labels")
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError(f"probe needs at least 2 classes, got {classes.tolist()}")
    if counts.min() < n_folds:
        raise ValueError(f"class {int(classes[np.argmin(counts)])} has {int(counts.min())} "
                         f"examples but n_folds={n_folds}; collect more mixtures per class")

    keys = feature_group(group)
    matrix, used = features_to_matrix(rows, keys)

    if model == "tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    elif model == "gbm":
        clf = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
    else:
        raise ValueError(f"unknown model {model!r}; use 'tree' or 'gbm'")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, matrix, y, cv=cv, scoring="accuracy")

    return {
        "group": group,
        "features": list(used),
        "model": model,
        "n": int(y.size),
        "n_classes": int(classes.size),
        "chance": float(1.0 / classes.size),
        "accuracy": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "folds": [float(s) for s in scores],
    }


def wilson_interval(accuracy: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % interval for a proportion.

    Quoted on every accuracy in this project. At n = 1500 an accuracy near 50 % resolves only
    to about +-2.5 points, and a per-class recall at n = 300 only to about +-5.7 -- so a two-
    or three-point difference between two variants is not a result, and saying so in the
    report is cheaper than defending it in a viva.
    """
    if n <= 0:
        raise ValueError("wilson_interval needs n > 0")
    p, z2 = float(accuracy), z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def interpret_probe(group: str, accuracy: float, chance: float,
                    tolerance: float = 1.5) -> tuple[str, str]:
    """Turn a probe accuracy into ``(verdict, explanation)``.

    The two families mean opposite things, so the same number gets opposite readings.
    """
    above = accuracy > chance * tolerance
    if group == "artefact_strict":
        if above:
            return ("BROKEN", (
                f"the strict artefact family reaches {accuracy:.1%} against {chance:.1%} "
                "chance, but after a fixed-length crop divided by its RMS its two features "
                "are constant by construction (level 0 dB, length SEG_LEN). Above chance here "
                "does not mean a subtle leak -- it means the mitigation did not run, or the "
                "crop length varies, or these features were computed on the wrong signal. "
                "Fix the pipeline; do not tune around this."))
        return ("OK", (
            f"the strict artefact family scores {accuracy:.1%} against {chance:.1%} chance, "
            "as it must: after mitigation its features are constant, so there is nothing to "
            "learn. The fixed-length RMS-normalised crop is doing its job."))
    if group == "artefact":
        if above:
            return ("LEAK", (
                f"the artefact family alone reaches {accuracy:.1%} against {chance:.1%} "
                "chance. These features describe how the mixture was assembled, not who is "
                "talking, so the counter can score without modelling speech. Every downstream "
                "number is void until the mitigation (fixed-length crop divided by its RMS) "
                "is applied and this probe returns to chance."))
        return ("OK", (
            f"the artefact family scores {accuracy:.1%} against {chance:.1%} chance, so the "
            "level and duration cues are gone and the evaluation is sound."))
    if group == "acoustic":
        if above:
            return ("SIGNAL", (
                f"the acoustic family reaches {accuracy:.1%} against {chance:.1%} chance. "
                "This is the task working, not a leak: summing N sparse speech signals tends "
                "toward Gaussian, so crest factor and kurtosis fall monotonically with N. "
                "This number is the bar the neural counter must clear."))
        return ("WEAK", (
            f"the acoustic family only reaches {accuracy:.1%} against {chance:.1%} chance. "
            "Either the mixing settings have destroyed the signal (additive noise and "
            "per-source gain jitter are the usual culprits) or there are too few mixtures."))
    return ("MIXED", f"{group} scores {accuracy:.1%} against {chance:.1%} chance; this family "
                     "mixes artefact and acoustic cues, so read the two separately.")


def speaker_disjointness(splits: dict[str, Sequence[str]]) -> dict:
    """Pairwise speaker overlap between named splits. Every count must be zero.

    The predecessor's audit for this was inside the dead code, so it never ran on a real
    store, and the only passing test exercised a synthetic corpus whose generator makes the
    splits disjoint by construction. This returns the overlap so a caller can assert on it.
    """
    names = sorted(splits)
    sets = {k: set(str(s) for s in v) for k, v in splits.items()}
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(sets[a] & sets[b])
            pairs.append({"a": a, "b": b, "n_shared": len(shared),
                          "shared": shared[:20]})
    return {
        "sizes": {k: len(v) for k, v in sets.items()},
        "pairs": pairs,
        "ok": all(p["n_shared"] == 0 for p in pairs),
    }


__all__ = ["ALL_FEATURES", "naive_scores", "probe", "wilson_interval",
           "interpret_probe", "speaker_disjointness"]
