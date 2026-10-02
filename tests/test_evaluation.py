"""The evaluation additions: slot selection, the oracle-mask ceiling, and the paired test.

``usable_si_sdri`` lets the references choose which slots to score; inference keeps the
loudest. ``slot_selection`` is what measures the difference, so it gets a test that knows the
right answer in both directions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def _slots(speaker_db: float, spare_db: float, seed: int = 0):
    """Two real speakers in slots 3 and 1 at ``speaker_db``, three spares at ``spare_db``."""
    rng = np.random.default_rng(seed)
    T = 8000
    refs = np.zeros((5, T))
    refs[:2] = rng.standard_normal((2, T))
    mix = refs[:2].sum(0)
    est = rng.standard_normal((5, T))
    est /= np.sqrt((est ** 2).mean(-1, keepdims=True))
    level = lambda db: 10.0 ** (db / 20.0) * np.sqrt((mix ** 2).mean())  # noqa: E731
    est *= level(spare_db)
    est[3] = refs[0] / np.sqrt((refs[0] ** 2).mean()) * level(speaker_db)
    est[1] = refs[1] / np.sqrt((refs[1] ** 2).mean()) * level(speaker_db)
    return est, refs, mix


def test_slot_selection_knows_the_right_answer() -> None:
    from countsep.metrics import loudest_slots, slot_selection

    est, refs, mix = _slots(speaker_db=-30.0, spare_db=-32.0)
    good = slot_selection(est, refs, mix, 2)
    assert good["correct"], "the loudest two ARE the speaker slots, but selection said no"
    assert abs(good["gap_db"] - 2.0) < 1e-6, f"gap should be +2 dB, got {good['gap_db']}"
    assert abs(good["kept_db"] + 30.0) < 1e-6, f"kept level should be -30 dB, {good['kept_db']}"
    assert sorted(loudest_slots(est, 2).tolist()) == [1, 3]

    est, refs, mix = _slots(speaker_db=-33.0, spare_db=-31.0)
    bad = slot_selection(est, refs, mix, 2)
    assert not bad["correct"], "spares are louder than the speakers, but selection said yes"
    assert bad["gap_db"] < 0, f"a losing selection must have a negative gap, {bad['gap_db']}"

    full = slot_selection(est, np.vstack([refs[:2], est[[0, 2, 4]]]), mix, 5)
    assert full["correct"] and full["gap_db"] is None, "N=5 keeps every slot: no gap exists"


def test_oracle_masks_recover_spectrally_disjoint_sources() -> None:
    """Two tones an octave apart: an ideal mask should separate them almost perfectly."""
    from countsep.baselines import ideal_mask_estimates
    from countsep.metrics import si_sdr

    t = np.arange(24000) / 8000.0
    sources = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 1760 * t)])
    noise = 0.01 * np.random.default_rng(0).standard_normal(t.size)
    mix = sources.sum(0) + noise
    for mode in ("irm", "ibm"):
        est = ideal_mask_estimates(mix, sources, noise, mode=mode)
        assert est.shape == sources.shape
        scores = si_sdr(est, sources)
        assert float(scores.min()) > 20.0, f"{mode} failed on disjoint tones: {scores}"


def test_mcnemar_counts_only_discordant_items() -> None:
    from countsep.baselines import mcnemar_exact

    a = [True] * 30 + [False] * 5 + [True] * 10
    b = [True] * 30 + [False] * 5 + [False] * 10
    res = mcnemar_exact(a, b)
    assert res["only_a_correct"] == 10 and res["only_b_correct"] == 0
    assert res["p_value"] < 0.01, f"10 vs 0 discordant should be significant: {res}"
    assert mcnemar_exact(a, a)["p_value"] == 1.0, "identical classifiers must give p = 1"


def test_consistent_slots_find_talkers_that_loudness_misses() -> None:
    """The measured failure: spares LOUDER than the talkers. Loudness keeps the spares; the
    mixture-rebuild rule must still find slots 1 and 3, with or without a noise slot, at any
    output level."""
    from countsep.metrics import consistent_slots, loudest_slots, projection_slots

    est, refs, mix = _slots(speaker_db=-34.0, spare_db=-31.0)
    assert set(loudest_slots(est, 2).tolist()) != {1, 3}, "fixture must fool loudness"
    assert set(consistent_slots(est, mix, 2).tolist()) == {1, 3}, "rebuild missed the talkers"
    assert set(projection_slots(est, mix, 2).tolist()) == {1, 3}, "projection missed them"

    rng = np.random.default_rng(1)
    noise = 0.3 * rng.standard_normal(mix.shape)
    est[1] *= 1e-3                                  # level must not matter
    picked = consistent_slots(est, mix + noise, 2, noise_est=noise)
    assert set(picked.tolist()) == {1, 3}, f"with a noise slot and a rescaled talker: {picked}"
    assert consistent_slots(est, mix, 0).size == 0
    assert set(consistent_slots(est, mix, 5).tolist()) == set(range(5))


def test_mixture_is_sources_plus_noise_and_recipes_rerender_exactly() -> None:
    """The mixing invariant, and the frozen-set promise: a CSV round trip renders the same audio."""
    import tempfile

    from conftest import store_and_bank
    from countsep.mixing import read_recipes, render_recipe, sample_recipe, write_recipes

    store, bank = store_and_bank("test")
    rng = np.random.default_rng(3)
    recipes = [sample_recipe(store, bank, n, rng, seg_len=24000) for n in (1, 2, 3, 4, 5)]
    for r in recipes:
        out = render_recipe(r, store, bank, seg_len=24000)
        err = float(np.abs(out["mix"] - (out["sources"].sum(0) + out["noise"])).max())
        assert err < 1e-5, f"mix != sources + noise (max error {err:.2e}) at N={r['n_src']}"

    path = os.path.join(tempfile.mkdtemp(prefix="countsep-recipes-"), "recipes.csv")
    write_recipes(recipes, path)
    for before, after in zip(recipes, read_recipes(path)):
        a = render_recipe(before, store, bank, seg_len=24000)["mix"]
        b = render_recipe(after, store, bank, seg_len=24000)["mix"]
        assert np.array_equal(a, b), f"recipe {before['mix_id']} re-renders differently from CSV"


if __name__ == "__main__":
    sys.exit(run_checks({
        "mix = sources + noise; recipes re-render":
            test_mixture_is_sources_plus_noise_and_recipes_rerender_exactly,
        "slot selection knows the right answer": test_slot_selection_knows_the_right_answer,
        "mixture rebuild finds talkers loudness misses":
            test_consistent_slots_find_talkers_that_loudness_misses,
        "oracle masks recover disjoint sources": test_oracle_masks_recover_spectrally_disjoint_sources,
        "McNemar counts only discordant items": test_mcnemar_counts_only_discordant_items,
    }))
