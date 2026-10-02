"""`scripts/10_slide_figures.py` must draw from the reports it is given and say so.

The slide figures are only worth regenerating if every number on them can be traced to the
file it came from. So these checks do not just look for PNGs: they read `sources.json` and
`numbers.md` back and assert that a figure drawn from the evaluation report names that report,
that a figure drawn from a recorded value is labelled `recorded:`, and that the headline
numbers are the committed evaluation's, not something the script made up.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks, store_and_bank, tiny_store  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "10_slide_figures.py")


def _run(*argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    return subprocess.run([sys.executable, SCRIPT, *argv], capture_output=True, text=True,
                          encoding="utf-8", env=env, cwd=REPO_ROOT, timeout=600)


def test_figures_are_drawn_from_the_committed_reports() -> None:
    out = tempfile.mkdtemp(prefix="slidefigs-")
    eval_report = os.path.join(REPO_ROOT, "report", "runs", "eval_report.json")
    recipes = os.path.join(REPO_ROOT, "data", "recipes_test.csv")
    proc = _run("--recipes_test", recipes, "--eval", eval_report, "--out", out)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    made = set(os.listdir(out))
    for name in ("slide04_test_set.png", "slide04_snr.png", "slide05_preprocessing.png",
                 "slide06_overview.png", "slide07_counter.png", "slide08_separator.png",
                 "slide09_confusion.png", "slide09_separation.png",
                 "slide10_shared_vs_two.png", "slide11_pooling.png",
                 "sources.json", "numbers.md"):
        assert name in made, f"{name} was not written; got {sorted(made)}"

    with open(os.path.join(out, "sources.json"), encoding="utf-8") as fh:
        sources = json.load(fh)
    assert any(eval_report in s for s in sources["figures"]["slide09_confusion.png"]), \
        "the confusion matrix does not name the report it was drawn from"
    assert all(s.startswith("recorded:") for s in sources["figures"]["slide11_pooling.png"]), \
        "with no pool_* runs attached, the pooling chart must say it uses the recorded proxy"
    assert "slide03_problem.png" in sources["skipped"], \
        "without a store the problem picture must be skipped, not drawn from nothing"

    with open(eval_report, encoding="utf-8") as fh:
        ev = json.load(fh)
    with open(os.path.join(out, "numbers.md"), encoding="utf-8") as fh:
        numbers = fh.read()
    assert f"{100 * ev['counting']['accuracy']:.1f} %" in numbers, \
        "numbers.md does not carry the evaluation's counting accuracy"
    assert f"{ev['slot_selection_rules']['pooled']['rebuild']:+.2f} dB" in numbers, \
        "numbers.md does not carry the delivered (rebuild) SI-SDRi"
    assert "487,941" in numbers and "5,249,073" in numbers, \
        "the parameter counts must be measured from the models"


def test_the_pictures_use_a_real_clip_from_the_store() -> None:
    from countsep.mixing import sample_recipe, write_recipes

    store, bank = store_and_bank("test")
    rng = np.random.default_rng(3)
    top = min(3, len(store.speakers))
    rows = [sample_recipe(store, bank, n, rng, mix_id=f"test_n{n}_{i:05d}")
            for n in range(1, top + 1) for i in range(4)]
    out = tempfile.mkdtemp(prefix="slidefigs-")
    recipes = os.path.join(out, "recipes_test.csv")
    write_recipes(rows, recipes)
    proc = _run("--store", tiny_store(), "--recipes_test", recipes, "--split", "test",
                "--out", os.path.join(out, "figs"))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]

    with open(os.path.join(out, "figs", "sources.json"), encoding="utf-8") as fh:
        sources = json.load(fh)
    assert "slide03_problem.png" in sources["figures"], \
        f"the problem picture was not drawn: {sources['skipped']}"
    assert any("clip test_n" in s for s in sources["figures"]["slide03_problem.png"]), \
        "the problem picture does not say which test clip it shows"
    assert any("clip test_n" in s for s in sources["figures"]["slide05_preprocessing.png"])


if __name__ == "__main__":
    sys.exit(run_checks({
        "figures are drawn from the committed reports":
            test_figures_are_drawn_from_the_committed_reports,
        "the pictures use a real clip from the store":
            test_the_pictures_use_a_real_clip_from_the_store,
    }))
