"""The built `.ipynb` files must never be stale, and the staleness check must actually work.

You edit `notebooks/src/*.py` and run `tools/build_notebooks.py`. If you forget the second
step, the `.ipynb` in the repo still has the old cells -- and since Kaggle imports the
`.ipynb`, you would be running last week's cells against this week's code. The log looks
completely normal while doing it, which is what makes it expensive.

Two defences, both tested here:

1. **This suite fails** if any notebook is stale, so it cannot be committed quietly.
2. Every notebook carries a **fingerprint** of the source it was built from, and the bootstrap
   cell recomputes that fingerprint from the freshly-cloned repo at run time. A notebook
   imported before the last push announces it in the first ten seconds.

Defence 2 has a trap: the fingerprint function is written out **twice** -- once in
`tools/build_notebooks.py` and once in `notebooks/src/_bootstrap.py` -- because the bootstrap
cannot import the tool (importing it would clone a repo) and the tool cannot import the
bootstrap (importing it *runs* the clone). Two copies of one function drift. So the last test
below execs both and asserts they agree, which turns a silent drift into a red test.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "notebooks", "src")
OUT_DIR = os.path.join(REPO_ROOT, "notebooks")


def _sources() -> list[str]:
    return sorted(f for f in os.listdir(SRC_DIR)
                  if f.endswith(".py") and not f.startswith("_"))


def _notebook(name: str) -> dict:
    with open(os.path.join(OUT_DIR, name.replace(".py", ".ipynb")), "r",
              encoding="utf-8") as fh:
        return json.load(fh)


def test_every_source_has_a_built_notebook() -> None:
    for name in _sources():
        path = os.path.join(OUT_DIR, name.replace(".py", ".ipynb"))
        assert os.path.exists(path), f"{name} has never been built; run tools/build_notebooks.py"


def test_no_notebook_is_stale() -> None:
    """The one that catches a forgotten rebuild."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    from build_notebooks import cells_fingerprint

    for name in _sources():
        nb = _notebook(name)
        stamped = None
        for cell in nb["cells"]:
            text = "".join(cell["source"])
            if "CELLS_SHA" in text:
                stamped = text.split('CELLS_SHA = "')[1].split('"')[0]
                break
        assert stamped, f"{name}: built notebook carries no CELLS_SHA stamp"
        current = cells_fingerprint(SRC_DIR, name)
        assert stamped == current, (
            f"{name} is STALE: the .ipynb was built from {stamped}, the source is now "
            f"{current}. Run: python tools/build_notebooks.py")


def test_notebooks_are_valid_and_self_contained() -> None:
    """Kaggle uploads a notebook alone, so the bootstrap must be inside each one."""
    for name in _sources():
        nb = _notebook(name)
        assert nb.get("cells"), f"{name}: no cells"
        assert nb.get("nbformat") == 4, f"{name}: wrong nbformat"
        text = "".join("".join(c["source"]) for c in nb["cells"])
        assert "REPO = bootstrap()" in text, (
            f"{name}: the bootstrap was not expanded -- is the '# %include _bootstrap.py' "
            f"line present in the source?")
        assert "def run(" in text, f"{name}: the run() helper is missing"
        for cell in nb["cells"]:
            assert cell["cell_type"] in ("code", "markdown", "raw"), \
                f"{name}: bad cell type {cell['cell_type']}"


def test_every_notebook_runs_preflight_before_spending() -> None:
    """A notebook that starts work without checking its inputs is how a session gets wasted."""
    for name in _sources():
        text = "".join("".join(c["source"]) for c in _notebook(name)["cells"])
        assert "scripts/preflight.py" in text, (
            f"{name} never calls preflight. Every notebook should check its store, its "
            f"recipes and its accelerator before doing anything expensive.")
        assert "CELLS_SRC" in text and "CELLS_SHA" in text, (
            f"{name} does not pass its stamp to preflight, so the staleness check is off.")


def test_notebooks_reference_scripts_that_exist() -> None:
    """A renamed script silently turns a notebook cell into a 'file not found' 20 minutes in."""
    scripts = set(os.listdir(os.path.join(REPO_ROOT, "scripts")))
    for name in _sources():
        text = "".join("".join(c["source"]) for c in _notebook(name)["cells"])
        for token in text.split():
            if token.startswith("scripts/") and token.endswith(".py"):
                called = token.split("/", 1)[1]
                assert called in scripts, f"{name} calls scripts/{called}, which does not exist"


def test_the_two_fingerprint_copies_agree() -> None:
    """`_bootstrap.py` and `build_notebooks.py` each define `cells_fingerprint`. They must match.

    The bootstrap cannot be imported (importing it clones a repo), so its copy is exec'd with
    only the pieces it needs, stopping before the module-level `bootstrap()` call.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    from build_notebooks import cells_fingerprint as from_tool

    with open(os.path.join(SRC_DIR, "_bootstrap.py"), "r", encoding="utf-8") as fh:
        source = fh.read()
    head = source.split("def cells_status", 1)[0]          # stop before anything with effects
    namespace: dict = {}
    exec(compile(head, "_bootstrap.py", "exec"), namespace)     # noqa: S102 - that is the point
    from_bootstrap = namespace["cells_fingerprint"]

    for name in _sources():
        assert from_tool(SRC_DIR, name) == from_bootstrap(SRC_DIR, name), (
            f"{name}: the two cells_fingerprint copies disagree. They are duplicated on "
            f"purpose and must stay byte-identical.")


if __name__ == "__main__":
    sys.exit(run_checks({
        "every source has a built notebook": test_every_source_has_a_built_notebook,
        "no notebook is stale": test_no_notebook_is_stale,
        "notebooks are valid and self-contained": test_notebooks_are_valid_and_self_contained,
        "every notebook runs preflight": test_every_notebook_runs_preflight_before_spending,
        "notebooks reference scripts that exist": test_notebooks_reference_scripts_that_exist,
        "the two fingerprint copies agree": test_the_two_fingerprint_copies_agree,
    }))
