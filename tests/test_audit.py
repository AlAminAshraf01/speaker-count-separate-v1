"""The audit must actually run, actually write its report, and actually fail when it should.

Its predecessor exited 0 having written nothing, so "the script ran" is not the property worth
testing here -- "the script produced the artefact it claims to produce" is. Every check below
runs `scripts/02_audit_and_baselines.py` as a **subprocess**, the way a notebook cell does, and
inspects the exit code and the JSON on disk rather than any in-process return value.

Against the tiny synthetic corpus this takes under two minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks, tiny_store  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "02_audit_and_baselines.py")

_CACHE: dict[str, tuple[int, str, dict]] = {}


def _run_audit(per_class: int = 25, extra: tuple[str, ...] = ()) -> tuple[int, str, dict]:
    """Run the audit end to end; return ``(exit_code, stdout, report_or_empty)``."""
    key = f"{per_class}-{extra}"
    if key in _CACHE:
        return _CACHE[key]

    out_dir = tempfile.mkdtemp(prefix="countsep-audit-")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(REPO_ROOT, "src"), os.path.join(REPO_ROOT, "tools"),
         env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--store", tiny_store(), "--probe_split", "train-100",
         "--per_class", str(per_class), "--folds", "3", "--out", out_dir, *extra],
        env=env, cwd=REPO_ROOT, capture_output=True, text=True)

    path = os.path.join(out_dir, "audit_report.json")
    report: dict = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    result = (proc.returncode, (proc.stdout or "") + (proc.stderr or ""), report)
    _CACHE[key] = result
    return result


def test_audit_exits_zero_on_a_clean_store() -> None:
    code, output, _ = _run_audit()
    assert code == 0, f"audit exited {code}:\n{output[-3000:]}"


def test_audit_actually_writes_its_report() -> None:
    """The exact property the predecessor lacked."""
    _code, _out, report = _run_audit()
    assert report, "audit_report.json is missing or empty -- this is the predecessor's bug"


def test_report_has_every_section_populated() -> None:
    """A report that exists but is half-empty is the same failure, one step later."""
    _code, _out, report = _run_audit()
    for key in ("naive", "probes", "bar", "sections", "speaker_reuse"):
        assert report.get(key), f"report section {key!r} is missing or empty"
    for key in ("artefact:raw", "artefact_strict:mitigated",
                "acoustic:mitigated", "everything:mitigated"):
        assert key in report["probes"], f"probe {key!r} did not run"
        assert "accuracy" in report["probes"][key], f"probe {key!r} has no accuracy"


def test_mitigation_holds_the_strict_artefact_probe_at_chance() -> None:
    """After a fixed-length crop divided by its RMS, level and length are constant.

    So this probe has literally nothing to learn and must sit at chance. This is the check
    that licenses every other number the project reports.
    """
    _code, _out, report = _run_audit()
    res = report["probes"]["artefact_strict:mitigated"]
    assert res["verdict"] == "OK", (
        f"strict artefact probe scored {res['accuracy']:.1%} against {res['chance']:.1%} "
        f"chance and was judged {res['verdict']}: {res.get('explanation', '')}")
    assert abs(res["accuracy"] - res["chance"]) < 0.10, (
        f"strict artefact probe at {res['accuracy']:.1%} is too far from chance "
        f"{res['chance']:.1%}; its features are supposed to be constant")


def test_unmitigated_audio_still_shows_the_leak() -> None:
    """The probe must be *capable* of detecting a leak, or 'no leak' means nothing.

    Run on audio with its original level restored, the artefact family should score well above
    chance -- that is the cue the mitigation exists to remove. A probe that never fires is not
    evidence of safety, it is evidence of a broken probe.
    """
    _code, _out, report = _run_audit()
    raw = report["probes"]["artefact:raw"]
    assert raw["accuracy"] > raw["chance"] * 1.5, (
        f"the artefact probe only reached {raw['accuracy']:.1%} on UNMITIGATED audio "
        f"(chance {raw['chance']:.1%}). It is supposed to find the level cue there; if it "
        f"cannot, its clean verdict on mitigated audio is worthless.")


def test_acoustic_probe_records_a_bar_above_the_naive_predictors() -> None:
    _code, _out, report = _run_audit()
    bar = report["bar"]["acoustic_gbm"]
    best_naive = max(r["accuracy"] for r in report["naive"].values())
    assert bar > best_naive, (
        f"the acoustic bar ({bar:.1%}) is not above the best naive predictor "
        f"({best_naive:.1%}); there is no signal to model")


def test_naive_predictors_are_at_chance_on_a_balanced_set() -> None:
    """Sanity: with equal mixtures per class, every naive predictor is exactly 1/K."""
    _code, _out, report = _run_audit()
    for name, res in report["naive"].items():
        assert abs(res["accuracy"] - 0.20) < 0.02, \
            f"naive predictor {name} scored {res['accuracy']:.1%}, expected 20% on 5 balanced classes"


def test_strict_mode_fails_loudly_on_a_missing_split() -> None:
    """--strict must turn a broken audit into a non-zero exit, not a warning in the log."""
    code, output, report = _run_audit(extra=("--splits", "train-100"))
    assert code != 0, "audit exited 0 despite being unable to check disjointness"
    assert report.get("problems"), "the report records no problem despite exiting non-zero"
    assert "PROBLEMS FOUND" in output


if __name__ == "__main__":
    sys.exit(run_checks({
        "audit exits zero on a clean store": test_audit_exits_zero_on_a_clean_store,
        "audit actually writes its report": test_audit_actually_writes_its_report,
        "report has every section populated": test_report_has_every_section_populated,
        "mitigation holds strict probe at chance": test_mitigation_holds_the_strict_artefact_probe_at_chance,
        "unmitigated audio still shows the leak": test_unmitigated_audio_still_shows_the_leak,
        "acoustic bar beats the naive predictors": test_acoustic_probe_records_a_bar_above_the_naive_predictors,
        "naive predictors are at chance": test_naive_predictors_are_at_chance_on_a_balanced_set,
        "strict mode fails loudly": test_strict_mode_fails_loudly_on_a_missing_split,
    }))
