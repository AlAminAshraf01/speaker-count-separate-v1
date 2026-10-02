"""Checkpoint and resume: the contract a 12-hour Kaggle session depends on.

Both trainers used to write ``"best.pt" if is_best else "last.pt"`` -- one file or the other,
never both. An improving epoch therefore left ``last.pt`` stale (or absent, when every epoch
so far had improved), a resume started from an older epoch with an older ``best_metric``, and
the first epoch to beat that stale value overwrote the real ``best.pt`` with a worse model.
The reported runs finished inside one session and were never resumed, so their numbers are
unaffected; the next run to hit the time limit would not have been.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def _tiny():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=10)
    return model, opt, sched


def test_last_is_written_every_epoch_and_best_is_a_copy() -> None:
    """After an IMPROVING epoch, last.pt must exist and be that epoch."""
    from countsep.checkpoint import find_resume, load_checkpoint, save_epoch_checkpoint

    model, opt, sched = _tiny()
    ckpt = tempfile.mkdtemp(prefix="countsep-ckpt-")
    best = float("-inf")
    for epoch, metric in enumerate([0.5, 0.7, 0.6, 0.8, 0.75], start=1):
        is_best = metric > best
        best = max(best, metric)
        save_epoch_checkpoint(ckpt, is_best=is_best, model=model, optimizer=opt,
                              scheduler=sched, epoch=epoch, global_step=epoch,
                              best_metric=best, history=[{"epoch": e} for e in range(1, epoch + 1)])
        last = load_checkpoint(os.path.join(ckpt, "last.pt"), restore_rng=False)
        assert last["epoch"] == epoch, (
            f"after epoch {epoch} (improving: {is_best}) last.pt holds epoch {last['epoch']}")

    best_state = load_checkpoint(os.path.join(ckpt, "best.pt"), restore_rng=False)
    last_state = load_checkpoint(os.path.join(ckpt, "last.pt"), restore_rng=False)
    assert best_state["epoch"] == 4, f"best.pt is epoch {best_state['epoch']}, expected 4"
    assert last_state["best_metric"] == 0.8, "last.pt forgot the true best metric"
    assert len(last_state["history"]) == 5, "history was not carried in the checkpoint"
    assert find_resume(None, work_dir=ckpt, search_inputs=False) == os.path.join(ckpt, "last.pt")


def test_resume_search_stays_in_its_own_run_directory() -> None:
    """The furthest-along last.pt of ANOTHER run must not be picked up."""
    from countsep.checkpoint import find_resume, run_dir_filter, save_checkpoint

    model, _, _ = _tiny()
    root = tempfile.mkdtemp(prefix="countsep-inputs-")
    for run, step in (("counter", 100), ("sep", 900), ("sep_n2", 5000), ("pool_meanstd", 7000)):
        save_checkpoint(os.path.join(root, "nb-output", run, "ckpt", "last.pt"),
                        model=model, global_step=step)
    empty = tempfile.mkdtemp(prefix="countsep-work-")

    def pick(out_dir: str | None) -> str:
        found = find_resume(None, work_dir=empty, input_root=root,
                            contains=run_dir_filter(out_dir) if out_dir else None)
        return found.replace("\\", "/")

    assert "/sep/ckpt/" in pick("/kaggle/working/sep")
    assert "/counter/ckpt/" in pick("/kaggle/working/counter")
    assert "/sep_n2/ckpt/" in pick("/kaggle/working/sep_n2")
    assert "/pool_meanstd/" in pick(None), \
        "this test cannot detect the bug: without the filter it should take the longest run"


def test_a_changed_schedule_refuses_to_resume() -> None:
    """OneCycleLR restores total_steps, so a different --epochs must stop, not crash later."""
    from countsep.checkpoint import check_resume_schedule, load_checkpoint, save_checkpoint

    model, opt, sched = _tiny()
    path = os.path.join(tempfile.mkdtemp(prefix="countsep-sched-"), "last.pt")
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched)
    state = load_checkpoint(path, restore_rng=False)
    check_resume_schedule(state, 10)
    try:
        check_resume_schedule(state, 20)
    except SystemExit:
        return
    raise AssertionError("resuming a 10-step schedule as 20 steps was allowed")


def test_counter_config_round_trips_and_old_checkpoints_still_load() -> None:
    """New checkpoints carry the whole config; the reported ones only carry the pooling."""
    from countsep.counter import ModelConfig, config_from_checkpoint

    custom = ModelConfig(channels=(16, 32), gru_hidden=32, pooling="meanstd")
    # a checkpoint's cfg goes through torch.save, which keeps tuples; JSON would give lists
    as_saved = {"cfg": {"model": {**dataclasses.asdict(custom), "channels": [16, 32]}}}
    assert config_from_checkpoint(as_saved) == custom, "full counter config did not round-trip"
    legacy = config_from_checkpoint({"cfg": {"pooling": "attentive"}})
    assert legacy == ModelConfig(pooling="attentive"), "a pooling-only checkpoint broke"
    assert config_from_checkpoint({}) == ModelConfig(), "a config-less checkpoint broke"


if __name__ == "__main__":
    sys.exit(run_checks({
        "last.pt every epoch, best.pt a copy": test_last_is_written_every_epoch_and_best_is_a_copy,
        "resume stays in its own run dir": test_resume_search_stays_in_its_own_run_directory,
        "a changed schedule refuses to resume": test_a_changed_schedule_refuses_to_resume,
        "counter config round-trips": test_counter_config_round_trips_and_old_checkpoints_still_load,
    }))
