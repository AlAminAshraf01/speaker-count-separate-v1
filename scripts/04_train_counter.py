#!/usr/bin/env python3
"""Train the neural counter (Tier B). One task, one objective, one precision.

    python scripts/02_train.py --store /kaggle/input/.../store --pooling eigen \\
        --epochs 20 --out /kaggle/working/run1

    # the pooling comparison from docs/DESIGN.md section 3 -- four cells, one flag each
    for p in meanstd attentive covariance eigen; do
        python scripts/02_train.py --store ... --pooling $p --epochs 10 --out runs/$p
    done

WHAT IS DELIBERATELY ABSENT, AND WHY
------------------------------------
There is no separation loss, no silence term, no noise slot, no permutation search and no loss
weighting, because there is only one task. In the sibling project the counting term supplied
**0.53 %** of the gradient into the shared trunk and separation supplied 86.14 % (633:1 into
the encoder); here cross-entropy is 100 % of it. That single change closes findings 1, 2, 3 and
4 of ``docs/DIAGNOSIS.md`` by construction, and it is free.

PRECISION IS A FIRST-CLASS CONCERN HERE
---------------------------------------
The sibling model trained under fp16 autocast and validated in fp32 -- two different functions,
44.9 % against 20.00 %, agreeing on 16.1 % of predictions. So:

* ``--amp`` is **off by default**. This model is small; on a T4 the speed-up does not justify
  re-opening that failure mode. Turn it on deliberately, not by inheritance.
* Whatever precision is chosen, **validation uses the same one as training**.
* At startup, and again at the end, the script asserts fp32 and autocast predictions agree on
  >=95 % of a batch and **refuses to continue if they do not** -- before spending GPU quota,
  not after.

THE BAR
-------
Validation accuracy is compared every epoch against ``--bar``, the hand-crafted baseline from
``scripts/01_audit_and_baselines.py``. A neural counter that does not beat a gradient-boosted
tree over sixteen scalars has bought nothing, and the training log should say so out loud
rather than leaving it to be discovered in the write-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from _common import (add_common_args, banner, build_store_and_bank, code_version,
                     require_store, resolve)


def evaluate(model, loader, device, *, amp: bool, n_classes: int) -> dict:
    """Accuracy, MAE and the confusion matrix, in the precision the caller names."""
    import torch

    model.eval()
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    by_snr: list[tuple[float, int]] = []
    with torch.no_grad():
        for batch in loader:
            wav = batch["mix"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(wav)["logits"]
            pred = logits.float().argmax(-1).cpu().numpy()
            true = batch["cls"].numpy()
            for t, p in zip(true, pred):
                confusion[int(t), int(p)] += 1
            by_snr.extend(zip(batch["snr_db"].numpy().tolist(),
                              (pred == true).astype(int).tolist()))

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    idx = np.arange(n_classes)
    mae = float((confusion * np.abs(idx[:, None] - idx[None, :])).sum() / max(total, 1))
    off_by_one = float(sum(confusion[i, j] for i in idx for j in idx
                           if abs(i - j) <= 1) / max(total, 1))
    recall = [float(confusion[i, i] / max(confusion[i].sum(), 1)) for i in idx]
    return {"accuracy": correct / max(total, 1), "mae": mae, "off_by_one": off_by_one,
            "per_class_recall": recall, "confusion": confusion.tolist(), "n": total,
            "snr_pairs": by_snr}


def check_precision_agreement(model, loader, device, *, amp: bool) -> float:
    """Fraction of predictions on which fp32 and autocast agree. Must be >= 0.95.

    This is the gate the sibling project needed and did not have. It costs one batch.
    """
    import torch

    model.eval()
    batch = next(iter(loader))
    wav = batch["mix"].to(device)
    with torch.no_grad():
        hi = model(wav)["logits"].float().argmax(-1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            lo = model(wav)["logits"].float().argmax(-1)
    return float((hi == lo).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--pooling", default="eigen",
                    choices=["meanstd", "attentive", "covariance", "eigen"])
    ap.add_argument("--train_split", default="train-100")
    ap.add_argument("--dev_split", default="dev")
    ap.add_argument("--recipes_dev", default=None, help="frozen dev recipes CSV")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast. OFF by default -- see the module docstring.")
    ap.add_argument("--bar", type=float, default=0.693,
                    help="the hand-crafted baseline this must beat (from script 01)")
    ap.add_argument("--time_budget_h", type=float, default=10.5)
    ap.add_argument("--out", default="/kaggle/working/run")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch

    from countsep.checkpoint import TimeBudget, find_resume, load_checkpoint, save_checkpoint
    from countsep.constants import N_CLASSES, N_LIST
    from countsep.datasets import DEFAULT_MIXING, DynamicMixDataset, FrozenMixDataset, build_loader
    from countsep.counter import ModelConfig, build_model
    from countsep.utils import json_dump_atomic, pick_device, seed_everything

    banner(f"02 - train the counter   (pooling={args.pooling}, amp={args.amp})")
    seed_everything(args.seed)
    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(None if args.device == "auto" else args.device)
    budget = TimeBudget(args.time_budget_h)

    # ---------------------------------------------------------------- data
    train_store, train_bank = build_store_and_bank(
        store_root, args.train_split, noise_kinds=DEFAULT_MIXING["noise_kinds"])
    dev_store, dev_bank = build_store_and_bank(
        store_root, args.dev_split, noise_kinds=DEFAULT_MIXING["noise_kinds"])

    train_set = DynamicMixDataset(train_store, train_bank,
                                  steps=args.steps_per_epoch * args.batch_size, seed=args.seed)
    recipes_dev = resolve(args.recipes_dev)
    if recipes_dev and os.path.exists(recipes_dev):
        dev_set = FrozenMixDataset(dev_store, dev_bank, recipes_dev)
        print(f"  dev: frozen recipes {recipes_dev} ({len(dev_set)} mixtures, "
              f"{dev_set.counts_per_n()})")
    else:
        dev_set = DynamicMixDataset(dev_store, dev_bank, steps=1500, seed=args.seed + 1)
        dev_set.set_epoch(999)     # a fixed salt, so dev is the same set every epoch
        print(f"  dev: no frozen recipes given, using a fixed-salt dynamic set "
              f"({len(dev_set)} mixtures). Generate frozen recipes for a reportable number.")

    # persistent=False is load-bearing -- see build_loader. With persistent workers,
    # set_epoch() never reaches them and all 20 epochs reuse one fixed set of mixtures.
    train_loader = build_loader(train_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=True,
                                persistent=False)
    dev_loader = build_loader(dev_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=max(1, args.num_workers - 1), persistent=False)

    # ---------------------------------------------------------------- model
    model = build_model(ModelConfig(pooling=args.pooling)).to(device)
    print(f"  {model.describe()}")
    print(f"  device: {device}   |   the model it replaces: 5.30 M params, 10.6 GFLOP/s audio")

    # ---------------------------------------------------------------- the precision gate
    banner("precision gate   (the check the sibling project did not have)")
    agreement = check_precision_agreement(model, dev_loader, device, amp=args.amp)
    print(f"  fp32 vs {'fp16 autocast' if args.amp else 'fp32 (amp off)'}: "
          f"predictions agree on {agreement:.1%} of a batch")
    if agreement < 0.95:
        print("\nFAILED: the two precisions disagree. Training under one and evaluating under\n"
              "the other is exactly how the sibling model came to score 44.9 % and 20.00 %\n"
              "from a single checkpoint. Fix this before spending GPU quota.", file=sys.stderr)
        return 2
    print("  OK -- the model is one function, not two.")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch,
        pct_start=0.15)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch, best = 0, -1.0
    resume = find_resume(None, work_dir=ckpt_dir)
    if resume:
        state = load_checkpoint(resume, model=model, optimizer=optimiser, scheduler=scheduler)
        start_epoch = int(state.get("epoch", 0))
        best = float(state.get("best_metric", -1.0))
        print(f"  resumed from {resume} at epoch {start_epoch} (best {best:.4f})")

    history: list[dict] = []
    banner("training")
    print(f"  the bar to beat: {args.bar:.1%}  (hand-crafted features + GBM, script 01)\n")

    for epoch in range(start_epoch, args.epochs):
        train_set.set_epoch(epoch)
        model.train()
        started, running, seen, correct = time.time(), 0.0, 0, 0

        for step, batch in enumerate(train_loader):
            wav = batch["mix"].to(device, non_blocking=True)
            target = batch["cls"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                logits = model(wav)["logits"]
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            running += float(loss.detach()) * target.numel()
            correct += int((logits.detach().float().argmax(-1) == target).sum())
            seen += int(target.numel())
            if step + 1 >= args.steps_per_epoch:
                break

        # Validation in the SAME precision as training. Not a detail -- see the docstring.
        metrics = evaluate(model, dev_loader, device, amp=args.amp, n_classes=N_CLASSES)
        elapsed = (time.time() - started) / 3600.0
        row = {"epoch": epoch + 1, "train_loss": running / max(seen, 1),
               "train_acc": correct / max(seen, 1), "val_acc": metrics["accuracy"],
               "val_mae": metrics["mae"], "val_off_by_one": metrics["off_by_one"],
               "hours": elapsed}
        history.append(row)

        verdict = "beats the bar" if metrics["accuracy"] > args.bar else "BELOW THE BAR"
        print(f"  epoch {epoch + 1:3d}  loss {row['train_loss']:.4f}  "
              f"train {row['train_acc']:.1%}  val {metrics['accuracy']:.1%}  "
              f"MAE {metrics['mae']:.3f}  +-1 {metrics['off_by_one']:.1%}  "
              f"[{verdict}]  {elapsed * 60:.1f} min")

        is_best = metrics["accuracy"] > best
        if is_best:
            best = metrics["accuracy"]
        save_checkpoint(os.path.join(ckpt_dir, "best.pt" if is_best else "last.pt"),
                        model=model, optimizer=optimiser, scheduler=scheduler,
                        epoch=epoch + 1, global_step=(epoch + 1) * args.steps_per_epoch,
                        best_metric=best, history=history,
                        scaler=scaler, wall_h=budget.elapsed_h(),
                        cfg_dict={"pooling": args.pooling, "amp": args.amp, "lr": args.lr,
                                  "batch_size": args.batch_size, "mixing": DEFAULT_MIXING})

        if budget.expired():
            print(f"\n  stopping cleanly: {budget.elapsed_h():.2f} h of a "
                  f"{args.time_budget_h} h budget used. Re-run to resume from {ckpt_dir}.")
            break

    # ---------------------------------------------------------------- verdict
    banner("result")
    # Score the checkpoint that SHIPS, not whatever weights are in memory when the loop
    # ends. best.pt is what notebook 04 loads, and the last epoch is usually not it --
    # this run ended at 83.3% with its best at 85.1%. Printing one headline beside
    # another model's confusion breakdown invites a report sentence that is not true of
    # any single checkpoint.
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model=model)
        print(f"  scoring {best_path}, the checkpoint notebook 04 will load")
        print()
    final = evaluate(model, dev_loader, device, amp=args.amp, n_classes=N_CLASSES)
    agreement = check_precision_agreement(model, dev_loader, device, amp=args.amp)
    print(f"  best val accuracy ....... {best:.1%}")
    print(f"  MAE ..................... {final['mae']:.3f}")
    print(f"  off-by-one .............. {final['off_by_one']:.1%}")
    print(f"  per-class recall ........ "
          + "  ".join(f"N={n}:{r:.0%}" for n, r in zip(N_LIST, final['per_class_recall'])))
    print(f"  fp32/autocast agreement . {agreement:.1%} (end of training)")
    print(f"\n  chance ......................... 20.0%")
    print(f"  hand-crafted + GBM (the bar) ... {args.bar:.1%}")
    print(f"  this model ..................... {best:.1%}   "
          f"{'<-- worth keeping' if best > args.bar else '<-- BOUGHT NOTHING'}")
    if best <= args.bar:
        print("\n  A neural counter that does not beat a gradient-boosted tree over sixteen")
        print("  scalars is a finding, not a failure -- report it as one. See docs/DESIGN.md.")

    report = {"code_version": code_version(), "pooling": args.pooling, "amp": args.amp,
              "best_val_accuracy": best, "bar": args.bar, "beats_bar": bool(best > args.bar),
              "best_checkpoint": {k: v for k, v in final.items() if k != "snr_pairs"},
              "precision_agreement": agreement, "history": history,
              "mixing": DEFAULT_MIXING, "params": model.count_parameters()}
    path = os.path.join(out_dir, "train_report.json")
    json_dump_atomic(report, path)
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        print(f"\nFAILED: report missing or truncated at {path}", file=sys.stderr)
        return 2
    print(f"\nreport -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
