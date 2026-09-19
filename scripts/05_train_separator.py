#!/usr/bin/env python3
"""Train the separator. Separation only -- no counting head, no counting term.

    python scripts/05_train_separator.py --store /kaggle/input/.../store \\
        --recipes_dev data/recipes_dev.csv --epochs 30 --out /kaggle/working/sep

WHY THIS IS A SEPARATE SCRIPT FROM THE COUNTER
----------------------------------------------
v0 trained one model on both jobs and both jobs suffered. Measured there:

* the counting term reached the shared trunk with **0.53 %** of the gradient against
  separation's 86.14 % -- 633:1 at the encoder -- so the counter never learned;
* and the auxiliary objectives cost separation **-8.7 dB** on a single-batch ablation. The
  pooled model reached **0.08 dB** on the official Libri2Mix test set while a fixed-N=2
  separation-only control reached **5.49 dB** in 8 epochs against the pooled model's 38.

Sharing bought almost nothing to offset that: dropping the separator from v0's model saved
only **8.4 %** of forward FLOPs, because the TCN is the expense.

So: one job per model, 100 % of the gradient each, joined at inference by
``countsep.pipeline``.

WHAT ELSE IS FIXED HERE
-----------------------
* **Hard clamp, not soft.** v0's ``soft_clamp`` multiplied by the SI-SDR gradient *peaks* at
  the clamp point, handing an already-solved N=1 item ~16x the gradient of a hard N=5 item.
  ``hard_clamp`` is flat above tau, so solved examples stop competing for the optimiser.
* **The silence term is a per-item average**, not per leftover slot. v0's denominator gave the
  single N=1 item in a balanced batch 40 % of that term and the N=5 item none.
* **One precision.** ``--amp`` is off by default, ``GlobalLayerNorm`` computes its statistics
  in fp32 regardless of autocast, and ``eps`` is 1e-5 rather than a value that is literally
  0.0 in fp16.

THE BAR
-------
Published Conv-TasNet on LibriMix 8 kHz min is **14.76 dB** SI-SDRi at N=2, with 200 epochs on
train-360. On a free-tier budget expect considerably less; v0's honest fixed-N=2 control got
5.49 dB in 2.53 GPU-h. Report the gap as budget, not as a mystery.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from _common import (add_common_args, banner, build_store_and_bank, code_version,
                     require_store, resolve)


def evaluate(model, loader, loss_fn, device, *, amp: bool, max_n_src: int) -> dict:
    """SI-SDRi per speaker count, plus the loss terms, in the caller's precision."""
    import torch

    from countsep.metrics import usable_si_sdri

    model.eval()
    per_n: dict[int, list[float]] = {}
    totals: dict[str, float] = {}
    dropped = [0]
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            moved = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(moved["mix"])
            loss, logs = loss_fn(out, moved)
            for k, v in logs.items():
                if isinstance(v, float) and np.isfinite(v):
                    totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            est = out["est"].float().cpu().numpy()
            refs = batch["refs"].numpy()
            mix = batch["mix"].numpy()
            for i, n in enumerate(batch["n_src"].tolist()):
                # usable_si_sdri returns (improvements, n_dropped). It LEAVES OUT sources the
                # mixture already equals -- a clean single-speaker mixture IS its own target,
                # so "improvement" there measures EPS, not separation. In v0 that one bug made
                # the headline read -8.68 dB instead of +1.2 dB. The drop count is reported
                # rather than hidden, because silently dropping examples is its own dishonesty.
                scores, n_drop = usable_si_sdri(est[i, :max_n_src], refs[i], mix[i], int(n))
                dropped[0] += int(n_drop)
                finite = [float(v) for v in np.asarray(scores).ravel() if np.isfinite(v)]
                if finite:
                    per_n.setdefault(int(n), []).extend(finite)

    means = {n: float(np.mean(v)) for n, v in sorted(per_n.items())}
    pooled = float(np.mean([v for vals in per_n.values() for v in vals])) if per_n else float("nan")
    return {"si_sdri_per_n": means, "si_sdri": pooled, "n_dropped": int(dropped[0]),
            "logs": {k: v / max(n_batches, 1) for k, v in totals.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--preset", default="paper", choices=["paper", "small", "tiny"])
    ap.add_argument("--train_split", default="train-100")
    ap.add_argument("--dev_split", default="dev")
    ap.add_argument("--recipes_dev", default=None)
    ap.add_argument("--n_list", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--w_sil", type=float, default=1.0)
    ap.add_argument("--w_noise", type=float, default=0.2)
    ap.add_argument("--clamp_si_sdr", type=float, default=30.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="fp16. OFF by default -- see docstring.")
    ap.add_argument("--time_budget_h", type=float, default=10.5)
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--out", default="/kaggle/working/sep")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch

    from countsep.checkpoint import TimeBudget, find_resume, load_checkpoint, save_checkpoint
    from countsep.constants import MAX_N_SRC
    from countsep.datasets import (DEFAULT_MIXING, DynamicMixDataset, FrozenMixDataset,
                                   build_loader)
    from countsep.losses import RectangularPITLoss
    from countsep.separator import PRESETS, build_separator
    from countsep.utils import json_dump_atomic, pick_device, seed_everything

    banner(f"05 - train the SEPARATOR   (preset={args.preset}, amp={args.amp})")
    seed_everything(args.seed)
    store_root = require_store(args)
    out_dir = resolve(args.out) or args.out
    os.makedirs(out_dir, exist_ok=True)
    device = pick_device(None if args.device == "auto" else args.device)
    budget = TimeBudget(args.time_budget_h)

    train_store, train_bank = build_store_and_bank(
        store_root, args.train_split, noise_kinds=DEFAULT_MIXING["noise_kinds"])
    dev_store, dev_bank = build_store_and_bank(
        store_root, args.dev_split, noise_kinds=DEFAULT_MIXING["noise_kinds"])

    train_set = DynamicMixDataset(train_store, train_bank, want="separate",
                                  n_list=args.n_list,
                                  steps=args.steps_per_epoch * args.batch_size, seed=args.seed)
    recipes_dev = resolve(args.recipes_dev)
    if recipes_dev and os.path.exists(recipes_dev):
        dev_set = FrozenMixDataset(dev_store, dev_bank, recipes_dev, want="separate",
                                   n_list=args.n_list,
                                   limit=args.val_batches * args.batch_size)
        print(f"  dev: frozen {recipes_dev} ({len(dev_set)} mixtures)")
    else:
        dev_set = DynamicMixDataset(dev_store, dev_bank, want="separate", n_list=args.n_list,
                                    steps=args.val_batches * args.batch_size, seed=args.seed + 1)
        dev_set.set_epoch(999)
        print(f"  dev: fixed-salt dynamic set ({len(dev_set)} mixtures)")

    train_loader = build_loader(train_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=True,
                                prefetch_factor=2)
    dev_loader = build_loader(dev_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=max(1, args.num_workers - 1), persistent=False,
                              prefetch_factor=2)

    model = build_separator(PRESETS[args.preset]).to(device)
    print(f"  {model.describe()}")
    print(f"  device: {device}   |   training on N in {args.n_list}")

    loss_fn = RectangularPITLoss(max_n_src=MAX_N_SRC, predict_noise=True,
                                 w_sep=1.0, w_sil=args.w_sil, w_count=0.0,
                                 w_noise=args.w_noise, clamp_si_sdr=args.clamp_si_sdr)
    print(f"  loss: w_sep 1.0, w_sil {args.w_sil}, w_noise {args.w_noise}, "
          f"w_count 0.0 (counting is a separate model), HARD clamp at {args.clamp_si_sdr} dB")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch,
        pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch, best = 0, float("-inf")
    resume = find_resume(None, work_dir=ckpt_dir)
    if resume:
        state = load_checkpoint(resume, model=model, optimizer=optimiser,
                                scheduler=scheduler, scaler=scaler)
        start_epoch = int(state.get("epoch", 0))
        best = float(state.get("best_metric", float("-inf")))
        print(f"  resumed from {resume} at epoch {start_epoch} (best {best:.3f} dB)")

    history: list[dict] = []
    banner("training")
    print("  published Conv-TasNet on LibriMix 8k min, N=2: 14.76 dB SI-SDRi "
          "(200 epochs, train-360)\n")

    for epoch in range(start_epoch, args.epochs):
        train_set.set_epoch(epoch)
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for step, batch in enumerate(train_loader):
            moved = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                out = model(moved["mix"])
            loss, logs = loss_fn(out, moved)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            seen += 1
            if step + 1 >= args.steps_per_epoch:
                break

        metrics = evaluate(model, dev_loader, loss_fn, device, amp=args.amp,
                           max_n_src=MAX_N_SRC)
        elapsed = (time.time() - started) / 3600.0
        row = {"epoch": epoch + 1, "train_loss": running / max(seen, 1),
               "val_si_sdri": metrics["si_sdri"], "per_n": metrics["si_sdri_per_n"],
               "hours": elapsed}
        history.append(row)
        per_n = "  ".join(f"N{n}:{v:+.1f}" for n, v in metrics["si_sdri_per_n"].items())
        drop = f"  ({metrics['n_dropped']} degenerate refs dropped)" if metrics["n_dropped"] else ""
        print(f"  epoch {epoch + 1:3d}  loss {row['train_loss']:8.3f}  "
              f"SI-SDRi {metrics['si_sdri']:+6.2f} dB   [{per_n}]  "
              f"{elapsed * 60:.0f} min{drop}")

        is_best = metrics["si_sdri"] > best
        if is_best:
            best = metrics["si_sdri"]
        save_checkpoint(os.path.join(ckpt_dir, "best.pt" if is_best else "last.pt"),
                        model=model, optimizer=optimiser, scheduler=scheduler, scaler=scaler,
                        epoch=epoch + 1, global_step=(epoch + 1) * args.steps_per_epoch,
                        best_metric=best, history=history, wall_h=budget.elapsed_h(),
                        cfg_dict={"model": vars(PRESETS[args.preset]), "amp": args.amp,
                                  "preset": args.preset, "n_list": args.n_list,
                                  "mixing": DEFAULT_MIXING})
        if budget.expired():
            print(f"\n  stopping cleanly at {budget.elapsed_h():.2f} h. Re-run to resume.")
            break

    banner("result")
    print(f"  best dev SI-SDRi ........ {best:+.2f} dB")
    print(f"  published (N=2, 200 ep) . +14.76 dB")
    print("\n  The gap is budget, not mystery: published numbers use 200 epochs on train-360.")
    print("  Report yours with the epoch count and the training set beside it.")

    report = {"code_version": code_version(), "preset": args.preset, "amp": args.amp,
              "best_val_si_sdri": best, "history": history, "n_list": args.n_list,
              "params": int(sum(p.numel() for p in model.parameters())),
              "loss": {"w_sil": args.w_sil, "w_noise": args.w_noise, "w_count": 0.0,
                       "clamp": args.clamp_si_sdr, "clamp_kind": "hard"}}
    path = os.path.join(out_dir, "separator_report.json")
    json_dump_atomic(report, path)
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        print(f"\nFAILED: report missing or truncated at {path}", file=sys.stderr)
        return 2
    print(f"\nreport -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
