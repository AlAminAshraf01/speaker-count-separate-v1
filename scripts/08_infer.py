#!/usr/bin/env python3
"""Audio in, a speaker count and one clean track per speaker out. The system, used.

    # your own recording, any format, any sample rate
    python scripts/08_infer.py --counter counter/ckpt/best.pt \\
        --separator sep/ckpt/best.pt --input meeting.m4a --out separated

    # or a mixture drawn from the frozen test set, where the true count is known
    python scripts/08_infer.py --counter ... --separator ... \\
        --from_recipes data/recipes_test.csv --store /kaggle/input/.../store --n 3

Everything else in this repo reports averages over 1,500 mixtures. This reports one file, and
that is the point: an average cannot be listened to. It is also the only place the two models
are used the way the system is meant to be used -- ``countsep.pipeline`` joins them, and
without this script nothing calls it.

WHY THE COUNT IS PRINTED HERE AND WAS NOT IN v0
-----------------------------------------------
v0's demo deliberately removed its "N speakers detected" line, because in fp32 that checkpoint
answered **1** for 1445 of 1500 test mixtures: the line printed the same number whoever was
talking, which looks like a result and carries no information. In v1 the counter is a separate
model with its own gradient, so the number means something and is printed -- next to the full
probability distribution, so an uncertain answer looks uncertain instead of confident.

WHY A LONG FILE IS NOT FED IN ONE BLOCK
---------------------------------------
Both models are trained on fixed 3 s RMS-normalised crops. v0 measured that handing the counter
one long block instead of overlapping windows took it from a working score to **0 correct out
of 300** -- the normalisation and the receptive field both assume the training length. So
``separate_long`` windows the file, permutation-aligns each window to its predecessor (without
which a speaker changes track partway through: measured at 0.908 correlation with speaker A
over the first half and 1.000 with speaker B over the second), and averages the per-window
count probabilities rather than voting, so a window landing in a pause abstains.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from _common import (add_common_args, autodetect_store, banner, build_store_and_bank,
                     code_version, resolve)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--counter", required=True)
    ap.add_argument("--separator", required=True)
    ap.add_argument("--input", default=None, help="any audio file, any sample rate")
    ap.add_argument("--from_recipes", default=None,
                    help="build the demo mixture from a frozen recipe file instead")
    ap.add_argument("--n", type=int, default=None,
                    help="with --from_recipes, pick a mixture with this many talkers")
    ap.add_argument("--index", type=int, default=0, help="which matching recipe to use")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--out", default="/kaggle/working/separated")
    ap.add_argument("--max_seconds", type=float, default=60.0)
    ap.add_argument("--all_slots", action="store_true",
                    help="also write the slots the counter rejected")
    args = ap.parse_args()

    import torch

    from countsep.audio import read_wav, write_wav
    from countsep.constants import N_LIST, SR
    from countsep.pipeline import load_pipeline, separate_long
    from countsep.utils import ensure_dir, json_dump_atomic, pick_device

    banner("08 - count, then separate   (one file, no training)")
    if not args.input and not args.from_recipes:
        print("Give it audio: --input FILE, or --from_recipes data/recipes_test.csv",
              file=sys.stderr)
        return 2

    out_dir = ensure_dir(resolve(args.out) or args.out)
    device = pick_device(None)
    system = load_pipeline(resolve(args.counter) or args.counter,
                           resolve(args.separator) or args.separator, device=device)

    # ------------------------------------------------------------------ the audio
    truth = None
    if args.input:
        path = resolve(args.input) or args.input
        if not os.path.exists(path):
            print(f"No such file: {path}", file=sys.stderr)
            return 2
        wav, _ = read_wav(path, sr=SR, mono=True)
        source = os.path.basename(path)
    else:
        from countsep.datasets import FrozenMixDataset
        store_root = args.store or autodetect_store()
        if store_root is None:
            print("--from_recipes also needs --store", file=sys.stderr)
            return 2
        store, bank = build_store_and_bank(store_root, args.test_split)
        recipes = resolve(args.from_recipes) or args.from_recipes
        dataset = FrozenMixDataset(store, bank, recipes, want="separate")
        picks = [i for i in range(len(dataset))
                 if args.n is None or int(dataset.recipes[i]["n_src"]) == args.n]
        if not picks:
            print(f"No mixture with n_src={args.n} in {recipes}", file=sys.stderr)
            return 2
        chosen = picks[args.index % len(picks)]
        item = dataset[chosen]
        wav = item["mix"].numpy()
        truth = int(item["n_src"])
        source = f"{os.path.basename(recipes)}[{chosen}]"

    limit = int(args.max_seconds * SR)
    if len(wav) > limit:
        print(f"  trimming {len(wav) / SR:.1f}s to the first {args.max_seconds:.0f}s "
              f"(raise --max_seconds to process more)")
        wav = wav[:limit]
    print(f"  input     : {source}   {len(wav) / SR:.2f} s @ {SR} Hz")

    # ------------------------------------------------------------------ run it
    out = separate_long(system, torch.from_numpy(np.ascontiguousarray(wav)).float().to(device))
    n_hat = int(out.n_hat[0])
    probs = out.count_probs[0].cpu().numpy()
    power = out.slot_power_db[0].cpu().numpy()

    banner("how many people are talking?")
    for cls, n in enumerate(N_LIST):
        bar = "#" * int(round(40 * float(probs[cls])))
        flag = "  <-- answer" if n == n_hat else ""
        plural = "s" if n > 1 else " "
        print(f"  {n} speaker{plural}  {float(probs[cls]):6.1%}  {bar}{flag}")
    if truth is not None:
        verdict = "CORRECT" if truth == n_hat else f"WRONG (off by {abs(truth - n_hat)})"
        print(f"\n  true count: {truth}    predicted: {n_hat}    {verdict}")
    if float(probs.max()) < 0.5:
        print("\n  Note: no class above 50 %. The model is genuinely unsure about this clip,")
        print("  which is a more useful thing to report than a confident wrong answer.")

    # ------------------------------------------------------------------ write the tracks
    banner("the separated tracks")
    print("  Slots are ranked by power, and the loss pushes surplus slots toward -30 dB, so")
    print(f"  the real talkers are the loud ones. A clear drop after slot {n_hat} agrees with")
    print("  the count; no drop means the counter and the separator disagree, which is worth")
    print("  hearing for yourself.\n")
    for k, db in enumerate(power, start=1):
        kept = "kept   " if k <= n_hat else "surplus"
        print(f"  slot {k}  {float(db):+7.1f} dB   {kept}")

    # One gain for every file, so the tracks keep their relative loudness -- which is
    # information. Normalising each track on its own would make a whisper and a shout
    # come out identical.
    tracks = out.sources[0].cpu().numpy()
    loudest = max(float(np.abs(wav).max()), float(np.abs(tracks[:n_hat]).max()), 1e-9)
    gain = 0.9 / loudest
    written = {"mixture.wav": wav}
    for k in range(n_hat):
        written[f"speaker_{k + 1:02d}.wav"] = tracks[k]
    if args.all_slots:
        for k in range(n_hat, len(tracks)):
            written[f"surplus_{k + 1:02d}.wav"] = tracks[k]
    if out.noise is not None:
        written["noise.wav"] = out.noise[0].cpu().numpy()
    for name, data in written.items():
        write_wav(os.path.join(out_dir, name), (data * gain).astype(np.float32), SR)
    print(f"\n  {len(written)} files -> {out_dir}")

    report = {
        "code_version": code_version(), "source": source,
        "seconds": len(wav) / SR, "n_hat": n_hat, "true_n": truth,
        "count_probs": {str(n): float(probs[c]) for c, n in enumerate(N_LIST)},
        "slot_power_db": [float(v) for v in power],
        "files": sorted(written),
    }
    path = os.path.join(out_dir, "infer_report.json")
    json_dump_atomic(report, path)
    if not os.path.exists(path) or os.path.getsize(path) < 64:
        print(f"\nFAILED: report missing or truncated at {path}", file=sys.stderr)
        return 2
    print(f"  report  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
