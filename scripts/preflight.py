#!/usr/bin/env python3
"""Check before you spend. Thirty seconds, no GPU quota, writes nothing.

    python scripts/preflight.py --for data          # before 00 / 01
    python scripts/preflight.py --for cpu           # before 02 / 03
    python scripts/preflight.py --for train --store ... --recipes_dev ...

Every row is either OK, WARN or FAIL, and a FAIL returns a non-zero exit code so the notebook
cell stops instead of scrolling past. The checks exist because each of them has already cost
somebody a session: a notebook running cells imported before the last push, a GPU notebook that
never had a GPU attached, a store that was never packed, a model whose two precisions disagree.

That last one is the important one and it is unique to this project. The counter it replaces
scored 44.9 % under fp16 autocast and 20.00 % in fp32 **from the same checkpoint**, because it
trained under one precision and was evaluated in the other. Here that is a preflight check, so
it fails in thirty seconds rather than after eleven hours.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from _common import add_common_args, banner, code_version, resolve

Row = tuple[str, str, str]      # (name, status, detail)

PROFILES: dict[str, tuple[str, ...]] = {
    "data":  ("code", "cells", "no_gpu", "disk", "ram", "deps"),
    "cpu":   ("code", "cells", "disk", "ram", "deps", "store"),
    "train": ("code", "cells", "gpu", "disk", "ram", "deps", "store", "recipes", "precision"),
    "eval":  ("code", "cells", "disk", "deps", "store", "recipes", "checkpoint"),
}


def check_code(args) -> Row:
    return ("code", "OK", code_version())


def check_cells(args) -> Row:
    """Are the notebook cells older than the repo they just cloned?"""
    if not args.cells_src or not args.cells_sha:
        return ("cells", "WARN", "unstamped -- re-import the notebook to enable this check")
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "notebooks", "src")
    try:
        import hashlib
        digest = hashlib.sha256()
        for part in (args.cells_src, "_bootstrap.py"):
            with open(os.path.join(src_dir, part), "rb") as fh:
                digest.update(fh.read().replace(b"\r\n", b"\n"))
            digest.update(b"\0")
        current = digest.hexdigest()[:16]
    except OSError as exc:
        return ("cells", "WARN", f"cannot verify ({exc})")
    if current == args.cells_sha:
        return ("cells", "OK", f"match repo ({current})")
    return ("cells", "FAIL",
            f"STALE: cells are {args.cells_sha}, repo is {current}. Re-import the .ipynb "
            f"from notebooks/ -- you are running code from before the last push.")


def check_gpu(args) -> Row:
    try:
        import torch
    except ImportError:
        return ("gpu", "FAIL", "torch is not installed")
    if not torch.cuda.is_available():
        return ("gpu", "FAIL",
                "no GPU. Settings -> Accelerator -> GPU T4 x2, then re-run this cell.")
    props = torch.cuda.get_device_properties(0)
    return ("gpu", "OK", f"{props.name}, {props.total_memory / 1e9:.1f} GB, "
                         f"{torch.cuda.device_count()} device(s)")


def check_no_gpu(args) -> Row:
    """A CPU notebook with a GPU attached is burning a 30 h/week quota on soundfile."""
    try:
        import torch
    except ImportError:
        return ("no_gpu", "OK", "torch not imported")
    if torch.cuda.is_available():
        return ("no_gpu", "WARN",
                "a GPU is attached but this notebook does not need one -- it is spending "
                "quota for nothing. Settings -> Accelerator -> None.")
    return ("no_gpu", "OK", "no accelerator, as intended")


def check_disk(args) -> Row:
    free = shutil.disk_usage("/kaggle/working" if os.path.isdir("/kaggle/working") else ".").free
    gb = free / 1e9
    if gb < 2:
        return ("disk", "FAIL", f"{gb:.1f} GB free -- not enough to write a store")
    if gb < 6:
        return ("disk", "WARN", f"{gb:.1f} GB free (Kaggle gives 20 GB in /kaggle/working)")
    return ("disk", "OK", f"{gb:.1f} GB free")


def check_ram(args) -> Row:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            info = {k.strip(): v for k, v in (ln.split(":", 1) for ln in fh)}
        total = int(info["MemTotal"].split()[0]) / 1e6
        avail = int(info["MemAvailable"].split()[0]) / 1e6
    except (OSError, KeyError, ValueError):
        return ("ram", "OK", "not measurable on this platform")
    status = "WARN" if avail < 2.0 else "OK"
    return ("ram", status, f"{avail:.1f} of {total:.1f} GB available")


def check_deps(args) -> Row:
    missing = []
    for mod in ("numpy", "scipy", "sklearn", "soundfile"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return ("deps", "FAIL", f"missing: {', '.join(missing)}")
    import numpy
    import scipy
    import sklearn
    return ("deps", "OK", f"numpy {numpy.__version__}, scipy {scipy.__version__}, "
                          f"sklearn {sklearn.__version__}")


def check_store(args) -> Row:
    from _common import autodetect_store

    root = autodetect_store(resolve(args.store))
    if root is None:
        return ("store", "FAIL",
                "no packed store found. Run notebook 00, or attach its output dataset.")
    try:
        from countsep.pack import SourceStore
        sizes = {}
        for split in ("train-100", "dev", "test"):
            try:
                st = SourceStore(root, split, mmap=True)
                sizes[split] = (len(st.target_idx), len(st.speakers))
            except FileNotFoundError:
                pass
    except Exception as exc:                        # noqa: BLE001
        return ("store", "FAIL", f"{root}: {exc}")
    if not sizes:
        return ("store", "FAIL", f"{root} has no packed splits")
    detail = ", ".join(f"{k}: {u} utts/{s} spk" for k, (u, s) in sizes.items())
    return ("store", "OK", f"{root}  ({detail})")


def check_recipes(args) -> Row:
    paths = [p for p in (args.recipes_dev, args.recipes_test) if p]
    if not paths:
        return ("recipes", "WARN", "no --recipes_dev/--recipes_test given")
    out = []
    for p in paths:
        full = resolve(p) or p
        if not os.path.exists(full):
            return ("recipes", "FAIL", f"missing {full} -- run 01_make_frozen_sets.py")
        from countsep.mixing import read_recipes
        rows = read_recipes(full)
        per_n: dict[int, int] = {}
        for r in rows:
            per_n[int(r["n_src"])] = per_n.get(int(r["n_src"]), 0) + 1
        if len(set(per_n.values())) > 1:
            return ("recipes", "WARN",
                    f"{os.path.basename(full)} is NOT balanced: {per_n}. Chance is no longer "
                    f"1/K and the naive baseline changes.")
        out.append(f"{os.path.basename(full)}: {len(rows)} ({sorted(per_n)})")
    return ("recipes", "OK", "; ".join(out))


def check_checkpoint(args) -> Row:
    if not args.ckpt:
        return ("checkpoint", "WARN", "no --ckpt given")
    full = resolve(args.ckpt) or args.ckpt
    if not os.path.exists(full):
        return ("checkpoint", "FAIL", f"missing {full}")
    import torch
    state = torch.load(full, map_location="cpu", weights_only=False)
    return ("checkpoint", "OK",
            f"epoch {state.get('epoch', '?')}, best {state.get('best_metric', float('nan')):.4f}, "
            f"pooling {state.get('cfg', {}).get('pooling', '?')}")


def check_precision(args) -> Row:
    """THE check this project exists because of. See the module docstring."""
    try:
        import torch

        from countsep.model import build_model, precision_agreement, probe_batch
    except ImportError as exc:
        return ("precision", "WARN", f"cannot check ({exc})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    # probe_batch, NOT torch.randn: white-noise inputs are statistically identical to each
    # other, which shrinks the across-input spread ~7x and inflates the ratio below into a
    # false failure. See probe_batch's docstring.
    res = precision_agreement(model, probe_batch(per_class=13))
    if not res["finite"]:
        return ("precision", "FAIL", f"{res['dtype']} produced non-finite logits")
    status = "OK" if res["ratio"] < 0.5 else "FAIL"
    return ("precision", status,
            f"{res['dtype']} perturbs logits by {res['ratio']:.2f}x their across-input spread "
            f"(fail above 0.50); argmax agrees on {res['agree']:.0%}")


CHECKS = {"code": check_code, "cells": check_cells, "gpu": check_gpu, "no_gpu": check_no_gpu,
          "disk": check_disk, "ram": check_ram, "deps": check_deps, "store": check_store,
          "recipes": check_recipes, "checkpoint": check_checkpoint,
          "precision": check_precision}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--for", dest="profile", default="cpu", choices=sorted(PROFILES))
    ap.add_argument("--recipes_dev", default=None)
    ap.add_argument("--recipes_test", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cells_src", default=None)
    ap.add_argument("--cells_sha", default=None)
    args = ap.parse_args()

    banner(f"preflight --for {args.profile}   (no GPU quota, writes nothing)")
    rows: list[Row] = []
    for name in PROFILES[args.profile]:
        try:
            rows.append(CHECKS[name](args))
        except Exception as exc:                    # noqa: BLE001 - a broken check is a WARN
            rows.append((name, "WARN", f"check raised {type(exc).__name__}: {exc}"))

    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"  {status:<4} {name.ljust(width)}  {detail}")

    failed = [r for r in rows if r[1] == "FAIL"]
    warned = [r for r in rows if r[1] == "WARN"]
    print(f"\n  {len(rows) - len(failed) - len(warned)} OK, {len(warned)} WARN, "
          f"{len(failed)} FAIL")
    if failed:
        print("\nFix the FAIL rows before continuing -- they will not fix themselves later,"
              "\nand every one of them has already cost somebody a session.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
