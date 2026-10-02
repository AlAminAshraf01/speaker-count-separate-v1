"""Torch datasets: dynamic mixing for training, frozen recipes for evaluation.

Two deliberate departures from the sibling project's `datasets.py`.

**The batch is lean.** That one yielded `mix`, `refs` (a ``(5, 24000)`` float32 tensor),
`noise` (another ``(24000,)``), `n_src`, `cls`, `is_noisy`, `snr_db` and `mix_id` — because a
separator needs the isolated sources. A counter never looks at them. At batch 64 that is
**33 MB per batch of pure waste**, allocated, collated and pinned every step, on a machine
with 4 vCPU where the dataloader is the bottleneck. Here a training batch is the waveform and
the label, and the metadata needed to *report* (`n_src`, `snr_db`, `is_noisy`) rides along as
small scalars.

**The mixing settings are the ones the measurements justify**, not the ones inherited. See
:data:`DEFAULT_MIXING` — babble is gone, gain jitter is halved, the SNR floor is raised. Each
change is priced in ``docs/DATA.md`` §4.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import MAX_N_SRC, N_LIST, SEG_LEN, n_to_class
from .mixing import read_recipes, render_recipe, sample_recipe

DEFAULT_MIXING: dict = {
    "gain_db_range": (-2.5, 2.5),   # was (-5, 5): +-5 dB cost 9 accuracy points
    "snr_db_range": (5.0, 20.0),    # was (0, 20): the 0-5 dB band drowns the cue
    "p_clean": 0.25,                # a clean condition, reported separately
    "noise_kinds": ("white", "pink", "brown"),   # NO babble: babble is 4-8 real talkers
}
"""Mixing settings chosen against measured costs. See ``docs/DATA.md`` section 4.

``noise_kinds`` excluding babble is not a tuning choice, it is a correctness one. With the
sibling project's ``p_clean=0.2`` and five noise kinds drawn uniformly, ~20 % of training
mixtures carried babble, and ~10 % of all mixtures carried it below 10 dB SNR. A "2 speaker"
clip with six audible talkers behind it has a **factually wrong label**.
"""


WANTS: tuple[str, ...] = ("count", "separate")


def _to_batch(rendered: dict, want: str = "count", *, max_n_src: int = MAX_N_SRC,
              seg_len: int = SEG_LEN) -> dict:
    """Pack a rendered mixture into the dict the requested task needs -- and nothing more."""
    if want not in WANTS:
        raise ValueError(f"unknown want={want!r}; use one of {WANTS}")
    n_src = int(rendered["n_src"])
    item = {
        "mix": torch.from_numpy(np.ascontiguousarray(rendered["mix"])),
        "cls": torch.tensor(n_to_class(n_src), dtype=torch.int64),
        "n_src": torch.tensor(n_src, dtype=torch.int64),
        "is_noisy": torch.tensor(int(rendered["is_noisy"]), dtype=torch.int64),
        "snr_db": torch.tensor(float(rendered["snr_db"]), dtype=torch.float32),
        "mix_id": str(rendered["mix_id"]),
    }
    if want == "separate":
        refs = np.zeros((max_n_src, seg_len), dtype=np.float32)
        refs[:n_src] = rendered["sources"][:max_n_src]
        item["refs"] = torch.from_numpy(refs)
        item["noise"] = torch.from_numpy(np.ascontiguousarray(rendered["noise"]))
    return item


class DynamicMixDataset(Dataset):
    """Endless training set: a freshly sampled mixture per index, per epoch.

    Dynamic mixing is free here because the sources are already packed as one int16 memmap,
    and it means the model never sees the same mixture twice -- which matters more for
    counting than for separation, because with only 201 usable target speakers a counter can
    otherwise partly solve the task by memorising voices.
    """

    def __init__(self, store, bank, *, want: str = "count",
                 n_list: Sequence[int] = N_LIST, steps: int = 100_000,
                 seg_len: int = SEG_LEN, seed: int = 72,
                 gain_db_range: tuple[float, float] = DEFAULT_MIXING["gain_db_range"],
                 snr_db_range: tuple[float, float] = DEFAULT_MIXING["snr_db_range"],
                 p_clean: float = DEFAULT_MIXING["p_clean"],
                 min_crop_rms_ratio: float = 0.3) -> None:
        self.store, self.bank = store, bank
        self.want = str(want)
        self.n_list = tuple(int(n) for n in n_list)
        self.steps, self.seg_len, self.seed = int(steps), int(seg_len), int(seed)
        self.gain_db_range = tuple(gain_db_range)
        self.snr_db_range = tuple(snr_db_range)
        self.p_clean = float(p_clean)
        self.min_crop_rms_ratio = float(min_crop_rms_ratio)
        self.epoch_salt = 0

        if max(self.n_list) > len(store.speakers):
            raise ValueError(f"split {store.split!r} has {len(store.speakers)} target speakers "
                             f"but n_list needs {max(self.n_list)} distinct ones")
        if "babble" in getattr(bank, "kinds", ()):
            raise ValueError(
                "the noise bank includes 'babble', which is 4-8 real talkers -- every mixture "
                "drawn with it carries a wrong count label. Build the bank with "
                "kinds=DEFAULT_MIXING['noise_kinds']. See docs/DATA.md section 3.")

    def set_epoch(self, epoch: int) -> None:
        """Change the per-item seed salt so each epoch draws fresh mixtures."""
        self.epoch_salt = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng((self.seed, self.epoch_salt, int(idx)))
        n_src = int(rng.choice(self.n_list))          # uniform: the classes stay balanced
        recipe = sample_recipe(
            self.store, self.bank, n_src, rng, seg_len=self.seg_len,
            gain_db_range=self.gain_db_range, snr_db_range=self.snr_db_range,
            p_clean=self.p_clean, min_crop_rms_ratio=self.min_crop_rms_ratio,
            mix_id=f"dyn{self.epoch_salt}_{idx}_n{n_src}")
        return _to_batch(render_recipe(recipe, self.store, self.bank, seg_len=self.seg_len),
                         self.want, seg_len=self.seg_len)


class FrozenMixDataset(Dataset):
    """Deterministic evaluation set rendered from a committed recipe CSV.

    Byte-identical every run, which is what "frozen test set" has to mean if numbers reported
    in different weeks are going to be comparable.
    """

    def __init__(self, store, bank, recipes_path: str, *, want: str = "count",
                 seg_len: int = SEG_LEN,
                 n_list: Sequence[int] | None = None, limit: int | None = None) -> None:
        self.store, self.bank = store, bank
        self.want = str(want)
        self.seg_len = int(seg_len)
        self.recipes_path = str(recipes_path)
        rows = read_recipes(recipes_path)
        if n_list is not None:
            keep = {int(n) for n in n_list}
            rows = [r for r in rows if int(r["n_src"]) in keep]
        if limit is not None:
            # Stratified, NOT rows[:limit]. The recipe file is written grouped by speaker
            # count, so a head-slice of 40 rows is 40 one- and two-speaker mixtures and the
            # per-N table silently comes back empty for N=3..5. Round-robin across the counts
            # keeps a truncated run representative of the full set.
            by_n: dict[int, list[dict]] = {}
            for r in rows:
                by_n.setdefault(int(r["n_src"]), []).append(r)
            picked: list[dict] = []
            idx = 0
            while len(picked) < int(limit) and any(idx < len(v) for v in by_n.values()):
                for n in sorted(by_n):
                    if idx < len(by_n[n]) and len(picked) < int(limit):
                        picked.append(by_n[n][idx])
                idx += 1
            rows = picked
        if not rows:
            raise ValueError(f"no recipes left after filtering {recipes_path!r}")
        self.recipes = rows

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, idx: int) -> dict:
        return _to_batch(render_recipe(self.recipes[int(idx)], self.store, self.bank,
                                       seg_len=self.seg_len), self.want, seg_len=self.seg_len)

    def counts_per_n(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for r in self.recipes:
            out[int(r["n_src"])] = out.get(int(r["n_src"]), 0) + 1
        return dict(sorted(out.items()))

    def speakers(self) -> set[str]:
        """Every speaker appearing anywhere in this set.

        The count of these -- not the number of mixtures -- is the number of independent units
        for a bootstrap. For the test split it is 32, while the mixture count is 1500.
        """
        from .mixing import recipe_speakers
        return {s for r in self.recipes for s in recipe_speakers(r, self.store)}


def collate_count(items: Sequence[dict]) -> dict:
    """Stack tensors; keep ``mix_id`` as a list of strings."""
    out: dict = {}
    for key in items[0]:
        if key == "mix_id":
            out[key] = [str(it[key]) for it in items]
        else:
            out[key] = torch.stack([it[key] for it in items], dim=0)
    return out


def build_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, num_workers: int = 2,
                 drop_last: bool = False, pin_memory: bool | None = None,
                 persistent: bool = True, prefetch_factor: int = 4):
    """DataLoader tuned for a Kaggle box: 4 vCPU, and the mixer is the bottleneck.

    ``prefetch_factor`` is 4 rather than v0's 2. For a counting loader the batch is ~7x
    smaller (no references tensor, no noise tensor), so a deeper queue costs little memory and
    buys real tolerance for the jitter in rendering a 5-source mixture. Lower it to 2 for a
    separation loader if host RAM gets tight.
    """
    from .utils import set_worker_seed

    # persistent_workers + set_epoch is a silent data bug, so it is refused rather than
    # warned about. Workers hold a SNAPSHOT of the dataset taken when they spawned;
    # set_epoch mutates the parent's copy and never reaches them. Measured on this repo:
    # with num_workers=2, persistent=True, three consecutive epochs all rendered mix_id
    # "dyn0_*" -- the same mixtures every epoch. It cost the separator 29 of its 30 epochs
    # of fresh data (12,000 distinct mixtures instead of the 360,000 the notebook printed)
    # and the counter 19 of 20, and it is invisible: the loss still falls, because the
    # model is memorising a fixed set.
    if num_workers > 0 and persistent and hasattr(dataset, "set_epoch"):
        raise ValueError(
            f"{type(dataset).__name__} has set_epoch(), so persistent workers would freeze "
            f"it at whatever epoch they spawned in and every epoch would draw identical "
            f"mixtures. Pass persistent=False for a dynamically-mixed training loader.")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    kwargs: dict = {
        "batch_size": int(batch_size), "shuffle": bool(shuffle),
        "num_workers": int(num_workers), "collate_fn": collate_count,
        "pin_memory": bool(pin_memory), "drop_last": bool(drop_last),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent)
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["worker_init_fn"] = set_worker_seed
    return torch.utils.data.DataLoader(dataset, **kwargs)
