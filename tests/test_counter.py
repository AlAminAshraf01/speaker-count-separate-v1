"""Model invariants, most importantly: the network must be ONE function, not two.

The sibling project's counting head scored 44.9 % under fp16 autocast and 20.00 % in fp32 from
the *same checkpoint on the same inputs*, agreeing on 16.1 % of predictions. It was trained
under autocast and evaluated in fp32, so the function that was fitted and the function that was
measured were different. The cause was mechanical and is verified in `docs/DIAGNOSIS.md`:
`GlobalLayerNorm` was hand-written from `mean`/`var`/`sqrt`, which autocast does **not** promote
to fp32 the way it promotes `nn.LayerNorm`, and its `eps = 1e-8` is exactly `0.0` in fp16.

No accuracy metric catches that -- both numbers look like plausible accuracies. What catches it
is asserting the invariant directly: **run the same inputs through the same weights in both
precisions and require the predictions to agree.** That is `test_fp32_and_fp16_agree` below, and
it is the single most valuable test in this repo.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def _batch(n: int = 8, seed: int = 0) -> torch.Tensor:
    """RMS-normalised white noise. Fine for shape, finiteness and precision checks."""
    g = torch.Generator().manual_seed(seed)
    wav = torch.randn(n, 24000, generator=g)
    return wav / wav.pow(2).mean(-1, keepdim=True).sqrt()


def _distinct_batch(per_class: int = 4, seed: int = 0) -> torch.Tensor:
    """Acoustically DISTINCT inputs: sums of 1..5 sparse synthetic talkers.

    The distinction matters, and getting it wrong cost this repo a wrong claim. An earlier
    version of `test_logits_actually_vary_with_the_input` fed `torch.randn` waveforms, which
    are statistically identical to one another -- so a small across-batch logit spread was the
    *correct* answer, not evidence of a degenerate head. Measured on the same untrained model:
    white noise gives a spread of 3.0e-3, real N=1..5 mixtures give 2.3e-2, a factor of 7.4.
    A degeneracy test is only meaningful if its inputs actually differ.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(24000) / 8000.0
    out = []
    for n_src in (1, 2, 3, 4, 5):
        for _ in range(per_class):
            total = torch.zeros(24000)
            for _k in range(n_src):
                f0 = 85.0 + 145.0 * torch.rand(1, generator=g).item()
                wave = sum(torch.sin(2 * torch.pi * f0 * h * t) / h for h in range(1, 20))
                gate = torch.zeros(24000)
                pos = 0
                while pos < 24000:
                    on = int((0.12 + 0.33 * torch.rand(1, generator=g).item()) * 8000)
                    off = int((0.08 + 0.27 * torch.rand(1, generator=g).item()) * 8000)
                    end = min(24000, pos + on)
                    if end > pos:
                        gate[pos:end] = torch.hann_window(max(2, end - pos))[: end - pos]
                    pos = end + off
                src = wave * gate
                total = total + src / (src.pow(2).mean().sqrt() + 1e-9)
            out.append(total / (total.pow(2).mean().sqrt() + 1e-9))
    return torch.stack(out)


def test_every_pooling_builds_and_runs() -> None:
    from countsep.counter import POOLINGS, build_model

    wav = _batch()
    for name in POOLINGS:
        model = build_model(name).eval()
        with torch.no_grad():
            logits = model(wav)["logits"]
        assert logits.shape == (wav.shape[0], 5), f"{name}: got {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), f"{name}: non-finite logits"


def test_no_hand_rolled_normalisation_anywhere() -> None:
    """Autocast protects `nn.LayerNorm` and `nn.BatchNorm`. It cannot protect a hand-rolled one.

    A structural check on the source, because the failure it prevents is invisible at fp32 and
    only shows up as an accuracy difference much later, in a different script.
    """
    import countsep.counter as mod

    with open(mod.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    body = source.split('"""', 2)[-1]           # skip the module docstring, which discusses it
    for pattern in (".var(", "torch.var("):
        assert pattern not in body, (
            f"model.py computes {pattern} directly. Hand-written normalisation is what defeated "
            f"autocast 49 times in the sibling repo -- use nn.BatchNorm/nn.LayerNorm instead.")


def test_head_uses_batchnorm_not_layernorm() -> None:
    """LayerNorm over a pooled vector deletes the common-mode direction -- the count's code.

    BatchNorm normalises with running statistics across the batch, so a uniform shift that
    encodes "more talkers" survives. See finding 6.
    """
    from countsep.counter import build_model

    model = build_model()
    kinds = {type(m).__name__ for m in model.head.modules()}
    assert "BatchNorm1d" in kinds, f"head has no BatchNorm1d, only {kinds}"
    assert "LayerNorm" not in kinds, (
        "the head applies LayerNorm to its pooled vector, which subtracts that vector's own "
        "mean and deletes the common-mode direction a scalar count naturally lives in")


def test_fp32_and_fp16_agree() -> None:
    """THE regression test for the sibling project's headline failure.

    Same weights, same inputs, two precisions. If they disagree, the model trained under
    autocast is not the model evaluated in fp32, and any accuracy number is uninterpretable.

    Uses CPU bfloat16 autocast when there is no GPU. bfloat16 has fp32's exponent range so it
    is *gentler* than the fp16 the failure happened in -- meaning this is a weaker test locally
    than on a T4, and `scripts/` should re-run it on device. A failure here is therefore
    serious: the gentle version already broke.
    """
    from countsep.counter import POOLINGS, build_model

    wav = _distinct_batch(per_class=13, seed=3)      # 65 inputs, not 16 -- see below
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    for name in POOLINGS:
        model = build_model(name).eval()
        if device_type == "cuda":
            model, wav_d = model.cuda(), wav.cuda()
        else:
            wav_d = wav

        with torch.no_grad():
            hi = model(wav_d)["logits"].float()
            with torch.autocast(device_type=device_type, dtype=dtype):
                lo = model(wav_d)["logits"].float()

        assert torch.isfinite(lo).all(), f"{name}: autocast produced non-finite logits"

        # The PRIMARY assertion is on the logits, not on argmax. An untrained model's logits
        # are nearly tied across classes, so argmax flips on a perturbation far smaller than
        # any real decision margin -- an earlier version of this test used 16 samples and
        # failed `covariance` at 93.8 % on a single tie-break, which measured the tie and not
        # the arithmetic. The honest quantity is the numerical delta relative to the spread of
        # the logits across different inputs: how much of the signal is noise.
        # Measured on 65 distinct inputs, untrained: meanstd 0.06x, attentive 0.05x,
        # covariance 0.12x, eigen 0.08x. `covariance` is the most precision-sensitive of the
        # four, because eigh on a near-degenerate matrix is ill-conditioned -- worth knowing
        # before choosing it, and the reason this is checked per pooling.
        delta = (hi - lo).abs().max().item()
        spread = hi.std(0).mean().item()
        ratio = delta / max(spread, 1e-12)
        assert ratio < 0.5, (
            f"{name}: {dtype} perturbs the logits by {delta:.3e}, which is {ratio:.2f}x the "
            f"across-input logit spread ({spread:.3e}). Numerical noise of that size is "
            f"competing with the signal, which is how the sibling model came to score 44.9 % "
            f"in fp16 and 20.00 % in fp32 from one checkpoint.")

        agree = (hi.argmax(-1) == lo.argmax(-1)).float().mean().item()
        assert agree >= 0.90, (
            f"{name}: fp32 and {dtype} predictions agree on only {agree:.1%} of {len(wav)} "
            f"inputs (logit delta {ratio:.2f}x the spread). The sibling project's two "
            f"precisions agreed on 16.1 %.")


def test_logits_actually_vary_with_the_input() -> None:
    """A head whose output barely moves between inputs is degenerate, however it scores.

    The sibling model's fp32 head answered "1 speaker" for 1445 of 1500 test mixtures, which is
    a constant function wearing a classifier's clothes. That is detectable without a single
    label -- just push acoustically different inputs through and measure whether the output
    moves -- and it is worth checking continuously rather than discovering after a training run.

    Deliberately uses :func:`_distinct_batch` rather than white noise; see its docstring for
    why that distinction is not pedantic.
    """
    from countsep.counter import build_model

    model = build_model().eval()
    with torch.no_grad():
        logits = model(_distinct_batch(seed=5))["logits"]
    spread = logits.std(0).mean().item()
    assert spread > 1e-2, (
        f"logits vary by only {spread:.2e} across 20 acoustically distinct inputs "
        f"(N = 1..5); the head is not reading its input")


def test_stft_is_computed_in_fp32_under_autocast() -> None:
    """The front end is a fixed transform. Running it in fp16 buys nothing and hides drift."""
    from countsep.counter import LinearSTFT

    stft = LinearSTFT()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device_type, dtype=dtype):
        spec = stft(_batch(2))
    assert spec.dtype == torch.float32, f"STFT returned {spec.dtype} under autocast"


def test_parameter_count_stays_small() -> None:
    """The point of this project is that counting never needed 5.30 M parameters."""
    from countsep.counter import POOLINGS, build_model

    for name in POOLINGS:
        n = build_model(name).count_parameters()
        assert 1e5 < n < 1.5e6, f"{name}: {n / 1e6:.3f} M parameters is outside the intended band"


def test_dynamic_mixing_cannot_be_frozen_by_persistent_workers() -> None:
    """Persistent workers silently disable dynamic mixing, and the loss never tells you.

    ``DynamicMixDataset.set_epoch`` sets ``self.epoch_salt``, which seeds the per-item RNG.
    A DataLoader with ``num_workers > 0, persistent_workers=True`` hands each worker a
    SNAPSHOT of the dataset when it spawns and keeps those workers alive across epochs, so
    the mutation never crosses the process boundary.

    This shipped. Measured on this repo, three consecutive epochs at num_workers=2,
    persistent=True all rendered mix_id "dyn0_*" -- byte-identical mixtures every epoch. The
    separator trained on 12,000 distinct mixtures rather than the 360,000 its notebook
    printed, the counter on 128,000 rather than 2,560,000, and the only visible symptom was
    a counter that peaked at epoch 8 and then memorised its way to 99.5 % train accuracy
    against 83.3 % validation.

    Nothing about a loss going down could have caught it, which is why ``build_loader``
    refuses the combination instead of warning about it.
    """
    from countsep.datasets import build_loader

    class Mixes(torch.utils.data.Dataset):
        """Stands in for DynamicMixDataset: the epoch is mutable state on the instance."""

        epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> dict:
            return {"mix": torch.zeros(4)}

    try:
        build_loader(Mixes(), batch_size=2, shuffle=False, num_workers=2, persistent=True)
        raise AssertionError(
            "build_loader accepted persistent workers over a dataset with set_epoch(). "
            "Every epoch would draw identical mixtures and nothing would say so.")
    except ValueError as exc:
        assert "set_epoch" in str(exc) and "persistent" in str(exc), (
            f"the refusal must name the cause and the remedy; got: {exc}")

    # The safe combinations must still be allowed, or the guard is just breakage.
    build_loader(Mixes(), batch_size=2, shuffle=False, num_workers=2, persistent=False)
    build_loader(Mixes(), batch_size=2, shuffle=False, num_workers=0)

    class Frozen(torch.utils.data.Dataset):
        """A frozen set has no epoch to advance, so persistent workers are correct for it."""

        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> dict:
            return {"mix": torch.zeros(4)}

    assert not hasattr(Frozen(), "set_epoch"), "the control must not carry set_epoch"
    build_loader(Frozen(), batch_size=2, shuffle=False, num_workers=2, persistent=True)


if __name__ == "__main__":
    sys.exit(run_checks({
        "every pooling builds and runs": test_every_pooling_builds_and_runs,
        "no hand-rolled normalisation": test_no_hand_rolled_normalisation_anywhere,
        "head uses BatchNorm not LayerNorm": test_head_uses_batchnorm_not_layernorm,
        "fp32 and fp16 agree": test_fp32_and_fp16_agree,
        "logits vary with the input": test_logits_actually_vary_with_the_input,
        "STFT stays fp32 under autocast": test_stft_is_computed_in_fp32_under_autocast,
        "parameter count stays small": test_parameter_count_stays_small,
        "dynamic mixing survives the dataloader": test_dynamic_mixing_cannot_be_frozen_by_persistent_workers,
    }))
