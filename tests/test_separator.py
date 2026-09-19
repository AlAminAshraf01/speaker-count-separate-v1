"""Regression tests for the three v0 defects this version exists to fix.

Each of these would pass silently as "a training run that did not work very well". None of them
would have been caught by a loss going down, which is why they are asserted directly.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402


def test_hard_clamp_has_no_gradient_above_tau() -> None:
    """v0's clamp PEAKED at tau instead of vanishing there.

    Its docstring said it stopped "one easy example dominating the batch". Multiplied by the
    way the SI-SDR gradient itself grows (~10**(s/20)), the soft version's product runs
    0.999 / 9.091 / **15.811** / 9.091 at 0 / 20 / 30 / 40 dB -- maximised exactly at the clamp
    point. Since a 1-speaker mixture is solved by copying the input, it sits in that band and
    carried ~16x the gradient of a hard 5-speaker one.
    """
    from countsep.losses import hard_clamp, soft_clamp

    for s in (35.0, 40.0, 60.0):
        x = torch.tensor([s], requires_grad=True)
        hard_clamp(x, 30.0).backward()
        assert float(x.grad) == 0.0, (
            f"hard_clamp still passes gradient at {s} dB, above tau=30")

    below = torch.tensor([10.0], requires_grad=True)
    hard_clamp(below, 30.0).backward()
    assert abs(float(below.grad) - 1.0) < 1e-6, "hard_clamp should be identity below tau"

    # and confirm the v0 version really does peak, so the comparison in the docs stays true
    products = []
    for s in (0.0, 20.0, 30.0, 40.0):
        x = torch.tensor([s], requires_grad=True)
        soft_clamp(x, 30.0).backward()
        products.append(float(x.grad) * 10 ** (s / 20))
    assert products[2] == max(products), (
        f"soft_clamp no longer peaks at tau; the documented rationale is stale: {products}")


def test_silence_term_is_a_per_item_average() -> None:
    """v0 divided by the number of leftover SLOTS, which made it 1-speaker-weighted.

    Leftover slots run 4, 3, 2, 1, 0 for N = 1..5, so in a balanced batch the single N=1 item
    supplied 40 % of the term and the N=5 item supplied none.

    The check: build two batches whose per-item silence violation is identical but whose
    leftover-slot counts differ wildly (all N=1 versus all N=5-with-one-spare). A per-item
    average is insensitive to the slot count; a per-slot average is not.
    """
    from countsep.losses import RectangularPITLoss

    torch.manual_seed(0)
    loss_fn = RectangularPITLoss(w_sep=0.0, w_count=0.0, w_noise=0.0, w_sil=1.0)
    B, T = 6, 4000

    def sil_for(n_src: int) -> float:
        est = torch.zeros(B, 6, T)
        refs = torch.zeros(B, 5, T)
        mix = torch.randn(B, T) * 0.1
        # every item leaks the SAME loud signal into exactly ONE spare slot
        est[:, 4] = torch.randn(B, T) * 5.0
        batch = {"refs": refs, "mix": mix, "noise": torch.zeros(B, T),
                 "n_src": torch.full((B,), n_src), "cls": torch.full((B,), n_src - 1),
                 "is_noisy": torch.zeros(B, dtype=torch.long)}
        _total, logs = loss_fn({"est": est, "count_logits": None}, batch)
        return logs["sil"]

    one, four = sil_for(1), sil_for(4)
    assert abs(one - four) / max(abs(one), 1e-9) < 0.05, (
        f"the silence term still depends on the leftover-slot count: N=1 gives {one:.4f} and "
        f"N=4 gives {four:.4f} for the same per-item violation. That is v0's bug.")


def test_global_layer_norm_survives_autocast() -> None:
    """autocast promotes nn.LayerNorm to fp32 but cannot protect a hand-written one.

    v0 had 49 hand-written norms with eps=1e-8, which is exactly 0.0 in fp16, so every one of
    them divided by an unguarded sqrt(var). The same checkpoint then scored 44.9 % in fp16 and
    20.00 % in fp32. v1's version computes its statistics in fp32 regardless.
    """
    from countsep.constants import MODEL_EPS
    from countsep.separator import GlobalLayerNorm

    assert float(torch.tensor(MODEL_EPS, dtype=torch.float16)) > 0.0, (
        f"MODEL_EPS={MODEL_EPS} is not representable in fp16 -- the v0 mistake exactly")

    norm = GlobalLayerNorm(32).eval()
    x = torch.randn(4, 32, 500) * 30.0
    with torch.no_grad():
        hi = norm(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lo = norm(x.to(torch.bfloat16)).float()
    assert torch.isfinite(lo).all(), "gLN produced non-finite output under autocast"
    rel = (hi - lo).abs().max() / hi.std().clamp_min(1e-9)
    assert float(rel) < 0.5, f"gLN diverges under autocast by {float(rel):.2f}x its own spread"


def test_the_separator_has_no_count_head() -> None:
    """The whole point of v1: one job per model."""
    from countsep.separator import ModelConfig, build_separator

    model = build_separator(ModelConfig(n_filters=64, bottleneck=32, hidden=64, skip=32,
                                        n_blocks=2, n_repeats=1))
    names = [n for n, _ in model.named_parameters()]
    assert not any("count" in n for n in names), \
        f"the separator still carries counting parameters: {[n for n in names if 'count' in n]}"
    out = model(torch.randn(2, 8000))
    assert out["count_logits"] is None, "the separator claims to produce count logits"
    assert out["est"].shape[1] == model.n_slots


def test_separator_loss_runs_without_a_count_head() -> None:
    """The shared loss must accept count_logits=None rather than crashing on it."""
    from countsep.losses import RectangularPITLoss

    B, T = 3, 4000
    batch = {"refs": torch.randn(B, 5, T), "mix": torch.randn(B, T),
             "noise": torch.randn(B, T), "n_src": torch.tensor([1, 3, 5]),
             "cls": torch.tensor([0, 2, 4]), "is_noisy": torch.tensor([1, 0, 1])}
    total, logs = RectangularPITLoss(w_count=0.0)(
        {"est": torch.randn(B, 6, T), "count_logits": None}, batch)
    assert torch.isfinite(total), "separator loss is not finite"
    assert logs["count"] == 0.0, "a counting term leaked into the separator loss"


def test_pipeline_orders_slots_by_loudness_and_takes_n_hat() -> None:
    """Slot selection reads the thing the loss optimised: spares are pushed toward silence."""
    from countsep.counter import build_model as build_counter
    from countsep.pipeline import CountThenSeparate
    from countsep.separator import ModelConfig, build_separator

    system = CountThenSeparate(
        build_counter(),
        build_separator(ModelConfig(n_filters=64, bottleneck=32, hidden=64, skip=32,
                                    n_blocks=2, n_repeats=1)))
    mix = torch.randn(3, 24000)
    mix = mix / mix.pow(2).mean(-1, keepdim=True).sqrt()
    out = system(mix)
    assert bool((out.slot_power_db[:, :-1] >= out.slot_power_db[:, 1:]).all()), \
        "slots are not ordered loudest-first"
    assert out.n_hat.shape == (3,) and int(out.n_hat.min()) >= 1 and int(out.n_hat.max()) <= 5
    for i in range(3):
        assert len(out.take(i)) == int(out.n_hat[i])

    forced = system(mix, n_true=torch.tensor([2, 3, 4]))
    assert forced.n_hat.tolist() == [2, 3, 4], \
        "forcing the true count is how separation is measured apart from counting"


def test_long_recordings_keep_each_speaker_on_one_track() -> None:
    """``separate_long`` overlap-adds windows, and slots are re-ranked by loudness in EVERY
    window -- so the moment two talkers trade places in the volume ranking, their slots trade
    places, and overlap-add welds half of one voice onto half of another.

    Measured before the fix, on the two-tone probe rebuilt below: output track 0 correlated
    **0.908 with speaker A over the first half and 1.000 with speaker B over the second**. Per
    window the separation was flawless; end to end a track changed who it held.

    This check verifies its own teeth. It runs the same probe twice -- once normally, once with
    the alignment stubbed out -- and demands the second one FAIL. A regression test for a
    permutation bug that passes with the fix removed is not a test, and this project has
    already shipped one of those.
    """
    import countsep.pipeline as pipeline_mod
    from countsep.constants import SEG_LEN, SR
    from countsep.pipeline import CountThenSeparate, separate_long

    class FrequencySplitSeparator(torch.nn.Module):
        """Splits its input at 300 Hz. Linear, so it survives the RMS rescaling exactly."""

        class cfg:
            max_n_src = 5
            predict_noise = True

        def forward(self, x: torch.Tensor) -> dict:
            spec = torch.fft.rfft(x, dim=-1)
            freq = torch.fft.rfftfreq(x.shape[-1], d=1.0 / SR).to(x.device)
            lo = torch.fft.irfft(spec * (freq < 300), n=x.shape[-1], dim=-1)
            hi = torch.fft.irfft(spec * (freq >= 300), n=x.shape[-1], dim=-1)
            zero = torch.zeros_like(lo)
            return {"est": torch.stack([lo, hi, zero, zero, zero, zero], dim=1),
                    "count_logits": None}

    class AlwaysTwo(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict:
            return {"logits": torch.tensor([[0.0, 9.0, 0.0, 0.0, 0.0]]).expand(x.shape[0], 5)}

    total = SEG_LEN * 2                                  # 6 s -> three 50 %-overlapped windows
    t = torch.arange(total, dtype=torch.float64) / SR
    ramp = torch.linspace(0.0, 1.0, total, dtype=torch.float64)
    speakers = {                                         # they swap loudness mid-file
        "A": ((1.0 - ramp) * 0.9 + 0.05) * torch.sin(2 * torch.pi * 200 * t),
        "B": (ramp * 0.9 + 0.05) * torch.sin(2 * torch.pi * 400 * t),
    }
    mix = (speakers["A"] + speakers["B"]).to(torch.float32).unsqueeze(0)
    system = CountThenSeparate(AlwaysTwo(), FrequencySplitSeparator()).eval()

    def correlation(x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x - x.mean(), y - y.mean()
        scale = x.norm() * y.norm()
        return abs(float(x @ y / scale)) if float(scale) > 1e-12 else 0.0

    def every_track_holds_one_speaker() -> bool:
        """True when no output track changes which speaker it carries."""
        tracks = separate_long(system, mix).sources[0].double()
        half = total // 2
        for track in tracks[:2]:
            halves = []
            for piece in (slice(0, half), slice(half, total)):
                scores = {k: correlation(track[piece], v[piece]) for k, v in speakers.items()}
                halves.append(max(scores, key=scores.get))
                if scores[halves[-1]] < 0.7:
                    return False
            if halves[0] != halves[1]:
                return False
        return True

    assert every_track_holds_one_speaker(),         "a speaker changed output track partway through the file"

    # A global reordering of tracks is allowed -- "loudest first" is the contract, and it is
    # asserted separately. What must not happen is a track changing hands mid-recording.
    original = pipeline_mod._align_to_previous
    try:
        pipeline_mod._align_to_previous = lambda prev_tail, chunk: chunk
        assert not every_track_holds_one_speaker(),             "this test cannot detect the bug it exists for: it passes with alignment removed"
    finally:
        pipeline_mod._align_to_previous = original


if __name__ == "__main__":
    sys.exit(run_checks({
        "hard clamp has no gradient above tau": test_hard_clamp_has_no_gradient_above_tau,
        "silence term is a per-item average": test_silence_term_is_a_per_item_average,
        "gLN survives autocast": test_global_layer_norm_survives_autocast,
        "the separator has no count head": test_the_separator_has_no_count_head,
        "separator loss runs without a count head": test_separator_loss_runs_without_a_count_head,
        "pipeline orders slots and takes n_hat": test_pipeline_orders_slots_by_loudness_and_takes_n_hat,
        "long audio keeps each speaker on one track": test_long_recordings_keep_each_speaker_on_one_track,
    }))
