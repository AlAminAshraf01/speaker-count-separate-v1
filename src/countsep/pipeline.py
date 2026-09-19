"""Count first, then separate, then keep N slots. The v1 system, end to end.

    counter  : CountCRNN  ->  N_hat in 1..5
    separator: SepNet     ->  5 speaker slots + 1 noise slot
    output   : the N_hat loudest speaker slots

Two specialists rather than one multi-task model, and that is the central design decision of
v1. The measurement behind it, from ``docs/DIAGNOSIS.md``:

* In v0 the counting objective supplied **0.53 %** of the gradient reaching the shared trunk
  while separation supplied 86.14 % -- 633:1 at the encoder. Counting never stood a chance.
* Separation was hurt too, not just counting. ``14_objective_ablation.py`` in v0 priced the
  auxiliary objectives at **-8.7 dB** on a single batch, and the pooled model reached 0.08 dB
  on the official Libri2Mix test set where a fixed-N=2 separation-only control reached
  **5.49 dB in a third of the epochs**.
* Sharing saves almost nothing anyway: removing the separator from v0's model saved only
  **8.4 %** of forward FLOPs, because the TCN is the cost. A dedicated 0.49 M-parameter
  counter adds roughly 5 % on top of a 5.25 M separator.

So the two jobs are trained apart, each with 100 % of its own gradient, and joined at
inference. Nothing is shared except the mixing pipeline and the frozen evaluation sets.

**Why slot selection is by loudness.** The separator is trained with a rectangular PIT loss
that pushes surplus slots toward ``SILENCE_DB`` (-30 dB relative to the mixture), so at
inference the real speakers are the loud slots and the spares are quiet. Ranking by power and
keeping the top ``N_hat`` is therefore reading the thing the loss actually optimised. It also
means a miscount degrades gracefully: predicting 3 when the truth is 4 returns the three
loudest real speakers rather than a scrambled set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import SEG_LEN, SR, class_to_n


@dataclass
class PipelineOutput:
    """What the system returns for one batch."""

    n_hat: torch.Tensor          # (B,) predicted speaker count, 1..5
    count_probs: torch.Tensor    # (B, n_classes) softmax over the counter's logits
    sources: torch.Tensor        # (B, max_n_src, T) speaker slots, ordered loudest first
    noise: torch.Tensor | None   # (B, T) the noise slot, if the separator has one
    slot_power_db: torch.Tensor  # (B, max_n_src) slot power relative to the mixture, in dB

    def take(self, index: int) -> list[torch.Tensor]:
        """The ``n_hat`` estimated speakers for one item, as a list of waveforms."""
        return [self.sources[index, k] for k in range(int(self.n_hat[index]))]


class CountThenSeparate(torch.nn.Module):
    """Wraps a trained counter and a trained separator into one callable system."""

    def __init__(self, counter: torch.nn.Module, separator: torch.nn.Module) -> None:
        super().__init__()
        self.counter = counter
        self.separator = separator

    @torch.no_grad()
    def forward(self, mix: torch.Tensor, *, n_true: torch.Tensor | None = None) -> PipelineOutput:
        """mix: (B, T) RMS-normalised waveform.

        ``n_true`` forces the count instead of predicting it. That is not cheating as long as
        it is *reported separately*: it is the only way to measure separation quality
        independently of counting quality, and the report needs both numbers -- an end-to-end
        score and a "separation given a perfect counter" score. Without the second, a bad
        end-to-end number cannot be attributed to either half.
        """
        self.eval()
        logits = self.counter(mix)["logits"].float()
        probs = torch.softmax(logits, dim=-1)
        if n_true is None:
            n_hat = torch.tensor([class_to_n(int(c)) for c in logits.argmax(-1).cpu()],
                                 device=mix.device)
        else:
            n_hat = n_true.to(mix.device)

        out = self.separator(mix)
        est = out["est"].float()
        max_n = self.separator.cfg.max_n_src
        speakers = est[:, :max_n]
        noise_slot = est[:, max_n] if est.shape[1] > max_n else None

        mix_power = mix.float().pow(2).mean(dim=-1, keepdim=True)
        power_db = 10.0 * torch.log10(
            speakers.pow(2).mean(dim=-1) / (mix_power + 1e-12) + 1e-12)

        order = power_db.argsort(dim=-1, descending=True)
        speakers = torch.gather(
            speakers, 1, order.unsqueeze(-1).expand(-1, -1, speakers.shape[-1]))
        power_db = torch.gather(power_db, 1, order)

        return PipelineOutput(n_hat=n_hat, count_probs=probs, sources=speakers,
                              noise=noise_slot, slot_power_db=power_db)


def load_pipeline(counter_ckpt: str, separator_ckpt: str,
                  device: torch.device | str = "cpu") -> CountThenSeparate:
    """Build the system from two checkpoints, reading each model's config from its own file.

    Each checkpoint stores the config it was trained with, so a pooling operator or a TCN depth
    chosen months ago is reproduced rather than guessed. Guessing here is how a checkpoint gets
    silently loaded into the wrong architecture and scores at chance.
    """
    from .checkpoint import load_checkpoint
    from .counter import ModelConfig as CounterConfig
    from .counter import build_model as build_counter
    from .separator import ModelConfig as SepConfig
    from .separator import build_separator

    c_state = torch.load(counter_ckpt, map_location="cpu", weights_only=False)
    pooling = str(c_state.get("cfg", {}).get("pooling", "eigen"))
    counter = build_counter(CounterConfig(pooling=pooling))
    load_checkpoint(counter_ckpt, model=counter)

    s_state = torch.load(separator_ckpt, map_location="cpu", weights_only=False)
    s_cfg = s_state.get("cfg", {}).get("model", {})
    separator = build_separator(SepConfig(**s_cfg) if s_cfg else SepConfig())
    load_checkpoint(separator_ckpt, model=separator)

    return CountThenSeparate(counter, separator).to(device).eval()


def _align_to_previous(prev_tail: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
    """Permute ``chunk``'s slots so each one continues the speaker it overlaps.

    A separator has no idea that slot 2 in window 7 holds the same person as slot 4 in
    window 8, and :class:`CountThenSeparate` re-ranks slots by loudness inside *every*
    window -- so the moment two talkers trade places in the volume ranking, their slots
    trade places too. Overlap-adding that unaligned welds half of one voice onto half of
    another.

    Measured on a 6 s two-tone probe whose talkers swap loudness mid-file: without this
    step, output track 0 correlated **0.908 with speaker A over the first half and 1.000
    with speaker B over the second**. The identity of a track changed halfway through the
    file, which is worse than poor separation because it looks fine in every per-window
    metric.

    The remedy is the usual one: score every (previous slot, current slot) pair by
    normalised cross-correlation over the region the two windows share, then take the
    assignment maximising the total. Surplus slots hold near-silence and correlate with
    nothing, so they land wherever is left -- correct, since they are empty.
    """
    from scipy.optimize import linear_sum_assignment

    def _unit(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    sim = torch.einsum("bkt,blt->bkl", _unit(prev_tail), _unit(chunk[..., :prev_tail.shape[-1]]))
    order = torch.empty(sim.shape[:2], dtype=torch.long, device=chunk.device)
    for b in range(sim.shape[0]):
        # The cost matrix is square (slots against slots), so `rows` is always 0..K-1.
        _, cols = linear_sum_assignment(-sim[b].detach().cpu().numpy())
        order[b] = torch.as_tensor(cols, device=chunk.device)
    return torch.gather(chunk, 1, order.unsqueeze(-1).expand(-1, -1, chunk.shape[-1]))


@torch.no_grad()
def separate_long(system: CountThenSeparate, wav: torch.Tensor, *, seg_len: int = SEG_LEN,
                  hop: int | None = None, sr: int = SR) -> PipelineOutput:
    """Run the system over a recording longer than one segment, with overlap-add.

    **Never feed a long recording in one block.** Both models are trained on fixed 3 s
    RMS-normalised crops; v0 measured that handing the counter one long block instead of
    overlapping windows took it from a working score to **0 correct out of 300**, because the
    normalisation and the receptive field both assume the training length.

    Windows are **permutation-aligned** to their predecessor before being overlap-added, or
    a speaker changes track partway through the file -- see :func:`_align_to_previous` for
    the measurement. The finished tracks are then ranked loudest-first over the whole
    recording, so the ordering means the same thing it does for a single segment.

    The count is taken by averaging the per-window probabilities, which is more stable than
    voting: a window that lands in a pause is uncertain rather than confidently wrong, and
    averaging probabilities lets it abstain.
    """
    hop = int(hop or seg_len // 2)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    total = wav.shape[-1]
    if total <= seg_len:
        pad = seg_len - total
        block = torch.nn.functional.pad(wav, (0, pad)) if pad else wav
        rms = block.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        out = system(block / rms)
        out.sources = out.sources[..., :total] * rms.unsqueeze(1)
        return out

    starts = list(range(0, max(total - seg_len, 0) + 1, hop))
    if starts[-1] + seg_len < total:
        starts.append(total - seg_len)

    window = torch.hann_window(seg_len, device=wav.device).clamp_min(1e-3)
    acc = None
    norm = torch.zeros(1, 1, total, device=wav.device)
    probs = []
    prev_start, prev_chunk = None, None
    for start in starts:
        block = wav[..., start:start + seg_len]
        rms = block.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        out = system(block / rms)
        probs.append(out.count_probs)
        chunk = out.sources * rms.unsqueeze(1)
        if prev_chunk is not None:
            # The last start is snapped to `total - seg_len`, so the overlap with the
            # previous window is not always `seg_len - hop`. Measure it.
            overlap = min(prev_start + seg_len - start, seg_len)
            if overlap > 1:
                chunk = _align_to_previous(prev_chunk[..., seg_len - overlap:], chunk)
        if acc is None:
            acc = torch.zeros(chunk.shape[0], chunk.shape[1], total, device=wav.device)
        acc[..., start:start + seg_len] += chunk * window
        norm[..., start:start + seg_len] += window
        prev_start, prev_chunk = start, chunk

    mean_probs = torch.stack(probs).mean(0)
    n_hat = torch.tensor([class_to_n(int(c)) for c in mean_probs.argmax(-1).cpu()],
                         device=wav.device)
    sources = acc / norm.clamp_min(1e-8)

    # Alignment locked every window to the FIRST window's ranking, which is an arbitrary
    # three seconds. Rank the finished tracks by their power over the whole recording, in dB
    # relative to the mixture -- the same convention the single-segment branch returns.
    mix_power = wav.float().pow(2).mean(dim=-1, keepdim=True)
    power_db = 10.0 * torch.log10(sources.pow(2).mean(-1) / (mix_power + 1e-12) + 1e-12)
    slot_order = power_db.argsort(dim=-1, descending=True)
    sources = torch.gather(sources, 1, slot_order.unsqueeze(-1).expand(-1, -1, total))
    power_db = torch.gather(power_db, 1, slot_order)
    return PipelineOutput(n_hat=n_hat, count_probs=mean_probs, sources=sources,
                          noise=None, slot_power_db=power_db)
