"""CountCRNN: a small convolutional-recurrent speaker counter, and its pooling options.

Roughly 0.4-0.7 M parameters against the 5.30 M of the model this replaces, and about 13x
cheaper per second of audio -- because none of that model's capacity was spent on counting.
The topology follows Stoter et al.'s CountNet (the published 0.35 M CRNN reaches MAE 0.27 on
LibriCount), with three deliberate departures, each fixing a named finding in
``docs/DIAGNOSIS.md``.

**1. Every normalisation is an ``nn`` module.** ``nn.BatchNorm2d`` and ``nn.LayerNorm`` are on
   autocast's fp32 promotion list; a hand-written ``mean``/``var``/``sqrt`` is not. The sibling
   project re-implemented ``LayerNorm`` by hand 49 times and thereby opted out of that
   protection 49 times, with an epsilon that is exactly 0.0 in fp16. Its model then trained as
   one function and evaluated as another: 44.9 % in fp16, 20.00 % in fp32, agreeing on 16 % of
   predictions. There is not one hand-rolled normalisation in this file, and
   ``tests/test_model.py`` asserts fp32 and fp16 agree. (finding 3)

**2. No per-sample normalisation of the pooled vector.** The old head ran ``nn.LayerNorm`` over
   its 256-d pooled statistics, which subtracts that vector's own mean -- deleting the
   common-mode direction, which is the natural encoding of a scalar count. BatchNorm
   normalises with *running* statistics across the batch instead, so between-sample
   differences survive. (finding 6)

**3. Pooling that can express cardinality.** The old head pooled ``mean`` and ``std`` over
   time: the mean vector plus ``sqrt(diag(Cov))``. But speaker count is a rank property -- N
   talkers means N dominant components in the channel covariance -- and that lives in the
   *off-diagonal* terms it discarded. Worse, the strongest measured cue is fourth-moment:
   kurtosis spreads 5x across N = 1..5 while crest factor spreads only 1.15x. A second-moment
   diagonal read-out was aimed at a fourth-moment off-diagonal cue. (findings 5 and 6)

   So pooling is a first-class, swappable choice here, and the default is set by measurement
   rather than by argument -- see ``docs/DESIGN.md`` for the comparison on identical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import HOP_LENGTH, MODEL_EPS, N_BINS, N_CLASSES, N_FFT, WIN_LENGTH


@dataclass
class ModelConfig:
    """Everything that defines the network. Serialised into every checkpoint."""

    channels: tuple[int, ...] = (32, 64, 64)
    gru_hidden: int = 64
    pooling: str = "eigen"            # meanstd | attentive | covariance | eigen
    cov_rank: int = 32                # projection width for the second-order poolings
    head_hidden: int = 128
    dropout: float = 0.2
    n_classes: int = N_CLASSES

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {self.pooling!r}; have {sorted(POOLINGS)}")
        if len(self.channels) < 1:
            raise ValueError("need at least one conv stage")


# --------------------------------------------------------------------------- front end

class LinearSTFT(nn.Module):
    """Linear-magnitude STFT. No log, no mel, no MFCC, no per-clip normalisation.

    Linear magnitude is Stoter et al.'s measured winner for counting, and it is also the
    conservative choice given finding 6: a log or a per-clip normalisation is one more chance
    to delete a common-mode cue. The input waveform has already been RMS-normalised by the
    mixer, which is the anti-leak step, so no further level handling is wanted here.

    Registered as a module rather than done in the dataloader so the transform travels with
    the checkpoint and cannot silently differ between training and evaluation -- which is the
    generic form of the bug that produced finding 3.
    """

    def __init__(self, n_fft: int = N_FFT, win_length: int = WIN_LENGTH,
                 hop_length: int = HOP_LENGTH) -> None:
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) -> magnitude (B, 1, n_bins, n_frames), always in fp32.

        The STFT is forced to fp32 even under autocast. It is a fixed, cheap transform, and
        fp16 buys nothing here while giving numerical divergence somewhere to hide.
        """
        with torch.autocast(device_type=wav.device.type, enabled=False):
            spec = torch.stft(
                wav.float(), n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.win_length, window=self.window.float(),
                center=True, pad_mode="reflect", normalized=False, return_complex=True)
            return spec.abs().unsqueeze(1)


# --------------------------------------------------------------------------- pooling

class MeanStdPool(nn.Module):
    """Mean and standard deviation over time -- what the failed project used.

    Kept so the comparison in ``docs/DESIGN.md`` is a measurement on identical data rather
    than an assertion, and so the claim "this pooling is the problem" stays falsifiable.
    Keeps the mean vector and ``sqrt(diag(Cov))``; discards every off-diagonal term.
    """

    def __init__(self, dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.out_dim = 2 * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, 2C)."""
        return torch.cat([x.mean(1), x.std(1, unbiased=False)], dim=-1)


class AttentivePool(nn.Module):
    """Attentive statistics pooling (Okabe et al. 2018), the speaker-embedding standard.

    Learns per-frame weights and takes a weighted mean and standard deviation, so frames where
    talkers overlap can dominate the summary instead of being averaged against silence.
    Yousefi & Hansen (Interspeech 2021) measured ~3 % absolute over average pooling on speaker
    counting specifically, which is the closest published evidence to this task.

    Still a second-moment read-out -- it fixes *which frames* are summarised, not *which
    statistic*. That is why the covariance poolings below also exist.
    """

    def __init__(self, dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.attend = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, dim))
        self.out_dim = 2 * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attend(x), dim=1)
        mean = (weights * x).sum(1)
        var = (weights * (x - mean.unsqueeze(1)) ** 2).sum(1)
        return torch.cat([mean, var.clamp_min(MODEL_EPS).sqrt()], dim=-1)


class CovariancePool(nn.Module):
    """Second-order pooling: the full channel covariance, off-diagonal terms included.

    Projects to ``cov_rank`` channels, mean-centres over time, forms ``C = U U^T / T``, and
    maps it to a flat space with the matrix logarithm before vectorising the upper triangle.
    The log-Euclidean map matters: covariance matrices live on a curved manifold, and feeding
    raw entries to a linear layer treats that manifold as if it were flat.

    This is the direct structural answer to finding 5. The evidence for it is real but *not*
    unanimous, and the honest summary is worth keeping next to the code: Stafylakis et al.
    improved VoxCeleb-O EER from 1.40 % to 1.16 % with Gram-matrix pooling and SoCov reports
    ~15.5 % relative EER reduction, but Wang et al. (ISCSLP 2021) compared standard deviation,
    covariance and lp-norm pooling head to head and found plain standard deviation best. So
    this is a hypothesis the repo tests, not a result it assumes.
    """

    def __init__(self, dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        rank = int(cfg.cov_rank)
        self.project = nn.Linear(dim, rank)
        self.rank = rank
        self.register_buffer("triu", torch.triu_indices(rank, rank), persistent=False)
        self.out_dim = rank * (rank + 1) // 2 + rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.project(x.float())
            z = z - z.mean(1, keepdim=True)
            cov = torch.einsum("bti,btj->bij", z, z) / z.shape[1]
            eye = torch.eye(self.rank, device=z.device, dtype=z.dtype)
            # trace-normalise so the read-out is invariant to overall gain, then ridge the
            # spectrum so eigh is well-conditioned and its gradient is finite
            cov = cov * (self.rank / cov.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(MODEL_EPS)
                         ).view(-1, 1, 1)
            cov = cov + 1e-3 * eye
            evals, evecs = torch.linalg.eigh(cov)
            log_cov = evecs @ torch.diag_embed(evals.clamp_min(1e-4).log()) @ evecs.transpose(-1, -2)
            flat = log_cov[:, self.triu[0], self.triu[1]]
            return torch.cat([flat, z.mean(1)], dim=-1)


class EigenPool(nn.Module):
    """The covariance eigenvalue *spectrum* alone -- an explicit cardinality read-out.

    If N concurrent talkers really do produce N dominant components, then the shape of the
    normalised eigenvalue spectrum is the count, and everything else in the covariance is
    nuisance. This pools to ``2 * rank + 1`` numbers instead of ``rank(rank+1)/2``, which is a
    far stronger inductive bias and far fewer parameters downstream.

    Includes the **effective rank** (Roy & Vetterli), ``exp(-sum p_i log p_i)`` over the
    trace-normalised spectrum -- a continuous, differentiable estimate of "how many directions
    carry energy", which is as close to a closed-form speaker counter as this gets.

    **This is the default, and it won by measurement.** On identical speaker-disjoint data,
    10 CPU epochs: eigen 74.4 % (MAE 0.305), attentive 67.5 %, meanstd 66.5 %, and the full
    :class:`CovariancePool` 20.0 % -- collapsed. It is also the *smallest* of the four
    (0.488 M params) and the fastest to train, because it pools to ``2*rank + 1`` numbers
    instead of ``rank(rank+1)/2``.

    The contrast with :class:`CovariancePool` is the interesting part and is worth keeping in
    mind before "improving" this: both read the same covariance, but the full log-Euclidean
    vectorisation hands the head 560 correlated dimensions and fails, while the eigenvalue
    spectrum hands it 65 and wins. The *eigenvectors are nuisance* -- they say which directions
    the talkers occupy, which is exactly what a counter should be invariant to. Throwing them
    away is not compression, it is the inductive bias.
    """

    def __init__(self, dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        rank = int(cfg.cov_rank)
        self.project = nn.Linear(dim, rank)
        self.rank = rank
        self.out_dim = 2 * rank + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.project(x.float())
            z = z - z.mean(1, keepdim=True)
            cov = torch.einsum("bti,btj->bij", z, z) / z.shape[1]
            eye = torch.eye(self.rank, device=z.device, dtype=z.dtype)
            evals = torch.linalg.eigvalsh(cov + 1e-3 * eye).clamp_min(1e-6)
            spectrum = evals / evals.sum(-1, keepdim=True)     # scale-invariant by construction
            log_spectrum = spectrum.log()
            effective_rank = torch.exp(-(spectrum * log_spectrum).sum(-1, keepdim=True))
            return torch.cat([log_spectrum, z.std(1, unbiased=False), effective_rank], dim=-1)


POOLINGS: dict[str, type] = {
    "meanstd": MeanStdPool,
    "attentive": AttentivePool,
    "covariance": CovariancePool,
    "eigen": EigenPool,
}


# --------------------------------------------------------------------------- network

class CountCRNN(nn.Module):
    """Linear STFT -> 2-D convs -> BiGRU over time -> pooling -> 5-way softmax."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.stft = LinearSTFT()

        blocks: list[nn.Module] = []
        in_ch, bins = 1, N_BINS
        for out_ch in cfg.channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),          # autocast promotes this; a hand-rolled norm is not
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1)),            # pool FREQUENCY only -- time resolution is the signal
            ]
            in_ch, bins = out_ch, bins // 2
        self.conv = nn.Sequential(*blocks)
        self.feature_dim = in_ch * bins

        self.gru = nn.GRU(self.feature_dim, cfg.gru_hidden, batch_first=True, bidirectional=True)
        gru_out = 2 * cfg.gru_hidden

        self.pool = POOLINGS[cfg.pooling](gru_out, cfg)
        self.head = nn.Sequential(
            nn.Linear(self.pool.out_dim, cfg.head_hidden),
            nn.BatchNorm1d(cfg.head_hidden),     # BatchNorm, NOT LayerNorm -- see finding 6
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.n_classes),
        )

    def forward(self, wav: torch.Tensor, return_internals: bool = False) -> dict:
        """wav: (B, T) waveform -> dict with 'logits' (B, n_classes)."""
        spec = self.stft(wav)                       # (B, 1, F, T)
        feat = self.conv(spec)                      # (B, C, F', T)
        seq = feat.permute(0, 3, 1, 2).flatten(2)   # (B, T, C*F')
        seq, _ = self.gru(seq)                      # (B, T, 2H)
        pooled = self.pool(seq)
        logits = self.head(pooled.to(self.head[0].weight.dtype))
        out = {"logits": logits}
        if return_internals:
            out.update({"spec": spec, "seq": seq, "pooled": pooled})
        return out

    def count_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return (f"CountCRNN(channels={self.cfg.channels}, gru={self.cfg.gru_hidden}, "
                f"pool={self.cfg.pooling}, pooled_dim={self.pool.out_dim}) "
                f"{self.count_parameters() / 1e6:.3f} M params")


def probe_batch(per_class: int = 4, seed: int = 0, seg_len: int = 24000) -> torch.Tensor:
    """A canonical batch of acoustically DISTINCT inputs: sums of 1..5 sparse talkers.

    Used by `tests/test_model.py` and `scripts/preflight.py` for any check of the form "does
    the output move when the input changes", including the fp32-vs-fp16 agreement gate. It
    lives here rather than in either caller because the distinction it encodes is easy to get
    wrong and expensive when you do.

    **Do not substitute `torch.randn`.** White-noise waveforms are statistically identical to
    one another, so a model's output legitimately barely moves across them and every
    sensitivity measured against that baseline is inflated. Measured on the same untrained
    model: white noise gives an across-input logit spread of 3.0e-3 and these give 2.3e-2, a
    factor of **7.4**. An earlier version of the precision gate used white noise and reported
    a perturbation of 0.87x the signal where the true figure is 0.12x -- it would have failed
    every run for a reason that did not exist.

    Synthetic and cheap on purpose: no store, no dataset, no disk, so the checks that use it
    run anywhere in under a second.
    """
    g = torch.Generator().manual_seed(int(seed))
    t = torch.arange(seg_len) / 8000.0
    out: list[torch.Tensor] = []
    for n_src in (1, 2, 3, 4, 5):
        for _ in range(int(per_class)):
            total = torch.zeros(seg_len)
            for _k in range(n_src):
                f0 = 85.0 + 145.0 * torch.rand(1, generator=g).item()
                wave = sum(torch.sin(2 * torch.pi * f0 * h * t) / h for h in range(1, 20))
                gate = torch.zeros(seg_len)
                pos = 0
                while pos < seg_len:               # real silences, not a continuous dip
                    on = int((0.12 + 0.33 * torch.rand(1, generator=g).item()) * 8000)
                    off = int((0.08 + 0.27 * torch.rand(1, generator=g).item()) * 8000)
                    end = min(seg_len, pos + on)
                    if end > pos:
                        gate[pos:end] = torch.hann_window(max(2, end - pos))[: end - pos]
                    pos = end + off
                src = wave * gate
                total = total + src / (src.pow(2).mean().sqrt() + 1e-9)
            out.append(total / (total.pow(2).mean().sqrt() + 1e-9))
    return torch.stack(out)


def precision_agreement(model: CountCRNN, wav: torch.Tensor | None = None,
                        dtype: torch.dtype | None = None) -> dict:
    """Compare a model's fp32 and autocast outputs. The gate this project exists because of.

    Returns ``{agree, delta, spread, ratio, dtype}``. ``ratio`` -- the numerical perturbation
    divided by the across-input spread of the logits -- is the number to judge on, because
    ``agree`` counts argmax flips and an untrained model's classes are nearly tied, so it
    flips on perturbations far below any real decision margin.

    The sibling project's counter scored 44.9 % in fp16 and 20.00 % in fp32 from one
    checkpoint, agreeing on 16.1 % of predictions, because it trained under one precision and
    was evaluated in the other.
    """
    device = next(model.parameters()).device
    if wav is None:
        wav = probe_batch()
    wav = wav.to(device)
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    was_training = model.training
    model.eval()
    with torch.no_grad():
        hi = model(wav)["logits"].float()
        with torch.autocast(device_type=device.type, dtype=dtype):
            lo = model(wav)["logits"].float()
    if was_training:
        model.train()
    delta = (hi - lo).abs().max().item()
    spread = hi.std(0).mean().item()
    return {"agree": float((hi.argmax(-1) == lo.argmax(-1)).float().mean()),
            "delta": delta, "spread": spread, "ratio": delta / max(spread, 1e-12),
            "finite": bool(torch.isfinite(lo).all()), "dtype": str(dtype)}


def build_model(cfg: ModelConfig | dict | str | None = None) -> CountCRNN:
    """Build from a config, a dict, or a pooling name."""
    if cfg is None or isinstance(cfg, ModelConfig):
        return CountCRNN(cfg)
    if isinstance(cfg, str):
        return CountCRNN(ModelConfig(pooling=cfg))
    return CountCRNN(ModelConfig(**cfg))
