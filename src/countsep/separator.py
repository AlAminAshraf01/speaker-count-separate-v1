"""SepNet: Conv-TasNet with a max-N mask head and a noise slot. No count head.

Luo & Mesgarani (2019) with two changes, and one repair.

* the mask head emits ``max_n_src`` speaker slots regardless of how many talkers are actually
  present -- surplus slots are pushed to silence by the loss;
* one extra **noise slot**, so background noise has somewhere to go instead of smearing across
  the speech slots (+66 k parameters, +1.3 %).

**The repair: there is no count head, and that is the whole point of v1.** In v0 a counting
head hung off this trunk, and it was measured to receive **0.53 %** of the gradient flowing
into the shared layers while separation received 86.14 % -- 633:1 at the encoder. The features
it read were therefore shaped almost entirely by separation, it never learned, and it settled
on the class marginal (answering "1 speaker" for 1445 of 1500 test mixtures in fp32).

v1 trains a dedicated counter instead (``countsep.counter.CountCRNN``, 0.49 M params). Dropping
the head from here costs nothing -- v0 measured that removing the separator saves only 8.4 % of
forward FLOPs, so the two directions are not symmetric: the TCN is the expense, and a separate
counter adds only ~5 % on top of it while getting 100 % of its own gradient.

``GlobalLayerNorm`` is also repaired; see its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import MAX_N_SRC, MODEL_EPS, SR


@dataclass
class ModelConfig:
    """Conv-TasNet hyperparameters. Defaults are the paper's Table I configuration."""

    n_filters: int = 512      # N  encoder basis size
    kernel: int = 16          # L  encoder window in samples (2 ms at 8 kHz)
    bottleneck: int = 128     # B  TCN channel width
    hidden: int = 512         # H  depthwise-conv channel width
    skip: int = 128           # Sc skip-connection width
    conv_kernel: int = 3      # P  depthwise kernel
    n_blocks: int = 8         # X  blocks per repeat
    n_repeats: int = 3        # R  repeats
    max_n_src: int = MAX_N_SRC
    predict_noise: bool = True
    mask_act: str = "relu"    # relu | sigmoid | softmax
    norm: str = "gLN"         # gLN | cLN
    causal: bool = False

    def __post_init__(self) -> None:
        if self.kernel % 2 != 0:
            raise ValueError("kernel must be even (stride is kernel // 2)")
        if self.causal:
            self.norm = "cLN"


PRESETS: dict[str, ModelConfig] = {
    "paper": ModelConfig(),
    "small": ModelConfig(n_filters=256, kernel=32, hidden=256, n_blocks=8, n_repeats=2),
    "tiny": ModelConfig(n_filters=128, kernel=32, bottleneck=64, hidden=128, skip=64,
                        n_blocks=4, n_repeats=2),
}


# --------------------------------------------------------------------------- norms

class GlobalLayerNorm(nn.Module):
    """gLN: normalise over channels *and* time, per sample. Non-causal.

    REPAIRED FROM v0, AND THE REPAIR IS THE POINT
    ----------------------------------------------
    This layer is hand-written from ``mean``/``var``/``sqrt`` because Conv-TasNet's gLN
    normalises over channels AND time, which ``nn.LayerNorm`` does not do. That is legitimate.
    What was not legitimate is that ``torch.autocast`` maintains an fp32 promotion list, and
    ``layer_norm`` is on it -- but a hand-rolled equivalent is not. v0 had **49 instances** of
    this layer, so it opted out of that protection 49 times, and its ``eps = 1e-8`` is
    **exactly 0.0 in fp16** (verified: ``float16(1e-8) == 0.0`` and ``v + 1e-8 == v``). Every
    one of them divided by an unguarded ``sqrt(var)``.

    The model then trained under fp16 autocast and was evaluated in fp32 -- two different
    functions. The same checkpoint scored 44.9 % and 20.00 % on counting, agreeing on 16.1 % of
    predictions, and ``torch.argmax`` on all-NaN logits returns index 0, which is class
    "1 speaker" -- the exact signature observed.

    Two lines fix it:

    * the statistics are computed in **fp32 regardless of autocast**, which is what
      ``nn.LayerNorm`` gets for free;
    * ``eps`` defaults to ``MODEL_EPS`` (1e-5), which is representable in fp16.

    ``tests/test_precision.py`` asserts fp32 and fp16 forward passes agree.
    """

    def __init__(self, channels: int, eps: float = MODEL_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T). Statistics in fp32; output returned in the input's dtype.

        WHY THIS IS FOLDED INSTEAD OF WRITTEN OUT
        -----------------------------------------
        The readable form -- subtract, divide, scale, shift -- costs **three full-size
        tensors per call**, because autograd saves ``xf`` for the variance, ``xf - mean``
        for the division, and the normalised result for the multiply. At the paper preset
        and batch 12 one of those is 70.29 MiB, there are 49 gLN instances, and the total
        came to **10.02 GiB of the 18.05 GiB** a single forward+backward retained -- on a
        14.56 GiB T4. It OOMed in block 19 of 24 on the first batch.

        ``mean`` and ``var`` reduce over channels *and* time, so they are ``(B, 1, 1)``,
        while ``gamma`` and ``beta`` are ``(1, C, 1)``. The whole affine therefore folds
        into ``(B, C, 1)`` coefficients -- kilobytes, not megabytes:

            gamma * (x - mean) / d + beta  ==  x * (gamma/d) + (beta - mean*gamma/d)

        which ``addcmul`` applies in one pass. Exact algebra, not an approximation.
        Measured: three saved full-size tensors become one, gLN drops to 3.30 GiB, the
        model's activations to 11.33 GiB, and batch 12 fits with ~1.77 GiB to spare. It is
        also *faster* (30.6 -> 25.1 ms forward) -- one full-size write instead of four.

        THE ONE THING THIS GIVES UP, AND WHY IT IS SAFE HERE
        ---------------------------------------------------
        ``x*scale - mean*scale`` subtracts two nearly-equal quantities when the input has a
        large DC offset, where ``x - mean`` would be exact by Sterbenz. Error grows as
        ``|mean| / sqrt(var + eps)``: harmless at 1, ~46 ulp at 1e2, and total loss of
        ``beta`` past 1e10.

        Measured at all 49 real call sites on the paper preset, that ratio is **0.38-0.70**
        -- max 0.677 at ``pre_norm`` -- and it stays pinned at 0.68 across 300 real training
        steps while the median *falls* (0.438 -> 0.245). Training walks away from the bad
        regime rather than into it, and it is structurally bounded: the worst site is
        ``relu(encoder)``, whose mean/std is capped near 0.7 for a half-Gaussian, and gLN
        removes the mean, so a DC offset can only re-enter through one conv bias. That is
        about six orders of magnitude of margin. **If a future change puts a large constant
        offset in front of a norm, revisit this.**

        The fp32 guarantee is not weakened -- it is strengthened. The old affine sat
        *outside* the ``enabled=False`` block and ran with autocast live; this one is inside
        it, so every statistic and the affine are fp32 regardless of autocast.
        """
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            var, mean = torch.var_mean(xf, dim=(1, 2), keepdim=True, unbiased=False)
            scale = self.gamma * torch.rsqrt(var + self.eps)   # (B, C, 1) -- kilobytes
            bias = self.beta - mean * scale                    # (B, C, 1) -- kilobytes
            out = torch.addcmul(bias, xf, scale)               # the ONLY full-size tensor
        return out.to(dtype)


class ChannelwiseLayerNorm(nn.Module):
    """cLN: normalise over channels only, per frame. Causal-safe.

    Same fp32 repair as :class:`GlobalLayerNorm` -- see its docstring.
    """

    def __init__(self, channels: int, eps: float = MODEL_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T). Statistics in fp32; output returned in the input's dtype."""
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            mean = xf.mean(dim=1, keepdim=True)
            var = xf.var(dim=1, keepdim=True, unbiased=False)
            out = (xf - mean) / torch.sqrt(var + self.eps)
        return (self.gamma * out.to(dtype) + self.beta)


def make_norm(kind: str, channels: int) -> nn.Module:
    """Build the normalisation layer named by ``kind``."""
    if kind == "gLN":
        return GlobalLayerNorm(channels)
    if kind == "cLN":
        return ChannelwiseLayerNorm(channels)
    if kind == "BN":
        return nn.BatchNorm1d(channels)
    raise ValueError(f"unknown norm: {kind!r}")


# --------------------------------------------------------------------------- tcn

class TemporalBlock(nn.Module):
    """One dilated depthwise-separable residual block."""

    def __init__(self, bottleneck: int, hidden: int, skip: int, kernel: int,
                 dilation: int, norm: str, causal: bool, use_residual: bool = True) -> None:
        super().__init__()
        self.causal = causal
        self.padding = (kernel - 1) * dilation

        self.conv_in = nn.Conv1d(bottleneck, hidden, 1)
        self.prelu_in = nn.PReLU()
        self.norm_in = make_norm(norm, hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel, dilation=dilation,
                                   groups=hidden, padding=0)
        self.prelu_out = nn.PReLU()
        self.norm_out = make_norm(norm, hidden)
        # The last block in the stack has nothing downstream to feed, so its residual
        # branch would be dead weight (~66 k parameters that never receive a gradient).
        self.res_out = nn.Conv1d(hidden, bottleneck, 1) if use_residual else None
        self.skip_out = nn.Conv1d(hidden, skip, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, B_ch, T) -> (residual, skip)."""
        y = self.norm_in(self.prelu_in(self.conv_in(x)))
        if self.causal:
            y = F.pad(y, (self.padding, 0))
        else:
            left = self.padding // 2
            y = F.pad(y, (left, self.padding - left))
        y = self.norm_out(self.prelu_out(self.depthwise(y)))
        residual = x if self.res_out is None else x + self.res_out(y)
        return residual, self.skip_out(y)


class TCN(nn.Module):
    """``n_repeats`` stacks of ``n_blocks`` blocks with exponentially growing dilation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        n_total = cfg.n_repeats * cfg.n_blocks
        specs = [(r, b) for r in range(cfg.n_repeats) for b in range(cfg.n_blocks)]
        self.blocks = nn.ModuleList([
            TemporalBlock(cfg.bottleneck, cfg.hidden, cfg.skip, cfg.conv_kernel,
                          2 ** b, cfg.norm, cfg.causal,
                          use_residual=(i < n_total - 1))
            for i, (_r, b) in enumerate(specs)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sum of every block's skip output, (B, Sc, T)."""
        total: torch.Tensor | None = None
        for block in self.blocks:
            x, skip = block(x)
            total = skip if total is None else total + skip
        assert total is not None
        return total


# --------------------------------------------------------------------------- heads

# --------------------------------------------------------------------------- network

class SepNet(nn.Module):
    """Encoder -> TCN -> mask head -> decoder. Separation only, by design."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stride = cfg.kernel // 2

        self.encoder = nn.Conv1d(1, cfg.n_filters, cfg.kernel, stride=self.stride, bias=False)
        self.pre_norm = make_norm(cfg.norm, cfg.n_filters)
        self.bottleneck = nn.Conv1d(cfg.n_filters, cfg.bottleneck, 1)
        self.tcn = TCN(cfg)
        self.mask_prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(cfg.skip, self.n_slots * cfg.n_filters, 1)
        self.decoder = nn.ConvTranspose1d(cfg.n_filters, 1, cfg.kernel,
                                          stride=self.stride, bias=False)

    # -- shape helpers -----------------------------------------------------
    @property
    def n_slots(self) -> int:
        """Speaker slots plus the optional noise slot."""
        return self.cfg.max_n_src + int(self.cfg.predict_noise)

    @property
    def noise_slot(self) -> int | None:
        """Index of the noise slot, or None when it is disabled."""
        return self.cfg.max_n_src if self.cfg.predict_noise else None

    def _pad_len(self, n_samples: int) -> int:
        """Right-padding that makes the encoder/decoder lengths line up exactly."""
        kernel, stride = self.cfg.kernel, self.stride
        if n_samples < kernel:
            return kernel - n_samples
        rem = (n_samples - kernel) % stride
        return 0 if rem == 0 else stride - rem

    def _apply_mask_act(self, masks: torch.Tensor) -> torch.Tensor:
        act = self.cfg.mask_act
        if act == "relu":
            return F.relu(masks)
        if act == "sigmoid":
            return torch.sigmoid(masks)
        if act == "softmax":
            return torch.softmax(masks, dim=1)
        raise ValueError(f"unknown mask_act: {act!r}")

    # -- forward -----------------------------------------------------------
    def forward(self, mix: torch.Tensor, return_internals: bool = False) -> dict:
        """mix: (B, T) or (B, 1, T) -> dict with 'est' (B, n_slots, T).

        ``count_logits`` is None: this model does not count. ``countsep.counter`` does, and
        ``countsep.pipeline`` puts the two together. The key is kept so the shared loss and
        the v0 checkpoints stay readable.
        """
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        batch, _, n_samples = mix.shape

        pad = self._pad_len(n_samples)
        x = F.pad(mix, (0, pad)) if pad else mix

        enc = F.relu(self.encoder(x))                       # (B, N, F)
        feat = self.tcn(self.bottleneck(self.pre_norm(enc)))  # (B, Sc, F)

        masks = self.mask_conv(self.mask_prelu(feat))
        masks = masks.view(batch, self.n_slots, self.cfg.n_filters, -1)
        masks = self._apply_mask_act(masks)

        masked = enc.unsqueeze(1) * masks                    # (B, S, N, F)
        flat = masked.reshape(batch * self.n_slots, self.cfg.n_filters, -1)
        est = self.decoder(flat).reshape(batch, self.n_slots, -1)
        est = est[..., :n_samples]

        out = {"est": est, "count_logits": None}
        if return_internals:
            out["masks"] = masks
            out["enc"] = enc
            out["feat"] = feat
        return out

    # -- introspection -----------------------------------------------------
    def count_params(self, trainable_only: bool = True) -> int:
        """Number of parameters."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(int(p.numel()) for p in params)

    def flops_per_second_of_audio(self, sr: int = SR) -> float:
        """Analytic forward FLOPs for one second of audio (1 MAC = 2 FLOP)."""
        cfg = self.cfg
        frames = sr / self.stride
        macs = cfg.n_filters * cfg.kernel * frames                 # encoder
        macs += cfg.n_filters * cfg.bottleneck * frames            # bottleneck 1x1
        per_block = (cfg.bottleneck * cfg.hidden
                     + cfg.hidden * cfg.conv_kernel                # depthwise
                     + cfg.hidden * cfg.bottleneck                 # residual 1x1
                     + cfg.hidden * cfg.skip)                      # skip 1x1
        macs += per_block * cfg.n_blocks * cfg.n_repeats * frames
        macs += cfg.skip * self.n_slots * cfg.n_filters * frames    # mask head
        macs += self.n_slots * cfg.n_filters * cfg.kernel * frames  # decoder
        return float(macs * 2.0)

    def describe(self) -> str:
        """One-line summary for logs and the report."""
        gflops = self.flops_per_second_of_audio() / 1e9
        return (f"SepNet(N={self.cfg.n_filters}, L={self.cfg.kernel}, "
                f"B={self.cfg.bottleneck}, H={self.cfg.hidden}, X={self.cfg.n_blocks}, "
                f"R={self.cfg.n_repeats}, slots={self.n_slots}) "
                f"{self.count_params() / 1e6:.2f} M params, "
                f"{gflops:.1f} GFLOP per second of audio")


def build_separator(cfg: ModelConfig | str | dict) -> SepNet:
    """Build a model from a ModelConfig, a preset name, or a plain dict."""
    if isinstance(cfg, str):
        if cfg not in PRESETS:
            raise KeyError(f"unknown preset {cfg!r}; have {sorted(PRESETS)}")
        cfg = PRESETS[cfg]
    elif isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return SepNet(cfg)
