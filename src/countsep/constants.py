"""Global constants. Everything here is 8 kHz, mono, 3-second segments, N = 1..5.

The STFT settings are not arbitrary and are not ours. Stoter et al. measured, over 36 trained
networks, that **linear-magnitude STFT beats log-mel, log-STFT and MFCC** for speaker counting,
and that a **25 ms window with a 10 ms hop** is the optimum -- independently confirmed by
TaCNet's 5-40 ms sweep, which bottoms out at 25 ms. Counting needs frequency resolution to
tell a one-source bin from a multi-source bin, which is exactly what perceptual warping throws
away. Do not "improve" this to log-mel because every other audio task uses log-mel.

There are **two** epsilons here, and collapsing them into one is a mistake this project made
and caught:

* ``MODEL_EPS = 1e-5`` guards divisions inside the torch model. It has to be *representable in
  fp16*, where the sibling project's 1e-8 is exactly 0.0 and ``v + 1e-8 == v`` -- so its 49
  hand-written normalisations divided by an unguarded ``sqrt(var)``.
* ``EPS = 1e-12`` guards float64 DSP in the numpy mixing and feature code.

Using 1e-5 for the DSP too looks harmless and is not. ``mixing`` normalises by
``1 / (rms(mix) + eps)``, so the mixture's residual level after normalisation is
``-20*log10(1 + eps/rms)`` -- a quantity that depends on ``rms``, which grows as ``sqrt(N)``.
At ``eps = 1e-5`` that residue runs from -8.686e-05 dB at N=1 to -3.884e-05 dB at N=5:
monotone in N, far above float32 resolution, and a depth-3 decision tree reads it at **39.2 %
against 20 % chance**. The leakage audit caught it within a minute of the constant changing.
At ``eps = 1e-12`` the residue is ~1e-11 dB and there is nothing to read.

The general lesson, and it is the same one as §3 of ``docs/DIAGNOSIS.md`` from the other
direction: **an epsilon is not a free parameter.** Too small for the working precision and it
is zero; too large and it is signal.
"""

from __future__ import annotations

SR: int = 8000
"""Sample rate, Hz. Fixed everywhere -- never resample internally."""

SEG_SECONDS: float = 3.0
"""Segment length in seconds. Half of the anti-leak mitigation: every input is the same length,
so clip duration cannot carry the count. The other half is dividing by the RMS."""

SEG_LEN: int = 24000
"""SEG_SECONDS * SR."""

MAX_N_SRC: int = 5
"""Number of speaker output slots on the separator. Only the mask head depends on this:
the TCN is 92-96 % of the FLOPs and is N-independent, so five slots cost ~4 % more than two."""

SILENCE_DB: float = -30.0
"""Silence floor for surplus separator slots and for the P-SI-SNR pad, dB relative to the mixture."""

N_LIST: tuple[int, ...] = (1, 2, 3, 4, 5)
"""Speaker counts the model is trained on. Class index is the position in this tuple."""

N_CLASSES: int = len(N_LIST)

# ---------------------------------------------------------------- front end

N_FFT: int = 256
"""FFT size at 8 kHz, giving 129 linear frequency bins up to Nyquist (4 kHz)."""

WIN_LENGTH: int = 200
"""25 ms at 8 kHz -- Stoter et al.'s measured optimum for counting."""

HOP_LENGTH: int = 80
"""10 ms at 8 kHz. Gives 299 frames for a 3 s segment."""

N_BINS: int = N_FFT // 2 + 1
"""129."""

N_FRAMES: int = SEG_LEN // HOP_LENGTH + 1
"""301 for a 3 s segment, before any trimming."""

# ---------------------------------------------------------------- numerics

EPS: float = 1e-12
"""Guard for float64 DSP (mixing, features). Small enough that the level residue it leaves in
a normalised mixture (~1e-11 dB) is far below float32 resolution and carries no cue."""

MODEL_EPS: float = 1e-5
"""Guard for divisions inside the torch model. Representable in fp16, unlike 1e-8 which is
exactly 0.0 there. Never use this in the DSP path -- see the module docstring."""

INT16_SCALE: float = 32767.0

CAP_SECONDS: float = 8.0
"""Maximum stored excerpt per source utterance (the highest-energy window)."""

CAP_LEN: int = 64000

MIN_UTT_SECONDS: float = 1.0
"""Utterances shorter than this are dropped by the packer -- they make degenerate crops."""

# ---------------------------------------------------------------- the bar

NAIVE_ACCURACY: float = 1.0 / N_CLASSES
"""Chance, and also the majority-class predictor, on a balanced 5-class set: 20 %."""


def n_to_class(n: int) -> int:
    """Map a speaker count to its class index."""
    return N_LIST.index(int(n))


def class_to_n(c: int) -> int:
    """Map a class index back to its speaker count."""
    return N_LIST[int(c)]
