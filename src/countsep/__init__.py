"""countsep — count the talkers, then separate them. Two specialists, one pipeline.

v1 of the EEE 402 Group 10 project. It replaces a single multi-task network that did both jobs
and did neither well, and every structural difference here traces to a measurement rather than
a preference. The full post-mortem is in ``docs/DIAGNOSIS.md``; the four that shaped this
package:

1. **The tasks were starving each other.** In v0 the counting objective reached the shared
   trunk with **0.53 %** of the gradient while separation took 86.14 % — 633:1 at the encoder.
   Counting settled on the class marginal and answered "1 speaker" for 1445 of 1500 test
   mixtures. Separation was hurt too: the auxiliary objectives were priced at **−8.7 dB**, and
   the pooled model scored 0.08 dB on official Libri2Mix where a separation-only control got
   5.49 dB in a fifth of the epochs.
   → v1 trains :mod:`countsep.counter` and :mod:`countsep.separator` apart, each with 100 % of
   its own gradient, and joins them in :mod:`countsep.pipeline`. Sharing was never buying much:
   removing the separator from v0's model saved only 8.4 % of forward FLOPs.

2. **Two loss terms were structurally biased toward N=1.** ``soft_clamp`` multiplied by the
   SI-SDR gradient *peaks* at the clamp point rather than vanishing there, so a trivially-solved
   one-speaker item carried ~16× the gradient of a hard five-speaker one; and the silence term
   divided by the number of leftover *slots*, which gave the single N=1 item in a balanced batch
   40 % of that term and the N=5 item none.
   → :func:`countsep.losses.hard_clamp`, and a per-item silence average.

3. **The model trained as one function and was measured as another.** ``autocast`` promotes
   ``nn.LayerNorm`` to fp32 but cannot protect a hand-written one, and v0 had 49 hand-written
   norms with ``eps = 1e-8`` — which is *exactly 0.0* in fp16. Same checkpoint: 44.9 % in fp16,
   20.00 % in fp32, agreeing on 16.1 % of predictions.
   → ``GlobalLayerNorm`` computes its statistics in fp32 regardless of autocast, ``MODEL_EPS``
   is representable in fp16, AMP is off by default, and a preflight check refuses to train when
   the two precisions disagree.

4. **The leakage audit had never actually run.** A ``def`` landed inside ``main()``, every probe
   became unreachable code after a return, and the script exited 0 while writing nothing — for
   months, with two headline numbers quoted from it that it had never produced.
   → ``scripts/02_audit_and_baselines.py`` verifies its own report before exiting, and
   ``tests/test_scripts_are_reachable.py`` fails the build on that exact bug shape.

And one number worth keeping in view: sixteen hand-crafted acoustic scalars plus a
gradient-boosted tree count speakers at **69.3 %** where v0's 5.3 M-parameter network managed
20.00 %. ``countsep.features`` and ``countsep.baselines`` are that system, they run on CPU in
minutes, and nothing neural here is allowed to claim victory without beating them.
"""

__version__ = "1.0.0"
