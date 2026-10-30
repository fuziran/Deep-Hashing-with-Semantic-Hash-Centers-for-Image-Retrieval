# Training-only audit of fused hash-target signal

B1+NFD remains fixed. Equal two-view averaging improved 32-bit seed-60
validation retrieval, while global-prototype and NFD-neighbor view weights
both lost to that control. The previous NVC screen found very few training
bit changes at the already fitted checkpoint. These observations do not yet
justify another 300-epoch run.

The next diagnostic asks whether a frozen two-view hash target contains useful
information **beyond a frozen single-view target** on the exact 9,000 training
images. For each image, it computes the selected NFD hash student's logits on
the standard center crop, its horizontal flip, and the existing independent
strong augmentation. The fused target is the arithmetic mean of the first two
raw-logit vectors. Both teacher targets are compared against the same strong
view, at fixed Bernoulli-distillation temperature 2.

`scripts/diagnose_i3_fused_target_signal.py` reports:

- how often fusion changes the single-view target code;
- the mean soft-target probability gap introduced by fusion;
- the bit corrections and errors each target would transfer to the strong view,
  using frozen B1 class centers as a diagnostic reference;
- each target's KL to the strong-view output;
- the extra output-logit gradient from fusion relative to single-target
  distillation, and its scale relative to center BCE.

Those are raw-output gradients, not backbone gradients; center BCE excludes
the NFD and quantization terms. No optimizer or parameter update is created.
Validation, final query and database loaders are never iterated. The dataset
split, pixel-content identity, selected checkpoint and frozen B1 center
provenance are checked. A per-training-image CSV preserves the audit.

This is not an efficacy gate or a new training method. Very small additional
target differences would give little support for a fusion-specific student
distillation experiment. Nonzero differences merely establish available
signal; they do not establish retrieval gain or originality. An audit at the
final fitted checkpoint cannot describe all earlier training epochs.

Cross-view hash distillation already appears in
[Deep Hash Distillation (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740345.pdf),
and [CroVCA](https://arxiv.org/abs/2510.27584) studies cross-view code alignment.
A future candidate must distinguish its mechanism from these methods and use
matched single-target and equal-fusion controls. Do not call an existing
distillation recipe a third innovation merely because it is added to NFD.

## Completed seed-60 audit (user-reported)

The user returned the remote report from
`save/i3_fused_target_signal32_seed60_20261030_232510`. Six remote unit tests,
the dry run and the full 9,000-image training-only audit completed normally.
The local checkout did not independently read the remote JSON.

| Diagnostic | Result |
| --- | ---: |
| Fusion changes the single-view binary target | 0 images, 0 bits |
| Single/fused target wrong-bit fraction vs B1 centers | 0 / 0 |
| Strong-view wrong-bit fraction | 0.000319444458 (about 92 bits) |
| Strong-view exact-center image fraction | 0.998444438 (about 14 affected images) |
| Single/fused target repair fraction | 0.000319444458 / 0.000319444458 |
| Single/fused target damage fraction | 0 / 0 |
| Added soft-target absolute probability gap | 0.000671624672 |
| Single target to strong-view KL | 0.006734146737 |
| Fused target to strong-view KL | 0.006552292500 |

The extra fused-target raw-logit gradient norm was 0.0634264, with a ratio
0.117356 relative to center-BCE raw-logit gradients. This is nonzero soft
signal, not a full-network gradient ratio or evidence of retrieval improvement.
It does not negate the identical binary targets and identical correction set.
The KL reduction is about 0.000181854, or 2.7% of the single-target KL.

**Decision: do not start a 300-epoch fused-target distillation experiment from
this audit.** There is no demonstrated additional binary correction signal,
and the measured extra soft-target difference alone does not justify the cost
or establish a separate contribution. This decision concerns the current
fixed checkpoint and view policy; it is not a theorem that all earlier epochs
or all distillation approaches lack useful signal. Preserve equal two-view
averaging as an engineering control and stop further seed-60 formula searches
within this fusion/distillation route.
