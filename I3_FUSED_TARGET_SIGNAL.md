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
