# I3 candidate: reliability-gated neighborhood feature distillation

This is a **candidate**, not a demonstrated improvement or a novelty claim. The
historical 32-bit seed-60 NFD validation run is weaker than ordinary feature
distillation, particularly among validation images whose binary codes change
under horizontal flip. That association does not prove that flip instability
causes the loss. A frozen teacher gate offers a falsifiable way to ask whether
the NFD neighborhood projection should be trusted for every training view.

## Mechanism and controls

Keep the frozen B1 similarity, centers, classifier, NFD basis, student ResNet34,
training schedule, code length and ordinary Hamming evaluation. For each training
view, the frozen I1 teacher predicts on the image and its horizontal flip. The
gate is one only if both predictions match the training label. The teacher is
always in eval mode, is detached, and is never optimized. On the student/teacher
normalized feature difference, use the existing NFD trace-normalized loss with
per-sample `alpha_i = nfd_alpha * gate_i`. Gate zero is exactly ordinary feature
distillation; gate one is exactly fixed NFD. The teacher sees the *actual training
augmentation* online, so a one-pass preflight is only a feasibility estimate.

`--rgnfd-mode gated` uses the individual gate. `mean` replaces every gate in a
batch with that batch's mean, matching its average neighborhood strength while
removing sample selection. `fixed` keeps the original NFD loss and performs the
extra frozen-teacher flip computation as a compute-matched control. All modes
use paired initialization, fresh training and validation-only checkpoint
selection. The launcher always passes `--skip-query`; the final query is not
iterated during the gate audit or the pilot.

## Feasibility gate

`scripts/diagnose_rgnfd_gate.py` checks exact raw-data and split fingerprints,
loads the audited old classifier, computes the two-view gate on training-only
augmentations and separately on validation center crops, and optionally joins
the existing frozen-student per-query validation audit by image ID. It reports
whether 5–95% of training examples pass the gate. This range is a pragmatic
resource screen for a nearly constant gate, not an accuracy or significance
criterion. Validation group comparisons are exploratory and cannot set a tuned
gate threshold or confirm retrieval improvement.

The new variant is `run.py --variant b3rnd`; the launcher is
`scripts/run_i3_rgnfd.py`. It inherits the exact seed-60 NFD reference config,
requires the gate report, records its checksum and verifies data, teacher and
projection identities. A pilot must be compared with the saved ordinary-FD and
fixed-NFD runs, then with `mean` before claiming sample-specific value. Seed-60
query has already been viewed historically; do not use it to choose this method.

Confidence-aware distillation and cross-view hash alignment have prior art.
The gate and its interaction with NFD are hypotheses requiring the controls and
independent seed confirmation before any paper claim.
