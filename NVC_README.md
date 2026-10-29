# I3 candidate: NFD-neighborhood view consistency

This branch implements a **candidate**, not an established improvement or a
novelty claim. The fixed basis is B1+NFD. CIFAR-100 32-bit seed-60 is used only
for validation screening; the final query is never iterated by the preflight or
pilot launcher. The earlier R-NFD correctness gate failed its feasibility check
(training gate 1.0) and is not used here.

## Mechanism

All three pilot modes use the same frozen B1 relation/centers, old I1 teacher,
NFD projection and paired ResNet34 initialization. The original weak-view B1+NFD
loss is unchanged. A second student forward receives a stronger training-only
view: resize, random resized crop with scale 0.65–1.0 and ratio 0.85–1.15,
horizontal flip and modest color jitter. Every mode adds the same strong-view
B1 center loss (default weight 0.2, warmup 40 to epoch 60).

The NFD teacher also encodes the strong view without gradients. For class y, its
already-frozen NFD neighbor basis Q_y gives the fraction of the teacher's view
displacement lying in the class-neighbor directions. The semantic mode weights
consistency by exp(-projected-fraction / 0.05). The plain mode uses weight one.
The augment mode computes the same views and teacher weight but sets the code
consistency coefficient to zero, isolating the extra augmentation/center loss.

The code term selects weak-view bits that match the correct class center and
whose signed strong-view logits fall below margin 0.5. It applies a softplus
margin penalty on those at-risk strong-view bits. This directly addresses code
changes without propagating weak-view bits that already contradict the center.
The default consistency weight is 0.02, ramped at epochs 41–60. These values
are initial screening settings, not tuned or proven optimal.

## Preflight and decision

`scripts/diagnose_i3_nvc_preflight.py` reads the old selected NFD student and
frozen teacher; it iterates one training-only pass of paired views. It checks
data/split/artifact identities and records the changed-image fraction, at-risk
bit fraction, teacher strong-view accuracy and semantic-weight spread. It writes
`preflight.json`, including a pragmatic feasibility flag. The flag requires
respectively >=0.02, >=0.005, >=0.60 and p90-p10 >=0.05. These are resource
screens, not accuracy claims. If it fails, do not launch 300-epoch pilots.

`scripts/run_i3_nvc.py` launches one `augment`, `plain` or `semantic` control
per invocation, inheriting the reference NFD hyperparameters and artifacts. It
requires a passed preflight, checks hashes and always sets `--skip-query`.
Each run independently trains 300 epochs and selects a checkpoint by validation
mAP@ALL. Compare all three validation metrics and train-view diagnostics with
the saved NFD reference. A gain from semantic mode over NFD alone is insufficient
to establish the NFD-specific component: it must also beat the augmentation-only
and plain-consistency controls. Subsequent 16/64-bit and fresh-seed confirmation
must use a frozen method, and query is evaluated only after selection.

Weak/strong hash consistency has prior art in Deep Hash Distillation (ECCV 2022)
and CroVCA (CVPR Workshops 2026). This implementation tests whether frozen NFD
neighbor directions provide useful sample-specific supervision in this SHC
setting. It does not claim to originate cross-view hash alignment.
