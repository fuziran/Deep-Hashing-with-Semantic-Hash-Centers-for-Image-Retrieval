# NFD-neighbor view-fusion screen

The fixed 32-bit B1+NFD student gains +0.013955, +0.010729 and +0.012616
validation mAP@ALL, @100 and @1000 from simple two-view logit averaging on
both validation and database images. This is exploratory on the checkpoint-
selection validation set. The final query was not evaluated. Two-view averaging
is the cost-matched control and is not a new method.

This screen asks whether NFD's frozen teacher-prototype neighbor geometry gives
better **label-free** view weights. The student already produces a 512-d feature
with each hash-logit vector, so no teacher forward or additional student view
is needed. The jointly averaged feature predicts one anchor prototype. For
each view, its support for that anchor is the cosine gap to the strongest
competing prototype among the five frozen NFD neighbors. Negative gaps are
clamped to zero. The two gaps yield a bounded original-view weight

`w = 0.5 + 0.25*(r_original-r_flipped)/(r_original+r_flipped+1e-6)`.

Thus w is in [0.25,0.75], and both views still contribute. The raw logits are
combined before sign. The output remains exactly 32 bits and retrieval uses
unchanged Hamming ranking. There is no true label or validation-derived
parameter in the fusion rule. The prototype and neighbor artifacts are checked
against the original NFD run.

`scripts/diagnose_i3_neighbor_fusion.py` scores four symmetric policies on the
same frozen validation/database split:

1. original one-view codes;
2. equal two-view averaging;
3. all-prototype margin weighting with the same bounded formula;
4. NFD-neighbor margin weighting.

The original score must reproduce the saved NFD validation mAP. The main
comparison is policy 4 minus policies 2 and 3, all at the same two student
forwards and 32-bit storage. Do not advance this route unless policy 4 beats
both controls on mAP@ALL by at least 0.005, does not reduce @100 or @1000, and
changes some codes relative to equal averaging. This is a development gate,
not statistical proof. The same validation set selected the checkpoint and is
being used for this method screen. Any promising rule needs frozen settings,
fresh paired seeds, and a careful novelty check before final-query evaluation.

Adaptive test-time augmentation and prototype-guided hashing have prior art;
this specific combination is only a candidate until both efficacy and a
defensible contribution beyond those methods are established.

## Seed-60 screening result (user-reported, 2026-10-30)

The remote screen completed and reproduced the original selected validation
score. Values below were returned by the user from `neighbor_view_fusion.json`;
the local checkout did not independently read the remote artifact.

| Frozen policy | mAP@ALL | mAP@100 | mAP@1000 | Mean false top-100 |
| --- | ---: | ---: | ---: | ---: |
| Original one view | 0.567751 | 0.668755 | 0.647944 | 34.538 |
| Equal two-view average | **0.581706** | **0.679484** | **0.660560** | **33.278** |
| All-prototype weighting | 0.573891 | 0.670225 | 0.655021 | 34.026 |
| NFD-neighbor weighting | 0.575965 | 0.671443 | 0.656283 | 33.803 |

NFD-neighbor minus equal averaging was -0.005741, -0.008041 and -0.004277
for the three mAP metrics. Its code differed from equal averaging on 13.4% of
validation images and 10.66% of database images, so the failure was not an
inactive branch. The descriptive paired bootstrap intervals for these three
deltas were entirely negative, but the validation set had selected the frozen
checkpoint and these are not independent confirmatory intervals.

**Decision: stop this candidate.** It fails the registered equal-fusion gate
on all three metrics. Do not run a final-query evaluation, new seeds, or a
parameter search for this formulation. Equal two-view averaging remains a
useful engineering control, not a substantiated third innovation.
