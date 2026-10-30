# Frozen NFD hash-readout pilot

This pilot leaves the view-fusion/distillation family. The NFD code aligns
the student's pre-hash features, while the hash head receives the original
center/quantization objective. The proposed question is whether the aligned
teacher-prototype geometry can constrain the feature-to-bit mapping directly.
This is a candidate experiment, not an established third innovation.

Use the selected 32-bit seed-60 NFD backbone once to encode the 9,000 training,
1,000 validation and 45,000 database images under the original single-view
transform. Fit three new linear heads using only training features/labels and
the existing frozen I1 geometry. The final query loader is never iterated.
There are three closed-form fits and no backbone update or 300-epoch run.

For L2-normalized training features X and frozen class-center targets Y,
the ordinary readout minimizes squared center error plus ridge regularization.
An unregularized bias is recovered from the training feature/target means.
The ridge coefficient is fixed at the mean eigenvalue of the centered training
feature covariance (trace / dimension), without validation selection.

Two geometric heads add prototype-difference equations
`(p_y-p_c) W ~= H_y-H_c`, with no bias in the difference equation:

- the global control uses all ordered distinct class pairs;
- the NFD candidate uses the original five frozen neighbors of each class.

The added Gram/cross terms are divided by the number of pairs and their mean
squared prototype-difference norm, matching the added input-Gram trace across
the graph controls. Their fixed coefficient is 0.1. All three heads share the
same training data, backbone, feature normalization and ridge coefficient.
The existing trained head is evaluated as the fourth arm and must reproduce
its saved validation mAP before any candidate result is accepted.

Inference for the fitted heads uses one student forward, feature normalization,
the fitted linear map and sign. Storage remains 32 bits per image and retrieval
uses the same stable Hamming rule. The original checkpoint is untouched. Fitted
heads and their training-only provenance are stored in `readout_heads.pt`.
This is additional supervised calibration after the original backbone fit;
do not describe its fitting budget as identical to the untouched original head.

Before viewing results, the development gate is fixed: advance the NFD head
only if its mAP@ALL exceeds the original, ordinary ridge and global-geometry
heads by at least 0.005 each, with neither @100 nor @1000 lower than those
controls. If ordinary ridge alone wins, that is evidence for generic head
calibration, not an independent NFD-neighbor contribution. If this single
configuration fails, stop rather than scan ridge, graph weight or neighbors
on the same validation split. Paired bootstrap intervals are descriptive:
this validation split selected the frozen backbone and was used in prior
method development. Any promising result needs frozen settings, fresh paired
seeds and later final-query evaluation.

Analytical supervised hashing, relation transfer and semantic-center mapping
all have prior art, including
[SDH (CVPR 2015)](https://openaccess.thecvf.com/content_cvpr_2015/html/Shen_Supervised_Discrete_Hashing_2015_CVPR_paper.html),
[RKD (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html),
and [semantic hash centers (ACCV 2022)](https://openaccess.thecvf.com/content/ACCV2022/html/Wang_Image_Retrieval_with_Well-Separated_Semantic_Hash_Centers_ACCV_2022_paper.html).
The narrow candidate here is the explicit use of the existing NFD-aligned
prototype differences to calibrate the final head. This pilot is not a
systematic novelty review or proof of an original contribution.

## Completed seed-60 pilot (user-reported)

The user returned the completed remote pilot summary. The original head's
validation score was reproduced and `query_evaluated` was false. The local
checkout did not independently read the remote JSON or checkpoint outputs.

| Head | mAP@ALL | mAP@100 | mAP@1000 | Mean false top-100 |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.567751 | 0.668755 | 0.647944 | 34.538 |
| Ordinary ridge | 0.568787 | 0.668291 | 0.648684 | 34.422 |
| Global geometry | 0.569098 | 0.669032 | 0.648689 | 34.373 |
| NFD geometry | 0.568245 | 0.666750 | 0.647491 | 34.491 |

NFD geometry minus ordinary ridge was -0.000542, -0.001540 and -0.001193.
NFD geometry minus global geometry was -0.000852, -0.002282 and -0.001198.
Relative to the original head, the NFD candidate's ALL gain was only 0.000494,
while @100 and @1000 declined. It therefore fails the registered gate.
The small gains of the other refits are exploratory and do not establish a
new mechanism or recovery of the continuous-to-binary retrieval gap.

**Decision: stop this fixed readout candidate.** Do not run final query, expand
to 16/64 bits or scan the graph/ridge coefficients for this formulation on
the same validation set. This result does not prove that every possible
hash-head architecture or fitting method is ineffective. No verified third
innovation has been obtained from this pilot.
