# I3 search: frozen NFD prototype-margin screen

The NVC pilot gate failed on the selected CIFAR-100 32-bit seed-60 NFD student:
only 0.001667 of training images changed any hash bit across the paired views,
and the at-risk bit fraction was 0.000441. The teacher-view gate had a wide
spread (p90-p10 = 0.4471), and the strong-view teacher accuracy was 0.9991.
These are one-pass observations of the **selected frozen checkpoint**, not a
measurement of the whole training trajectory. Do not run the three 300-epoch
NVC pilots based on this screen.

`scripts/diagnose_i3_prototype_margin.py` is the next read-only screen. It uses
the existing NFD student, I1 teacher and cached I1 class prototypes, and reads
only the fixed training and validation splits. It compares cosine similarity to
the true-class prototype with the nearest different-class prototype. The
`actionable_fraction` counts examples where the student margin is below 0.1
and the teacher margin is at least 0.1. It also reports the proportion of the
student's nearest competing classes in the already-frozen NFD neighbor list.

These are feasibility diagnostics for a possible dense, feature-level
class-neighbor loss. Prototype accuracy and margins are **not retrieval mAP**;
validation already participated in selecting the frozen checkpoint. A favorable
screen would justify implementing a candidate with matched ordinary-prototype,
NFD-neighbor, and no-extra-loss controls. It would not establish a new method's
novelty or retrieval benefit. Prototype contrastive learning and class-center
separation already have prior art, including CVPR 2023's minimal-distance
separated hash centers; a paper claim needs a narrower mechanism and ablations.
