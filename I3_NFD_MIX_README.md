# I3 screening candidate: NFD plus semantic center Mixup

The fixed baseline remains B1+NFD. The selected 32-bit NFD student perfectly
classifies all 9,000 training images against frozen I1 prototypes, while its
validation prototype accuracy is 74.4%. The frozen teacher is lower on the
same validation proxy (72.8%). NVC and prototype-margin screens did not show
useful training signal at the selected frozen checkpoint. This motivates a
bounded test of additional *training examples*, not extra pressure on the
already fitted original training examples. These proxy accuracies are not mAP.

`b3nmix` combines the existing B1 center/quantization and NFD losses with an
auxiliary soft-center loss on images mixed from different classes. It reuses
the existing `SemanticMixup` implementation and frozen B1 relation/centers.
The semantic arm samples nearby classes using the frozen I1 relation. Its
mixing fraction is capped by the Hamming distance between the two centers.
The `uniform_budgeted` control samples classes uniformly with the **same**
Hamming budget and loss weight, so only neighbor selection changes. The
original B1+NFD run is the no-Mixup control. The `plain` mode already in B2M
uses a different amplitude policy and is therefore not the primary control.

The mechanism has no new teacher, model, inference path or retrieval rule.
Each auxiliary batch runs after the original B1+NFD backward pass with batch
normalization running statistics held fixed. Separate RNG states preserve the
paired main training order. Both arms use the same 300-epoch budget and select
by validation mAP@ALL; the launcher always skips the final query.

This is a **screening candidate**, not a confirmed third innovation. Semantic
Mixup in hashing has prior art (for example SEGA, NeurIPS 2025). Even if it
beats B1+NFD, the class-neighbor mechanism needs to beat the matched uniform
control and B1+Mixup before any narrow novelty or independent contribution
claim. If the 32-bit validation pilot fails, stop rather than tune from query.
If it succeeds, freeze settings and run all bit lengths and new paired seeds,
then evaluate query once for every preregistered group.
