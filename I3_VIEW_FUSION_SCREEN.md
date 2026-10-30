# Frozen NFD view-fusion screen

The B1+NFD student remains the fixed baseline. This script is a low-cost
feasibility check after the NFD+semantic-Mixup pilot failed: 300 completed
epochs, selected epoch 20, and validation mAP@ALL 0.513021 versus 0.567751
for NFD. It does not train a model or establish a third innovation.

`scripts/diagnose_i3_view_fusion.py` loads the selected 32-bit seed-60 NFD
checkpoint. It encodes the same validation and original database images with
the standard center crop and its horizontal flip. The simple control averages
the two raw hash-logit vectors before taking the sign. All arms still produce
32-bit codes and use the existing stable Hamming evaluator:

- original validation code against original database codes;
- fused validation code against original database codes;
- fused validation code against fused database codes.

The original score must exactly reproduce the saved selected validation score.
Split IDs, labels, checkpoint digest and dataset-content digest are checked.
The final query loader is created for split verification but never iterated.
The full report and per-validation-image AP are saved under the requested
output directory. Paired bootstrap intervals are descriptive because the same
validation set selected the frozen checkpoint.

Simple two-view fusion is an established technique and has double image-encoding
cost; it is a diagnostic control, not a novelty claim. If symmetric fusion does
not clearly improve all three validation metrics, stop this route. If it does,
the next method must add a defensible B1/NFD-specific mechanism, beat this
simple control at the same encoding cost, and survive fresh paired seeds before
any final-query evaluation.
