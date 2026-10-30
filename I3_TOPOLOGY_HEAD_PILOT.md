# Five-epoch topology-anchored nonlinear hash-head pilot

The fixed base is the selected 32-bit seed-60 B1+NFD checkpoint. Prior view
weighting, fused-target signal and linear readout pilots did not establish a
third innovation. This pilot tests a different mechanism: learn the class
representative codes and nonlinear readout jointly, while retaining B1's
pairwise center geometry rather than requiring every bit to equal a fixed
class center.

The small MLP and alignment/diversity control are informed by
[CroVCA](https://arxiv.org/abs/2510.27584) and its
[official implementation](https://github.com/ilyassmoummad/cross-view-code-alignment).
This is an adaptation, not a paper reproduction or a claim of state-of-the-art
performance. It uses a frozen NFD ResNet, two fixed cached training views, and
the existing symmetric binary Hamming evaluator. CroVCA's paper setup includes
foundation encoders and asymmetric retrieval. The candidate is not verified as
an original contribution by adopting that published head recipe.

One original weak and one existing strong augmentation are encoded for each
of the 9,000 training images. Validation (1,000) and original database (45,000)
are encoded under their original single-view transform. The final query loader
is never iterated. Only the training feature cache and frozen training-derived
B1/NFD artifacts enter head optimization. Cache identity includes checkpoint,
data pixels/labels, split, view source and collection seeds/workers. The cache
is saved for audit and can be reused after a run fails without re-encoding.

Four heads share the architecture Linear(512,512) -> ReLU -> Linear(512,32) -> BatchNorm,
paired initialization and batch order. Their fixed budget is 5 epochs each,
batch size 256, AdamW, cosine LR 1e-3 to 1e-6, weight decay 1e-2:

| Arm | Objective |
| --- | --- |
| center_mlp | Original B1 center BCE and 1e-4 quantization term; capacity/budget control |
| cv_control | Class-batch-mean hard-code alignment and coding-rate regularization, weight 0.1 |
| global_topology | cv_control plus B1 center-affinity target loss, uniform class-pair weights, coefficient 0.1 |
| nfd_topology | Same topology target and coefficient; half uniform pair weight, half frozen NFD-neighbor graph weight |

Inputs are L2 normalized. For supervised alignment the current class
mean of the paired batch's output logits supplies a detached binary target.
Coding-rate regularization follows the published MCR-style construction with
epsilon 0.05. This learns new codes instead of distilling a saturated frozen
training target.

For topology, each class's representative input is the mean of its **student
training features**, avoiding a new assumption that teacher and student feature
coordinates are identical. Its predicted soft code is tanh(head(prototype)).
The weighted off-diagonal squared error compares its pairwise affinity with
`H1 @ H1.T / bits`. This target permits a common bit permutation/sign change
and does not pin an individual learned bit to a particular B1 bit. The
prototype-only forward receives gradients with BN running statistics frozen.
All pair weights sum to one. No validation labels, database labels, teacher
forward or query input is used to build this loss.

Every head is evaluated at the fixed final epoch 5. There is no head checkpoint
selection using validation. The original selected head must reproduce its
saved validation metrics before training begins. Outputs include feature
cache, each fitted head/history, per-validation-image AP and all five method
results, including the original head. Inference uses one view and the MLP;
codes remain 32 bits and Hamming ranking is unchanged. The extra head has
more parameters than the original linear head, so all matched controls and
the extra post-training calibration budget must be reported.

The development gate is fixed before results: nfd_topology must exceed the
original, center_mlp, cv_control and global_topology mAP@ALL by at least 0.005
each, with neither @100 nor @1000 below those comparators. A gain shared by
the control is not an independent NFD-topology gain. If this fixed recipe
fails, do not scan coefficients, epochs, hidden width or pair weights on this
validation split. The backbone was already selected on the same validation
split, so all paired intervals are descriptive. A promising candidate still
needs novelty review, frozen settings and fresh paired seeds before query.
