# NFD versus ordinary SAM: a fixed direction screen

B1 and NFD remain the accepted first and second contributions. The negative
topology-head pilot is closed. This screen tests whether an existing generalization
optimizer is worth pursuing on the original student training problem. It is not
innovation three, a reproduction of classification-paper scores, or evidence that
previous failures were caused by sharpness.

The method is non-adaptive L2 SAM from
[Foret et al., ICLR 2021](https://arxiv.org/abs/2010.01412), independently implemented.
The [official SAM code](https://github.com/google-research/sam) is a reference;
there is no new dependency. SAM wraps the original RMSprop update instead of
switching the base optimizer to SGD. Network-weight quantization and binary
retrieval-code quantization are different problems; related work such as
[SAQ](https://arxiv.org/abs/2111.12273) also prevents treating a generic
sharpness/quantization combination as a novelty claim.

## Registered comparison

Only the deterministic CIFAR-100 32-bit seed-60 reference is accepted, with its
original 300 epochs, batch size, workers, image transformations, learning rate,
cosine schedule, validation interval, B1 teacher/centers/relations and NFD
alpha=0.75, weight=0.01, topk=5, warmup=20 and ramp-end=40. Both students are
newly initialized using the paired original hash-stage seed and ImageNet weights.
The reference student's trained checkpoint is not loaded.

| Arm | Radius | Objective | Student passes / batch | Base updates / batch |
| --- | ---: | --- | ---: | ---: |
| control | 0 | Original center BCE + quantization + scheduled NFD | 1 | 1 |
| sam | 0.05 | Ordinary SAM on that exact full objective | 2 | 1 |

Weight decay stays 1e-5 in RMSprop and is not added to the SAM ascent objective.
All student parameters with gradients participate in the single global L2 ascent;
this is not adaptive SAM, per-layer normalization, gradient clipping or a
hash-specific modification. At zero gradient the perturbation is zero. Training
statistics report the unperturbed objective, with separate perturbed loss/gap and
perturbation/gradient norms. These are local training diagnostics, not proofs of
flatness, robustness or retrieval gain.

The same augmented images and frozen teacher features are reused by the second
pass. Teacher inference happens once per active-NFD batch. Student BatchNorm
uses batch statistics for both passes but updates running buffers only once;
the second pass is not switched to eval mode. Its RNG state is replayed, then
restored to the state following the first pass. Perturbed parameters are restored
by exact copies before RMSprop updates, including on a failed second pass.

The two arms have equal epochs, images and optimizer updates, **not equal training
computation**. Together they use about three ordinary runs' worth of student
passes. Record measured wall time and memory; no runtime multiplier is guaranteed.
This first direction screen cannot attribute gains specifically to flatness or
prove superiority at an equal-compute budget. Any later method claim needs that
comparison too.

## Entry and artifacts

`bash scripts/start_i3_sam_screen.sh CONFIG 0` runs the environment check,
SAM/NFD tests, dry preflight and disposable GPU smoke check in order. Only
after all succeed does it launch the sequential screen with nohup, recording
the latest output/log/PID pointers in logs/i3_sam_screen_latest.*. It works
from the teacher environment because all Python commands explicitly use SHC.

`scripts/run_i3_sam_screen.py --nfd-config CONFIG --output-root NEW_ROOT` runs
control and sam sequentially on one GPU. It refuses an existing output directory.
`--dry-run` checks the inherited protocol, actual dataset/split and frozen caches,
without iterating loaders or training. `--smoke-test` additionally runs one disposable
real GPU SAM update at full NFD strength to check numerics/memory before a long run.
It evaluates no validation/query/database images and does not initialize either
formal arm. Both options leave NEW_ROOT unused. No DINO weights/environment or
new teacher training is required.

The existing ResNet34 ImageNet file must be present in models_ckpt or torch hub
cache. The launcher fingerprints it and the reference/source before running.
Each arm saves the original config, selected model, selection receipt, projection
and per-epoch history, plus an initialization identity. The history adds all-batch
sample-order hashes and each epoch's first two augmented images' hash. These
checks verify order and a reproducibility sample, not every augmented pixel.

The final report refuses mismatched initializations, shared protocol/artifacts,
epoch orders or sampled augmented images. `control.log` and `sam.log` contain
training details; `completed_arms.json`, `summary.csv` and `screen_report.json`
record progress and results. `--report NEW_ROOT` rechecks completed artifacts.
No full optimizer/RNG resume support is added; interrupted arms cannot be resumed
from best_model_state.pt.

Both selections use the original validation mAP@ALL and original cutoffs. Final
query is always skipped. The fixed development gate requires SAM to exceed both
the fresh control and historical NFD mAP@ALL by at least 0.005, with neither @100
nor @1000 lower. This threshold is a screening decision, not statistical significance.
The historical baseline is context, not the contemporaneous control. The familiar
validation set has already been used extensively, so a positive result is
exploratory and requires frozen settings and fresh paired seeds before confirmation.
Failure stops this fixed SAM recipe; it does not rule out all generalization methods.

## Validation scope

Synthetic CPU tests cover a manually computed SAM update, radius-zero exact legacy
reduction, BN/dropout/RNG behavior, frozen teacher calls/gradients, parameter rollback,
full tiny NFD training, initialization/input pairing, launcher guards, dry-run and
smoke dispatch, and completed-report rejection of mismatched pairs. Real GPU memory,
numerical behavior and CIFAR-100 efficacy must be checked remotely. The smoke check
exists for that purpose; local tests do not establish an improvement.

Local verification: 41 tests passed (11 SAM, 8 NFD, plus baseline/query regression
checks) on Windows with PyTorch 2.14.0+cpu; the Bash starter passed `bash -n`.
No real CIFAR-100 training or GPU smoke test was run on this workstation.

## Historical NFD manifests

Some original NFD config.json files predate the dataset_content_sha256 field.
The launcher accepts those without modifying them. It obtains the expected pixel/
label digest from the existing NFD bottleneck diagnostic.json, checking its
checkpoint SHA256, training-source SHA256, split, variant, bit length and seed
against the reference. The current raw data is still independently fingerprinted
and must match that report. A digest is never taken from the current data and
treated as its own historical verification.

By default the historical report is resolved through
logs/i3_bottleneck_latest.path -> nfd/diagnostic.json. Alternatively supply
--reference-audit FILE to the Python launcher, or FILE as the third argument to
start_i3_sam_screen.sh. Missing or mismatched historical evidence stops preflight.
The chosen report path/hash and expected data identity are recorded in the plan.
The compatibility fix passed 22 SAM/NFD CPU tests, including three additional
legacy-manifest tests for lookup, identity rejection and actual-data comparison.

## Completed seed-60 direction screen (user-reported)

The user returned the remote report for
save/i3_sam_screen32_seed60_20261031_142135_706. Both arms completed 300 epochs,
selected epoch 300 by validation mAP@ALL, and skipped final query. The report
passed initialization, epoch-order and sampled augmented-image pairing checks.
The local checkout did not independently read the remote model/report files.

| Method | mAP@ALL | mAP@100 | mAP@1000 | Epoch-loop seconds |
| --- | ---: | ---: | ---: | ---: |
| Fresh B1+NFD control | 0.5677513634 | 0.6687548945 | 0.6479442334 | 4817.642 |
| B1+NFD + ordinary SAM, rho=0.05 | 0.5859671046 | 0.6825764238 | 0.6605279599 | 10348.372 |
| SAM minus control | +0.0182157412 | +0.0138215293 | +0.0125837265 | |

The fresh control exactly reproduced all three historical validation scores.
The gains are 1.8216, 1.3822 and 1.2584 percentage points. Epoch-loop time was
about 80.29 versus 172.47 minutes, a 2.148x ratio. That history field excludes
validation and setup time, so it is not total end-to-end runtime.

**Decision: the fixed development gate passed; preserve rho=0.05 and treat
ordinary SAM as a promising strong control.** This single familiar-validation
result establishes neither independent confirmation nor a hash-specific novelty
or flatness mechanism. Do not search more radii on this split or label the
existing SAM algorithm innovation three. Follow-up should first check paired
training-randomness repeats with frozen settings. Keep split/teacher/centers
fixed for that experiment and report its scope accurately: training-seed repeats
are not new data splits or independent held-out confirmation. The current --seed
controls both data splitting and hash-stage RNG; changing it directly would
change the split and violate the frozen parent cache protocol. An explicit
hash-stage seed must be introduced and tested before launching those repeats.
Any later hash-specific candidate must beat ordinary SAM and account for its
extra compute; final query remains reserved for a frozen protocol.

## User-frozen seed-60 query evaluation (user-reported)

The user explicitly chose to keep seed=60 rather than add training-seed repeats,
and authorized evaluating the already selected 32-bit control and SAM models.
This supersedes the earlier proposed repeat-before-query sequence. Additional
seeds are not a prerequisite for the user's chosen single-seed experiment.
Both selected models were epoch 300; rho remained 0.05. The returned evaluation
counts were 5,000 query images against 45,000 database images for both arms.

| Method | Query mAP@ALL | Query mAP@100 | Query mAP@1000 |
| --- | ---: | ---: | ---: |
| B1+NFD control | 0.583456 | 0.688053 | 0.665630 |
| B1+NFD + ordinary SAM | 0.607899 | 0.706845 | 0.686009 |
| Reported SAM minus control | +0.024443 | +0.018792 | +0.020378 |

Values and deltas are transcribed at the terminal's displayed precision; the
delta was calculated from unrounded metrics. Do not reconstruct extra precision
from the rounded rows. The reported gains are 2.444, 1.879 and 2.038 percentage
points. The control reproduced the CIFAR-100 32-bit B1+NFD query row in the
user's innovation-one document. The local checkout did not independently load
the remote evaluation artifacts.

**Decision: preserve B1+NFD+SAM as an empirically positive single-seed 32-bit
configuration.** The result supports further work on the training-generalization
direction under this protocol. It is not an original SAM algorithm, multi-seed
evidence, or a causal demonstration that sharpness produced the improvement.
Training-loop cost remains about 2.148x the control; inference architecture,
one-view encoding, code length and ordinary Hamming ranking remain unchanged.

Keep seed=60 and rho=0.05 fixed for any planned 16/64-bit transfer evaluation;
use each bit length's matching B1/NFD artifacts and validation-based selection.
The current screen launcher accepts only 32-bit references, so do not assume it
already supports those runs. Record any future method proposed after seeing
these query results as further development, not as an untouched blind test.
