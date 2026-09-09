# I2-NFD implementation

Implemented on `codex/i2-nfd`. This is an experimental candidate, not a validated improvement.
No real-data training or final-query evaluation was run during implementation.

## Entry and controls

Use `run.py --variant b2nfd` with the audited same-seed B1 config, similarity cache,
center cache, and classifier cache. Existing `load_frozen_b1` validates the common
protocol and frozen center/relation content. NFD additionally validates classifier
cache metadata and its state SHA256, and records prototype, relation-file and basis
SHA256 values. The teacher is constructed without pretrained downloads.

| Option | Default | Meaning |
| --- | --- | --- |
| `--teacher-cache` | required | Existing classifier cache from the parent B1 |
| `--nfd-alpha` | 0.75 | Neighborhood weighting; **0 is ordinary feature distillation** |
| `--nfd-weight` | 0.01 | Maximum auxiliary weight; **0 disables the auxiliary path** |
| `--nfd-topk` | 5 | Other classes selected by frozen I1 similarity |
| `--nfd-warmup` | 20 | Last disabled epoch |
| `--nfd-ramp-end` | 40 | First epoch at full weight |

Keep `--paired-stage-seeds` and use `--skip-query` during development. All shared
arguments must match the parent, including workers (the CLI default is 6, whereas
the current parent used 4). NFD is mutually exclusive with the old innovation-two
modules and cannot use `--resume-sbr`.

## Launch template (not executed)

Run only after the proposed teacher-information check and a remote synthetic-test
pass. Supply actual same-seed audited paths for the four cache/config variables.
For the current seed=60 parent, the common settings are:

```bash
python -u run.py --variant b2nfd \
  --parent-b1-config "$PARENT_CONFIG" \
  --similarity-cache "$SIMILARITY_CACHE" \
  --center-cache "$CENTER_CACHE" \
  --teacher-cache "$CLASSIFIER_CACHE" \
  --dataset cifar-100-new-seg --root ./data/cifar-100-python \
  --num-classes 100 --code-length 32 --seed 60 \
  --classify-epoch 300 --epoch 300 --lr 7e-5 --batch-size 64 \
  --num-workers 4 --test-map 5 --topK -1 100 1000 --gpu 0 \
  --relation-mode full --relation-kappa 30 --relation-temperature 1 \
  --relation-weight-floor 0.05 --paired-stage-seeds --skip-query \
  --nfd-alpha 0.75 --nfd-weight 0.01 --nfd-topk 5 \
  --nfd-warmup 20 --nfd-ramp-end 40 --output-dir "$NEW_OUTPUT_DIR"
```

For the ordinary-distillation control, change only `--nfd-alpha 0` and use another
new output directory. Do not re-use a completed run directory. Both variants train
a fresh, paired student; neither continues training the parent's best model.
No comparison should claim a new mechanism solely from beating B1: compare the
neighborhood version to ordinary distillation with the same teacher and budget.

## Training and output behavior

The student has a single forward/BN update per batch. Teacher inference uses the
same augmented input, `eval()` and `no_grad()`, and starts only once the auxiliary
weight is positive. Teacher preparation preserves RNG; the main entry resets the
paired hash-stage seed after preparation. Only student parameters enter RMSprop.
NFD sends gradients to the backbone, not directly to `hash_layer` or the teacher.
The original center BCE and quantization objective stay active.

Projection uses float64 SVD with a deterministic class-ID tie break and threshold
`max(1e-12, 1e-8 * max_singular_value)`, then stores a float32 basis. The zero-rank
case falls back to ordinary normalized feature MSE. A trace normalization controls
overall quadratic-form scale, not the actual gradient norm.

`config.json` includes NFD version/parameters/provenance; `nfd_projection.pt` stores
bases, ranks and neighbors. Per-epoch history includes component losses and first-
batch clean/weighted-NFD gradient diagnostics. Zero-weight epochs have zero NFD
loss and no teacher forward; absent projected-error fields are not measurements.
Best student weights are saved at each improved validation mAP@ALL. Per-epoch
history is also saved; **NFD does not yet support full optimizer/RNG resume**.
A best-model file is not a full training checkpoint. Keep the terminal connected
for formal runs until a separate resume workflow is implemented and verified.

The teacher and projector are absent from the saved student `state_dict`; ordinary
ResNet inference and Hamming evaluation remain compatible. This branch has a new
source digest. Old evaluation scripts pinned to the SBR source should not have
their checks bypassed; use the relevant original branch for historical reproduction.

## Verification

```bash
python -m unittest tests.test_b2nfd_feature_distill -v
python -m unittest discover -s tests -v
```

Local verification: 8 NFD tests and 73 total tests passed on Windows,
PyTorch 2.2.2+cpu / torchvision 0.17.2+cpu. Coverage includes formula/trace,
rank-zero fallback, ordinary-distillation reduction, invalid inputs, teacher BN
and gradient isolation, state/forward compatibility, actual parent-cache checks,
RNG preservation, distinct run identities, no artifact regeneration, and tiny
training with exactly matching disabled/B1 losses and parameters. Tiny integration
evaluation uses a stub that rejects query calls. CUDA 2.1.2+cu121 and real CIFAR-100
training have not been exercised locally; these checks do not establish retrieval
performance or GPU memory cost.
