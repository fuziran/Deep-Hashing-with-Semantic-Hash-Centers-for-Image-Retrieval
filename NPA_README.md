# I3-NPA: neighborhood-protected implicit feature augmentation

This branch adds `--variant b3npa`: I1 + the existing NFD + I3-NPA.
It is an experimental candidate. CPU synthetic tests validate implementation;
they do not establish CIFAR-100 retrieval improvement or CUDA compatibility.
No real-data training, query evaluation, commit or push was performed here.

## Method and exact scope

Use the same frozen B1 relation, centers and classifier teacher as the paired NFD
reference. Train a fresh student with the same hash-stage initialization and data
order, rather than continuing the reference's best weights. NFD parameters should
be copied from the actual reference, not reset to CLI defaults.

Before hash training, run the frozen eval teacher on the training-only relation
loader with center crop and horizontal flip. Normalize each view, average them,
normalize the average, and accumulate one observation per training image. Exact
image-ID coverage is checked against the training loader. No validation, query or
database images contribute to these statistics. The existing dataset loader in
this checkout supports CIFAR-100 only; this change does not add NABirds/Cars or
multi-label loaders.

CPU float64 parallel-Welford accumulation estimates unbiased class covariances.
With eta = `--npa-shrinkage`, combine each covariance with the pooled *within-class*
covariance, weighted by class degrees of freedom. Between-class variation is not
included in the pooled covariance.

For class y, reuse the exact padded NFD orthonormal basis Q_y. Let P_y = Q_y Q_y^T.
The three modes are:

* `protected`: A_y = (I-P_y) C_y (I-P_y).
* `covariance`: A_y = C_y (no neighborhood protection).
* `isotropic`: A_y = I/d.

All final covariance matrices have trace one. Thus the controls match total
perturbation energy, rather than letting projection win simply by reducing noise.
Protection's retained trace fraction *before* normalization is recorded. A class
with no variation, fewer than two images, or retained fraction <= 1e-6 is rejected
rather than amplified or silently replaced. Zero-rank NFD protection naturally
reduces to the covariance control. No low-rank approximation is used in v1.

For raw student feature f_i and hash head z_i = W f_i + b, define the local
perturbation covariance

    Sigma_i = rho^2 * stop_gradient(||f_i||^2) * A_y.

This reconciles normalized teacher statistics with the existing raw-feature hash
head. The perturbation RMS norm is rho times the student feature norm. Covariances
model empirical class variation, not calibrated predictive/model uncertainty;
Gaussian perturbations and protected subspaces do not guarantee label preservation.

For center bit h in {-1,+1}, let v_ik = w_k^T Sigma_i w_k. The added bit loss is

    R_ik = softplus(-2*h*z + 2*v) - softplus(-2*h*z)
    L = L_B1 + lambda_NFD(t)*L_NFD + lambda_NPA(t)*mean(R_ik).

Gaussian moment generation and Jensen's inequality give the first softplus as an
upper bound on expected *logistic* bit BCE. Factor 2 matches sigmoid(2*z) =
(tanh(z)+1)/2. The original probability BCE/quantization implementation remains
unchanged; finite-precision saturated BCELoss is not identical to logit BCE, so
the total numerical training loss is not claimed to be a global bound in those
extremes. The increment uses float64 subtraction before converting back to the
student dtype, and averages across samples and bits. The feature-norm scale is
explicitly detached; this is a local frozen-scale surrogate during each update.

I3 gradients reach W and the student backbone through z. Teacher, covariance and
Q stay frozen. Only student parameters enter the optimizer. Students have one
forward/BatchNorm update per batch, even when both auxiliary losses are enabled.
There are no sampled extra features, second student view, new teacher, modified
hash center, bit dropping, reranking or changed Hamming tie handling.

Setting `--npa-weight 0` **or** `--npa-rho 0` skips covariance preparation and the
NPA training path. Paired synthetic runs verify identical B2NFD losses and final
student parameters. The source/run identities still differ by variant, as intended.

## Initial candidate settings

| Option | Default | Meaning |
| --- | --- | --- |
| `--npa-mode` | protected | Full method or covariance/isotropic controls |
| `--npa-rho` | 0.05 | Relative RMS feature perturbation |
| `--npa-shrinkage` | 0.2 | Weight of pooled within-class covariance |
| `--npa-weight` | 1.0 | Maximum increment weight, restricted to [0,1] |
| `--npa-warmup` | 40 | Last epoch with NPA disabled |
| `--npa-ramp-end` | 60 | First epoch at full weight |

These are unvalidated starting settings, not a search result. Epochs 1-40 follow
the paired NFD training path; epochs 41-60 ramp up the increment. Keep the NFD
schedule from the reference. Do not choose settings by repeatedly evaluating query.

At C=100,d=512, frozen float32 covariance buffers occupy about 100 MiB on the
training device. Float64 CPU statistics require about 200 MiB for M2, plus means,
temporary covariance/projection matrices and copies. Two teacher views are run
once over the training subset before training, in addition to the original NFD
teacher cost. Per-batch covariance products are computed for present classes only.
Measure actual preparation time, peak CPU/GPU memory and training throughput.

## Local commit and push (PowerShell, repository root)

The branch has already been created. Stage only these task files, avoiding the
existing untracked data, reports, output directories and Git bundle:

```powershell
git branch --show-current
git add -- run.py train.py utils/experiment.py utils/neighborhood_protected_augmentation.py scripts/run_i3_npa.py tests/test_b3npa_augmentation.py tests/test_i3_launcher.py NPA_README.md README.md
git diff --cached --stat
git commit -m "Implement I3 neighborhood-protected augmentation on NFD"
git push -u origin update-reliable-relation-nfd-distillation-i3-neighborhood-protected-augmentation
```

## Remote execution (Linux, existing SHC environment)

From the remote repository root, fetch and switch to this branch. Preserve any
unrelated remote local changes before switching; Git will refuse conflicts.

```bash
git fetch origin
git switch update-reliable-relation-nfd-distillation-i3-neighborhood-protected-augmentation
git pull --ff-only
python -m unittest discover -s tests -v
```

The launcher finds exactly one `b2nfd` `config.json` per requested code length,
dataset and seed under `--baseline-root`. It inherits epoch budget, seed, split
settings, optimizer settings, loader settings, all NFD options and the four parent
artifact paths separately for each code length. It does not select by score.
All paths are checked before the first job starts. The training entry then checks
parent configuration, cache metadata and tensor/state content as in NFD.

Assuming the reference configs/caches are still under `./save`, first print and
inspect the complete inherited training commands without starting GPU work:

```bash
python scripts/run_i3_npa.py \
  --baseline-root ./save --dataset cifar-100-new-seg --seed 60 \
  --bits 16 32 64 --gpu 0 \
  --output-root ./save/i3_npa_seed60_v1 \
  --mode protected --rho 0.05 --shrinkage 0.2 --weight 1.0 \
  --warmup 40 --ramp-end 60 --evaluate-query --dry-run
```

For the requested three final experiments, run the same command without
`--dry-run`. `nohup` keeps the batch running if the SSH terminal disconnects:

```bash
nohup python -u scripts/run_i3_npa.py \
  --baseline-root ./save --dataset cifar-100-new-seg --seed 60 \
  --bits 16 32 64 --gpu 0 \
  --output-root ./save/i3_npa_seed60_v1 \
  --mode protected --rho 0.05 --shrinkage 0.2 --weight 1.0 \
  --warmup 40 --ramp-end 60 --evaluate-query \
  > i3_npa_seed60_v1.launch.log 2>&1 &
```

Jobs run **sequentially**, not concurrently on one GPU. Each inherits its own
reference's epoch budget (300 if those references used 300). `--evaluate-query`
evaluates the final query once, after selecting the best checkpoint by validation
mAP@ALL. Omit it during development to use validation only. All three metrics come
from the same selected model, never separate best checkpoints per metric.

`./save` is a search root, not a claim about where the remote results actually
reside. Change it if needed. For duplicate configs (e.g. ordinary distillation and
NFD candidates in the same tree), replace auto-discovery with explicit paths:

```bash
python scripts/run_i3_npa.py \
  --nfd-config /actual/path/to/16bit_NFD/config.json \
  --nfd-config /actual/path/to/32bit_NFD/config.json \
  --nfd-config /actual/path/to/64bit_NFD/config.json \
  --seed 60 --bits 16 32 64 --gpu 0 \
  --output-root ./save/i3_npa_seed60_v1 --evaluate-query --dry-run
```

The explicit paths above are placeholders; the automatic command needs none.
Resolve stale cache paths in copied manifests instead of bypassing content checks.
Use `--data-root /actual/cifar-100-python` only if the original dataset path moved.
Existing output roots are rejected. Do not reuse a completed/partial output root.

To run just one length, use `--bits 32` and a new output root. Controls use exactly
the same launcher with `--mode covariance` / `--mode isotropic` and distinct output
roots. For a disabled control, use `--weight 0` and another output root.

## Outputs and checks

At the batch output root:

* `launch_plan.json`: exact inherited command and source NFD config per job.
* `16bit.log`, `32bit.log`, `64bit.log`: unbuffered per-run training logs.
* `summary.csv`: completed jobs, selected epoch and three query mAP metrics (blank
  query columns when evaluation is disabled), updated after each completed job.

Inside each `16bit_protected/runs/...` (and analogous 32/64-bit run):

* `config.json`: NFD/NPA settings, parent/split provenance and covariance digest.
* `npa_statistics.pt`: frozen covariance, teacher means, counts, retained trace and
  hashes; `nfd_projection.pt` retains the original neighborhood bases.
* `training_history.json`: original loss, NFD loss, NPA increment, logit variance,
  schedule and first-batch gradient diagnostics. NPA's gradient reference is the
  combined center+quantization+NFD objective; NFD's reference remains the original
  center+quantization objective. Disabled epochs do not measure NPA variance.
* `best_model_state.pt`, `metrics.json`, and, when enabled, original retrieval
  outputs/per-query AP. Model state contains only the ordinary student.

The original NFD epoch history/best-weight behavior is retained. **Full optimizer,
scheduler and RNG checkpoint resume is not implemented for NFD/NPA.** A best-model
file cannot resume the exact training trajectory. `nohup` prevents terminal hangup,
not machine/process failures; a failed job must be rerun in a new output directory.

For the example batch:

```bash
tail -f ./save/i3_npa_seed60_v1/16bit.log
cat ./save/i3_npa_seed60_v1/summary.csv
```

Implementation checks cover streaming statistics vs direct covariance, PSD and
trace, protected directions, energy-matched controls, Gaussian-quadrature loss
bound, extreme logits, bit averaging, gradients, frozen teacher/BN, preparation
RNG preservation and train-ID checks, disabled exact NFD reduction, single student
forward, real entry/cache dispatch and launcher preflight/inheritance/summary.

The CPU suite passes 88 tests in the local PyTorch 2.14.0+cpu environment. CUDA and
real CIFAR-100 results remain for remote verification. Establish improvement using
paired NFD comparisons across all nine metrics, additional seeds and the covariance
control; reduced auxiliary loss alone is not retrieval evidence.
