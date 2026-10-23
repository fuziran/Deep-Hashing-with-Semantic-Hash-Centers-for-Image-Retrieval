# I3-FCD: foundation-model discriminative distillation on B1+NFD

Branch: `update-reliable-relation-nfd-distillation-i3-foundation-discriminative-distillation`.
Entry: `run.py --variant b3fcd`. This is an experimental candidate; no retrieval
gain is claimed. The previous NPA implementation remains available for historical
reproduction and is not activated by FCD.

## Implementation

Keep the audited I1 relation, centers, old classifier and NFD unchanged. Train a
fresh paired ResNet34 student. Add a training-only linear classifier after its
continuous hash codes. Its initialization is `weight = 4*H/code_length, bias=0`,
preserving RNG and giving all code lengths the same initial self-center logit.
The auxiliary head is separate from the student's inference state.

A separate offline preparation process loads a frozen official DINOv3 ViT-B/16
(or explicitly requested DINOv2 ViT-B/14 with registers) from a local repository
and a local weight file, with strict state loading and no automatic weight fetch.
It extracts center-crop and flipped features for the exact training subset and
validation subset. Normalize each view, average them, and normalize again. Fit
a zero-initialized linear classifier on training features only (AdamW, lr=0.03,
weight decay=0.001, 200 full-batch steps); select by validation accuracy every
10 steps, also checking the final step. Cache **training-only** class logits.

The cache is a weak-view teacher. Original random-crop training views are left
unchanged: this is cached weak-view-to-original-student supervision, not online
same-view foundation inference. The old NFD still uses its original same-view
frozen classifier. A per-image table of 9000x100 float32 logits is about 3.4 MiB.
No DINO inference is performed during the 300-epoch hash runs.

For auxiliary prediction `a=A(tanh(z))`, use:

    L = L_B1+NFD + lambda_FCD(t) * (CE(a,y) + TCKD(a,t) + NCKD(a,t)).

TCKD uses the binary distribution [true-class probability, sum of other classes].
NCKD uses the normalized distribution of all non-target classes. KL is evaluated
with stable log-softmax and multiplied by T^2; the two components have coefficient
one. Teacher logits are detached. No wrong class is made a positive target.

Modes:

* `ce`: only the auxiliary classifier and hard training labels (no KD loss).
* `plain`: ordinary decoupled KD with the same new teacher.
* `neighbor`: on the top-k non-target I1 neighbors, set
  `w_yc = 1 + gamma*max(0,S_yc)`, leaving other weights at one. Before NCKD,
  multiply **both** non-target distributions by these weights and renormalize.
  TCKD is unchanged. This bounds weights to [1,2] for gamma<=1; it does not match
  actual gradient norms to plain KD. Record and inspect the gradient diagnostics.

The initial candidate uses max weight=0.05, T=2, gamma=1, top-k=5, warmup=40,
ramp-end=60. Epochs 1-40 do not use FCD, epochs 41-60 ramp up. These settings are
unvalidated starting values, not tuned optima. `--fcd-weight 0` skips the new cache,
head and loss; synthetic tests verify exact NFD loss/parameter reduction.

FCD gradients reach both the original hash layer and backbone through tanh(z).
Only the student and new auxiliary head enter RMSprop, with the original learning
rate/weight decay/scheduler. There is one student forward/BN update per batch.
After training, discard the auxiliary head and use the original student + sign +
ordinary Hamming retrieval. The source digest includes the new training module.

External pretrained data and teacher-preparation computation are **additional
resources**. This is not a same-information-budget improvement over B1+NFD.
Foundation KD and DKD already exist; the neighborhood mechanism needs the plain
KD control before it can be described as a supported independent contribution.

## Teacher information check

Preparation never iterates the final-query loader or the official database loader.
Train and validation image IDs must be disjoint. It reports classification errors
corrected/regressed relative to the original I1 classifier. A second fixed-budget
diagnostic trains a zero-initialized 32-bit linear hash head on each frozen feature
set, using the same B1 centers, AdamW settings and 200 steps. That head uses
BCEWithLogits(2z) and is evaluated at its final step, not selected by query/mAP.

The diagnostic gallery is **training images only**, queried by validation images;
it is not the formal full-database mAP protocol. Preparation passes only if:

1. New teacher validation accuracy is higher, with more old errors corrected than
   new errors introduced.
2. New frozen-feature hash probe has higher validation mAP@ALL and no decrease in
   @100 or @1000 against the old frozen-feature probe.

This is an operational screening rule, not a statistical guarantee or proof that
distillation will improve final mAP. The teacher classifier is selected on the
same validation set used for the check, so the check is developmental evidence.
All outputs are saved even on failure, then preparation exits nonzero and the
launcher refuses formal runs. Do not bypass the check to claim teacher superiority.

`foundation_cache.pt` contains plain CPU tensors and scalar/dict provenance for
compatibility with the original SHC environment. Loading verifies split, dataset,
seed, old classifier digest, preprocessing sizes, exact training IDs/labels, actual
training-image bytes and tensor digests. File, foundation weights and teacher-head
hashes, preparation source hash, model-repo commit and Torch version are recorded.
Only load trusted local torch caches/checkpoints.

## Commit and push on local Windows PowerShell

The new branch has already been created and checked out. Stage these files only:

```powershell
Set-Location D:\SHC\SHC-Baseline-run
git add -- README.md FCD_README.md run.py train.py utils/experiment.py utils/foundation_distill.py scripts/prepare_foundation_teacher.py scripts/run_i3_fcd.py tests/test_foundation_distill.py
git diff --cached --stat
git commit -m "Implement foundation discriminative distillation on B1 NFD"
git push -u origin update-reliable-relation-nfd-distillation-i3-foundation-discriminative-distillation
```

## Remote checkout and exact reference paths

From the existing remote repository `/mnt/d/fzr/SHC`, using its original SHC
environment, run:

```bash
git fetch origin
git switch update-reliable-relation-nfd-distillation-i3-foundation-discriminative-distillation
git pull --ff-only
python -m unittest discover -s tests -v

NFD16='./save_nfd_16_64_seed60_ZZdT2Ou0/nfd_16bit/runs/cifar-100-new-seg_B2NFD_16bit_seed60_ac0f8d38af38/config.json'
NFD32='./save_nfd_seed60_bINBuclZ/nfd/runs/cifar-100-new-seg_B2NFD_32bit_seed60_639b7ea1eaa3/config.json'
NFD64='./save_nfd_16_64_seed60_ZZdT2Ou0/nfd_64bit/runs/cifar-100-new-seg_B2NFD_64bit_seed60_715253eea766/config.json'
```

These are the exact original NFD references previously supplied in this chat, not
the `fine_aux` family or ordinary-FD controls. The 32-bit reference's historical
metrics file had no saved query metrics; its equality to the earlier screenshot's
32-bit value has not been established. Commands inherit the actual reference's
protocol, not screenshot scores. Each parent still gets its original audit checks.

## One-time DINOv3 setup and weight acquisition

DINOv3's official training/evaluation environment requires PyTorch >=2.7.1; keep
the existing SHC environment intact and prepare the cache in a separate one. The
official model weights require access through Meta's download process. Obtain
the **ViT-B/16, LVD-1689M** backbone URL through the official repository's download
link. Do not use a satellite checkpoint, random model, or a different architecture.

* Official model/access instructions: https://github.com/facebookresearch/dinov3
* PyTorch matching versions: https://pytorch.org/get-started/previous-versions/

The following commands use the official PyTorch 2.7.1 + torchvision 0.22.1 CUDA
11.8 wheels in a separate environment. They do not replace the original SHC install:

```bash
conda create -n SHC_FCD_teacher python=3.11 -y
conda activate SHC_FCD_teacher
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'numpy<2' pillow scipy matplotlib

mkdir -p external models_ckpt
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
read -r -p 'Paste the official DINOv3 ViT-B/16 LVD-1689M download URL: ' DINO_URL
wget -O models_ckpt/dinov3_vitb16.pth "$DINO_URL"
unset DINO_URL
```

If the repository/weight/environment already exists, reuse it rather than recreate
it. The URL is the one required account-specific input; no public URL is invented
or embedded. A small access-denied HTML file will fail strict checkpoint loading.
The program records weight SHA256, so the arbitrary local filename is not its
identity. An explicitly requested DINOv2 control is supported by changing both
repo/weights and `--foundation-model dinov2_vitb14_reg`; no automatic fallback occurs.

## Prepare and check the teacher once

Run from the same repository root, still in `SHC_FCD_teacher`. Bash variables above
remain available in the same terminal after switching Conda environments:

```bash
python -u scripts/prepare_foundation_teacher.py \
  --nfd-config "$NFD32" \
  --foundation-repo ./external/dinov3 \
  --foundation-model dinov3_vitb16 \
  --foundation-weights ./models_ckpt/dinov3_vitb16.pth \
  --extraction-batch-size 32 --probe-steps 200 --gpu 0 \
  --output-dir ./save/fcd_teacher_seed60_v1

cat ./save/fcd_teacher_seed60_v1/teacher_report.json
```

Read `passed` and the two diagnostic comparisons. Failed preparation writes a
report/cache for inspection but exits with an explanation. Use a fresh output
directory for a new preparation; completed/partial directories are not overwritten.
If preparation OOMs, retry a **new directory** with smaller extraction batch size;
the original audited training batch size is not changed.

## Requested three code lengths

After the teacher check passes, switch back to the original SHC environment and
preflight all three explicit references. There is no ambiguous auto-discovery:

```bash
conda activate SHC

FCD_ARGS=(
  --nfd-config "$NFD16"
  --nfd-config "$NFD32"
  --nfd-config "$NFD64"
  --teacher-cache ./save/fcd_teacher_seed60_v1/foundation_cache.pt
  --bits 16 32 64 --seed 60 --gpu 0
  --mode neighbor --weight 0.05 --temperature 2 --gamma 1 --topk 5
  --warmup 40 --ramp-end 60
  --output-root ./save/i3_fcd_seed60_v1
  --evaluate-query
)

python scripts/run_i3_fcd.py "${FCD_ARGS[@]}" --dry-run
```

Preflight checks reference/cache paths and the information-check flag. Each real
training entry additionally verifies parent metadata and actual image/cache
content. If successful, in the same terminal:

```bash
nohup python -u scripts/run_i3_fcd.py "${FCD_ARGS[@]}" \
  > i3_fcd_seed60_v1.launch.log 2>&1 &
echo "Launcher PID: $!"
```

The launcher runs 16 -> 32 -> 64 sequentially, inheriting each reference's seed,
300-epoch budget, data order, optimizer and NFD parameters. Each model is selected
by validation mAP@ALL; `--evaluate-query` then computes all three query metrics
once from that same model. Omit `--evaluate-query` for development. Prefer a
32-bit validation-only pilot and the plain/CE controls before all final queries;
the full command above supplies the user's requested three final runs.

For controls, invoke the launcher with `--mode plain` or `--mode ce` and a new
output root. Pass only the desired reference and `--bits 32` for a single-code
pilot. Disabled control: `--weight 0`; it does not need a foundation cache. Do not
reuse completed or partial output roots. No checkpoint from B1/NFD is loaded as
the student's starting weights.

## Outputs, limits and verification

Teacher preparation: `foundation_cache.pt`, `foundation_classifier.pt`,
`teacher_report.json` and a split manifest. There is no final-query evaluation.

Batch training root: `launch_plan.json`, `16bit.log`/`32bit.log`/`64bit.log`, and
`summary.csv` updated after each completed run. Each per-code run saves its
`config.json`, original `best_model_state.pt` and retrieval outputs,
`fcd_selected_auxiliary_head.pt` for the selected epoch, NFD projection artifact,
metrics and per-epoch history. CE/TCKD/NCKD, effective neighbor weights and
first-batch student-gradient diagnostics are logged. Hash inference state never
contains the new teacher or auxiliary head.

```bash
tail -f i3_fcd_seed60_v1.launch.log
tail -f ./save/i3_fcd_seed60_v1/16bit.log
cat ./save/i3_fcd_seed60_v1/summary.csv
```

Full optimizer/RNG resume is not implemented for this NFD-based variant. A best
student checkpoint is not an exact resume checkpoint. `nohup` protects against
terminal hangup, not process/machine failure.

Local CPU verification passes 98 tests (PyTorch 2.14.0+cpu), including independent
DKD formula checks, extreme logits, teacher detachment, label/ID/image corruption,
exact disabled reduction, single forward/BN update, real audited parent dispatch,
synthetic full offline preparation and three-run launcher/summary. No official
DINO weights, real CIFAR data or CUDA training were executed locally. Real-model
compatibility, teacher superiority and retrieval improvement remain remote checks.

References: DINOv3 https://arxiv.org/abs/2508.10104 ; DKD (CVPR 2022)
https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html .
