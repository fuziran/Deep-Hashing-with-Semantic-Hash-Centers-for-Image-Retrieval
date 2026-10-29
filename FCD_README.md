# I3-FCD: foundation-model discriminative distillation on B1+NFD

Branch: `update-reliable-relation-nfd-distillation-i3-foundation-discriminative-distillation`.
Entry: `run.py --variant b3fcd`. This is an experimental candidate; no retrieval
gain is claimed. The previous NPA implementation remains available for historical
reproduction and is not activated by FCD.

For the current 32-bit CE/plain/neighbor screen and independent query evaluation,
use the complete workflow at the end of this file. The historical three-code
command below is an optional later experiment, not the initial screen.

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

When the fixed hash probe fails despite better classification, run
`scripts/diagnose_fcd_logits.py` before considering another hash training job.
It reuses cached DINOv3 training logits and the saved classifier, infers only
validation DINOv3 features plus the original teacher's train/validation logits,
then compares hard class-center codes and soft T=2 expected-center codes under
the same validation-vs-training-only retrieval protocol. The report includes
paired bootstrap intervals over validation queries. This is exploratory evidence
about the actual distillation logits; it does not replace the failed fixed-probe
gate or evaluate the final query/database.

For a failed original gate, the normal FCD launcher still refuses the cache.
An explicit `--exploratory-diagnostic /path/to/diagnostic.json` enables a
validation-only pilot if that report belongs to the same failed cache and NFD
reference, was generated by the current diagnostic source, and shows positive
paired-bootstrap lower bounds for all three T=2 direct-center metrics. The
training manifest records the failed original gate, exploratory status, report
path and SHA256. The pilot still starts a fresh paired student and preserves the
original loss and 300-epoch budget; it does not mark the teacher check as passed.
Use `plain` first to test whether the better teacher logits actually improve a
student. Run `ce` and `neighbor` controls only if the plain pilot merits further
work. Because this route was chosen after seeing validation diagnostics, report
its results as exploratory; defer final query evaluation and additional bit
lengths until after the method and comparisons are frozen.

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
python -m pip install 'numpy<2' -r external/dinov3/requirements.txt --index-url https://pypi.org/simple --timeout 60 --retries 3
read -r -p 'Paste the official DINOv3 ViT-B/16 LVD-1689M download URL: ' DINO_URL
wget -O models_ckpt/dinov3_vitb16.pth "$DINO_URL"
unset DINO_URL
```

If the repository/weight/environment already exists, reuse it rather than recreate
it. The URL is the one required account-specific input; no public URL is invented
or embedded. A small access-denied HTML file will fail strict checkpoint loading.
The local Hub entrypoint imports evaluation modules, so install the official
requirements.txt even when using only the ViT-B/16 backbone.
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

The original FCD CPU verification passed 98 tests (PyTorch 2.14.0+cpu), including independent
DKD formula checks, extreme logits, teacher detachment, label/ID/image corruption,
exact disabled reduction, single forward/BN update, real audited parent dispatch,
synthetic full offline preparation and three-run launcher/summary. No official
DINO weights, real CIFAR data or CUDA training were executed locally. Real-model
compatibility, teacher superiority and retrieval improvement remain remote checks.

References: DINOv3 https://arxiv.org/abs/2508.10104 ; DKD (CVPR 2022)
https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html .

## 32bit 三组筛选与独立 query 评估

本次新增 `scripts/evaluate_query.py` 与 `scripts/run_i3_fcd_screen.sh`。
筛选脚本按测试、教师准备及信息检查、三组参数检查、ce/plain/neighbor顺序训练执行。
任何步骤失败立即停止；三组均不传 `--evaluate-query`，不自动运行最终查询评价。
三个学生各自继承原 NFD 的300轮预算、初始化、数据顺序和 NFD 参数。
本地修改必须先同步到服务器，再开始新训练。

### 同步本地修改

在本地 PowerShell 执行。只暂存本次代码与文档，避免加入已有数据、输出与研究报告：

```powershell
Set-Location D:\SHC\SHC-Baseline-run
git add -- .gitattributes network.py train.py data/data_loader.py utils/experiment.py scripts/evaluate_query.py scripts/run_i3_fcd_screen.sh tests/test_query_evaluation.py tests/test_foundation_distill.py FCD_README.md
git diff --cached --stat
git commit -m "Add audited standalone query evaluation and FCD screen workflow"
git push -u origin update-reliable-relation-nfd-distillation-i3-foundation-discriminative-distillation
```

以下命令在远程 Linux/WSL Bash 中依次执行：

```bash
cd /mnt/d/fzr/SHC
git fetch origin
git switch update-reliable-relation-nfd-distillation-i3-foundation-discriminative-distillation
git pull --ff-only
git log -1 --oneline

NFD32='./save_nfd_seed60_bINBuclZ/nfd/runs/cifar-100-new-seg_B2NFD_32bit_seed60_639b7ea1eaa3/config.json'
test -f "$NFD32"
```

### 教师环境与官方权重

原 SHC 环境不升级。若已存在 `SHC_FCD_teacher`，跳过创建；否则：

```bash
conda create -n SHC_FCD_teacher python=3.11 -y
conda run -n SHC_FCD_teacher python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
conda run -n SHC_FCD_teacher python -m pip install 'numpy<2' pillow scipy matplotlib
```

官方 DINOv3 的 Hub 入口还会导入评估模块，实际需要仓库 requirements.txt 中的依赖（包括 torchmetrics）；只按 Hub 声明的 torch、numpy 安装会在加载时失败。
模型须为 **DINOv3 ViT-B/16 LVD-1689M**，下载权限与对应URL由官方流程提供：
https://github.com/facebookresearch/dinov3 。不提供猜测的权重下载地址。

```bash
mkdir -p external models_ckpt logs
if [ ! -d external/dinov3 ]; then
  git clone https://github.com/facebookresearch/dinov3.git external/dinov3
fi
conda run -n SHC_FCD_teacher python -m pip install 'numpy<2' -r external/dinov3/requirements.txt --index-url https://pypi.org/simple --timeout 60 --retries 3
if [ ! -s models_ckpt/dinov3_vitb16.pth ]; then
  read -r -s -p '粘贴官方 ViT-B/16 LVD-1689M 权重下载URL: ' DINO_URL
  printf '\n'
  wget -q -O models_ckpt/dinov3_vitb16.pth "$DINO_URL"
  unset DINO_URL
fi
test -s models_ckpt/dinov3_vitb16.pth
conda run -n SHC_FCD_teacher python -c 'import torch,torchvision; print(torch.__version__,torchvision.__version__); assert torch.cuda.is_available(), "Teacher CUDA unavailable"'
conda run -n SHC python -c 'import torch,torchvision; print(torch.__version__,torchvision.__version__); assert torch.cuda.is_available(), "SHC CUDA unavailable"'
```

### 后台执行完整32bit流程

本段不需要手动切换 Conda 环境；脚本使用 `conda run` 调用两个现有环境。
若原训练环境不叫 SHC，将 `SHC_TRAIN_ENV=SHC` 改成实际环境名。
目录记录在日志目录的 `.output_path` 文件中，换终端后可以读取。

```bash
SCREEN_STAMP=$(date +%Y%m%d_%H%M%S)
SCREEN_ROOT="$(pwd)/save/i3_fcd_screen32_seed60_${SCREEN_STAMP}"
SCREEN_LOG="$(pwd)/logs/i3_fcd_screen32_seed60_${SCREEN_STAMP}.log"
printf '%s\n' "$SCREEN_ROOT" > "${SCREEN_LOG}.output_path"

nohup env SHC_TRAIN_ENV=SHC SHC_TEACHER_ENV=SHC_FCD_teacher SHC_GPU=0 \
  bash scripts/run_i3_fcd_screen.sh \
  "$NFD32" ./external/dinov3 ./models_ckpt/dinov3_vitb16.pth "$SCREEN_ROOT" \
  > "$SCREEN_LOG" 2>&1 &
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$!" "$SCREEN_LOG" "$SCREEN_ROOT"
```

脚本包含以下实际教师准备参数：extraction batch=32，probe steps=200，GPU=0。
教师检查要求 `passed=true`，缓存位于 `SCREEN_ROOT/teacher/foundation_cache.pt`。
若提取显存不足，在新的输出根目录重试，并设置 `SHC_EXTRACTION_BATCH=16`；不修改学生训练batch。
FCD三组固定 weight=0.05、temperature=2、gamma=1、topk=5、warmup=40、ramp-end=60。
教师只做一次准备，随后三组顺序复用同一缓存。

```bash
tail -f "$SCREEN_LOG"
# 教师结束后：
cat "$SCREEN_ROOT/teacher/teacher_report.json"
# 进入相应训练后：
tail -f "$SCREEN_ROOT/ce/32bit.log"
tail -f "$SCREEN_ROOT/plain/32bit.log"
tail -f "$SCREEN_ROOT/neighbor/32bit.log"
# 全部完成后：
cat "$SCREEN_ROOT/validation_comparison.csv"
```

`validation_comparison.csv` 包含原 NFD 与三组新方法的选中轮次、验证ALL、@100、@1000。
每组的三项指标均取自按验证ALL选中的同一轮，不能逐项挑最高值。
若教师检查失败，查看 teacher.log/report；若测试失败，查看 tests.log；训练失败查看相应32bit.log。
输出根目录不复用，且当前仍不支持完整训练断点恢复。

### 单独评价已完成训练的最佳模型

以下是后续显式评价入口，**不属于上面的筛选脚本**。只在决定评价冻结模型时执行。
脚本不接受其他epoch、任意checkpoint或跳过摘要检查的选项。
支持本次更新后完成、带校验记录的 paired NFD/FCD；历史运行没有这些记录时，保留原评价流程，不能补写伪造的训练校验记录。

新训练的 metrics.json 保存最佳模型、训练配置、完整训练历史的 SHA256。
数据加载阶段记录完整像素与标签摘要。评价会严格核对这些记录、训练源码、按验证ALL首次最大值选中的轮次、数据内容和划分摘要；模型通过 strict=True 完整加载，构造时不下载预训练权重。
不需要 DINO权重、教师缓存、中心缓存或辅助分类头，不运行优化器，也不遍历训练/验证loader。
默认结果目录为该run下新建的 `final_query/`；已存在时拒绝覆盖。
原训练 config/metrics/history/model 保持不变。

```bash
# SCREEN_ROOT 指向上面那次已完成的筛选输出。
# 先检查三组；dry-run检查模型和数据，不进行query前向：
for mode in ce plain neighbor; do
  RUN_DIR=$(conda run -n SHC python -c \
    'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); assert len(r)==1; print(r[0]["run_dir"])' \
    "$SCREEN_ROOT/$mode/summary.csv")
  conda run --no-capture-output -n SHC python scripts/evaluate_query.py \
    --run-dir "$RUN_DIR" --gpu 0 --dry-run || break
done

# 明确决定评价三组冻结模型后再执行；不重训：
for mode in ce plain neighbor; do
  RUN_DIR=$(conda run -n SHC python -c \
    'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); assert len(r)==1; print(r[0]["run_dir"])' \
    "$SCREEN_ROOT/$mode/summary.csv")
  conda run --no-capture-output -n SHC python -u scripts/evaluate_query.py \
    --run-dir "$RUN_DIR" --gpu 0 || break
done
```

如数据仅迁移位置，可增加 `--data-root /实际/cifar-100-python`；内容必须完全一致。
`final_query/metrics.json` 记录三个query指标及来源校验，`retrieval_outputs.pt` 保存query/database二值码、标签、显式样本ID、每query AP及PR数据，并保存本次划分副本。
最终成功以新评价目录中完整的 metrics.json 为准，原训练 metrics.json 的 query_evaluated=false 不回写。
新流程本地合成验证通过不等同真实DINO权重、CUDA或CIFAR-100指标验证。
本次新增8项独立评价及数据摘要测试，以及独立DINOv3环境版本门槛测试；合成教师准备测试隔离了真实DINOv3运行时要求，避免要求原SHC环境升级。
107项完整CPU回归在PyTorch 2.2.2 / torchvision 0.17.2通过；PyTorch 2.14.0 CPU也完成完整回归及最终FCD测试复核。启动脚本通过Bash语法检查，以及正常顺序、教师失败停止、plain训练失败停止三种模拟流程检查。

## Seed-60 32-bit exploratory follow-up: isolated teacher-logit strength

The validation-only screen selected CE at 0.573391 mAP@ALL, plain DKD at
0.572528, neighbor DKD at 0.570476, and the paired B1+NFD reference at
0.567751. CE uses hard labels but no teacher logits. Over epochs 60–300,
the median cosine between the weighted FCD gradient and the current main-loss
gradient was +0.9610 for CE, -0.7678 for plain, and -0.7358 for neighbor.
This motivates a single controlled ablation before expanding code lengths.

`--kd-scale` in `scripts/run_i3_fcd.py` (forwarded as `--fcd-kd-scale`) changes
the auxiliary loss from `CE + TCKD + NCKD` to
`CE + kd_scale * (TCKD + NCKD)`. The existing outer `--weight 0.05`,
warmup, and NFD settings stay fixed. The default `kd_scale=1` retains the
previous formula. The new scale is recorded in the run manifest, and the raw
teacher terms plus their scaled contribution are recorded in training history.
It must be finite and in [0,1]. A pilot at `kd_scale=0.05` tests whether a
smaller teacher-logit gradient adds value beyond the completed CE control;
it is not itself evidence of a new method. Keep query evaluation disabled for
this exploratory, failed-original-gate teacher cache.

The completed 300-epoch `kd_scale=0.05` plain pilot selected epoch 300 and
reached validation mAP@ALL 0.568570, mAP@100 0.671359, and mAP@1000 0.647856.
All three are below the completed CE control (0.573391, 0.675099, 0.653127).
The result also falls below the original plain DKD mAP@ALL 0.572528. Shrinking
the DKD gradient alone did not recover the CE control; the observed negative
gradient cosine is therefore a diagnostic association, not a demonstrated
cause. Stop this FCD configuration's code-length and weight search.

`scripts/diagnose_i3_validation.py` compares the frozen B1+NFD and CE models
on the original validation vs database protocol. It checks the saved selected
epoch, full history, available selection receipts, split and pixel hash,
strictly loads student checkpoints, and checks recomputed mAP against the
recorded validation metrics. It never iterates final query images. Outputs
include per-query and per-class AP, top-100 false-positive counts, boundary
Hamming tie counts, and descriptive paired bootstrap intervals. Historical
NFD runs may lack a selection receipt; the report records that limitation.
Validation queries already selected both models, so bootstrap intervals are
descriptive and are not independent confirmation.
