# B1 关系估计阶段的裁切辅助视图实验

本分支只实现现有裁切图作为辅助视图的预实验，不执行前景分割，不将裁切图替换进分类器训练或哈希训练。默认不传辅助参数时保持原 B1 数学流程。

## 实现规则

- 原图 root 始终指向原始 CIFAR-100。辅助 PNG 用单独目录和清单提供。
- 清单中的样本必须属于当前 seed 和 val-per-class 对应的训练索引，标签、原图像素摘要、辅助像素摘要、RGB/32×32 格式均须匹配。重复、未审核、缺文件、路径越界等情况直接报错。
- q 为审核通过的二值标志；未列入清单的图片 q=0。已列入但文件缺失是错误，不回退。
- 原图与翻转图计算权重、有效证据量、视觉原型及融合 gate。仅概率分支用 `(1-alpha)*a_B1 + alpha*a_aux`，默认 alpha=0.2。辅助图单视图推理，使用原有 mask-strategy 和 temperature。
- 辅助图使用与原图关系估计相同的 Resize/CenterCrop/归一化。裁切后可能再次中心裁切，因此必须检查预览。
- alpha=0 跳过混合运算，恢复原 B1 关系数值；仍读取和审计全部已声明的辅助文件，并输出辅助推理诊断。
- 当前只允许 `--variant b1 --relation-mode full` 配合辅助图，以免消融模式含义混淆。
- 数据加载不再回退到其他默认数据目录；显式 root 错误就报错。
- 数据像素与标签摘要进入缓存标识；辅助内容摘要、alpha、预处理配置进入关系和中心缓存标识，分类器缓存不依赖辅助图。

## 远程准备

同步本分支的修改，包括新增的 `utils/foreground_aux.py`、测试、打包脚本与本说明。所有命令在项目根目录运行，Linux Bash 示例中的路径按实际情况设置。不要将本地临时测试环境上传。

本地已生成 `output/foreground_aux_seed60.zip`，可复制到远程并解压到项目根目录，得到 `foreground_aux_seed60/`。已核对 56 张图的原图摘要、辅助摘要、标签及 seed60 训练归属；没有运行模型推理或训练。辅助内容摘要为 `cbcf17fea4a00d490963a5dc0be911d8fa4de6bb022690d6af95ea7af51b7069`。

可直接使用完整裁切包的 `metadata/applied_train_crops.csv` 和 `images/`，也可只打包现有 56 张训练辅助图：

```bash
python tools/package_foreground_aux.py \
  --crop-package /path/to/cifar-100-cut \
  --output ./foreground_aux_seed60
```

把生成的目录复制到远程项目目录。它只有辅助图和审核清单，不是 CIFAR 原始训练目录。这个 seed60 清单不可用于其他种子：换种子需要重新生成其训练划分对应的审核清单。

```bash
ORIGINAL_ROOT=./data/cifar-100-python
AUX_MANIFEST=./foreground_aux_seed60/approved_train.csv
AUX_ROOT=./foreground_aux_seed60/images
export CUBLAS_WORKSPACE_CONFIG=:4096:8

COMMON=(--variant b1 --paired-stage-seeds \
  --dataset cifar-100-new-seg --root "$ORIGINAL_ROOT" \
  --num-classes 100 --code-length 32 --seed 60 --val-per-class 10 \
  --classify-epoch 300 --epoch 300 --test-map 5 --lr 7e-5 \
  --batch-size 64 --num-workers 4 --gpu 0 --topK -1 100 1000)
AUX=(--aux-manifest "$AUX_MANIFEST" --aux-root "$AUX_ROOT" --aux-alpha 0.2)
```

完整分类器缓存恢复时直接构建空模型再加载权重，不下载 ImageNet 初始权重。正常分类器训练仍沿用原有预训练初始化。

## 分类器缓存的限制

`--classifier-cache` 可以跨输出目录复用分类器；它检查数据摘要、划分、训练配置及元数据摘要，记录文件与模型状态摘要。不根据文件名猜测缓存是否来自原图。

**旧版本缓存没有 data_sha256，不能完成原图内容来源核验，因此会明确拒绝。不要手工补写摘要冒充已验证来源。** 如果远程只有旧缓存，需要先用新代码在原图上单独准备一次分类器，或继续保留旧结果作为历史参考。本地没有执行这一步。

下面的命令会训练分类器，仅在确实缺少新格式缓存时由你执行；它不训练哈希网络：

```bash
CLASSIFIER_DIR=$(mktemp -d ./save_i1_original_classifier_XXXXXX)
python -u run.py "${COMMON[@]}" --stage classifier --output-dir "$CLASSIFIER_DIR"
CLASSIFIER_CACHE=$(find "$CLASSIFIER_DIR/ClassificationNet" -maxdepth 1 -name '*.pt' -type f)
test -f "$CLASSIFIER_CACHE" || exit 1
```

如果已有匹配的新格式缓存，直接设置 `CLASSIFIER_CACHE=/path/to/audited_classifier.pt`，无需重训。

## 先做诊断，不训练

```bash
DIAG_DIR=$(mktemp -d ./save_i1_aux_diag_seed60_XXXXXX)
python -u run.py "${COMMON[@]}" "${AUX[@]}" \
  --classifier-cache "$CLASSIFIER_CACHE" \
  --relation-diagnostics-only --output-dir "$DIAG_DIR"
```

`--relation-diagnostics-only` 在缺少缓存时直接失败，绝不启动分类器训练，也不生成中心或训练哈希网络。不要用普通 `--stage similarity` 代替：后者在缓存缺失时允许训练分类器。

检查 `runs/<本次运行>/` 下：

- `aux_previews/`：每个辅助样本一张四列图，依次为原 PNG、辅助 PNG、原图最终模型输入、辅助图最终模型输入。
- `auxiliary_diagnostics.json`：总数量、逐类数量及覆盖率、逐样本预测/真实类置信度/原权重、概率矩阵及最终融合矩阵变化。
- `S_prob_original.csv`、`S_prob.csv`、`S1_original.csv`、`S1.csv`：同一个冻结分类器下加入辅助图前后的关系矩阵。
- `stage1_diagnostics.pt`：含上述矩阵、视觉原型、gate、有效证据量和配置的完整诊断。
- `config.json`：解析后的原图路径、数据/辅助/分类器摘要等。

`S1_original` 是当前分类器上的无辅助 B1 关系矩阵，不意味着已复现历史训练的 mAP。诊断本身不计算检索 mAP。

## 确认后新增一组完整实验

```bash
RUN_DIR=$(mktemp -d ./save_i1_aux_seed60_XXXXXX)
LOG_DIR="logs/$(basename "$RUN_DIR")"
mkdir -p "$LOG_DIR"
nohup python -u run.py "${COMMON[@]}" "${AUX[@]}" \
  --classifier-cache "$CLASSIFIER_CACHE" --output-dir "$RUN_DIR" \
  > "$LOG_DIR/b1_aux_32bit_seed60.log" 2>&1 &
echo $! > "$LOG_DIR/launcher.pid"
printf '结果目录：%s\n日志目录：%s\n' "$RUN_DIR" "$LOG_DIR"
```

新输出目录会重新计算辅助关系矩阵，但复用指定分类器。维持 paired-stage-seeds；验证 mAP@ALL 用于选择 epoch，query 仅用于最终报告，不据其反复调整 alpha。

如果重新训练了分类器，历史原图 B1 不能自动视为完全配对对照。需要严格配对时，用同一 `CLASSIFIER_CACHE`、同一 COMMON 参数、不传 AUX，再在独立输出目录跑原图 B1。是否追加这组由你的缓存来源和复现需求决定。

## 代码验证

```bash
python -m unittest tests.test_b0_core tests.test_i1_relation tests.test_foreground_aux -v
```

这些测试使用小型 CPU 张量与临时图片，不运行数据集训练，不下载预训练模型。
