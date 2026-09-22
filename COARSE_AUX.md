# 固定 B1 中心的粗类别辅助训练

本分支只实现建议文档的首选路线：固定创新点一的 S1、H1，在 Stage 3 的
ResNet34 池化特征上增加训练用辅助分类头。没有加入前景裁切或 LSH。
这是一项待验证的候选方案，尚无新增真实数据实验结果。

## 实现与默认行为

- `--aux-weight 0`（默认）：不创建辅助头，保留 B1 中心损失和量化损失。
- `--aux-target coarse`：由官方 CIFAR-100 原始 `train` 文件的细类/粗类标签
  验证 100→20 映射；每个细类只能属于一个粗类，每个粗类必须含五个细类。
  不从图片或最终查询结果推测层级，不修改数据划分和细类别相关性。
- `--aux-target fine`：100 类辅助头，用来区分一般辅助分类与层级信息的收益。
  两类辅助头共享同一接入点和优化设置，但输出维数不同，参数量不完全相等。
- `L = 原 B1 损失 + λ(epoch) * CE(辅助头(共享池化特征), 辅助标签)`。
- `--aux-warmup-epochs 10`：从第 1 轮的最大权重 / 10 线性增加至第 10 轮的最大权重；
  设为 0 则从第一轮直接使用最大权重。辅助头初始化不消耗主任务随机数流。
- 推理仍只输出原码长二值码，保留原汉明距离排序和并列处理；查询不使用真实标签选桶。
- 必须显式提供匹配的 S1、H1 缓存。校验种子、划分、码长、关系配置、S1 内容摘要及
  H1 的来源元数据；跨 Stage 3 代码提交允许复用，不重跑 Stage 1/2。
- 固定中心模式在 Stage 3 开始前重置种子和训练顺序，保证本分支的三组对照一致。
  `--paired-stage-seeds` 是否传入仍须与来源 S1 的生成协议相同，因为它影响缓存校验。
  历史未按此方式重置 Stage 3 的结果不能替代本轮 `λ=0` 对照。

## 远程运行前准备

1. 同步本分支代码，以及原 B1 的同种子、同码长缓存与配置。
2. S1 是 `SimilarityMatrix/*.pt` 中带 `value`/`metadata` 的 B1 缓存，
   H1 是 `HashCenters/*_SHC_*.pt`；不要用 `stage1_diagnostics.pt` 或无元数据裸张量代替。
3. 根据原 B1 `config.json` 设置 seed、码长、分类训练轮数、学习率、图像变换、
   mask/relation 参数、验证划分、batch size、worker 数、验证间隔和确定性设置。
   参数不匹配时会报错，不应删除校验来强行复用。每个种子使用自身 S1/H1。
4. 可先运行 `python -m unittest discover -s tests -v`。这是合成张量单元测试，
   不需要 CIFAR 数据、GPU 或下载预训练权重。

## 第一轮：三个独立的验证实验

以下 Linux Bash 示例假设来源 B1 使用默认参数和 `--paired-stage-seeds`。
将路径换成远程实际路径；如来源没有该选项，从公共参数中删除它。
示例 `0.05` 和 10 轮渐增只是起始配置，不是已验证最优值。

```bash
S1=/absolute/path/to/b1/SimilarityMatrix/your_s1.pt
H1=/absolute/path/to/b1/HashCenters/your_h1_SHC.pt
COMMON=(--variant b1 --stage train --seed 60 --code-length 32
  --root /absolute/path/to/cifar-100-python
  --similarity-cache "$S1" --center-cache "$H1"
  --paired-stage-seeds --classify-epoch 300 --epoch 300
  --lr 7e-5 --batch-size 64 --num-workers 6 --test-map 5
  --aux-warmup-epochs 10 --validation-only)

# 固定中心 B1 对照
python run.py "${COMMON[@]}" --aux-weight 0 \
  --output-dir ./save_coarse_aux/seed60/b1_control

# 粗类别辅助方案
python run.py "${COMMON[@]}" --aux-target coarse --aux-weight 0.05 \
  --output-dir ./save_coarse_aux/seed60/coarse_w005

# 一般辅助分类对照
python run.py "${COMMON[@]}" --aux-target fine --aux-weight 0.05 \
  --output-dir ./save_coarse_aux/seed60/fine_w005
```

正权重要求 `--variant b1 --stage train --similarity-cache ... --center-cache ...`。
没有缓存时直接报错，不会悄悄生成新中心。每组实验使用独立输出目录；
已存在的同配置运行目录会被拒绝覆盖。此版本没有断点续训功能。

`--validation-only` 不执行最终查询评估。主加载器仍按现有协议读取 CIFAR 原始文件并
构造四个数据加载器，但最终查询加载器不参与训练、选轮次或参数搜索。

## 如何读结果

产物位于各输出目录的 `runs/<运行标识>/`：

- `config.json`：配置、源代码摘要、固定 S1/H1 文件摘要、中心内容摘要和标签映射。
- `training_history.json`：每轮损失，辅助权重/交叉熵/训练准确率，以及验证轮次的三项 mAP。
- `metrics.json`：按验证 `mAP@ALL` 选择的单一轮次及该轮全部验证指标、模型摘要。
- `best_model_state.pt`：只有原 ResNet 哈希网络，直接兼容 `ResNet(code_length).load_state_dict(...)`。
- `best_auxiliary_head.pt`：同一选中轮次的辅助头，单独存档，不用于检索。

`hierarchy_errors.top100/top1000` 记录“同粗类不同细类”与“跨粗类”误检数量及比例。
比例分母是所有查询的实际返回条目总数，不是误检总数。诊断标签只在排名完成后
统计，不改变候选集合或排序。指标仍按 100 个细类别计算，所有指标来自同一个选中模型。

首先比较粗辅助与 `λ=0` 的三项验证 mAP，再查看同粗类误检是否增加；
结合细分类辅助对照判断收益是否只是一般多任务学习效果。
不要以辅助分类准确率上升代替检索收益，也不要根据最终查询集调权重。
诊断需要额外 CPU 排序时间，但不增加查询编码分支。

## 确定方案后的最终查询评估

冻结方案后，用已经由验证集选中的模型直接评估，不必重新训练：

```bash
python evaluate_saved.py \
  --run-dir ./save_coarse_aux/seed60/coarse_w005/runs/实际运行目录 \
  --root /absolute/path/to/cifar-100-python --gpu 0
```

该命令检查源代码、模型 SHA256 和数据划分，输出到原运行目录下的 `final_query/`，
包含三项查询 mAP、误检诊断、二值码和逐查询 AP。不覆盖原验证记录或已存在的最终结果。
保留训练时的代码版本；已经在训练末尾完成查询评估的运行不应再次用此命令评估。

本地只执行代码与合成数据单元测试，真实训练和最终查询评估由远程机器执行。
