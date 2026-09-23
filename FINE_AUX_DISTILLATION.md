# B1＋细分类辅助＋普通蒸馏 / NFD

分支：`update-reliable-relation-fine-aux-distillation`，基于 `update-reliable-relation-nfd-distillation`。

## 方法与开关

统一入口 `--variant b2nfd`，沿用同种子 B1 的配置、冻结关系 S1、中心 H1 和分类器教师。
学生从配对初始化训练，不从 B1 最佳模型续训。

`L = L_center + lambd * L_quant + fine_aux_weight(epoch) * CE + nfd_weight(epoch) * L_distill`

细分类辅助为共享池化特征上的线性分类头，使用训练样本的原始单类别 one-hot 标签（CIFAR-100 为 100 类），不使用粗类别映射。
辅助 CE 梯度进入学生主干和辅助头，不直接进入哈希头；教师保持 eval/no_grad。
两项辅助损失共享同一次学生前向与 BN 更新。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--fine-aux-weight` | 0 | 辅助 CE 最大权重；0 不创建辅助头，恢复原蒸馏行为 |
| `--fine-aux-warmup` | 20 | 前 20 轮关闭细分类辅助 |
| `--fine-aux-ramp-end` | 40 | 第 40 轮起达到完整权重，期间线性递增 |
| `--nfd-alpha` | 0.75 | 0 为普通蒸馏；0.75 为 NFD |

细分类辅助仅支持 b2nfd 入口。设置 `--nfd-weight 0` 可做仅细分类辅助对照，仍要求完整冻结产物以保持同一审计流程。
辅助头初始化使用隔离随机数状态，避免改变学生初始化、采样和增强随机流。

## 运行示例（Linux Bash，未启动真实训练）

先设置四个变量，指向同一种子 B1 的真实产物：`PARENT_CONFIG`、`SIMILARITY_CACHE`、`CENTER_CACHE`、`CLASSIFIER_CACHE`。
下面共用参数适用于已有 seed=60、CIFAR-100、32bit、300 轮、workers=4 的配置；若父实验不同，按其 config.json 修改全部共用参数，缓存检查会拒绝不一致配置。
辅助权重 0.1 是可修改的起始实验值，尚未经性能验证。

```bash
common=(
  --variant b2nfd
  --parent-b1-config "$PARENT_CONFIG"
  --similarity-cache "$SIMILARITY_CACHE"
  --center-cache "$CENTER_CACHE"
  --teacher-cache "$CLASSIFIER_CACHE"
  --dataset cifar-100-new-seg --root ./data/cifar-100-python
  --num-classes 100 --code-length 32 --seed 60
  --classify-epoch 300 --epoch 300 --lr 7e-5 --batch-size 64
  --num-workers 4 --test-map 5 --topK -1 100 1000 --gpu 0
  --relation-mode full --relation-kappa 30 --relation-temperature 1
  --relation-weight-floor 0.05 --paired-stage-seeds --skip-query
  --fine-aux-weight 0.1 --fine-aux-warmup 20 --fine-aux-ramp-end 40
  --nfd-weight 0.01 --nfd-topk 5 --nfd-warmup 20 --nfd-ramp-end 40
)

# B1＋细分类辅助＋普通蒸馏
python -u run.py "${common[@]}" --nfd-alpha 0 --output-dir ./save_fine_aux_fd

# B1＋细分类辅助＋NFD
python -u run.py "${common[@]}" --nfd-alpha 0.75 --output-dir ./save_fine_aux_nfd
```

两组保持其余配置相同，使用各自的新输出目录。默认按 validation mAP@ALL 选择最佳轮次。

## 输出与兼容性

- `config.json`：细分类辅助参数、版本、策略及原蒸馏来源信息；启用的辅助配置参与 run_id。
- `training_history.json`：新增 `loss_fine_aux`、`fine_aux_weight`、`fine_aux_accuracy`。未启用轮次记 0，准确率此时不是测量值。总 loss 包含所有加权项。
- NFD 梯度诊断中的 clean 项在启用细分类辅助后表示 B1＋加权细分类 CE，与加权蒸馏梯度比较。
- `best_model_state.pt`：仅学生原有模型参数，可以严格加载到原 ResNet 进行检索。
- `best_fine_aux_state.pt`：对应同一最佳轮次的辅助头参数与 epoch，单独保存，不参与检索。

沿用 NFD 的限制：不支持完整优化器／随机状态断点续训；最佳模型文件不是完整训练检查点。

## 验证

本地 CPU 完整回归 77 项通过（PyTorch 2.14.0+cpu），包含两种 alpha 模式的小规模训练、辅助头更新、主干梯度、教师冻结、单次 BN 更新、零权重逐值等价、严格模型加载、不同辅助权重的运行身份和冻结产物复用。
没有启动真实数据训练、GPU 验证或 query 评估，因此尚不能判断检索指标收益。

```bash
python -m unittest tests.test_fine_aux_distillation tests.test_b2nfd_feature_distill -v
python -m unittest discover -s tests -v
```
