# 创新三：对比增强哈希一致性正则化

## 问题背景

原始训练：每张图像只经过一组标准增强（Resize → RandomCrop → RandomFlip），通过 center_loss 将哈希码拉向预定义的哈希中心。

**核心问题**：训练没有明确约束「同一图像的不同视角应输出相同哈希码」。这意味着：
1. 模型对增强敏感——轻微的颜色变化可能翻转部分 bit
2. 判别特征学习缺乏 invariance（不变性）——泛化到测试集时鲁棒性不足
3. 不同类别之间的哈希码边界（margin）不够清晰

## 解决方案

引入 **NT-Xent（Normalized Temperature-scaled Cross Entropy）** 对比损失，即 SimCLR 中的对比一致性损失。

### 双视角数据增强（data_loader.py）

训练集中每张图像返回两个独立增强的视角：
- **视角 1（view1）**：标准增强（与原来相同）— `transform`
- **视角 2（view2）**：更强增强 — `transform_aug`

```python
# _build_augment_transform() — 强增强配置
transforms.Compose([
    transforms.Resize(resize_size),
    transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),  # 随机裁剪比例更大
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.2, hue=0.1),            # 颜色抖动
    transforms.RandomGrayscale(p=0.2),                          # 随机灰度化
    transforms.ToTensor(),
    normalize,
])
```

`CIFAR100HashDataset` 当 `dual_view=True` 时（由 `alpha_cont > 0` 自动激活）返回：
```python
return img1, img2, label, idx_tensor   # 4-tuple
```

测试集和数据库集始终返回标准 3-tuple，**推理无任何改变**。

### NT-Xent 对比损失（train.py）

```python
def _contrastive_loss(u1, u2, tau):
    u1_norm = F.normalize(u1.tanh(), dim=1)    # [B, bit]，L2 归一化
    u2_norm = F.normalize(u2.tanh(), dim=1)    # [B, bit]
    sim = u1_norm @ u2_norm.T / tau            # [B, B] 相似度矩阵

    # 正样本对：对角线 (i, i)；负样本：同行其余 B-1 个
    labels = torch.arange(B, device=device)
    loss = (cross_entropy(sim, labels) + cross_entropy(sim.T, labels)) / 2
    return loss
```

**含义**：
- 正样本对：同一图像的两个视角 `(u1[i], u2[i])` 的哈希码应尽量相似
- 负样本对：同 batch 中不同图像的哈希码应尽量不同
- 温度参数 `tau`（默认 0.07）：控制分布锐度，值越小则对难负例的惩罚越强

### 完整训练损失

```
Loss = CSQLoss(u1)                     （视角1对哈希中心的对齐损失）
     + alpha_cont · NT-Xent(u1, u2)   （视角间一致性损失）
```

视角 2 的哈希码 `u2` 只参与对比损失，不直接参与中心对齐，避免引入多余梯度。

### Batch 格式的向后兼容

训练循环自动检测 batch 的元素数量：
```python
if use_contrastive and len(batch_data) == 4:
    image, image2, label, ind = batch_data
    ...
else:
    image, label, ind = batch_data   # 原始行为，alpha_cont=0 时触发
```

设置 `--alpha-cont 0` 即可完全退回原始行为（不启用双视角，不计算对比损失）。

## 改动文件

| 文件 | 改动内容 |
|------|---------|
| `data/data_loader.py` | `CIFAR100HashDataset` 添加 `dual_view`、`transform_aug` 参数；`__getitem__` 根据 `dual_view` 返回 3-tuple 或 4-tuple；新增 `_build_augment_transform` 函数；`load_data` 根据 `args.alpha_cont` 自动激活双视角 |
| `train.py` | 新增 `_contrastive_loss` 函数；训练循环检测 batch 格式，双视角时额外计算对比损失 |
| `run.py` | 新增 `--alpha-cont`（默认 0.1）和 `--tau`（默认 0.07）参数 |

## 超参数选择建议

| 参数 | 默认值 | 调整方向 |
|------|-------|---------|
| `--alpha-cont` | 0.1 | 若训练不稳定，降至 0.05；多标签数据集（COCO）可升至 0.2 |
| `--tau` | 0.07 | 更小（0.05）→ 对难负例更严格；更大（0.1）→ 训练更平稳 |

**禁用**：`--alpha-cont 0`

## 预期效果与代价

| 指标 | 效果 |
|------|------|
| mAP@ALL | +1.5 ~ 3.0%（增强不变性提升泛化） |
| MSCOCO 多标签 | +2.0 ~ 4.0%（多标签语义更复杂，对比损失收益更大） |
| 训练时间 | 约 +20%（每 batch 额外一次前向传播 u2） |
| 推理时间 | **零开销**（测试集仍是单视角） |

## 与创新四的协同

当同时启用创新三和创新四时，`u_main = u1`（视角 1 的哈希码）被同时用于：
1. 计算 center_loss（创新二）
2. 计算对比损失（创新三）
3. 更新 AdaptiveHashCenter 的统计量（创新四）

三者共用同一次前向传播，不增加额外开销。
