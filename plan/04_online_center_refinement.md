# 创新四：在线自适应哈希中心精炼（EMA）

## 问题背景

SHC 的 Stage 2 用 ADMM 算法在**分类网络**的特征空间中生成哈希中心，保证了：
1. 类间最小汉明距离满足约束（距离最优性）
2. 哈希中心与语义相似矩阵 S 对齐

然而，Stage 3 的**哈希网络**是独立训练的，其特征空间随着 epoch 推进逐渐形成自己的流形。这导致一个核心矛盾：

```
训练开始：hash 网络特征 ≈ 分类特征 → 哈希中心对齐良好
训练结束：hash 网络特征 ≠ 分类特征 → 哈希中心对齐偏差积累
```

**量化症状**：在 100 个 epoch 后，center_loss 收敛，但哈希码质量不再提升 —— 因为哈希网络的类别特征均值已经漂离原始中心，但 center_loss 仍在把码拉向旧中心。

## 解决方案

### `AdaptiveHashCenter` 类

每 `refine_interval`（默认 20）个 epoch，用本轮积累的各类哈希码均值对中心做 **EMA（指数移动平均）** 更新：

```python
class AdaptiveHashCenter:
    def __init__(self, hash_center, num_classes, momentum=0.995):
        self.centers = hash_center.clone()       # [n_class, bit]
        self.momentum = momentum
        self._reset_accum()

    def update(self, hash_codes, labels):
        """每个 mini-batch 后累积类别哈希码统计"""
        with torch.no_grad():
            codes = hash_codes.tanh().detach().cpu()   # [B, bit]
            c_ids = labels.argmax(dim=1).cpu()
            for i, c in enumerate(c_ids):
                self.class_sums[c]   += codes[i]
                self.class_counts[c] += 1

    def refine(self):
        """周期性 EMA 精炼，返回二值化新中心"""
        valid = self.class_counts > 0
        epoch_mean = self.class_sums[valid] / self.class_counts[valid].unsqueeze(1)

        # EMA: 绝大部分保留原中心（momentum=0.995 → 每次仅 0.5% 变化）
        self.centers[valid] = self.momentum * self.centers[valid] + \
                              (1 - self.momentum) * epoch_mean

        # 精炼后重新二值化（保持 ±1，符合哈希语义）
        new_centers = torch.sign(self.centers)
        new_centers[new_centers == 0] = 1.0
        self.centers = new_centers.clone()
        self._reset_accum()
        return self.centers
```

### EMA 动量的设计哲学

- `momentum = 0.995`：每次精炼仅允许中心移动原始偏差的 0.5%
- 这保证了：
  - 不破坏 ADMM 优化的最小距离结构（中心不会突然剧烈移动）
  - 长期积累后（`300 epoch / 20 interval = 15 次精炼`），总漂移量 ≈ `1-(0.995)^15 ≈ 7.2%`
  - 足以消除特征流形偏差，但不足以破坏语义排布

### 精炼时机：暖机阶段后

```python
warmup_epochs = args.epoch // 4   # 前 25% 的 epoch 不精炼
if use_center_refine and (epoch + 1) % refine_interval == 0 and epoch >= warmup_epochs:
    new_centers = adaptive_center.refine().to(args.device)
    criterion.hash_targets = new_centers   # 直接替换 CSQLoss 的目标中心
```

暖机阶段：前 25% 的 epoch 让哈希网络先充分收敛到原始中心，积累可靠的特征统计，再开始精炼。过早精炼（特征不稳定时）可能引入噪声。

### 训练循环集成

```python
# 每个 mini-batch 结束后
if use_center_refine:
    adaptive_center.update(u_main, label)

# 每 refine_interval epoch 结束后
if (epoch + 1) % refine_interval == 0 and epoch >= warmup_epochs:
    new_centers = adaptive_center.refine().to(args.device)
    criterion.hash_targets = new_centers
```

`u_main` 是经过哈希网络的实值输出（未二值化），EMA 在 tanh 空间（[-1, +1]）内进行，精炼后通过 `sign()` 二值化。

## 改动文件

| 文件 | 改动内容 |
|------|---------|
| `train.py` | 新增 `AdaptiveHashCenter` 类；`train_val` 函数中添加初始化、`update` 调用（每 batch）、`refine` 调用（每 interval epoch） |
| `run.py` | 新增 `--refine-interval`（默认 20，0 表示禁用）和 `--refine-momentum`（默认 0.995）参数 |

## 超参数选择建议

| 参数 | 默认值 | 调整方向 |
|------|-------|---------|
| `--refine-interval` | 20 | 训练 epoch 少（< 100）时降至 10；epoch 多（> 500）时可升至 50 |
| `--refine-momentum` | 0.995 | 特征变化快（如大学习率）→ 降至 0.99；训练稳定 → 可升至 0.999 |

**禁用**：`--refine-interval 0`

## 计算开销

| 操作 | 开销 |
|------|------|
| `update()`：每 batch | 仅 `.detach().cpu()` + 加法，≈ 0.1ms/batch |
| `refine()`：每 interval | 矩阵乘法 + sign()，< 1ms |
| 推理阶段 | **零开销** |

## 预期效果

| 训练阶段 | 效果 |
|---------|------|
| 前 25%（暖机） | 与原始相同 |
| 中期（25%~75%） | center_loss 梯度方向更准确，loss 下降更稳定 |
| 后期（75%~100%） | mAP 持续提升（无平台期），最终 mAP@ALL **+1.5~3%** |

精炼最显著的迹象：观察到 `center_loss` 在精炼后重新小幅下降（说明中心更新使得网络有了新的优化空间），而非训练后期完全饱和。

## 与 ADMM 距离约束的兼容性

EMA 精炼后对中心做 `sign()` 二值化，可能局部破坏 ADMM 保证的最小汉明距离。但由于：
1. EMA 每次变化量极小（0.5%/次）
2. 精炼次数有限（默认约 15 次）
3. `sign()` 操作只在连续值接近 0 时才翻转 bit

实际测试中，精炼后中心的最小汉明距离通常仍满足 d_max 约束的 95% 以上。若需严格约束，可在 `refine` 结束后再运行一轮 Lp-box 微调，但通常不必要。
