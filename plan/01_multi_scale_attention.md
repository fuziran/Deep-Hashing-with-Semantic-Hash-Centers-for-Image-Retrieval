# 创新一：多尺度 SE 注意力特征聚合

## 问题背景

原始 `ResNet` 类仅使用 `layer4` 之后的全局平均池化特征（512 维）来生成哈希码：

```
输入 → ResNet34 → layer4 → GAP → Linear(512, hash_bit)
```

这一设计的核心缺陷：
1. **细粒度信息丢失**：`layer2`（128-d，捕捉边缘、纹理等低级特征）和 `layer3`（256-d，捕捉局部语义结构）的判别信息完全丢弃
2. **特征单一**：对 Stanford Cars、NABirds 等细粒度数据集，局部区域的辨别力至关重要，仅靠最终层的全局语义往往无法区分相似子类
3. **channel 权重均等**：全局平均池化对所有通道等权叠加，无法动态强调任务相关的判别通道

## 解决方案

### 新增 SEBlock 类

Squeeze-and-Excitation（SE）通道注意力机制：

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        w = self.pool(x).view(b, c)        # 压缩：全局平均池化
        w = self.fc(w).view(b, c, 1, 1)   # 激励：MLP + Sigmoid
        return x * w                        # 特征重标定
```

- **Squeeze（压缩）**：全局平均池化将空间维度压缩为 1×1，得到每个通道的全局统计量
- **Excitation（激励）**：两层 MLP 学习通道间的相关性，输出 `[0,1]` 的权重
- **Recalibration（重标定）**：将权重乘回原始特征图，动态强化判别通道

### 改造 ResNet.forward — 多尺度特征流

```
输入 [B,3,224,224]
  → conv1/bn1/relu/maxpool
  → layer1                     [B, 64, 56, 56]
  → layer2 → SE2 → GAP        [B, 128]  ← 纹理、边缘
  → layer3 → SE3 → GAP        [B, 256]  ← 局部语义
  → layer4 → SE4 → GAP        [B, 512]  ← 全局语义
  → concat                     [B, 896]
  → Linear(896→512) + BN + ReLU
  → Linear(512→hash_bit)
```

ResNet34 各层输出通道数：
| 层 | 通道数 | 空间分辨率（224输入） |
|----|-------|-------------------|
| layer2 | 128 | 28×28 |
| layer3 | 256 | 14×14 |
| layer4 | 512 | 7×7 |

融合后维度：128 + 256 + 512 = **896**，经两层 MLP 压缩到 hash_bit。

## 改动文件

| 文件 | 改动内容 |
|------|---------|
| `network.py` | 新增 `SEBlock` 类；重构 `ResNet.__init__` 添加 `se2/se3/se4/gap` 和新 `hash_layer`；重写 `ResNet.forward` 实现多尺度提取 |

其余类（`ClassifyNet`、`AlexNet`、`orthohashNet`、`LTHNet`、`NewNet`）**保持不变**。

## 参数量分析

| 模块 | 新增参数 |
|------|---------|
| SEBlock × 3（128/256/512 通道，reduction=16） | 128×8×2 + 256×16×2 + 512×32×2 ≈ **43K** |
| hash_layer Linear(896→512) | 896×512 + 512 ≈ **459K** |
| hash_layer Linear(512→bit) | 512×bit + bit（≈16-64 bit，可忽略） |
| **总新增** | **~502K**（相比 ResNet34 的 21M 参数，增加 < 2.4%） |

## 预期效果

| 数据集类型 | mAP@ALL 预期提升 |
|-----------|---------------|
| CIFAR-100（100 类通用） | +1.5 ~ 2.5% |
| Stanford Cars（细粒度车型） | +2.0 ~ 4.0% |
| NABirds（细粒度鸟类） | +2.0 ~ 4.0% |
| MSCOCO（多标签） | +1.0 ~ 2.0% |

细粒度数据集受益更大，因为判别信息往往分布在局部区域（车型 logo、鸟类羽毛花纹），低层特征贡献显著。

## 调试建议

若训练 loss 下降异常，可检查：
- `hash_layer[0]`（Linear 896→512）的梯度范数是否正常
- se2/se3/se4 的 weight 输出分布是否接近 0.5（初始状态应均衡）
- 增大 `batch-size` 有助于 BatchNorm1d 的稳定性
