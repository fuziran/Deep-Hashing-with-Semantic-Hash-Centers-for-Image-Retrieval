# SHC 创新改进计划

> 基于 ACM TOIS 2025 论文"Deep Hashing with Semantic Hash Centers for Image Retrieval"，针对当前实现提出五个系统性创新点，旨在提升 mAP@ALL / mAP@100 / mAP@1000 等核心指标。

---

## 项目现状分析

| 模块 | 当前做法 | 瓶颈 |
|------|---------|------|
| 特征提取 | ResNet34 → GAP → Linear(512, bit) | 只用最终层，局部判别信息丢失 |
| 损失函数 | BCE(center) + λ·(\|u\|-1)² | 无 bit 均衡/独立约束，哈希码利用率低 |
| 语义相似矩阵 | 分类网络 softmax 输出 + 归一化 | 仅捕捉一阶邻域相似，忽略传递语义 |
| 哈希中心 | 离线生成后完全固定 | 与实际特征流形对齐缺口随训练扩大 |
| 数据增强 | Resize + RandomCrop + Flip | 无哈希一致性正则，泛化能力受限 |

---

## 创新点一：多尺度注意力特征聚合（Multi-Scale Attention Feature Fusion）

### 动机
当前 ResNet34 结构只使用 `layer4` 之后的全局平均池化特征（512 维），丢弃了 `layer2`（128 维） 和 `layer3`（256 维）包含的细粒度中间语义信息。在细粒度识别数据集（Stanford Cars, NABirds）上，局部判别区域对检索至关重要。

### 具体改进方案

**文件**: `network.py` — `ResNet` 类

**改动一：加入多尺度特征提取**
```python
class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet34"):
        super(ResNet, self).__init__()
        # ... 保留原有 backbone 加载 ...

        # 通道对齐卷积（将 layer2、layer3 统一到 256 维）
        self.align2 = nn.Conv2d(128, 256, 1)   # layer2 输出 128 通道
        self.align3 = nn.Conv2d(256, 256, 1)   # layer3 输出 256 通道

        # SE 通道注意力模块（轻量）
        self.se2 = SEBlock(256)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        # 多尺度融合后投影到 hash_bit
        self.hash_layer = nn.Sequential(
            nn.Linear(512 + 256 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, hash_bit),
        )

    def forward(self, x):
        x = self.conv1(x);  x = self.bn1(x);  x = self.relu(x);  x = self.maxpool(x)
        x = self.layer1(x)
        f2 = self.layer2(x)                    # [B, 128, H2, W2]
        f3 = self.layer3(f2)                   # [B, 256, H3, W3]
        f4 = self.layer4(f3)                   # [B, 512, H4, W4]

        # GAP + SE
        f2 = self.se2(self.align2(f2))
        f2 = f2.mean(dim=[2, 3])               # [B, 256]
        f3 = self.se3(self.align3(f3))
        f3 = f3.mean(dim=[2, 3])               # [B, 256]
        f4 = self.se4(f4)
        f4 = f4.mean(dim=[2, 3])               # [B, 512]

        feat = torch.cat([f2, f3, f4], dim=1)  # [B, 1024]
        return self.hash_layer(feat)
```

**改动二：SE Block 实现**
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w
```

### 预期效果
- CIFAR-100: mAP@ALL **+1.5~2.5%**
- Stanford Cars / NABirds: mAP@ALL **+2~4%**（细粒度受益更大）
- 模型参数量增加 < 5%，训练时间增加约 10%

---

## 创新点二：Bit 均衡 + Bit 独立正则化损失（Balanced & Decorrelated Hash Loss）

### 动机
原始 `Q_loss = (|u| - 1)²` 仅强制实值哈希码接近 ±1（量化），但：
1. 没有约束各 bit 的使用均衡性 → 某些 bit 长期为 +1 或 -1，浪费编码容量
2. 没有约束 bit 间独立性 → bits 高度相关，信息冗余，有效编码维度远小于标称 bit 数

### 具体改进方案

**文件**: `train.py` — `CSQLoss.forward()`

**新损失函数**:
```python
def forward(self, u, y, ind, args):
    u_tanh = u.tanh()
    hash_center = self.label2center(y)

    # 原始两项
    center_loss = self.criterion(0.5 * (u_tanh + 1), 0.5 * (hash_center + 1))
    Q_loss = (u_tanh.abs() - 1).pow(2).mean()

    # 新增：Bit 均衡损失
    # 每个 bit 在 batch 内的均值应接近 0（等概率 +1/-1）
    bit_mean = u_tanh.mean(dim=0)                      # [bit]
    balance_loss = bit_mean.pow(2).mean()

    # 新增：Bit 独立损失（去相关）
    # 归一化后的协方差矩阵应接近单位矩阵
    u_norm = u_tanh - u_tanh.mean(dim=0, keepdim=True)
    cov = (u_norm.T @ u_norm) / u_tanh.size(0)        # [bit, bit]
    cov_diag = torch.diag(cov)
    corr = cov / (cov_diag.unsqueeze(0) * cov_diag.unsqueeze(1)).sqrt().clamp(1e-8)
    eye = torch.eye(corr.size(0), device=corr.device)
    independence_loss = (corr - eye).pow(2).mean()

    return (center_loss
            + args.lambd * Q_loss
            + args.alpha_bal * balance_loss
            + args.alpha_ind * independence_loss)
```

**超参数建议**（`run.py` 新增）:
```python
parser.add_argument('--alpha-bal', default=0.01, type=float, help='bit balance weight')
parser.add_argument('--alpha-ind', default=0.001, type=float, help='bit independence weight')
```

### 预期效果
- mAP@ALL: **+1~2%**（码字有效信息量提升）
- 短码（16 bit）收益最显著，因为低维时 bit 冗余问题更严重

---

## 创新点三：对比增强哈希一致性正则（Contrastive Hash Consistency Regularization）

### 动机
当前训练只用单一视角（标准增强）的图像。同一图像的不同增强版本应产生相同的哈希码，但现有损失无法显式强制这一一致性。引入对比一致性损失可以 (1) 强化特征的增强不变性，(2) 隐式拉开不同类别哈希码的距离。

### 具体改进方案

**文件**: `train.py` — 训练循环  
**文件**: `data/data_loader.py` — 支持双视角输出

**Step 1: 修改 DataLoader 支持双视角**
```python
class CIFAR100HashDataset(Dataset):
    def __getitem__(self, item):
        real_idx = int(self.indices[item])
        img = Image.fromarray(self.images[real_idx])
        label_id = int(self.labels[real_idx])

        # 返回两个不同增强视角
        img1 = self.transform(img)
        img2 = self.transform_aug(img)   # 更强增强：ColorJitter + GaussianBlur

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[label_id] = 1.0
        return img1, img2, label, torch.tensor(real_idx, dtype=torch.long)
```

**Step 2: 在训练循环加入一致性损失**
```python
# train.py — 训练步骤
image1, image2, label, ind = batch
u1 = net(image1)       # 视角1的哈希向量
u2 = net(image2)       # 视角2的哈希向量

# 原有中心损失（对 u1）
loss_main = criterion(u1, label.float(), ind, args)

# 对比一致性损失（InfoNCE 近似）
# 同一图像的两个视角哈希码应尽量接近
u1_norm = F.normalize(u1.tanh(), dim=1)
u2_norm = F.normalize(u2.tanh(), dim=1)
sim_matrix = u1_norm @ u2_norm.T / args.tau       # [B, B]
labels_cont = torch.arange(sim_matrix.size(0), device=args.device)
loss_cont = (F.cross_entropy(sim_matrix, labels_cont) +
             F.cross_entropy(sim_matrix.T, labels_cont)) / 2

loss = loss_main + args.alpha_cont * loss_cont
```

**新增超参数**:
```python
parser.add_argument('--alpha-cont', default=0.1, type=float, help='contrastive loss weight')
parser.add_argument('--tau', default=0.07, type=float, help='contrastive temperature')
```

**强增强配置**（训练集）:
```python
transform_aug = transforms.Compose([
    transforms.Resize(args.resize_size),
    transforms.RandomResizedCrop(args.crop_size, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    normalize,
])
```

### 预期效果
- mAP@ALL: **+1.5~3%**（增强不变性带来更好泛化）
- MSCOCO 多标签数据集预期收益更大（+2~4%）
- 训练速度降低约 20%（每批次双倍前向传播）

---

## 创新点四：在线自适应哈希中心精炼（Online Adaptive Hash Center Refinement）

### 动机
现有流程：哈希中心在训练前完全离线生成，训练阶段固定不变。问题在于：
- 哈希中心是基于分类网络特征空间生成的，而哈希网络特征空间与之不完全一致
- 随着训练深入，哈希网络逐渐形成自己的特征流形，但哈希中心无法追踪这种变化
- 导致后期训练的 center_loss 梯度方向与实际特征分布偏离

### 具体改进方案

**文件**: `train.py`

**做法**：每隔 `T` 个 epoch，用 EMA（指数移动平均）方法对哈希中心进行精炼：

```python
class AdaptiveHashCenter:
    def __init__(self, hash_center, num_classes, momentum=0.99):
        self.centers = hash_center.clone().float()   # [n_class, bit]
        self.momentum = momentum
        self.class_counts = torch.zeros(num_classes)
        self.class_sums = torch.zeros_like(self.centers)

    def update(self, hash_codes, labels):
        """在每个 mini-batch 后累积特征统计"""
        with torch.no_grad():
            hash_codes = hash_codes.tanh().detach().cpu()
            class_ids = labels.argmax(dim=1).cpu()
            for i, c in enumerate(class_ids):
                self.class_sums[c] += hash_codes[i]
                self.class_counts[c] += 1

    def refine(self):
        """每 T epoch 执行一次 EMA 精炼"""
        with torch.no_grad():
            valid = self.class_counts > 0
            epoch_mean = self.class_sums[valid] / self.class_counts[valid].unsqueeze(1)
            # EMA 更新
            self.centers[valid] = (self.momentum * self.centers[valid] +
                                   (1 - self.momentum) * epoch_mean)
            # 精炼后二值化（保持中心为 ±1）
            self.centers = torch.sign(self.centers)
            self.centers[self.centers == 0] = 1
            # 重置统计
            self.class_sums.zero_()
            self.class_counts.zero_()
        return self.centers


# train_val 函数中的修改
adaptive_center = AdaptiveHashCenter(hash_center, args.num_classes, momentum=0.995)
refine_interval = 20   # 每 20 个 epoch 精炼一次

for epoch in range(args.epoch):
    # ... 正常训练 ...
    for image, label, ind in train_loader:
        u = net(image)
        loss = criterion(u, label.float(), ind, args)
        adaptive_center.update(u, label)   # 累积统计
        # ...

    # 周期性精炼
    if (epoch + 1) % refine_interval == 0 and epoch > args.epoch // 4:
        new_centers = adaptive_center.refine()
        criterion.hash_targets = new_centers.to(args.device)
        print(f"[Epoch {epoch+1}] Hash centers refined.")
```

### 预期效果
- mAP@ALL: **+1.5~3%**（特别在训练后期，中心对齐收益明显）
- 对 CIFAR-100（100 类）和 NABirds（细粒度）效果最显著
- 不增加推理时间（只在训练时额外计算）

---

## 创新点五：高阶语义传播相似矩阵（High-Order Semantic Diffusion Similarity Matrix）

### 动机
当前相似矩阵仅捕捉一阶语义关系（分类网络直接预测的相邻类别相似度）。考虑传递关系："轿车"与"跑车"相似，"跑车"与"赛车"相似，则"轿车"与"赛车"也应具有中等相似度。这种传递关系（高阶语义）在当前矩阵中完全缺失。

### 具体改进方案

**文件**: `GenerateSimilarityMatrix.py` — 在 `GenerateSimilarityMatrix()` 结束前加入扩散步骤

**方法一：对称归一化图扩散**
```python
def high_order_diffusion(S, alpha=0.2, steps=2):
    """
    S: [n_class, n_class]，归一化后的相似矩阵（对角线=1）
    alpha: 高阶传播权重
    steps: 传播阶数
    """
    # 去掉对角线自环，构建纯相似矩阵
    S_off = S.clone()
    S_off.fill_diagonal_(0)

    # 行归一化（转移矩阵）
    row_sum = S_off.sum(dim=1, keepdim=True).clamp(min=1e-8)
    T = S_off / row_sum   # 随机游走转移矩阵

    # 高阶扩散：S_new = (1-α)·I + α·T + α²·T² + ...
    S_diffused = torch.eye(S.size(0), device=S.device)
    T_power = T.clone()
    coeff = alpha
    for _ in range(steps):
        S_diffused = S_diffused + coeff * T_power
        T_power = T_power @ T
        coeff *= alpha

    # 对称化 + 重新归一化到原始 [-1, 1] 范围
    S_diffused = (S_diffused + S_diffused.T) / 2
    S_diffused.fill_diagonal_(1.0)

    # 与原始 S 加权融合（保留原始强信号）
    S_final = 0.7 * S + 0.3 * S_diffused
    return S_final
```

**方法二：标签平滑增强版相似矩阵**

在现有归一化步骤后，额外加入 `k-NN` 图过滤，仅保留每类最相关的 `k` 个邻居：
```python
def sparse_topk_similarity(S, topk=10):
    """只保留每行 top-k 相似类，其余置 0"""
    S_sparse = torch.zeros_like(S)
    for i in range(S.size(0)):
        vals, idx = torch.topk(S[i], topk + 1)  # +1 包含自身
        S_sparse[i, idx] = vals
    S_sparse = (S_sparse + S_sparse.T) / 2
    S_sparse.fill_diagonal_(1.0)
    return S_sparse
```

**在 `GenerateSimilarityMatrix` 末尾调用**:
```python
# 原有归一化结束后
S[mask] = 1

# 新增：高阶语义扩散
S = high_order_diffusion(S, alpha=0.15, steps=3)
S = sparse_topk_similarity(S, topk=max(10, args.num_classes // 10))

torch.save(S, ...)
return S
```

### 预期效果
- mAP@ALL: **+1~2%**（哈希中心语义对齐更准确）
- 在类别层次结构清晰的数据集（Stanford Cars 细分车型、NABirds 鸟类分类）收益最显著：**+2~3%**
- 不增加任何推理时间，仅前置计算一次

---

## 综合预期收益汇总

| 创新点 | 涉及文件 | 指标预期提升 | 实现难度 | 训练开销 |
|--------|---------|------------|---------|---------|
| 1. 多尺度注意力特征 | `network.py` | mAP@ALL +1.5~4% | ★★☆ | +10% |
| 2. Bit 均衡&独立损失 | `train.py` | mAP@ALL +1~2% | ★☆☆ | <+1% |
| 3. 对比哈希一致性 | `train.py`, `data_loader.py` | mAP@ALL +1.5~3% | ★★☆ | +20% |
| 4. 在线哈希中心精炼 | `train.py` | mAP@ALL +1.5~3% | ★★☆ | <+5% |
| 5. 高阶语义相似扩散 | `GenerateSimilarityMatrix.py` | mAP@ALL +1~3% | ★☆☆ | 无 |
| **全部叠加（理论上界）** | — | mAP@ALL **+5~10%** | — | — |

> 注：各创新点设计上相互独立，可逐一消融实验验证贡献，再叠加最优组合。

---

## 实施顺序建议

```
阶段 A（基础改进，快速验证）：
  创新点 2（Bit 损失）→ 创新点 5（相似矩阵扩散）
  → 运行 baseline 对比实验

阶段 B（结构改进）：
  创新点 1（多尺度注意力）
  → 消融验证 SE / multi-scale 各自贡献

阶段 C（训练策略）：
  创新点 3（对比一致性）+ 创新点 4（在线中心精炼）
  → 联合实验，分析交互效果
```

---

## 关键消融实验设计

每个创新点独立开关（通过 `run.py` 参数控制），建议完整消融矩阵：

| 实验编号 | 创新1 | 创新2 | 创新3 | 创新4 | 创新5 |
|---------|------|------|------|------|------|
| baseline | ✗ | ✗ | ✗ | ✗ | ✗ |
| +Innov1 | ✓ | ✗ | ✗ | ✗ | ✗ |
| +Innov2 | ✗ | ✓ | ✗ | ✗ | ✗ |
| +Innov3 | ✗ | ✗ | ✓ | ✗ | ✗ |
| +Innov4 | ✗ | ✗ | ✗ | ✓ | ✗ |
| +Innov5 | ✗ | ✗ | ✗ | ✗ | ✓ |
| full | ✓ | ✓ | ✓ | ✓ | ✓ |

在 CIFAR-100 的 16/32/64 bit 三种码长上各跑一轮，对比 mAP@ALL / mAP@100 / mAP@1000。

---

*计划文件生成于 2026-06-29，请确认后执行实现。*
