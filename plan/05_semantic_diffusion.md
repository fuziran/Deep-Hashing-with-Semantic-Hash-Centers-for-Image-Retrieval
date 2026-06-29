# 创新五：高阶语义传播相似矩阵

## 问题背景

原始相似矩阵（`GenerateSimilarityMatrix.py`）的构建过程：
1. 训练一个分类网络
2. 对每张训练样本，取分类 logits（mask 掉真实类），做 softmax 后累加到对应行
3. 对矩阵做行归一化

此方法只建模**一阶**邻域语义关系（分类网络直接预测的相邻类别相似度），存在两个问题：

**问题 1：缺乏传递语义**
- 若"轿车"≈"跑车"，"跑车"≈"赛车"，则"轿车"与"赛车"应有中等相似度
- 原始矩阵仅有一跳的直接关系，忽略了此类传递关系

**问题 2：噪声类对的干扰**
- 分类网络对部分弱相关类别也会给出小的 softmax 值，积累后产生噪声相似度
- 这些噪声会轻微污染哈希中心的语义对齐

## 解决方案

### 函数一：`_high_order_diffusion`（高阶图扩散）

**数学原理**：
设 `T` 为行归一化的随机游走转移矩阵（去掉对角线自环），则 k 阶扩散矩阵为：

```
S_diffused = I + α·T + α²·T² + ... + αᵏ·Tᵏ
```

- `α = 0.15`（衰减因子，控制高阶影响强度）
- `steps = 3`（3 跳，捕捉三阶邻域关系）
- 最终与原矩阵加权融合：`S_final = 0.7·S_original + 0.3·S_diffused`

```python
def _high_order_diffusion(S, alpha=0.15, steps=3):
    S_off = S.clone()
    S_off.fill_diagonal_(0.0)
    T = S_off / S_off.sum(dim=1, keepdim=True).clamp(min=1e-8)

    S_diffused = torch.eye(n, device=device)
    T_power = T.clone()
    coeff = alpha
    for _ in range(steps):
        S_diffused = S_diffused + coeff * T_power
        T_power = T_power @ T
        coeff *= alpha

    S_diffused = (S_diffused + S_diffused.T) / 2.0
    S_diffused.fill_diagonal_(1.0)
    return 0.7 * S + 0.3 * S_diffused
```

**直觉解释**：
- 1 阶（`α·T`）：直接相邻类的传播，权重 0.15
- 2 阶（`α²·T²`）：间接相邻类，权重 0.0225
- 3 阶（`α³·T³`）：三跳邻近，权重 0.003

越远的传播权重越小，保证近邻的语义信号仍占主导。

### 函数二：`_sparse_topk_similarity`（Top-K 稀疏化）

对每个类只保留语义最相关的 `k` 个邻居（默认取 `max(10, n_class//10)`），将其余弱相关类的相似度置零：

```python
def _sparse_topk_similarity(S, topk):
    S_sparse = torch.zeros_like(S)
    for i in range(n):
        row = S[i].clone()
        row[i] = float('-inf')
        vals, idx = torch.topk(row, k=min(topk, n-1))
        mask = vals > 0
        S_sparse[i, idx[mask]] = vals[mask]
    return (S_sparse + S_sparse.T) / 2.0
```

这一步：
1. 去除噪声类对的干扰，相似矩阵更稀疏、更精准
2. 使哈希中心优化时的语义约束更聚焦

### 调用链路

在 `GenerateSimilarityMatrix` 函数的最后，原始归一化完成后追加调用：

```python
# 原有归一化（行归一化 + 对角线置 1）完成后
if use_diffusion:
    S = _high_order_diffusion(S, alpha=diff_alpha, steps=diff_steps)
    S = _sparse_topk_similarity(S, topk=topk_neighbours)
```

## 改动文件

| 文件 | 改动内容 |
|------|---------|
| `GenerateSimilarityMatrix.py` | 新增 `_high_order_diffusion`、`_sparse_topk_similarity` 两个函数；在 `GenerateSimilarityMatrix` 末尾追加调用（受 `args.use_diffusion` 控制） |

## 新增参数（`run.py`）

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `--use-diffusion` | `True` | 是否启用语义扩散 |
| `--diff-alpha` | `0.15` | 扩散衰减因子 |
| `--diff-steps` | `3` | 扩散跳数 |
| `--sim-topk` | `max(10, n//10)` | Top-K 邻居保留数 |

## 计算复杂度

- 扩散：O(n² × steps)，n=类别数。对 n=100，100²×3=30,000 次矩阵乘法，毫秒级
- Top-K：O(n²)，同样极快
- 整体：对预处理阶段（Stage 1），额外开销 < 0.1 秒

**推理阶段无任何开销**（相似矩阵只在训练前计算一次）

## 预期效果

| 数据集 | mAP@ALL 预期提升 |
|--------|---------------|
| Stanford Cars（车型分类层次结构清晰） | +2.0 ~ 3.0% |
| NABirds（鸟类分类有明确的科属关系） | +2.0 ~ 3.0% |
| CIFAR-100（类别层次相对简单） | +0.5 ~ 1.5% |

**核心收益**：哈希中心的语义排布更忠实于真实的类间关系层次，相关类别的哈希中心相互更近，无关类别更远。

## 验证方法

可在训练前打印矩阵统计量来确认扩散是否有效：
```python
print('扩散前 off-diag min/max/mean:', S[~mask].min(), S[~mask].max(), S[~mask].mean())
# 运行扩散
print('扩散后 off-diag min/max/mean:', S[~mask].min(), S[~mask].max(), S[~mask].mean())
```
期望：扩散后 min 值略升高（弱相关类获得少量传递相似度）、方差略降（分布更均匀）。
