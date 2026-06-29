# 创新二：Bit 均衡 + Bit 独立正则化损失

## 问题背景

原始 `CSQLoss` 的损失函数：

```
Loss = center_loss + λ · Q_loss
```

其中 `Q_loss = mean((|u| - 1)²)` 仅强迫实值哈希码接近 ±1（量化约束），但存在两个根本缺陷：

### 缺陷 1：Bit 均衡性缺失

若某个 bit 长期输出 +1（因训练数据分布偏斜），则该维度实质变成常量，不携带任何信息。

**理论上**，k bit 哈希码最多可区分 2ᵏ 个不同编码。若某些 bit 严重偏斜（如 90% 为 +1），实际有效编码数目远小于 2ᵏ，造成"虚胖"的编码容量。

### 缺陷 2：Bit 独立性缺失

若 bit_i 和 bit_j 高度相关（相关系数 r ≈ 1），则二者携带的信息几乎重叠，等效于只有 k-1 个独立 bit。

**Shannon 信息论**：最优二值编码应使每个 bit 独立且均衡，此时总编码熵最大（k bit）。

## 解决方案

在 `CSQLoss.forward` 中追加两个正则项（原有两项不变）：

### Innovation 2a：Bit 均衡损失（Balance Loss）

```python
bit_mean = u_tanh.mean(dim=0)           # [bit]，每个 bit 在当前 batch 的均值
balance_loss = bit_mean.pow(2).mean()   # 期望 bit_mean → 0
```

- **含义**：强制每个 bit 在 batch 内的均值趋向 0，即 +1 和 -1 等概率出现
- **权重** `alpha_bal`：默认 `0.01`，小于 center_loss，仅作柔性约束

### Innovation 2b：Bit 独立损失（Independence Loss）

```python
u_center = u_tanh - u_tanh.mean(dim=0, keepdim=True)
cov = (u_center.T @ u_center) / B          # [bit, bit] 协方差矩阵
diag = torch.diag(cov).clamp(min=1e-8)
corr = cov / (diag.unsqueeze(0) * diag.unsqueeze(1)).sqrt()  # 标准化相关
eye = torch.eye(bit, device=corr.device)
independence_loss = (corr - eye).pow(2).mean()   # 相关矩阵 → 单位矩阵
```

- **含义**：强制 bit 间的相关矩阵接近单位矩阵（各 bit 线性不相关）
- **权重** `alpha_ind`：默认 `0.001`，量级需小于 balance_loss 以避免过度约束

### 完整新损失

```
Loss = center_loss
     + λ        · Q_loss            （量化，原始）
     + α_bal    · balance_loss      （bit 均衡，新增）
     + α_ind    · independence_loss （bit 独立，新增）
```

## 改动文件

| 文件 | 改动内容 |
|------|---------|
| `train.py` | `CSQLoss.forward` 中追加 balance_loss 和 independence_loss 计算 |
| `run.py` | 新增 `--alpha-bal`（默认 0.01）和 `--alpha-ind`（默认 0.001）参数 |

## 超参数选择建议

| 参数 | 默认值 | 调整方向 |
|------|-------|---------|
| `--alpha-bal` | 0.01 | 若 loss 抖动加大，降至 0.005；若 bit 均衡改善不明显，升至 0.05 |
| `--alpha-ind` | 0.001 | 短码（16 bit）可升至 0.005；长码（64 bit）可降至 0.0005 |

**禁用方式**：设置 `--alpha-bal 0 --alpha-ind 0` 退化为原始 CSQLoss。

## 预期效果

| 码长 | mAP@ALL 预期提升 | 原因 |
|------|---------------|------|
| 16 bit | +1.5 ~ 2.5% | 短码 bit 冗余问题最严重，收益最大 |
| 32 bit | +1.0 ~ 2.0% | 中等码长，均衡改善明显 |
| 64 bit | +0.5 ~ 1.5% | 长码本身冗余相对较少 |

## 验证方法

训练结束后，可用以下代码检验 bit 分布：

```python
# 在 compute_result 后执行
binary_codes, _ = compute_result(database_loader, net, device)
bit_mean = binary_codes.float().mean(dim=0)    # 期望接近 0
print("Bit bias (should be ~0):", bit_mean.abs().mean().item())

# Bit 相关矩阵（理想情况接近单位矩阵）
corr = torch.corrcoef(binary_codes.float().T)
off_diag = corr[~torch.eye(corr.size(0), dtype=bool)]
print("Off-diag correlation mean:", off_diag.abs().mean().item())
```
