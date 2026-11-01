# NABirds-B：B1 + NFD + SAM

在 `NABirds-B-update-reliable-relation-nfd-distillation` 的 `9e48878f`
基础上迁移 CIFAR-100 已验证的普通 SAM 更新（原 CIFAR 分支 `a420456d`）。
这项迁移尚无 NABirds-B 实验结果；SAM 是已有方法，本代码不构成独立创新性证明。

## 固定实验协议

- 只新训练 seed=60 的 16、32、64 bit B1+NFD+SAM，各 300 轮；不重跑历史 NFD 对照。
- 普通非自适应 L2 SAM，rho=0.05，底层仍为 RMSprop、weight_decay=1e-5，原余弦调度不变。
- 完整目标为原中心损失 + 原量化损失 + 原定时 NFD；alpha=0.75，NFD 权重=0.01，
  topk=5，warmup=20，ramp_end=40。两次学生前向使用同一批增强图像和相同随机抽样，
  第二次不累计 BN 运行统计；NFD 启用时每批教师只前向一次，最后只更新一次优化器。
- 从各码长历史 NFD 的 config.json 继承 NABirds-B 555 类、原数据根路径、学习率、
  batch size、增强、验证频率和 paired_stage_seeds 等设置。原脚本使用 lr=1e-4、
  batch_size=64、num_workers=0，每类留 2 张验证图像；不会套用 CIFAR 数据协议。
- 验证集 mAP@ALL 选择最佳轮次，相同分数保留较早轮次。训练不计算最终 query；
  所有计划码长训练结束后，通过独立入口评估已选模型。
- 复用原 B1 关系、中心、分类教师缓存，并核对元数据和内容；不训练或下载教师。
  学生沿用原 ImageNet ResNet34 初始化，需要已有 `resnet34-b627a593.pth`。
- SAM 每批两次学生前向/反向，训练成本通常高于原 NFD；本地测试不预测远程耗时和精度。

原始 split_hash 算法保持不变，继续核对相对图片路径、类别及划分。新实验另外逐文件
计算内容 SHA256，绑定本次训练和独立评估。历史 NFD 未记录图片内容指纹，因此不能
据此追溯证明历史运行时图片字节相同。读取 query/database 文件计算指纹不进行模型评估。

## 远程启动

默认已有 NFD 目录为 `save_nabirds_b_i2_nfd`，其中每个码长只允许存在一个符合条件的
历史 NFD config.json；需要同时保留该实验的完整 300 轮 history、metrics 中的 query_metrics，
以及配置引用的 B1 父配置与三个缓存文件。原历史 metrics 可以没有 query_evaluated 字段，
但不得将 validation 数值代替 query 指标。

```bash
cd /mnt/d/fzr/SHC
git fetch origin
git switch NABirds-B-update-reliable-relation-nfd-distillation-i3-sharpness-aware-minimization
git pull --ff-only
bash scripts/start_nabirds_b_sam.sh "$PWD/save_nabirds_b_i2_nfd" 0
```

启动器默认使用 conda 环境 `SHC`；如原 NABirds 实验采用其他环境，可在启动前设置
`export SHC_ENV=实际环境名`。先检查 GPU 和回归测试，再核对历史缓存/数据，并对三个码长
各运行一个临时 GPU SAM 批次，包含完整 NFD 目标。临时批次不计入正式 300 轮。
全部通过后后台依次训练 16 → 32 → 64 bit。文件内容核对会多次扫描图片，可能需要数分钟。
默认目录不同则将启动器第一个参数改为实际历史 NFD 目录。

若该目录存在重复试验，Python 入口支持显式指定每个原配置，避免任意挑选：

```bash
conda run --no-capture-output -n SHC python -u scripts/run_nabirds_b_sam.py \
  --nfd-config /原16bit运行目录/config.json \
  --nfd-config /原32bit运行目录/config.json \
  --nfd-config /原64bit运行目录/config.json \
  --bits 16 32 64 --gpu 0 --output-root /新的输出目录 --dry-run
```

## 查看训练状态和结果

```bash
cd /mnt/d/fzr/SHC
SAM_ROOT=$(cat logs/nabirds_b_sam_latest.path)
SAM_LOG=$(cat logs/nabirds_b_sam_latest.log_path)
tail -n 20 "$SAM_LOG"
for BITS in 16 32 64; do
  if test -f "$SAM_ROOT/${BITS}bit.log"; then
    printf '\n===== %sbit =====\n' "$BITS"
    tail -n 8 "$SAM_ROOT/${BITS}bit.log"
  fi
done
```

总日志只报告码长切换；详细 batch/epoch 进度写入对应的 `${BITS}bit.log`。
新运行目录中保存 config、training_start、history、最佳学生权重、选模摘要和 NFD 投影信息。
每轮记录 SAM 目标、扰动范数、优化器/前向次数、批次顺序指纹；首批采样 NFD 梯度诊断。
完整训练报告命令：

```bash
conda run --no-capture-output -n SHC python -u scripts/run_nabirds_b_sam.py --report "$SAM_ROOT"
```

该命令只在全部计划码长完整训练、来源/数据指纹一致、选模记录一致时通过。

## 独立查询集评估

以下代码先确认训练完成，再启动三个码长的 query 评估；不重跑历史 NFD 对照。

```bash
(
  set -e
  cd /mnt/d/fzr/SHC
  SAM_ROOT=$(cat logs/nabirds_b_sam_latest.path)
  conda run --no-capture-output -n SHC python -u scripts/run_nabirds_b_sam.py --report "$SAM_ROOT"
  nohup conda run --no-capture-output -n SHC \
    python -u scripts/run_nabirds_b_sam.py --query "$SAM_ROOT" --gpu 0 \
    > "$SAM_ROOT/query_launcher.log" 2>&1 < /dev/null &
  printf 'PID=%s\nLOG=%s\n' "$!" "$SAM_ROOT/query_launcher.log"
)
```

`--query` 在任何 query 前向前检查全部三个已选模型和数据；独立评估不构造教师或优化器，
不下载初始化权重，不修改训练结果。query/database 数量取自实际 NABirds-B 划分，
不使用 CIFAR 的 5000/45000 常数。已有 query 输出目录时拒绝覆盖，请先检查已有日志。
训练器和 query 批处理均不自动断点续跑。

```bash
SAM_ROOT=$(cat logs/nabirds_b_sam_latest.path)
tail -n 20 "$SAM_ROOT/query_launcher.log"
for BITS in 16 32 64; do
  if test -f "$SAM_ROOT/query/${BITS}bit.log"; then
    printf '\n===== query %sbit =====\n' "$BITS"
    tail -n 8 "$SAM_ROOT/query/${BITS}bit.log"
  fi
done
conda run --no-capture-output -n SHC python -u scripts/run_nabirds_b_sam.py --query-report "$SAM_ROOT"
```

`--query-report` 只读已完成的评估结果，不能用于启动 query。汇总保存为
`query/query_report.json` 和 `query/query_report.csv`，包含各码长 mAP@ALL/@100/@1000
及相对历史 NFD query 的差值；历史对照来源、SHA256 一并记录。

## 验证

```bash
python -m unittest discover -s tests -q
```

回归覆盖原 NABirds B1/NFD、SAM 手工更新对照、rho=0 退化、随机状态/BN/教师复用、
异常回滚、三种码长的微型训练、历史配置继承、文件内容变更、选模完整性、独立 query
及三组 query 启动/汇总协议。合成 CPU 测试不代替远程真实 NABirds GPU 预检或实验结果。
