# Stanford Cars B 基线实验

分支：`baseline-run-Stanford_Cars-B`。以已运行 CIFAR-100/NABirds-B 的审计 B0 实现
`baseline-run-NABirds-B@b1963c62` 为基础，仅运行 B0 的 16、32、64 bit，不包含 B1/NFD/SAM。

## 实现与论文设置

- 保留已有 B0 的单视图、等权类别关系估计，默认屏蔽预测 argmax 后 softmax；
  保留原 MDS 初始化、语义中心离散更新和中心 BCE 加量化损失。
- 论文第13页给出 Cars-B：196类，5880张训练、1958张 query、8347张 database。
  本地 `data/stanford_cars-new-seg/images` 的实际文件计数与之吻合。
- 论文第15页采用 ImageNet ResNet34，非 CIFAR 数据集学习率1e-4，RMSprop，
  batch_size=64，分类和哈希各300轮，量化权重1e-4；启动器沿用这些设置。
- 延续已有 NABirds-B 审计协议，从原 train 中每类留2张验证，seed=60。
  所以实际训练5488张、验证392张，query1958张、database8347张。
  不改变原 query/database，不用 query 选择分类器或哈希模型。
- 分类器按验证准确率选模；哈希网络每5轮验证，按验证mAP@ALL选模，持平保留较早轮次。
  每个码长300轮结束后，自动用所选模型评估一次 query，输出ALL/100/1000三个指标。
- 训练增强：resize256、随机crop224、水平翻转和ImageNet归一化；评估用中心crop224。
  num_workers=0，便于和已有实验及另一台机器保持一致。
- 启用 `--paired-stage-seeds`：每个码长的哈希训练开始前重置seed和样本顺序，
  使分类器/关系缓存是否已存在不影响哈希初始化；这不改变损失或中心算法。
- 第一个16bit实验训练Cars-B自己的分类器并构建关系；32/64bit复用同一审计关系缓存，
  各自生成中心并训练独立哈希网络。不使用其他数据集的分类器、关系或中心。

这是现有审计B0向Cars-B的适配。每类留验证图像、验证选模和带保护的中心离散更新
沿用此前实验，不能将该流程称为逐行复现论文原始训练协议。

## 另一台远程机器准备

以下路径参考已有两个数据集的命令。若另一台机器项目路径不同，修改 `cd`；
数据根路径应包含 `train/`、`query/`、`database/` 三个目录，各按类别存放图片。
本地数据和预训练权重不随Git上传，需要在远程已有项目中保留或复制。

```bash
cd /mnt/d/fzr/SHC
git fetch origin
git switch baseline-run-Stanford_Cars-B
git pull --ff-only
git log -1 --oneline

conda run --no-capture-output -n SHC python -u -c \
  'import torch,torchvision; print("torch:",torch.__version__); print("torchvision:",torchvision.__version__); assert torch.cuda.is_available(), "CUDA不可用"; print("GPU:",torch.cuda.get_device_name(0)); print("计算检查:",(torch.ones(1,device="cuda")+1).item())'
```

沿用另一台机器能够运行原SHC实验的conda环境即可；不需要DINO或教师专用环境。
已验证的原实验组合为Python3.10、torch2.1.2+cu121、torchvision0.16.2+cu121、numpy<2，
但本分支不会替换用户已安装的GPU环境。

准备ImageNet ResNet34权重，已有本地或torch hub缓存时直接加载，不重复下载：

```bash
conda run --no-capture-output -n SHC python -u -c \
  'from network import _build_resnet; _build_resnet(); print("ResNet34初始化权重检查通过")'
```

优先读取项目内 `models_ckpt/resnet34-b627a593.pth`；否则使用torch hub缓存，缺失时
上述准备命令会下载ImageNet V1权重。默认权重文件约83MB，不是训练好的Cars-B教师。

## 启动三个码长

```bash
cd /mnt/d/fzr/SHC
CARS_ROOT="$PWD/data/stanford_cars-new-seg/images"
bash scripts/start_stanford_cars_b0.sh "$CARS_ROOT" 0
```

默认环境是 `SHC`。若原环境名称不同：

```bash
SHC_ENV=你的环境名 bash scripts/start_stanford_cars_b0.sh "$CARS_ROOT" 0
```

启动后立即打印PID、总日志和独立输出目录，并写入以下路径记录；GPU检查、测试、
数据检查和训练全部在后台写日志，不必等预检结束才能查看记录：

- `logs/stanford_cars_b0_latest.path`
- `logs/stanford_cars_b0_latest.log_path`
- `logs/stanford_cars_b0_latest.pid`

启动器检查原B0和新Cars回归测试、196类数据及计数、图片内容指纹、ImageNet初始化，
然后依次运行16→32→64bit。每2000张图片打印检查进度，防止长时间无提示。
不自动断点续训；请在同一任务尚未结束时保持已有进程运行。

只检查数据和协议、暂不训练：

```bash
conda run --no-capture-output -n SHC python -u scripts/run_stanford_cars_b0.py \
  --root "$CARS_ROOT" --output-root "$PWD/tmp/cars_b0_preflight_only" --gpu 0 --dry-run
```

dry-run 不迭代任何图像加载器、不前向模型、不创建正式输出目录。
它逐文件读取字节形成内容指纹，记录跨train/query/database完全相同的文件内容。
本地原始Cars-B完整检查记录到23对跨集合内容重复，包括不同标签的相同照片；本实现保留既有划分，
不自动去重或重新抽样。详细记录保存在 `splits/*_content_duplicates.json` 和实验计划中，
结果汇总给出重复内容配对数量，解释实验时应说明该原始数据现象。

## 查看状态与结果

```bash
cd /mnt/d/fzr/SHC
B0_ROOT=$(cat logs/stanford_cars_b0_latest.path)
B0_LOG=$(cat logs/stanford_cars_b0_latest.log_path)
tail -n 25 "$B0_LOG"
for BITS in 16 32 64; do
  if test -f "$B0_ROOT/${BITS}bit.log"; then
    printf '\n===== %sbit =====\n' "$BITS"
    tail -n 12 "$B0_ROOT/${BITS}bit.log"
  fi
done
nvidia-smi
```

总日志报告预检和码长切换，分组日志记录分类器、中心生成、哈希训练及最终查询集评估。
最终查询集计算时CPU排序较多，GPU利用率不代表整个评估的工作量。

三组结束后：

```bash
cd /mnt/d/fzr/SHC
B0_ROOT=$(cat logs/stanford_cars_b0_latest.path)
conda run --no-capture-output -n SHC python -u scripts/run_stanford_cars_b0.py --report "$B0_ROOT"
```

汇总读取三组已完成的query结果，不启动新训练或新评估。保存 `summary.json` 和 `summary.csv`，
包含码长、选中轮次、运行目录及三个mAP指标。检查300轮历史、验证选模、数据/配置/来源及
模型与检索产物SHA256，缺失或不一致时报告失败，避免将半成品当作最终结果。

## 验证范围

```bash
python -m unittest discover -s tests -q
```

覆盖CIFAR/NABirds原划分、B0关系和中心数学检查，以及Cars196类标签、验证留出、
查询集索引不参与训练、内容指纹/跨集合内容重复记录、三种码长B0参数、最佳权重冻结与
验证后单次query评估、CPU环境清单和汇总完整性。
本地CPU测试与数据检查不替代远程GPU完整300轮实验，也不预先承诺论文表中的数值。
