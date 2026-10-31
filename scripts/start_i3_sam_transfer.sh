#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
BASELINE_ROOT="${1:-$REPO_ROOT/save_nfd_16_64_seed60_ZZdT2Ou0}"
GPU="${2:-0}"
ANCHOR_RUN="${3:-$REPO_ROOT/save/i3_sam_screen32_seed60_20261031_142135_706/control/runs/cifar-100-new-seg_B2NFD_32bit_seed60_4981e9aee776}"
if [[ ! -d "$BASELINE_ROOT" || ! -f "$ANCHOR_RUN/config.json" || ! "$GPU" =~ ^[0-9]+$ ]]; then
  printf '原始16/64bit实验根目录或32bit对照模型不存在，或GPU编号无效。\nBASELINE=%s\nANCHOR=%s\n' "$BASELINE_ROOT" "$ANCHOR_RUN" >&2
  exit 1
fi
mkdir -p logs tmp
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
TRANSFER_ROOT="$REPO_ROOT/save/i3_sam_only16_64_seed60_$STAMP"
TRANSFER_LOG="$REPO_ROOT/logs/i3_sam_only16_64_seed60_$STAMP.log"
echo "1/4 检查原 SHC 环境并运行 SAM/NFD/迁移测试"
conda run --no-capture-output -n SHC python -u -c \
  'import sys,torch; i=int(sys.argv[1]); assert torch.cuda.is_available(), "CUDA 不可用"; print(torch.__version__,torch.cuda.get_device_name(i)); print((torch.ones(1,device="cuda:"+str(i))+1).item())' "$GPU"
conda run --no-capture-output -n SHC python -m unittest \
  tests.test_i3_sam tests.test_i3_sam_transfer tests.test_b2nfd_feature_distill -q
ARGS=(scripts/run_i3_sam_transfer.py --baseline-root "$BASELINE_ROOT" \
      --dataset-anchor-run "$ANCHOR_RUN" --output-root "$TRANSFER_ROOT" --gpu "$GPU")
echo "2/4 检查两个码长各自的配置和缓存，以及共同的数据身份"
conda run --no-capture-output -n SHC python -u "${ARGS[@]}" --dry-run
echo "3/4 分别运行16bit和64bit临时GPU批次检查"
conda run --no-capture-output -n SHC python -u "${ARGS[@]}" --smoke-test
echo "4/4 后台只启动16bit SAM -> 64bit SAM，各300轮；原NFD对照不重跑"
printf '%s\n' "$TRANSFER_ROOT" > logs/i3_sam_transfer_latest.path
printf '%s\n' "$TRANSFER_LOG" > logs/i3_sam_transfer_latest.log_path
nohup conda run --no-capture-output -n SHC python -u "${ARGS[@]}" \
  > "$TRANSFER_LOG" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/i3_sam_transfer_latest.pid
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$!" "$TRANSFER_LOG" "$TRANSFER_ROOT"
