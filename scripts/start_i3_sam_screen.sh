#!/usr/bin/env bash
# Tests -> data/cache preflight -> disposable GPU batch -> sequential 300-epoch pair.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
NFD_CONFIG="${1:-$REPO_ROOT/save_nfd_seed60_bINBuclZ/nfd/runs/cifar-100-new-seg_B2NFD_32bit_seed60_639b7ea1eaa3/config.json}"
GPU="${2:-0}"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda 不在当前 PATH 中；请先打开可使用 conda 的终端。" >&2
  exit 1
fi
if [[ ! -f "$NFD_CONFIG" || ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "NFD 配置不存在，或 GPU 编号无效：$NFD_CONFIG / $GPU" >&2
  exit 1
fi
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SCREEN_ROOT="$REPO_ROOT/save/i3_sam_screen32_seed60_$STAMP"
SCREEN_LOG="$REPO_ROOT/logs/i3_sam_screen32_seed60_$STAMP.log"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/tmp"

echo "1/5 检查原 SHC 环境及 GPU"
conda run --no-capture-output -n SHC python -u -c \
  'import sys, torch; i=int(sys.argv[1]); print("PyTorch:",torch.__version__); assert torch.cuda.is_available(), "CUDA 不可用"; print("GPU:",torch.cuda.get_device_name(i)); print("GPU计算:",(torch.ones(1,device="cuda:"+str(i))+1).item())' "$GPU"

echo "2/5 运行 SAM 和原 NFD 合成测试"
conda run --no-capture-output -n SHC python -m unittest \
  tests.test_i3_sam tests.test_b2nfd_feature_distill -q

ARGS=(scripts/run_i3_sam_screen.py --nfd-config "$NFD_CONFIG" \
      --output-root "$SCREEN_ROOT" --gpu "$GPU")
echo "3/5 检查原始数据、B1 缓存和训练设置"
conda run --no-capture-output -n SHC python -u "${ARGS[@]}" --dry-run

echo "4/5 用一个临时 SAM 批次检查完整 NFD 下的 GPU 数值和显存"
conda run --no-capture-output -n SHC python -u "${ARGS[@]}" --smoke-test

echo "5/5 后台顺序启动 control 300轮 -> sam 300轮"
printf '%s\n' "$SCREEN_ROOT" > "$REPO_ROOT/logs/i3_sam_screen_latest.path"
printf '%s\n' "$SCREEN_LOG" > "$REPO_ROOT/logs/i3_sam_screen_latest.log_path"
nohup conda run --no-capture-output -n SHC python -u "${ARGS[@]}" \
  > "$SCREEN_LOG" 2>&1 < /dev/null &
SCREEN_PID=$!
printf '%s\n' "$SCREEN_PID" > "$REPO_ROOT/logs/i3_sam_screen_latest.pid"
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$SCREEN_PID" "$SCREEN_LOG" "$SCREEN_ROOT"
echo "详细训练日志分别为结果目录下的 control.log 和 sam.log。"
