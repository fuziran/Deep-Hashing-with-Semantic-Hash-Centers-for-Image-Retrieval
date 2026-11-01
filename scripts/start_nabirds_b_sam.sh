#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
BASELINE_ROOT="${1:-$REPO_ROOT/save_nabirds_b_i2_nfd}"
GPU="${2:-0}"
ENV_NAME="${SHC_ENV:-SHC}"
if [[ ! -d "$BASELINE_ROOT" || ! "$GPU" =~ ^[0-9]+$ ]]; then
  printf '原NABirds-B NFD结果目录不存在，或GPU编号无效：%s / %s\n' "$BASELINE_ROOT" "$GPU" >&2
  exit 1
fi
mkdir -p logs tmp
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUTPUT_ROOT="$REPO_ROOT/save/nabirds_b_nfd_sam_seed60_$STAMP"
LOG="$REPO_ROOT/logs/nabirds_b_nfd_sam_seed60_$STAMP.log"
echo "1/4 检查原实验环境和GPU，运行NABirds/SAM回归测试"
conda run --no-capture-output -n "$ENV_NAME" python -u -c \
  'import sys,torch; i=int(sys.argv[1]); assert torch.cuda.is_available(), "CUDA 不可用"; print(torch.__version__,torch.cuda.get_device_name(i)); print((torch.ones(1,device="cuda:"+str(i))+1).item())' "$GPU"
conda run --no-capture-output -n "$ENV_NAME" python -m unittest discover -s tests -q
ARGS=(scripts/run_nabirds_b_sam.py --baseline-root "$BASELINE_ROOT" \
      --bits 16 32 64 --output-root "$OUTPUT_ROOT" --gpu "$GPU")
echo "2/4 核对原NABirds-B划分、555类配置及各码长B1/NFD缓存；文件内容校验可能需要数分钟"
conda run --no-capture-output -n "$ENV_NAME" python -u "${ARGS[@]}" --dry-run
echo "3/4 分别运行16/32/64bit临时SAM批次，检查完整NFD下的GPU显存与数值"
conda run --no-capture-output -n "$ENV_NAME" python -u "${ARGS[@]}" --smoke-test
echo "4/4 后台仅启动16bit SAM -> 32bit SAM -> 64bit SAM；已有对照和教师不重训"
printf '%s\n' "$OUTPUT_ROOT" > logs/nabirds_b_sam_latest.path
printf '%s\n' "$LOG" > logs/nabirds_b_sam_latest.log_path
nohup conda run --no-capture-output -n "$ENV_NAME" python -u "${ARGS[@]}" \
  > "$LOG" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/nabirds_b_sam_latest.pid
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$!" "$LOG" "$OUTPUT_ROOT"
