#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DATA_ROOT="${1:-$REPO_ROOT/data/stanford_cars-new-seg/images}"
GPU="${2:-0}"
ENV_NAME="${SHC_ENV:-SHC}"
if [[ ! -d "$DATA_ROOT" || ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "Cars-B 数据目录不存在或 GPU 编号无效：$DATA_ROOT / $GPU" >&2
  exit 1
fi
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUTPUT="$REPO_ROOT/save_stanford_cars_b0_$STAMP"
LOG="$REPO_ROOT/logs/stanford_cars_b0_$STAMP.log"
printf '%s\n' "$OUTPUT" > logs/stanford_cars_b0_latest.path
printf '%s\n' "$LOG" > logs/stanford_cars_b0_latest.log_path
printf '预检与训练进度写入此日志\n' > "$LOG"
nohup conda run --no-capture-output -n "$ENV_NAME" \
  python -u scripts/run_stanford_cars_b0.py --root "$DATA_ROOT" \
  --output-root "$OUTPUT" --gpu "$GPU" >> "$LOG" 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/stanford_cars_b0_latest.pid
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$!" "$LOG" "$OUTPUT"
