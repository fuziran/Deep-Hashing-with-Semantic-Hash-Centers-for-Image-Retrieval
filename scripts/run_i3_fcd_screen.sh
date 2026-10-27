#!/usr/bin/env bash
# Validation-only seed-60/32-bit screen. Uses two existing Conda environments.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo 'Usage: bash scripts/run_i3_fcd_screen.sh NFD32_CONFIG DINO_REPO DINO_WEIGHTS NEW_OUTPUT_ROOT' >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."
NFD_CONFIG="$(realpath -e -- "$1")"
FOUNDATION_REPO="$(realpath -e -- "$2")"
FOUNDATION_WEIGHTS="$(realpath -e -- "$3")"
OUTPUT_ROOT="$(realpath -m -- "$4")"
TRAIN_ENV="${SHC_TRAIN_ENV:-SHC}"
TEACHER_ENV="${SHC_TEACHER_ENV:-SHC_FCD_teacher}"
GPU_ID="${SHC_GPU:-0}"
EXTRACTION_BATCH="${SHC_EXTRACTION_BATCH:-32}"
CONDA_COMMAND="${CONDA_EXE:-conda}"

test -f "$NFD_CONFIG"
test -f "$(dirname -- "$NFD_CONFIG")/metrics.json"
test -f "$(dirname -- "$NFD_CONFIG")/training_history.json"
test -f "$FOUNDATION_REPO/hubconf.py"
test -f "$FOUNDATION_WEIGHTS"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Output already exists; choose a fresh directory: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p -- "$OUTPUT_ROOT"
printf '%s\n' "$NFD_CONFIG" > "$OUTPUT_ROOT/nfd_reference.txt"

echo "1/4 Checking original SHC environment. Log: $OUTPUT_ROOT/tests.log"
"$CONDA_COMMAND" run --no-capture-output -n "$TRAIN_ENV" \
  python -m unittest discover -s tests -v > "$OUTPUT_ROOT/tests.log" 2>&1
"$CONDA_COMMAND" run --no-capture-output -n "$TRAIN_ENV" python -c \
  'import json,sys; c=json.load(open(sys.argv[1])); assert (c["variant"],c["code_length"],c["seed"])==("b2nfd",32,60), "Expected seed-60 32-bit NFD reference"' \
  "$NFD_CONFIG"

echo "2/4 Preparing/checking teacher. Log: $OUTPUT_ROOT/teacher.log"
"$CONDA_COMMAND" run --no-capture-output -n "$TEACHER_ENV" \
  python -u scripts/prepare_foundation_teacher.py \
  --nfd-config "$NFD_CONFIG" \
  --foundation-repo "$FOUNDATION_REPO" --foundation-model dinov3_vitb16 \
  --foundation-weights "$FOUNDATION_WEIGHTS" \
  --extraction-batch-size "$EXTRACTION_BATCH" --probe-steps 200 --gpu "$GPU_ID" \
  --output-dir "$OUTPUT_ROOT/teacher" > "$OUTPUT_ROOT/teacher.log" 2>&1
# Preparation exits nonzero on failed information checks; set -e stops here.

COMMON=(
  --nfd-config "$NFD_CONFIG" --bits 32 --seed 60 --gpu "$GPU_ID"
  --teacher-cache "$OUTPUT_ROOT/teacher/foundation_cache.pt"
  --weight 0.05 --temperature 2 --gamma 1 --topk 5 --warmup 40 --ramp-end 60
)
echo '3/4 Preflighting ce, plain, neighbor (no query evaluation).'
for mode in ce plain neighbor; do
  "$CONDA_COMMAND" run --no-capture-output -n "$TRAIN_ENV" \
    python scripts/run_i3_fcd.py "${COMMON[@]}" --mode "$mode" \
    --output-root "$OUTPUT_ROOT/$mode" --dry-run \
    > "$OUTPUT_ROOT/$mode.preflight.log" 2>&1
done

echo '4/4 Training ce -> plain -> neighbor sequentially (validation only).'
for mode in ce plain neighbor; do
  echo "Starting $mode. Log: $OUTPUT_ROOT/$mode/32bit.log"
  "$CONDA_COMMAND" run --no-capture-output -n "$TRAIN_ENV" \
    python -u scripts/run_i3_fcd.py "${COMMON[@]}" --mode "$mode" \
    --output-root "$OUTPUT_ROOT/$mode" > "$OUTPUT_ROOT/$mode.launch.log" 2>&1
done

# All three validation metrics must come from each run's selected epoch.
"$CONDA_COMMAND" run --no-capture-output -n "$TRAIN_ENV" python - "$NFD_CONFIG" "$OUTPUT_ROOT" <<'PY'
import csv
import json
from pathlib import Path
import sys

reference, output = map(Path, sys.argv[1:])
rows = []
for name in ('nfd', 'ce', 'plain', 'neighbor'):
    if name == 'nfd':
        run = reference.parent
    else:
        with (output/name/'summary.csv').open(newline='') as stream:
            summaries = list(csv.DictReader(stream))
        if len(summaries) != 1:
            raise ValueError(f'Expected one 32-bit run for {name}')
        run = Path(summaries[0]['run_dir'])
    metrics = json.loads((run/'metrics.json').read_text())
    history = json.loads((run/'training_history.json').read_text())
    selected = next(row for row in history if row['epoch'] == metrics['selected_epoch'])['validation']
    rows.append(dict(method=name, run_dir=str(run), selected_epoch=metrics['selected_epoch'],
                     **{key: selected[key] for key in ('mAP@ALL', 'mAP@100', 'mAP@1000')}))
with (output/'validation_comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY
echo "Complete. Validation comparison: $OUTPUT_ROOT/validation_comparison.csv"
