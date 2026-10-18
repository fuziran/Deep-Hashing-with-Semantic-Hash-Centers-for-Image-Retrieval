#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8

NABIRDS_ROOT=/mnt/d/fzr/SHC/data/NABirds-B/NAbirds-new-seg/images
B1_OUTPUT=./save_nabirds_b_i1
NFD_OUTPUT=./save_nabirds_b_i2_nfd

if [[ ! -d "$NABIRDS_ROOT/train" || ! -d "$NABIRDS_ROOT/query" || ! -d "$NABIRDS_ROOT/database" ]]; then
  echo "NABirds-B train/query/database directories are missing: $NABIRDS_ROOT" >&2
  exit 1
fi

mkdir -p "$NFD_OUTPUT/logs"
python -m unittest discover -s tests -q

for bits in 16 32 64; do
  case "$bits" in
    16) parent_run=NABirds-B_B1_16bit_seed60_dc78fa39187b ;;
    32) parent_run=NABirds-B_B1_32bit_seed60_28e260e36f07 ;;
    64) parent_run=NABirds-B_B1_64bit_seed60_380c6e8224f2 ;;
  esac
  parent_config="$B1_OUTPUT/runs/$parent_run/config.json"
  if [[ ! -f "$parent_config" ]]; then
    echo "Missing B1 parent configuration: $parent_config" >&2
    exit 1
  fi

  mapfile -t cache_paths < <(python - "$parent_config" "$B1_OUTPUT" <<'PY'
import argparse
import json
import sys
from pathlib import Path

from utils.experiment import build_cache_metadata

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = Path(sys.argv[2])
parent = argparse.Namespace(**config)
classifier_hash = build_cache_metadata(parent, "classifier")["config_hash"]
center_hash = build_cache_metadata(parent, "centers")["config_hash"]
print(base / "SimilarityMatrix" / f"NABirds-B_{config['similarity_hash']}.pt")
print(base / "HashCenters" / f"NABirds-B_SHC_{center_hash}.pt")
print(base / "ClassificationNet" / f"NABirds-B_{classifier_hash}.pt")
PY
  )
  if [[ ${#cache_paths[@]} -ne 3 ]]; then
    echo "Could not derive the three B1 cache paths for $bits bits" >&2
    exit 1
  fi
  for path in "${cache_paths[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing B1 cache: $path" >&2
      exit 1
    fi
  done

  echo "===== Starting NABirds-B I2-NFD ${bits}-bit run ====="
  python -u run.py \
    --variant b2nfd \
    --parent-b1-config "$parent_config" \
    --similarity-cache "${cache_paths[0]}" \
    --center-cache "${cache_paths[1]}" \
    --teacher-cache "${cache_paths[2]}" \
    --dataset NABirds-B \
    --root "$NABIRDS_ROOT" \
    --num-classes 555 \
    --code-length "$bits" \
    --seed 60 \
    --lr 1e-4 \
    --classify-epoch 300 \
    --epoch 300 \
    --test-map 5 \
    --beta 1.0 \
    --lambd 1e-4 \
    --batch-size 64 \
    --num-workers 0 \
    --resize-size 256 \
    --crop-size 224 \
    --val-per-class 2 \
    --mask-strategy predicted_argmax \
    --center-update-strategy monotonic_discrete_search \
    --relation-mode full \
    --relation-kappa 30 \
    --relation-temperature 1 \
    --relation-weight-floor 0.05 \
    --nfd-alpha 0.75 \
    --nfd-weight 0.01 \
    --nfd-topk 5 \
    --nfd-warmup 20 \
    --nfd-ramp-end 40 \
    --topK -1 100 1000 \
    --gpu 0 \
    --output-dir "$NFD_OUTPUT" \
    2>&1 | tee "$NFD_OUTPUT/logs/${bits}bit.log"
done
