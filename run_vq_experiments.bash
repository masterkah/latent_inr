#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments}"
PLOTS_DIR="$OUT_DIR"
USE_AMP_TF32="${USE_AMP_TF32:-0}"

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(
    #"$ROOT_DIR/config/vqinr_siren_3000.json"
    "$ROOT_DIR/config/vqinr_relu_3000_codebook_64.json"
    "$ROOT_DIR/config/vqinr_relu_3000_codebook_128.json"
    "$ROOT_DIR/config/vqinr_relu_3000_codebook_256.json"
    "$ROOT_DIR/config/vqinr_relu_3000_codebook_512.json"
  )
fi

PSNR_FILES=()
CODEBOOK_FILES=()

for cfg in "${CONFIGS[@]}"; do
  if [ ! -f "$cfg" ]; then
    echo "Config not found: $cfg" >&2
    exit 1
  fi
  name="$(basename "$cfg")"
  name="${name%.json}"

  echo "==> Training $name"
  python "$ROOT_DIR/train_vq.py" -config "$cfg" -output-folder "$OUT_DIR" -use-amp-tf32 "$USE_AMP_TF32"

  run_dir="$OUT_DIR/run_vq_${name}"
  psnr_file="$run_dir/psnr_history.csv"
  codebook_file="$run_dir/codebook_usage.csv"

  if [ -f "$psnr_file" ]; then
    PSNR_FILES+=("$psnr_file")
  else
    echo "Warning: PSNR file missing: $psnr_file" >&2
  fi

  if [ -f "$codebook_file" ]; then
    CODEBOOK_FILES+=("$codebook_file")
  else
    echo "Warning: codebook file missing: $codebook_file" >&2
  fi

done

echo "==> Plotting"
VIS_CMD=(python "$ROOT_DIR/src/vq_visualization.py" --out-dir "$PLOTS_DIR" --psnr-avg --max-points 300)
if [ ${#PSNR_FILES[@]} -gt 0 ]; then
  VIS_CMD+=(--psnr-files "${PSNR_FILES[@]}")
fi
if [ ${#CODEBOOK_FILES[@]} -gt 0 ]; then
  VIS_CMD+=(--codebook-files "${CODEBOOK_FILES[@]}")
fi
"${VIS_CMD[@]}"

echo "Done. Plots saved to: $PLOTS_DIR"
