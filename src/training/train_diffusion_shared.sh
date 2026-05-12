#!/usr/bin/env bash
set -euo pipefail

# go to submission root
cd "$(dirname "$0")/../.."

# path to original project repo that contains robomimic
PROJECT_ROOT="/home/chetana/immimic_project/ImMimic-CoRL2025"

echo "============================================================"
echo "Training Diffusion Policy on merged shared co-training dataset"
echo "============================================================"
echo "Submission root: $(pwd)"
echo "Project root: ${PROJECT_ROOT}"

# make original repo importable
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/robomimic/robomimic/scripts/train.py" \
  --config src/training/diffusion_shared_cotrain.json
