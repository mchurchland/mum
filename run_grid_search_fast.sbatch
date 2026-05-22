#!/bin/bash
#SBATCH --job-name=ddm-grid-fast
#SBATCH --output=logs/ddm_grid_fast_%A_%a.out
#SBATCH --error=logs/ddm_grid_fast_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-2186%16

set -euo pipefail

mkdir -p logs grid_results_fast

python grid_search_ddm.py \
  --grid-index "${SLURM_ARRAY_TASK_ID}" \
  --grid-json grid_config.json \
  --block 0.2 \
  --conditions both \
  --sample-size 5000 \
  --dx 0.04 \
  --dt 0.004 \
  --skip-likelihood \
  --output-dir grid_results_fast
