#!/bin/bash
#SBATCH --job-name=ddm-grid
#SBATCH --output=logs/ddm_grid_%A_%a.out
#SBATCH --error=logs/ddm_grid_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-2186

set -euo pipefail

mkdir -p logs grid_results

python grid_search_ddm.py \
  --grid-index "${SLURM_ARRAY_TASK_ID}" \
  --grid-json grid_config.json \
  --block 0.2 \
  --conditions both \
  --sample-size 5000 \
  --output-dir grid_results
