#!/bin/bash
#SBATCH --job-name=ddm-grid
#SBATCH --output=logs/ddm_grid_%A_%a.out
#SBATCH --error=logs/ddm_grid_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-15

set -euo pipefail

mkdir -p logs grid_results

GRID_SIZE=$(python grid_search_ddm.py --grid-json grid_config.json --print-grid-size)

for ((GRID_INDEX=SLURM_ARRAY_TASK_ID; GRID_INDEX<GRID_SIZE; GRID_INDEX+=16)); do
  python grid_search_ddm.py \
    --grid-index "${GRID_INDEX}" \
    --grid-json grid_config.json \
    --block 0.2 \
    --conditions both \
    --sample-size 5000 \
    --output-dir grid_results
done
