#!/bin/bash
#SBATCH --job-name=ddm-fit-lr
#SBATCH --output=logs/ddm_fit_lr_%j.out
#SBATCH --error=logs/ddm_fit_lr_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

mkdir -p logs fit_results

python fit_left_right_ddm.py \
  --condition no_wiggle \
  --block 0.2 \
  --sample-size 5000 \
  --output-json fit_results/no_wiggle_left_right_fit.json
