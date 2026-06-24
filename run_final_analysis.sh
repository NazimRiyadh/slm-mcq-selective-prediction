#!/usr/bin/env bash
set -euo pipefail

python code/final_analysis/build_additional_analyses.py \
  --data-dir code/final_analysis/input_data \
  --out-dir code/final_analysis/analysis_outputs \
  --tables-dir paper/tables \
  --figures-dir paper/figures

python code/final_analysis/strict_feature_audit.py \
  --input code/final_analysis/input_data/locked_primary_selected.csv \
  --out-dir code/final_analysis/analysis_outputs \
  --bootstrap-reps 50000 \
  --seed 20260623

mkdir -p results/final_analysis
cp -f code/final_analysis/analysis_outputs/*.csv results/final_analysis/

echo "Final analyses completed successfully."
