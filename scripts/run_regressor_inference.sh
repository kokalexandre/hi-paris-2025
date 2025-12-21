#!/bin/bash
#SBATCH --job-name=math_infer
#SBATCH --partition=preemptable
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

source .venv/bin/activate

python src/models/mathscore_regressor_inference.py \
  --x_test_path data/processed/X_test_clean.csv \
  --model_dir autogluon_mathscore_regressor \
  --output_path submissions/mathscore_regressor_predictions.csv
