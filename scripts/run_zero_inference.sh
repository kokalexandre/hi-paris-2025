#!/bin/bash
#SBATCH --job-name=zero_infer
#SBATCH --partition=preemptable
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G

source .venv/bin/activate

python src/models/zero_classifier_inference.py \
    --x_test_path data/processed/X_test_clean.csv \
    --model_dir autogluon_zero_classifier \
    --output_path submissions/zero_classifier_predictions.csv \

