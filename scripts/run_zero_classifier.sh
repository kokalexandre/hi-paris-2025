#!/bin/bash
#SBATCH --job-name=zero_clf
#SBATCH --time=10:00:00
#SBATCH --partition=preemptable
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --output=logs/zero_clf_%j.out
#SBATCH --error=logs/zero_clf_%j.err

source .venv/bin/activate

python src/models/zero_classifier.py \
    --x_train_path data/processed/X_train_clean.csv \
    --y_train_path data/raw/y_train.csv \
    --output_dir autogluon_zero_classifier \
    --time_limit 28800
