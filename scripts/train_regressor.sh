#!/bin/bash
#SBATCH --job-name=math_reg
#SBATCH --partition=preemptable
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1             
#SBATCH --mem=0
#SBATCH --output=logs/math_reg_%j.out
#SBATCH --error=logs/math_reg_%j.err

source .venv/bin/activate

python src/models/mathscore_regressor.py \
    --x_train_path data/processed/X_train_clean.csv \
    --y_train_path data/raw/y_train.csv \
    --output_dir autogluon_mathscore_regressor \
    --num_bag_folds 4 \
    --num_stack_levels 1 \
    --time_limit 36000 

