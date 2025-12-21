# Hi!ckathon 2025

This repository contains my machine learning pipeline built for the Hi!ckathon 2025 challenge using a PISA tabular dataset. The dataset is not included in this repository.

## Result

I reached **joint second** in the **technical track**, achieving an **R2 of 0.79** on the final evaluation.

## Problem overview

Goal: predict `MathScore`, a continuous target.

A key property of the dataset is a large mass at exactly zero (`MathScore = 0`). To handle this, the solution is implemented as a two-stage model:

1. **Zero classifier**: predict whether `MathScore` is zero or non-zero (binary classification, optimized for F1).
2. **Regressor**: predict `MathScore` as a continuous value (regression, optimized for R2).
3. **Gating / post-processing**: if the classifier predicts zero, the final prediction is forced to `0.0`; otherwise we keep the regressor prediction.

## Data leakage and preprocessing

During early iterations, we identified **data leakage** through features that were directly or indirectly derived from the target. These features artificially boosted offline performance but did not generalize reliably.

To address this, the preprocessing step includes **dropping leakage-prone columns**, such as aggregated per-question scores/timings. Removing these columns improved the validity of the evaluation and ensured the pipeline matches a realistic prediction setting.

## Alternative architecture explored (not retained)

Before converging to the final two-stage pipeline above, we experimented with a different approach:

- First, detect zeros with a binary model.
- Then, train a regression model **only on non-zero samples** by filtering out the rows where `MathScore == 0`.

In practice, this approach was **not conclusive**: training the regressor only on non-zero samples led to significantly worse performance, reaching **R2 = 0.69**, compared to the final gated approach. The final pipeline was therefore retained.

## Repository structure

hi-paris-2025/
├── .gitignore                          # Excludes data, CSVs, AutoGluon artifacts, logs, etc.
├── LICENSE                             
├── README.md                           
│
├── notebooks/                          # Exploration and preprocessing notebooks (no dataset in repo)
│   ├── exploration.ipynb             
│   └── preprocessing.ipynb             # Preprocessing and leakage-related feature analysis
│
├── scripts/                            # SLURM job scripts (sbatch) for cluster execution
│   ├── train_zero_classifier.sh        # Train the zero vs non-zero classifier
│   ├── train_regressor.sh              # Train the MathScore regressor
│   ├── run_zero_inference.sh           # Run zero-classifier inference on X_test_clean
│   └── run_regressor_inference.sh      # Run regressor inference on X_test_clean
│
└── src/
    └── models/                         
        ├── zero_classifier.py          # Train binary model: MathScore_is_non_zero (metric: F1)
        ├── zero_classifier_inference.py# Infer 0/1 on X_test_clean
        ├── mathscore_regressor.py      # Train regression model for MathScore (metric: R2)
        ├── mathscore_regressor_inference.py # Infer continuous MathScore on X_test_clean (ID, MathScore)
        ├── merge_mathscore_predictions.py   predictions
        └── eval_r2.py                  # Compute R2 on y_test vs final predictions

## Environment setup

Recommended:
- Python 3.12
- AutoGluon 1.4.0
- pandas, numpy

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install autogluon
