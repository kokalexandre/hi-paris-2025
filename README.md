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

- `src/models/`
  - `zero_classifier.py`: trains an AutoGluon binary classifier (`MathScore_is_non_zero`) with `eval_metric=f1`
  - `zero_classifier_inference.py`: runs inference with the trained classifier
  - `mathscore_regressor.py`: trains an AutoGluon regressor with `eval_metric=r2`
  - `mathscore_regressor_inference.py`: runs inference with the trained regressor
  - `merge_mathscore_predictions.py`: combines regressor + classifier outputs into final predictions
  - `eval_r2.py`: computes R2 on a labeled test set
- `scripts/`: SLURM job scripts (sbatch) to train and run inference on a cluster
- `notebooks/`: exploration and preprocessing notebooks

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
