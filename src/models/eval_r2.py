import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Évalue le R2 sur y_test vs predictions.")
    p.add_argument("--y_test_path", type=str, required=True, help="CSV: ID + MathScore (vrai)")
    p.add_argument("--pred_path", type=str, required=True, help="CSV: ID,MathScore (prédit)")
    return p.parse_args()


def r2_score_np(y_true, y_pred):
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main():
    args = parse_args()

    y_test_raw = pd.read_csv(args.y_test_path)
    if y_test_raw.shape[1] < 2:
        raise ValueError("y_test doit contenir au moins 2 colonnes (ID + target).")

    y_test = y_test_raw.iloc[:, :2].copy()
    y_test.columns = ["ID", "MathScore_true"]

    pred_raw = pd.read_csv(args.pred_path)
    if pred_raw.shape[1] < 2:
        raise ValueError("predictions doit contenir au moins 2 colonnes (ID, MathScore).")

    pred = pred_raw.iloc[:, :2].copy()
    pred.columns = ["ID", "MathScore_pred"]

    y_test["ID"] = y_test["ID"].astype(int)
    pred["ID"] = pred["ID"].astype(int)

    merged = y_test.merge(pred, on="ID", how="inner", validate="one_to_one")

    if len(merged) != len(y_test) or len(merged) != len(pred):
        raise ValueError(
            f"Mismatch IDs / lignes après merge: y_test={len(y_test)}, preds={len(pred)}, merged={len(merged)}"
        )

    y_true = pd.to_numeric(merged["MathScore_true"], errors="raise").to_numpy()
    y_pred = pd.to_numeric(merged["MathScore_pred"], errors="raise").to_numpy()

    score = r2_score_np(y_true, y_pred)
    print(f"R2: {score}")


if __name__ == "__main__":
    main()
