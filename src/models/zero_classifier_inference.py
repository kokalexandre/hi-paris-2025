import argparse
import os

import pandas as pd
from autogluon.tabular import TabularPredictor


def detect_id_column(df):
    for col in ["Unnamed: 0", "ID", "id"]:
        if col in df.columns:
            return col
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inférence du modèle autogluon_zero_classifier sur X_test_clean.csv"
    )
    parser.add_argument(
        "--x_test_path",
        type=str,
        required=True,
        help="Chemin vers X_test_clean.csv",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="autogluon_zero_classifier",
        help="Répertoire du modèle AutoGluon entraîné",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Chemin du fichier .csv de sortie avec les prédictions",
    )
    parser.add_argument(
        "--include_proba",
        action="store_true",
        help="Si spécifié, ajoute aussi la probabilité de la classe 1 (non-zero).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    X_test = pd.read_csv(args.x_test_path)

    id_col = detect_id_column(X_test)
    if id_col is not None:
        ids = X_test[id_col].copy()
        X_features = X_test.drop(columns=[id_col])
    else:
        ids = None
        X_features = X_test

    predictor = TabularPredictor.load(args.model_dir)

    preds = predictor.predict(X_features)

    out_df = pd.DataFrame()
    if ids is not None:
        out_df[id_col] = ids
    out_df["MathScore_is_non_zero_pred"] = preds

    if args.include_proba:
        proba = predictor.predict_proba(X_features)[1]
        out_df["MathScore_is_non_zero_proba"] = proba

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_df.to_csv(args.output_path, index=False)

    print(f"Prédictions sauvegardées dans : {args.output_path}")


if __name__ == "__main__":
    main()
