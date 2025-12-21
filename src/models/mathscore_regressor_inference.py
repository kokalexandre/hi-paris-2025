import argparse
import os

import pandas as pd
from autogluon.tabular import TabularPredictor


def detect_id_column(df: pd.DataFrame) -> str:
    for col in ["Unnamed: 0", "ID", "id"]:
        if col in df.columns:
            return col
    raise ValueError("Aucune colonne ID trouvée.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Inférence du modèle autogluon_mathscore_regressor sur X_test_clean.csv"
    )
    p.add_argument("--x_test_path", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    X_test = pd.read_csv(args.x_test_path)
    id_col = detect_id_column(X_test)

    ids = X_test[id_col]
    X_features = X_test.drop(columns=[id_col])

    predictor = TabularPredictor.load(args.model_dir)

    preds = predictor.predict(X_features)
    preds = pd.Series(preds, name="MathScore")

    out_df = pd.DataFrame({"ID": ids.astype(int).to_numpy(), "MathScore": preds.to_numpy()})

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(args.output_path, index=False)
    print(f"CSV sauvegardé : {args.output_path}")


if __name__ == "__main__":
    main()
