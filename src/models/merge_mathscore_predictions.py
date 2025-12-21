import argparse
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge des prédictions régression et zero-classifier pour produire MathScore final."
    )
    p.add_argument("--reg_path", type=str, required=True, help="CSV: ID,MathScore (régression)")
    p.add_argument(
        "--zero_path",
        type=str,
        required=True,
        help="CSV: ID + colonne binaire (0/1) du zero-classifier",
    )
    p.add_argument("--output_path", type=str, required=True, help="CSV sortie: ID,MathScore")
    return p.parse_args()


def find_id_col(df: pd.DataFrame) -> str:
    for c in ["ID", "Unnamed: 0", "Unamed: 0", "id"]:
        if c in df.columns:
            return c
    return df.columns[0]


def main():
    args = parse_args()

    reg = pd.read_csv(args.reg_path)
    zero = pd.read_csv(args.zero_path)

    if len(reg) != len(zero):
        raise ValueError(f"Nombre de lignes différent: reg={len(reg)} vs zero={len(zero)}")

    reg_id = find_id_col(reg)
    zero_id = find_id_col(zero)

    reg_ids = reg[reg_id].astype(int).to_numpy()
    zero_ids = zero[zero_id].astype(int).to_numpy()

    if reg_ids.shape != zero_ids.shape or (reg_ids != zero_ids).any():
        raise ValueError("Les IDs ne correspondent pas (valeurs ou ordre différents) entre les deux fichiers.")

    reg_target_candidates = [c for c in reg.columns if c != reg_id]
    if len(reg_target_candidates) != 1:
        raise ValueError("Le fichier régression doit contenir exactement 2 colonnes: ID et MathScore.")
    reg_score_col = reg_target_candidates[0]

    zero_pred_candidates = [c for c in zero.columns if c != zero_id]
    if len(zero_pred_candidates) < 1:
        raise ValueError("Le fichier zero-classifier doit contenir au moins 2 colonnes: ID et prediction.")
    zero_pred_col = zero_pred_candidates[0]

    reg_scores = pd.to_numeric(reg[reg_score_col], errors="coerce").to_numpy()
    zero_preds = pd.to_numeric(zero[zero_pred_col], errors="coerce").to_numpy()

    if pd.isna(reg_scores).any():
        raise ValueError("NaN détectés dans les prédictions de régression (colonne MathScore).")
    if pd.isna(zero_preds).any():
        raise ValueError("NaN détectés dans les prédictions du zero-classifier.")
    if not set(pd.unique(zero_preds)).issubset({0, 1}):
        raise ValueError(f"La prédiction zero-classifier doit être binaire 0/1. Valeurs trouvées: {sorted(set(zero_preds))[:10]}")

    final_scores = reg_scores.astype(float)
    final_scores[zero_preds.astype(int) == 0] = 0.0

    out = pd.DataFrame({"ID": reg_ids, "MathScore": final_scores})

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out.to_csv(args.output_path, index=False)
    print(f"Fichier écrit : {args.output_path}")


if __name__ == "__main__":
    main()
