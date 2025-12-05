import argparse
import os

import pandas as pd
from autogluon.tabular import TabularPredictor


def detect_id_column(df):
    for col in ["Unnamed: 0", "Unamed: 0", "ID", "id"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Aucune colonne id trouvée (attendu par ex. 'Unnamed: 0' ou 'Unamed: 0')."
    )


def get_resources_from_slurm():
    def to_int(val):
        if val is None:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    num_cpus = None
    for var in ["SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"]:
        num_cpus = to_int(os.getenv(var))
        if num_cpus is not None:
            break

    num_gpus = None
    for var in ["SLURM_GPUS_PER_TASK", "SLURM_GPUS", "SLURM_GPUS_ON_NODE"]:
        num_gpus = to_int(os.getenv(var))
        if num_gpus is not None:
            break

    return num_cpus, num_gpus


def prepare_training_data(x_path, y_path):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    id_col = detect_id_column(X)
    if id_col not in y.columns:
        raise ValueError(
            f"La colonne id '{id_col}' n'est pas présente dans y_train."
        )

    target_candidates = [c for c in y.columns if c != id_col]
    if len(target_candidates) != 1:
        raise ValueError(
            "y_train doit contenir exactement une colonne cible en plus de l'id."
        )
    target_col = target_candidates[0]

    y_binary_col = "MathScore_is_non_zero"
    y[y_binary_col] = (y[target_col] != 0).astype(int)

    train_df = X.merge(y[[id_col, y_binary_col]], on=id_col, how="inner")
    train_df = train_df.drop(columns=[id_col])

    return train_df, y_binary_col


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero vs non-zero classifier pour MathScore avec AutoGluon."
    )
    parser.add_argument(
        "--x_train_path",
        type=str,
        required=True,
        help="Chemin vers X_train_clean.csv",
    )
    parser.add_argument(
        "--y_train_path",
        type=str,
        required=True,
        help="Chemin vers y_train.csv (contient id + MathScore continu).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Répertoire de sortie pour le modèle AutoGluon.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=None,
        help="Temps limite (en secondes) pour l'entraînement AutoGluon.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_data, label_col = prepare_training_data(
        args.x_train_path, args.y_train_path
    )

    num_cpus, num_gpus = get_resources_from_slurm()
    print(f"Ressources détectées -> num_cpus={num_cpus}, num_gpus={num_gpus}")
    print(f"Time limit = {args.time_limit}")

    predictor = TabularPredictor(
        label=label_col,
        problem_type="binary",
        eval_metric="f1",
        positive_class=1,
        path=args.output_dir,
    )

    fit_kwargs = {
        "train_data": train_data,
        "presets": "best_quality",
        "verbosity": 2,
    }
    if args.time_limit is not None:
        fit_kwargs["time_limit"] = args.time_limit
    if num_cpus is not None:
        fit_kwargs["num_cpus"] = num_cpus
    if num_gpus is not None and num_gpus > 0:
        fit_kwargs["num_gpus"] = num_gpus

    predictor.fit(**fit_kwargs)

    print(f"Modèle entraîné et sauvegardé dans : {args.output_dir}")


if __name__ == "__main__":
    main()
