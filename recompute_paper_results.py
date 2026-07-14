from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


MODEL_ORDER = ("Flux", "GPT", "Gemini", "SDXL")
GROUND_TRUTH = {
    "Flux": "Female",
    "GPT": "Male",
    "Gemini": "Female",
    "SDXL": "Male",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun SI-FPM feature separability and generated-image driver outputs."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "IMDB - Paper de Metricas Faciais/Gerar CSV - IMDB/"
            "outputs_imdb_shpm/outputs_imdb_shpm/imdb_shpm_reference_valid.csv"
        ),
        help="IMDB reference CSV with gender and 30 SI-FPM features.",
    )
    parser.add_argument(
        "--generated-features",
        type=Path,
        default=Path(
            "IMDB - Paper de Metricas Faciais/Comparação - IMDB/workspace (7)/"
            "orkspace/sam3/sam3/outputs_shpm_original/extracted_shpm_features.csv"
        ),
        help="CSV with already extracted SI-FPM features for generated images.",
    )
    parser.add_argument(
        "--generated-shpm",
        type=Path,
        default=Path(
            "IMDB - Paper de Metricas Faciais/Comparação - IMDB/workspace (7)/"
            "orkspace/sam3/sam3/outputs_shpm_original/shpm_original_results.csv"
        ),
        help="CSV with SHPM proximity scores for generated images.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("Imagem_Quadro"),
        help="Output directory consumed by gerar_quadro_shpm.py.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper"),
        help="Directory for manuscript-facing derived CSV summaries.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Stratified CV random seed.")
    return parser.parse_args()


def model_from_filename(filename: str) -> str:
    lower = filename.lower()
    if "flux" in lower:
        return "Flux"
    if "gemini" in lower:
        return "Gemini"
    if "sdxl" in lower:
        return "SDXL"
    if "gpt" in lower:
        return "GPT"
    return Path(filename).stem


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "filename",
        "gender",
        "area_face",
        "status",
        "error_message",
        "model",
    }
    return [
        column
        for column in df.columns
        if column not in excluded and "_vs_" in column
    ]


def numeric_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    return out


def cosine_rows(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    row_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    denom = row_norms * vector_norm
    denom[denom == 0] = np.nan
    scores = matrix @ vector / denom
    return np.nan_to_num(scores, nan=-1.0)


def evaluate_nearest_centroid(
    reference_df: pd.DataFrame,
    features: list[str],
    folds: int,
    seed: int,
) -> dict[str, float]:
    data = reference_df[["gender", *features]].dropna().copy()
    labels = data["gender"].to_numpy()
    x = data[features].to_numpy(dtype=float)
    y = np.array([1 if label == "Female" else 0 for label in labels])

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accuracies: list[float] = []

    for train_idx, test_idx in splitter.split(x, y):
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        female_centroid = x_train[y_train == 1].mean(axis=0)
        male_centroid = x_train[y_train == 0].mean(axis=0)
        female_scores = cosine_rows(x_test, female_centroid)
        male_scores = cosine_rows(x_test, male_centroid)
        predicted = (female_scores >= male_scores).astype(int)
        accuracies.append(float((predicted == y_test).mean() * 100))

    return {
        "n": float(len(data)),
        "female_n": float((data["gender"] == "Female").sum()),
        "male_n": float((data["gender"] == "Male").sum()),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies, ddof=1)),
    }


def compute_driver_scores(
    generated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    features: list[str],
    top_n: int = 3,
) -> dict[str, list[dict[str, float | str]]]:
    ref = reference_df[["gender", *features]].dropna().copy()
    female_rows = ref[ref["gender"] == "Female"]
    male_rows = ref[ref["gender"] == "Male"]
    eps = 1e-6

    stats: dict[str, dict[str, float]] = {}
    for feature in features:
        female_values = female_rows[feature].to_numpy(dtype=float)
        male_values = male_rows[feature].to_numpy(dtype=float)
        female_median = float(np.median(female_values))
        male_median = float(np.median(male_values))
        female_iqr = float(np.percentile(female_values, 75) - np.percentile(female_values, 25))
        male_iqr = float(np.percentile(male_values, 75) - np.percentile(male_values, 25))
        pooled_iqr = (female_iqr + male_iqr) / 2 + eps
        stats[feature] = {
            "female_median": female_median,
            "male_median": male_median,
            "female_iqr": max(female_iqr, eps),
            "male_iqr": max(male_iqr, eps),
            "weight": abs(female_median - male_median) / pooled_iqr,
        }

    drivers: dict[str, list[dict[str, float | str]]] = {}
    for _, row in generated_df.iterrows():
        model = row["model"]
        scores: list[dict[str, float | str]] = []
        for feature in features:
            value = row.get(feature)
            if pd.isna(value):
                continue
            s = stats[feature]
            distance_to_female = abs(float(value) - s["female_median"]) / s["female_iqr"]
            distance_to_male = abs(float(value) - s["male_median"]) / s["male_iqr"]
            # Positive values pull the generated image toward the Male reference group;
            # negative values pull it toward the Female reference group.
            evidence = s["weight"] * (distance_to_female - distance_to_male)
            scores.append(
                {
                    "metric": feature,
                    "evidence": float(evidence),
                    "predicted_group": "Male" if evidence > 0 else "Female",
                }
            )
        drivers[model] = sorted(scores, key=lambda item: abs(float(item["evidence"])), reverse=True)[:top_n]
    return drivers


def compute_generated_assignments(shpm_df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, row in shpm_df.iterrows():
        model = model_from_filename(str(row["filename"]))
        if model not in MODEL_ORDER:
            continue
        score_female = float(row["shpm_female"])
        score_male = float(row["shpm_male"])
        assigned = "Female" if score_female >= score_male else "Male"
        ground_truth = GROUND_TRUTH[model]
        rows.append(
            {
                "model": model,
                "score_female": f"{score_female:.4f}",
                "score_male": f"{score_male:.4f}",
                "assigned": assigned,
                "ground_truth": ground_truth,
                "correct": "yes" if assigned == ground_truth else "no",
            }
        )
    return sorted(rows, key=lambda item: MODEL_ORDER.index(item["model"]))


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    reference_df = pd.read_csv(args.reference)
    full_features = feature_columns(reference_df)
    reference_df = numeric_feature_frame(reference_df, full_features)
    no_hair_features = [feature for feature in full_features if "hair" not in feature]

    full_eval = evaluate_nearest_centroid(reference_df, full_features, args.folds, args.seed)
    no_hair_eval = evaluate_nearest_centroid(reference_df, no_hair_features, args.folds, args.seed)

    separability_rows = [
        {
            "feature_configuration": "All 30 features",
            "feature_count": len(full_features),
            **full_eval,
        },
        {
            "feature_configuration": "No-hair features",
            "feature_count": len(no_hair_features),
            **no_hair_eval,
        },
    ]
    separability_path = args.paper_dir / "feature_separability_results.csv"
    pd.DataFrame(separability_rows).to_csv(separability_path, index=False)

    generated_raw = pd.read_csv(args.generated_features)
    generated_raw["model"] = generated_raw["filename"].map(model_from_filename)
    generated_features = [feature for feature in full_features if feature in generated_raw.columns]
    generated_df = numeric_feature_frame(generated_raw[["model", *generated_features]], generated_features)
    generated_df = generated_df.set_index("model").loc[list(MODEL_ORDER)].reset_index()
    generated_vectors_path = args.figure_dir / "generated_shpm_vectors.csv"
    generated_df.to_csv(generated_vectors_path, index=False)

    drivers = compute_driver_scores(generated_df, reference_df, generated_features)
    evidence_path = args.figure_dir / "dominant_shpm_evidence.csv"
    with evidence_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("model", "metric", "evidence", "predicted_group", "notes"),
        )
        writer.writeheader()
        for model in MODEL_ORDER:
            for item in drivers.get(model, []):
                writer.writerow(
                    {
                        "model": model,
                        "metric": item["metric"],
                        "evidence": f"{float(item['evidence']):.4f}",
                        "predicted_group": item["predicted_group"],
                        "notes": "computed by recompute_paper_results.py",
                    }
                )

    shpm_df = pd.read_csv(args.generated_shpm)
    assignments = compute_generated_assignments(shpm_df)
    assignments_path = args.figure_dir / "class_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("model", "score_female", "score_male", "assigned", "ground_truth", "correct"),
        )
        writer.writeheader()
        writer.writerows(assignments)

    print("Feature separability:")
    for row in separability_rows:
        print(
            f"  {row['feature_configuration']}: n={int(row['n'])}, "
            f"accuracy={row['mean_accuracy']:.1f} ± {row['std_accuracy']:.1f}"
        )
    print(f"Wrote {separability_path}")
    print(f"Wrote {generated_vectors_path}")
    print(f"Wrote {evidence_path}")
    print(f"Wrote {assignments_path}")


if __name__ == "__main__":
    main()
