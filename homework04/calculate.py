"""Reproduce ML Zoomcamp 2026 HW04 calculations from the pinned course CSV.

Pass a local copy of the official dataset as the positional argument. The
script validates its SHA-256 and writes results.json plus the requested
precision/recall plot beside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent
EXPECTED_DATA_SHA256 = "2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9"


def write_precision_recall_svg(rows: list[dict[str, float]], path: Path) -> None:
    """Create the requested threshold plot using only the Python standard library."""
    width, height = 800, 520
    left, right, top, bottom = 76, 28, 48, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_pos(value: float) -> float:
        return left + value * plot_width

    def y_pos(value: float) -> float:
        return top + (1.0 - value) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="400" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">HW04: Precision and recall by threshold</text>',
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        x, y = x_pos(value), y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>')
        parts.append(f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.1f}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#334155" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155" stroke-width="1.5"/>',
        f'<text x="{left + plot_width/2:.2f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Decision threshold</text>',
        f'<text x="20" y="{top + plot_height/2:.2f}" transform="rotate(-90 20 {top + plot_height/2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14">Metric value</text>',
    ])
    for key, color in (("precision", "#2563eb"), ("recall", "#dc2626")):
        points = " ".join(f'{x_pos(row["threshold"]):.2f},{y_pos(row[key]):.2f}' for row in rows)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
    parts.extend([
        f'<line x1="{width-220}" y1="{top+10}" x2="{width-190}" y2="{top+10}" stroke="#2563eb" stroke-width="3"/>',
        f'<text x="{width-184}" y="{top+14}" font-family="sans-serif" font-size="13">Precision</text>',
        f'<line x1="{width-220}" y1="{top+32}" x2="{width-190}" y2="{top+32}" stroke="#dc2626" stroke-width="3"/>',
        f'<text x="{width-184}" y="{top+36}" font-family="sans-serif" font-size="13">Recall</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        help="local path to the official course_lead_scoring_2026.csv file",
    )
    args = parser.parse_args()
    data_path = args.csv.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Official pinned dataset is required: {data_path}")
    dataset_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if dataset_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError(
            "Dataset checksum does not match the current official 2026 release: "
            f"{dataset_sha256}"
        )

    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import KFold, train_test_split

    df = pd.read_csv(data_path)
    target = "converted"
    if target not in df.columns or df[target].isna().any():
        raise ValueError("The official target column must exist and contain no missing labels.")

    feature_columns = [column for column in df.columns if column != target]
    categorical = list(
        df[feature_columns]
        .select_dtypes(include=["object", "string", "category"])
        .columns
    )
    numerical = [column for column in feature_columns if column not in categorical]
    missing_before = df[feature_columns].isna().sum().astype(int).to_dict()

    # HW04: categorical missing values become the literal "NA"; numerical
    # missing values become 0.0. The target is never imputed.
    for column in categorical:
        df[column] = df[column].astype("string").fillna("NA").astype(str)
    df[numerical] = df[numerical].fillna(0.0)

    # Use the exact two course calls: a held-out 20% test split, then 25% of
    # the remaining rows for validation. Only train/validation are used below.
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )
    y_train = df_train[target].to_numpy()
    y_val = df_val[target].to_numpy()

    # Q1: numerical feature ROC AUC on training data. Invert the score only
    # when its direction is below chance, exactly as the instructions say.
    numerical_candidates = [
        "lead_score",
        "number_of_courses_viewed",
        "interaction_count",
        "annual_income",
    ]
    q1_scores: dict[str, dict[str, float | bool]] = {}
    for column in numerical_candidates:
        raw_auc = float(roc_auc_score(y_train, df_train[column]))
        inverted = raw_auc < 0.5
        directed_auc = float(
            roc_auc_score(y_train, -df_train[column] if inverted else df_train[column])
        )
        q1_scores[column] = {
            "raw_auc": raw_auc,
            "direction_inverted": inverted,
            "directed_auc": directed_auc,
        }
    q1_best_feature = max(q1_scores, key=lambda column: q1_scores[column]["directed_auc"])

    def fit_and_predict_auc(
        train_frame: pd.DataFrame,
        train_target: np.ndarray,
        predict_frame: pd.DataFrame,
        c_value: float = 1.0,
    ) -> np.ndarray:
        # DictVectorizer learns categories only from the current training fold.
        vectorizer = DictVectorizer()
        encoded_train = vectorizer.fit_transform(
            train_frame[feature_columns].to_dict(orient="records")
        )
        encoded_predict = vectorizer.transform(
            predict_frame[feature_columns].to_dict(orient="records")
        )
        model = LogisticRegression(
            solver="liblinear", C=c_value, max_iter=1000
        )
        model.fit(encoded_train, train_target)
        return model.predict_proba(encoded_predict)[:, 1]

    # Q2: fit the specified logistic regression on the training split, then
    # score its probabilities on validation with ROC AUC.
    validation_scores = fit_and_predict_auc(df_train, y_train, df_val, c_value=1.0)
    q2_validation_auc = float(roc_auc_score(y_val, validation_scores))

    # Q3/Q4: evaluate every threshold from 0.00 to 1.00 inclusive by 0.01.
    threshold_rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.0, 1.0, 101):
        prediction = validation_scores >= threshold
        precision = float(precision_score(y_val, prediction, zero_division=0))
        recall = float(recall_score(y_val, prediction, zero_division=0))
        f1 = float(f1_score(y_val, prediction, zero_division=0))
        threshold_rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision,
                "recall": recall,
                "abs_precision_recall_difference": abs(precision - recall),
                "f1": f1,
            }
        )

    valid_intersections = [
        row
        for row in threshold_rows
        if not (row["precision"] == 0.0 and row["recall"] == 0.0)
    ]
    smallest_gap = min(row["abs_precision_recall_difference"] for row in valid_intersections)
    closest_thresholds = [
        row["threshold"]
        for row in valid_intersections
        if math.isclose(
            row["abs_precision_recall_difference"], smallest_gap, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    max_f1 = max(row["f1"] for row in threshold_rows)
    max_f1_thresholds = [
        row["threshold"]
        for row in threshold_rows
        if math.isclose(row["f1"], max_f1, rel_tol=0.0, abs_tol=1e-12)
    ]

    # Q5/Q6: five shuffled folds over df_full_train only. The held-out test set
    # is kept out of fold scoring, threshold selection, and C tuning.
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    def cross_validated_auc(c_value: float) -> list[float]:
        scores: list[float] = []
        for train_indices, validation_indices in kfold.split(df_full_train):
            fold_train = df_full_train.iloc[train_indices]
            fold_val = df_full_train.iloc[validation_indices]
            fold_y_train = fold_train[target].to_numpy()
            fold_y_val = fold_val[target].to_numpy()
            fold_scores = fit_and_predict_auc(
                fold_train, fold_y_train, fold_val, c_value=c_value
            )
            scores.append(float(roc_auc_score(fold_y_val, fold_scores)))
        return scores

    q5_fold_scores = cross_validated_auc(1.0)
    q6_by_c: dict[str, dict[str, float | list[float]]] = {}
    c_values = [0.000001, 0.001, 1.0]
    for c_value in c_values:
        fold_scores = cross_validated_auc(c_value)
        q6_by_c[f"{c_value:g}"] = {
            "fold_auc": fold_scores,
            "mean_auc": float(np.mean(fold_scores)),
            "population_std": float(np.std(fold_scores)),
            "mean_auc_rounded_3": round(float(np.mean(fold_scores)), 3),
            "std_rounded_3": round(float(np.std(fold_scores)), 3),
        }
    best_mean = max(result["mean_auc"] for result in q6_by_c.values())
    q6_best_c = [
        float(c_value)
        for c_value in c_values
        if math.isclose(
            q6_by_c[f"{c_value:g}"]["mean_auc"], best_mean, rel_tol=0.0, abs_tol=1e-12
        )
    ]

    result = {
        "assignment": "ML Zoomcamp 2026 Homework 4: Evaluation Metrics for Classification",
        "data_source": "DataTalksClub/machine-learning-zoomcamp/cohorts/2026/data/course_lead_scoring_2026.csv",
        "dataset_sha256": dataset_sha256,
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "rows": int(len(df)),
        "missing_values_before_imputation": {
            column: int(count) for column, count in missing_before.items()
        },
        "feature_columns": feature_columns,
        "split_sizes": {
            "train": int(len(df_train)),
            "validation": int(len(df_val)),
            "test_held_out_unused": int(len(df_test)),
            "full_train_for_cv": int(len(df_full_train)),
            "each_cv_fold_train": int(len(df_full_train) * 4 / 5),
            "each_cv_fold_validation": int(len(df_full_train) / 5),
        },
        "Q1_directed_training_roc_auc_by_numeric_feature": q1_scores,
        "Q1_highest_feature": q1_best_feature,
        "Q2_validation_roc_auc": q2_validation_auc,
        "Q2_validation_roc_auc_rounded_3": round(q2_validation_auc, 3),
        "Q3_precision_recall_smallest_gap": {
            "minimum_absolute_difference": smallest_gap,
            "closest_thresholds": closest_thresholds,
            "selected_threshold_for_single_answer": min(closest_thresholds),
        },
        "Q4_max_f1": {
            "maximum_f1": max_f1,
            "thresholds_with_maximum_f1": max_f1_thresholds,
            "selected_threshold_for_single_answer": min(max_f1_thresholds),
        },
        "Q5_5_fold_auc_C_1": {
            "fold_auc": q5_fold_scores,
            "population_std": float(np.std(q5_fold_scores)),
            "population_std_rounded_3": round(float(np.std(q5_fold_scores)), 3),
        },
        "Q6_auc_by_C": q6_by_c,
        "Q6_best_C_options_by_exact_mean": q6_best_c,
        "threshold_grid": "0.00 through 1.00 inclusive, step 0.01",
        "notes": [
            "The held-out test split was created as instructed but not used for model selection or answers.",
            "CV uses df_full_train, KFold(n_splits=5, shuffle=True, random_state=1), and population standard deviation.",
        ],
    }

    (HERE / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_precision_recall_svg(threshold_rows, HERE / "precision_recall_thresholds.svg")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
