"""
src/models/trainer.py

Trains one LightGBM regressor per Big-Five trait using cross-validation.
Saves trained models to MODEL_DIR as .pkl files.

Usage (CLI)
-----------
python -m src.models.trainer \
    --data data/labelled_users.csv \
    --output models/

Expected CSV columns
--------------------
username, O_score, C_score, E_score, A_score, N_score, <feature columns...>
Scores are on a 1–7 scale.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CV_FOLDS, LGBM_PARAMS, MODEL_DIR, TRAITS, SCORE_MIN, SCORE_MAX

logger = logging.getLogger(__name__)

# Columns that are metadata, not features
META_COLS = {"username", "_username", "_comment_count", "_word_count"}


def load_training_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load labelled data CSV.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (columns = psycholinguistic features)
    y : pd.DataFrame  — target matrix  (columns = O, C, E, A, N)
    """
    df = pd.read_csv(csv_path)

    target_cols = [f"{t}_score" for t in TRAITS]
    feature_cols = [c for c in df.columns if c not in META_COLS | set(target_cols)]

    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(f"CSV missing target columns: {missing_targets}")

    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    y.columns = TRAITS   # rename O_score → O, etc.

    # Clip scores to valid range
    y = y.clip(SCORE_MIN, SCORE_MAX)

    logger.info("Loaded %d samples, %d features", len(X), len(X.columns))
    return X, y


def build_pipeline() -> Pipeline:
    """
    Sklearn Pipeline: impute NaN → scale → LightGBM regressor.
    Imputation handles users with too-few comments for some features.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LGBMRegressor(**LGBM_PARAMS)),
    ])


def train_all_traits(
    X: pd.DataFrame,
    y: pd.DataFrame,
    output_dir: Path = MODEL_DIR,
) -> Dict[str, Dict]:
    """
    Train one pipeline per trait, evaluate with k-fold CV, save to disk.

    Returns
    -------
    dict  {trait: {"pipeline": Pipeline, "cv_mae": float, "cv_r2": float}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    for trait in TRAITS:
        logger.info("Training trait: %s", trait)
        pipeline = build_pipeline()
        y_trait  = y[trait].values

        # Cross-validation metrics
        mae_scores = -cross_val_score(
            pipeline, X, y_trait, cv=kf,
            scoring="neg_mean_absolute_error"
        )
        r2_scores = cross_val_score(
            pipeline, X, y_trait, cv=kf, scoring="r2"
        )

        # Final fit on all data
        pipeline.fit(X, y_trait)

        # Persist
        model_path = output_dir / f"model_{trait}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        logger.info(
            "  %s → MAE=%.3f ± %.3f  |  R²=%.3f ± %.3f  | saved to %s",
            trait,
            mae_scores.mean(), mae_scores.std(),
            r2_scores.mean(), r2_scores.std(),
            model_path,
        )

        results[trait] = {
            "pipeline": pipeline,
            "cv_mae":   float(mae_scores.mean()),
            "cv_r2":    float(r2_scores.mean()),
        }

    # Save feature names for inference-time alignment
    feature_names_path = output_dir / "feature_names.pkl"
    with open(feature_names_path, "wb") as f:
        pickle.dump(list(X.columns), f)

    logger.info("All models saved to %s", output_dir)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train Big-Five LightGBM models")
    parser.add_argument("--data",   required=True,        help="Path to labelled CSV")
    parser.add_argument("--output", default=str(MODEL_DIR), help="Directory to save models")
    args = parser.parse_args()

    X, y = load_training_data(args.data)
    results = train_all_traits(X, y, output_dir=Path(args.output))

    print("\n── Training Summary ─────────────────────────────────")
    for trait, metrics in results.items():
        print(f"  {trait}  MAE={metrics['cv_mae']:.3f}  R²={metrics['cv_r2']:.3f}")


if __name__ == "__main__":
    main()
