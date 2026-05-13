"""
src/models/predictor.py

Loads trained LightGBM pipelines and predicts Big-Five scores
for a new user given their feature vector.

Usage (CLI)
-----------
python -m src.models.predictor --username someuser --limit 300
"""

import argparse
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import MODEL_DIR, SCORE_MIN, SCORE_MAX, TRAITS, TRAIT_NAMES

logger = logging.getLogger(__name__)


@dataclass
class PersonalityProfile:
    username: str
    scores: Dict[str, float] = field(default_factory=dict)   # {"O": 4.2, ...}
    comment_count: int = 0
    word_count: int = 0
    low_data_warning: bool = False

    def trait_name(self, abbr: str) -> str:
        return TRAIT_NAMES.get(abbr, abbr)

    def summary(self) -> str:
        lines = [f"Personality profile for u/{self.username}",
                 f"Based on {self.comment_count} comments ({self.word_count} words)",
                 ""]
        for trait in TRAITS:
            score = self.scores.get(trait, float("nan"))
            bar   = "█" * int(round(score)) + "░" * (7 - int(round(score)))
            lines.append(f"  {self.trait_name(trait):<20} {bar}  {score:.2f}/7.00")
        if self.low_data_warning:
            lines.append("\n⚠  Low comment count — reliability is reduced.")
        return "\n".join(lines)


class Predictor:
    """
    Loads one pickled sklearn Pipeline per trait and exposes a predict()
    method that returns a PersonalityProfile.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self._pipelines: Dict[str, object] = {}
        self._feature_names = None
        self._load_models(model_dir)

    # ── Setup ──────────────────────────────────────────────────────────────

    def _load_models(self, model_dir: Path):
        fn_path = model_dir / "feature_names.pkl"
        if fn_path.exists():
            with open(fn_path, "rb") as f:
                self._feature_names = pickle.load(f)
            logger.info("Loaded %d feature names", len(self._feature_names))
        else:
            logger.warning("feature_names.pkl not found — using feature order as-is")

        for trait in TRAITS:
            path = model_dir / f"model_{trait}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self._pipelines[trait] = pickle.load(f)
                logger.info("Loaded model for trait %s", trait)
            else:
                logger.warning("Model file not found: %s", path)

    @property
    def models_loaded(self) -> bool:
        return len(self._pipelines) == len(TRAITS)

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, feature_df: pd.DataFrame) -> PersonalityProfile:
        """
        Parameters
        ----------
        feature_df : pd.DataFrame
            Output of feature_pipeline.run_pipeline() — shape (1, n_features).

        Returns
        -------
        PersonalityProfile
        """
        username      = feature_df.index[0] if feature_df.index[0] else "unknown"
        comment_count = int(feature_df.get("_comment_count", pd.Series([0]))[0])
        word_count    = int(feature_df.get("_word_count",    pd.Series([0]))[0])

        # Strip metadata columns before inference
        meta = {"_username", "_comment_count", "_word_count"}
        X    = feature_df.drop(columns=[c for c in meta if c in feature_df.columns])

        # Align to training feature order
        if self._feature_names:
            for col in self._feature_names:
                if col not in X.columns:
                    X[col] = float("nan")
            X = X[self._feature_names]

        scores: Dict[str, float] = {}
        for trait, pipeline in self._pipelines.items():
            raw_score = float(pipeline.predict(X)[0])
            # Clip to valid range
            scores[trait] = round(float(np.clip(raw_score, SCORE_MIN, SCORE_MAX)), 2)

        # Fallback heuristics if no models loaded
        if not scores:
            logger.warning("No models loaded — returning neutral scores")
            scores = {t: 4.0 for t in TRAITS}

        return PersonalityProfile(
            username=str(username),
            scores=scores,
            comment_count=comment_count,
            word_count=word_count,
            low_data_warning=(comment_count < 50),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--limit",    type=int, default=300)
    args = parser.parse_args()

    # Import here to avoid circular imports at module level
    from src.scraper.reddit_scraper import RedditScraper
    from src.features.feature_pipeline import run_pipeline

    scraper = RedditScraper()
    result  = scraper.fetch(args.username, limit=args.limit)

    if not result.success:
        print(f"Scraping failed: {result.error}")
        return

    feature_df = run_pipeline(args.username, result.comments)
    predictor  = Predictor()
    profile    = predictor.predict(feature_df)

    print(profile.summary())


if __name__ == "__main__":
    main()
