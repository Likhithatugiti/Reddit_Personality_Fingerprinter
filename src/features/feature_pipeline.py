"""
src/features/feature_pipeline.py

Orchestrates the full feature extraction pipeline:
  raw comments → clean → extract features → DataFrame row
"""

import logging
import pandas as pd
from typing import List, Dict

from src.features.preprocessor import preprocess_comments
from src.features.psycholinguistic import extract_all
from config import MIN_COMMENTS_REQUIRED

logger = logging.getLogger(__name__)


def run_pipeline(username: str, comments: List[str]) -> pd.DataFrame:
    """
    Full pipeline: raw comments → single-row feature DataFrame.

    Parameters
    ----------
    username : str
    comments : list of str  — raw comment texts from scraper

    Returns
    -------
    pd.DataFrame  — shape (1, n_features), index = [username]
    """
    if len(comments) < MIN_COMMENTS_REQUIRED:
        logger.warning(
            "u/%s has only %d comments (< %d). Reliability will be low.",
            username, len(comments), MIN_COMMENTS_REQUIRED
        )

    cleaned  = preprocess_comments(comments)
    features = extract_all(cleaned)

    # Add metadata (not used in model, useful for display)
    features["_username"]      = username
    features["_comment_count"] = len(cleaned)
    features["_word_count"]    = sum(len(c.split()) for c in cleaned)

    df = pd.DataFrame([features])
    df.index = [username]
    logger.info(
        "Feature extraction complete for u/%s: %d features from %d comments",
        username, len(features), len(cleaned)
    )
    return df


def batch_pipeline(users: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Run pipeline for multiple users.

    Parameters
    ----------
    users : dict  {username: [comment, ...]}

    Returns
    -------
    pd.DataFrame  — shape (n_users, n_features)
    """
    rows = []
    for username, comments in users.items():
        row = run_pipeline(username, comments)
        rows.append(row)
    return pd.concat(rows) if rows else pd.DataFrame()
