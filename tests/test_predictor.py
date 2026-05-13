"""
tests/test_predictor.py

Tests for the Predictor class.
Uses mock pipelines so no real model files are required.
"""

import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.predictor import Predictor, PersonalityProfile
from config import TRAITS, SCORE_MIN, SCORE_MAX


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_mock_predictor() -> Predictor:
    """Create a Predictor with mocked pipelines (no disk I/O)."""
    predictor = Predictor.__new__(Predictor)
    predictor._feature_names = None
    predictor._pipelines = {}

    for trait in TRAITS:
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([4.5])
        predictor._pipelines[trait] = mock_pipeline

    return predictor


def make_feature_df(username: str = "testuser") -> pd.DataFrame:
    """Create a minimal feature DataFrame matching pipeline output."""
    data = {
        "pronoun_i_rate":    [0.05],
        "hedging_rate":      [0.02],
        "certainty_rate":    [0.01],
        "vader_compound_mean": [0.3],
        "_comment_count":    [150],
        "_word_count":       [4500],
    }
    df = pd.DataFrame(data, index=[username])
    return df


# ── PersonalityProfile tests ──────────────────────────────────────────────────

def test_personality_profile_summary():
    profile = PersonalityProfile(
        username="testuser",
        scores={"O": 5.1, "C": 4.2, "E": 3.8, "A": 6.0, "N": 2.5},
        comment_count=200,
        word_count=5000,
    )
    summary = profile.summary()
    assert "testuser" in summary
    assert "Openness" in summary
    assert "5.10" in summary


def test_personality_profile_low_data_warning():
    profile = PersonalityProfile(
        username="sparse",
        scores={t: 4.0 for t in TRAITS},
        comment_count=20,
        low_data_warning=True,
    )
    summary = profile.summary()
    assert "Low" in summary or "reliability" in summary.lower()


# ── Predictor.predict tests ───────────────────────────────────────────────────

def test_predict_returns_profile():
    predictor  = make_mock_predictor()
    feature_df = make_feature_df()
    profile    = predictor.predict(feature_df)

    assert isinstance(profile, PersonalityProfile)
    assert profile.username == "testuser"
    assert set(profile.scores.keys()) == set(TRAITS)


def test_predict_scores_in_valid_range():
    predictor  = make_mock_predictor()
    feature_df = make_feature_df()
    profile    = predictor.predict(feature_df)

    for trait, score in profile.scores.items():
        assert SCORE_MIN <= score <= SCORE_MAX, (
            f"Score for {trait} = {score} out of range [{SCORE_MIN}, {SCORE_MAX}]"
        )


def test_predict_clips_extreme_scores():
    predictor = make_mock_predictor()
    # Override models to return out-of-range scores
    for trait in TRAITS:
        predictor._pipelines[trait].predict.return_value = np.array([99.0])

    feature_df = make_feature_df()
    profile    = predictor.predict(feature_df)

    for score in profile.scores.values():
        assert score <= SCORE_MAX


def test_predict_comment_count_propagated():
    predictor  = make_mock_predictor()
    feature_df = make_feature_df()
    profile    = predictor.predict(feature_df)
    assert profile.comment_count == 150


def test_predict_low_data_warning_triggered():
    predictor  = make_mock_predictor()
    feature_df = make_feature_df()
    feature_df["_comment_count"] = 30   # below threshold
    profile    = predictor.predict(feature_df)
    assert profile.low_data_warning


def test_models_loaded_property():
    predictor = make_mock_predictor()
    assert predictor.models_loaded


def test_models_not_loaded_when_empty():
    predictor = Predictor.__new__(Predictor)
    predictor._feature_names = None
    predictor._pipelines     = {}
    assert not predictor.models_loaded
