"""
tests/test_features.py

Unit tests for feature extraction functions.
Verifies that features are in expected ranges and that the
pipeline handles edge cases gracefully.
"""

import pytest
import numpy as np

from src.features.preprocessor import clean_comment, preprocess_comments
from src.features.psycholinguistic import extract_all, _pronoun_features, _sentiment_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_COMMENTS = [
    "I really think we should maybe reconsider this approach. I'm not sure it's optimal.",
    "Definitely! I always knew this would work. You guys are absolutely right about everything.",
    "We all need to support each other. Together we can achieve so much more as a community.",
    "Because I understand the reason behind this, I can see why people are concerned.",
    "WOW!! This is amazing!!! I can't believe how great this is!!!",
]


# ── Preprocessor tests ────────────────────────────────────────────────────────

def test_clean_removes_urls():
    text    = "Check out https://example.com for more info"
    cleaned = clean_comment(text)
    assert "https://" not in cleaned
    assert "for more info" in cleaned


def test_clean_removes_reddit_refs():
    text    = "See u/someuser's post on r/python"
    cleaned = clean_comment(text)
    assert "u/someuser" not in cleaned
    assert "r/python" not in cleaned


def test_clean_preserves_punctuation():
    text    = "Is this really true? I think so!"
    cleaned = clean_comment(text)
    assert "?" in cleaned
    assert "!" in cleaned


def test_preprocess_drops_empty():
    comments = ["Hello world", "", "   ", "x"]
    result   = preprocess_comments(comments)
    assert all(len(c) > 10 for c in result)


def test_clean_markdown_link():
    text    = "Here is [a link](https://reddit.com) to check"
    cleaned = clean_comment(text)
    assert "a link" in cleaned
    assert "https://" not in cleaned


# ── Feature extraction tests ──────────────────────────────────────────────────

def test_extract_all_returns_dict():
    features = extract_all(SAMPLE_COMMENTS)
    assert isinstance(features, dict)
    assert len(features) > 30


def test_extract_all_no_nan_on_good_input():
    features = extract_all(SAMPLE_COMMENTS)
    # Core features should not be NaN with valid input
    core_keys = [
        "pronoun_i_rate", "hedging_rate", "certainty_rate",
        "vader_compound_mean", "type_token_ratio",
    ]
    for k in core_keys:
        assert k in features, f"Missing feature: {k}"
        assert not np.isnan(features[k]), f"NaN in feature: {k}"


def test_extract_all_empty_returns_nan_dict():
    features = extract_all([])
    assert isinstance(features, dict)
    # All values should be NaN
    for k, v in features.items():
        assert np.isnan(v), f"Expected NaN for empty input, got {v} for {k}"


def test_pronoun_i_rate_high_for_first_person():
    tokens = ["i", "me", "my", "i", "i", "think", "so"]
    n      = len(tokens)
    feats  = _pronoun_features(tokens, n)
    assert feats["pronoun_i_rate"] > 0.4   # 3 first-person singular out of 7


def test_hedging_rate_nonzero():
    hedgy_comments = [
        "Maybe this is right. I think perhaps we could try.",
        "It might possibly work, sort of.",
    ]
    features = extract_all(hedgy_comments)
    assert features["hedging_rate"] > 0


def test_certainty_rate_nonzero():
    certain_comments = [
        "I definitely know this is absolutely correct.",
        "This is always the right choice, without question.",
    ]
    features = extract_all(certain_comments)
    assert features["certainty_rate"] > 0


def test_sentiment_negative_for_negative_text():
    negative_comments = [
        "I hate this. It's terrible and awful.",
        "This is the worst thing I've ever seen.",
    ]
    features = _sentiment_features(negative_comments)
    assert features["vader_neg_mean"] > features["vader_pos_mean"]


def test_exclamation_rate():
    exclamatory = ["WOW!! This is great!!!", "AMAZING!!!"]
    features    = extract_all(exclamatory)
    assert features["exclamation_rate"] > 0


def test_all_traits_in_nrc_output():
    features = extract_all(SAMPLE_COMMENTS)
    nrc_keys = [
        "nrc_anger_rate", "nrc_fear_rate", "nrc_joy_rate",
        "nrc_trust_rate", "nrc_positive_rate", "nrc_negative_rate",
    ]
    for k in nrc_keys:
        assert k in features, f"Missing NRC feature: {k}"


def test_rates_are_between_0_and_1():
    features = extract_all(SAMPLE_COMMENTS)
    rate_keys = [k for k in features if k.endswith("_rate")]
    for k in rate_keys:
        v = features[k]
        if not np.isnan(v):
            assert 0.0 <= v, f"{k} = {v} is negative"
