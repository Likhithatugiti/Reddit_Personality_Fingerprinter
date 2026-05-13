"""
src/features/psycholinguistic.py

Extracts 50+ psycholinguistic features from a user's comment corpus.
Each feature group is a separate function so they can be tested and
extended independently.

Feature groups
--------------
1.  Pronoun usage       (I, we, you, they rates)
2.  Hedging             (maybe, perhaps, sort of, ...)
3.  Certainty           (definitely, always, never, ...)
4.  Cognitive           (because, think, know, realize, ...)
5.  Social              (people, together, friend, ...)
6.  Sentiment           (VADER compound, pos/neg/neu ratios)
7.  Text complexity     (TTR, avg sentence length, Flesch-Kincaid)
8.  Punctuation         (! rate, ? rate, CAPS ratio)
9.  Emotion             (NRC Lexicon: anger, fear, joy, trust, ...)
10. Time orientation    (past/present/future tense ratios)

All rates are per-token (0–1 range) unless noted otherwise.
"""

import logging
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import textstat
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (
    CERTAINTY_WORDS,
    COGNITIVE_WORDS,
    HEDGING_WORDS,
    SOCIAL_WORDS,
)
from src.features.preprocessor import tokenise, get_sentences

logger = logging.getLogger(__name__)

_vader = SentimentIntensityAnalyzer()


# ── Master entry point ────────────────────────────────────────────────────────

def extract_all(comments: List[str]) -> Dict[str, float]:
    """
    Given a list of cleaned comment strings, compute all features.

    Returns
    -------
    dict  {feature_name: float}
        Always returns the full feature set (NaN for unavailable features).
    """
    if not comments:
        logger.warning("extract_all called with empty comment list")
        return _empty_feature_dict()

    full_text  = " ".join(comments)
    doc        = tokenise(full_text)
    tokens     = [t.text.lower() for t in doc if not t.is_space]
    token_count = max(len(tokens), 1)   # guard against zero division
    sentences  = get_sentences(full_text)

    features: Dict[str, float] = {}

    features.update(_pronoun_features(tokens, token_count))
    features.update(_hedging_features(tokens, token_count, full_text))
    features.update(_certainty_features(tokens, token_count, full_text))
    features.update(_cognitive_features(tokens, token_count, full_text))
    features.update(_social_features(tokens, token_count))
    features.update(_sentiment_features(comments))
    features.update(_complexity_features(full_text, sentences, tokens, token_count))
    features.update(_punctuation_features(full_text, token_count))
    features.update(_emotion_features(full_text))
    features.update(_tense_features(doc, token_count))

    logger.debug("Extracted %d features from %d comments", len(features), len(comments))
    return features


# ── Feature group implementations ────────────────────────────────────────────

def _pronoun_features(tokens: List[str], n: int) -> Dict[str, float]:
    """
    First-person singular (I, me, my, mine, myself) → high N, low A
    First-person plural  (we, us, our)              → high A, E
    Second-person        (you, your, yours)          → high E, A
    Third-person         (he, she, they, them)       → narrative tendency
    """
    first_sing  = {"i", "me", "my", "mine", "myself"}
    first_plur  = {"we", "us", "our", "ours", "ourselves"}
    second      = {"you", "your", "yours", "yourself", "yourselves"}
    third_sing  = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
    third_plur  = {"they", "them", "their", "theirs", "themselves"}

    counts = Counter(tokens)
    return {
        "pronoun_i_rate":           sum(counts[p] for p in first_sing) / n,
        "pronoun_we_rate":          sum(counts[p] for p in first_plur) / n,
        "pronoun_you_rate":         sum(counts[p] for p in second) / n,
        "pronoun_third_sing_rate":  sum(counts[p] for p in third_sing) / n,
        "pronoun_third_plur_rate":  sum(counts[p] for p in third_plur) / n,
    }


def _hedging_features(tokens: List[str], n: int, text: str) -> Dict[str, float]:
    """
    Hedging language (maybe, perhaps, I think) → low C, low E
    Uses both single-word and phrase matching.
    """
    text_lower = text.lower()
    count = sum(1 for t in tokens if t in HEDGING_WORDS)
    # Phrase-level match (e.g. "sort of", "kind of")
    phrase_count = sum(text_lower.count(phrase) for phrase in HEDGING_WORDS if " " in phrase)
    return {
        "hedging_rate":  (count + phrase_count) / n,
    }


def _certainty_features(tokens: List[str], n: int, text: str) -> Dict[str, float]:
    """
    Certainty language (definitely, always) → high C, low N
    """
    text_lower = text.lower()
    count = sum(1 for t in tokens if t in CERTAINTY_WORDS)
    phrase_count = sum(text_lower.count(p) for p in CERTAINTY_WORDS if " " in p)
    return {
        "certainty_rate": (count + phrase_count) / n,
    }


def _cognitive_features(tokens: List[str], n: int, text: str) -> Dict[str, float]:
    """
    Cognitive / causal words (because, think, understand) → high O
    """
    text_lower = text.lower()
    count = sum(1 for t in tokens if t in COGNITIVE_WORDS)
    phrase_count = sum(text_lower.count(p) for p in COGNITIVE_WORDS if " " in p)
    return {
        "cognitive_rate": (count + phrase_count) / n,
    }


def _social_features(tokens: List[str], n: int) -> Dict[str, float]:
    """
    Social-orientation words (people, together, friend) → high E, A
    """
    count = sum(1 for t in tokens if t in SOCIAL_WORDS)
    return {
        "social_rate": count / n,
    }


def _sentiment_features(comments: List[str]) -> Dict[str, float]:
    """
    Run VADER on each comment, then average.
    compound → valence; pos/neg/neu → emotion profile
    High compound + high pos → high A, high E
    High neg → high N
    """
    compounds, pos_list, neg_list, neu_list = [], [], [], []

    for comment in comments:
        scores = _vader.polarity_scores(comment)
        compounds.append(scores["compound"])
        pos_list.append(scores["pos"])
        neg_list.append(scores["neg"])
        neu_list.append(scores["neu"])

    return {
        "vader_compound_mean": float(np.mean(compounds)),
        "vader_compound_std":  float(np.std(compounds)),
        "vader_pos_mean":      float(np.mean(pos_list)),
        "vader_neg_mean":      float(np.mean(neg_list)),
        "vader_neu_mean":      float(np.mean(neu_list)),
        "sentiment_volatility": float(np.std(compounds)),   # alias for clarity
    }


def _complexity_features(
    text: str,
    sentences: List[str],
    tokens: List[str],
    n: int,
) -> Dict[str, float]:
    """
    Linguistic complexity → high O (elaborate language, rich vocabulary)
    High complexity can also indicate C (thoroughness).
    """
    # Type-token ratio (vocabulary richness)
    unique_tokens = len(set(tokens))
    ttr = unique_tokens / n

    # Average sentence length in words
    sent_lengths = [len(s.split()) for s in sentences if s]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0

    # Readability (higher = more complex)
    try:
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        flesch_reading  = textstat.flesch_reading_ease(text)
        avg_syllables   = textstat.avg_syllables_per_word(text)
    except Exception:
        flesch_kincaid = 0.0
        flesch_reading  = 0.0
        avg_syllables   = 0.0

    return {
        "type_token_ratio":     ttr,
        "avg_sentence_length":  avg_sent_len,
        "flesch_kincaid_grade": float(flesch_kincaid),
        "flesch_reading_ease":  float(flesch_reading),
        "avg_syllables_per_word": float(avg_syllables),
        "unique_word_count":    float(unique_tokens),
    }


def _punctuation_features(text: str, n: int) -> Dict[str, float]:
    """
    Exclamation marks → high E
    Question marks    → high O (curiosity), high A (engagement)
    ALL CAPS words    → high E, possibly high N (emotional arousal)
    """
    exclamation_rate = text.count("!") / n
    question_rate    = text.count("?") / n
    words            = text.split()
    caps_rate        = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)

    return {
        "exclamation_rate": exclamation_rate,
        "question_rate":    question_rate,
        "caps_rate":        caps_rate,
    }


def _emotion_features(text: str) -> Dict[str, float]:
    """
    NRC Emotion Lexicon — 8 basic emotions + positive/negative.
    anger, fear → N
    joy, trust  → A, E
    anticipation, surprise → O
    """
    try:
        nrc = NRCLex(text)
        raw = nrc.raw_emotion_scores
        total = max(sum(raw.values()), 1)
        emotions = ["anger", "fear", "anticipation", "trust",
                    "surprise", "sadness", "joy", "disgust",
                    "positive", "negative"]
        return {
            f"nrc_{e}_rate": raw.get(e, 0) / total
            for e in emotions
        }
    except Exception as exc:
        logger.warning("NRC extraction failed: %s", exc)
        return {f"nrc_{e}_rate": 0.0 for e in
                ["anger","fear","anticipation","trust",
                 "surprise","sadness","joy","disgust",
                 "positive","negative"]}


def _tense_features(doc, n: int) -> Dict[str, float]:
    """
    Past-tense usage    → reflective, possibly high N
    Future-tense usage  → planning, high C
    Present-tense usage → in-the-moment, high E
    """
    past    = sum(1 for t in doc if t.tag_ in ("VBD", "VBN"))
    present = sum(1 for t in doc if t.tag_ in ("VBP", "VBZ", "VBG"))
    future  = sum(1 for t in doc if t.text.lower() in ("will", "shall", "gonna", "going"))

    return {
        "past_tense_rate":    past    / n,
        "present_tense_rate": present / n,
        "future_tense_rate":  future  / n,
    }


# ── Empty feature dict (used as fallback) ────────────────────────────────────

def _empty_feature_dict() -> Dict[str, float]:
    """Return all features as NaN — predictor will handle imputation."""
    keys = [
        "pronoun_i_rate","pronoun_we_rate","pronoun_you_rate",
        "pronoun_third_sing_rate","pronoun_third_plur_rate",
        "hedging_rate","certainty_rate","cognitive_rate","social_rate",
        "vader_compound_mean","vader_compound_std","vader_pos_mean",
        "vader_neg_mean","vader_neu_mean","sentiment_volatility",
        "type_token_ratio","avg_sentence_length","flesch_kincaid_grade",
        "flesch_reading_ease","avg_syllables_per_word","unique_word_count",
        "exclamation_rate","question_rate","caps_rate",
        "nrc_anger_rate","nrc_fear_rate","nrc_anticipation_rate",
        "nrc_trust_rate","nrc_surprise_rate","nrc_sadness_rate",
        "nrc_joy_rate","nrc_disgust_rate","nrc_positive_rate","nrc_negative_rate",
        "past_tense_rate","present_tense_rate","future_tense_rate",
    ]
    return {k: float("nan") for k in keys}
