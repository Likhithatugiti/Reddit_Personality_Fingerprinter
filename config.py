"""
config.py
Central configuration for the Reddit Personality Fingerprinter.
All tunable parameters live here so nothing is hardcoded across modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_DIR   = BASE_DIR / "models"
OUTPUT_DIR  = BASE_DIR / "outputs"
DATA_DIR    = BASE_DIR / "data"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ── Reddit API ────────────────────────────────────────────────────────────────
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "PersonalityFingerprinter/1.0")

# ── Scraping ──────────────────────────────────────────────────────────────────
COMMENT_LIMIT        = 500          # max comments fetched per user
MIN_COMMENT_LENGTH   = 15           # characters; shorter comments are noise
MIN_COMMENTS_REQUIRED = 50          # warn if user has fewer than this
SUBREDDITS_BLACKLIST  = {           # skip deleted / bot accounts
    "[deleted]", "[removed]", "AutoModerator"
}

# ── NLP ───────────────────────────────────────────────────────────────────────
SPACY_MODEL          = "en_core_web_sm"
MAX_TEXT_LENGTH      = 1_000_000    # spaCy char limit guard

# ── Features ──────────────────────────────────────────────────────────────────
HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "seem",
    "appears", "probably", "likely", "i think", "i believe",
    "i guess", "i suppose", "sort of", "kind of", "somewhat"
}

CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "always", "never",
    "clearly", "obviously", "undoubtedly", "without doubt",
    "for sure", "no question", "guaranteed"
}

COGNITIVE_WORDS = {
    "because", "cause", "since", "reason", "therefore",
    "thus", "hence", "result", "think", "know", "consider",
    "understand", "realize", "believe", "wonder", "noticed"
}

SOCIAL_WORDS = {
    "they", "their", "friend", "people", "everyone", "someone",
    "together", "us", "we", "community", "others", "family"
}

# ── Model ─────────────────────────────────────────────────────────────────────
TRAITS               = ["O", "C", "E", "A", "N"]
TRAIT_NAMES          = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}
SCORE_MIN            = 1.0
SCORE_MAX            = 7.0
LGBM_PARAMS          = {
    "n_estimators":   400,
    "learning_rate":  0.05,
    "max_depth":      5,
    "num_leaves":     31,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "random_state":   42,
    "verbose":        -1,
}
CV_FOLDS             = 5

# ── Visualisation ─────────────────────────────────────────────────────────────
RADAR_COLORS = {
    "fill":   "rgba(99, 110, 250, 0.3)",
    "line":   "rgba(99, 110, 250, 0.9)",
}
