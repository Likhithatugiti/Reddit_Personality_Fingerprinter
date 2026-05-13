"""
src/features/preprocessor.py

Cleans raw Reddit comment text using spaCy.
Preserves enough linguistic signal for psycholinguistic feature extraction
(i.e. does NOT aggressively lemmatise or strip stop-words — personality
signals live in function words like "I", "maybe", "always").
"""

import re
import logging
from typing import List

import spacy
from spacy.tokens import Doc

from config import SPACY_MODEL, MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)

# Load once at import time
try:
    _nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    _nlp.max_length = MAX_TEXT_LENGTH
    logger.info("spaCy model '%s' loaded", SPACY_MODEL)
except OSError:
    raise OSError(
        f"spaCy model '{SPACY_MODEL}' not found. "
        f"Run: python -m spacy download {SPACY_MODEL}"
    )


# ── Public helpers ────────────────────────────────────────────────────────────

def clean_comment(text: str) -> str:
    """
    Light cleaning that removes Reddit-specific noise while preserving
    the linguistic patterns that matter for personality detection.

    What is removed
    ---------------
    - Markdown links:   [text](url)  → text
    - Bare URLs:        https://...  → ''
    - Reddit mentions:  u/username   → ''
    - Subreddit refs:   r/subreddit  → ''
    - Repeated spaces / newlines

    What is kept
    ------------
    - Punctuation (!, ?, . carry sentiment)
    - Capitalisation (for VADER)
    - All function words (I, we, maybe, always, …)
    """
    # Markdown link → keep anchor text
    text = re.sub(r'\[([^\]]+)\]\(https?://\S+\)', r'\1', text)
    # Bare URLs
    text = re.sub(r'https?://\S+', '', text)
    # Reddit user / subreddit refs
    text = re.sub(r'\b[ur]/\w+', '', text)
    # Block-quote marker
    text = re.sub(r'^>+\s?', '', text, flags=re.MULTILINE)
    # Excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_comments(comments: List[str]) -> List[str]:
    """
    Apply clean_comment to a list of strings.
    Drops comments that become empty after cleaning.
    """
    cleaned = [clean_comment(c) for c in comments]
    cleaned = [c for c in cleaned if len(c) > 10]
    return cleaned


def tokenise(text: str) -> Doc:
    """Return a spaCy Doc for a single text string."""
    return _nlp(text)


def tokenise_batch(texts: List[str], batch_size: int = 50) -> List[Doc]:
    """Efficient batch tokenisation via spaCy's pipe."""
    return list(_nlp.pipe(texts, batch_size=batch_size))


def get_sentences(text: str) -> List[str]:
    """
    Split text into sentences using spaCy's sentenciser.
    Falls back to period-split if sentenciser is not in the pipeline.
    """
    doc = _nlp(text)
    if doc.has_annotation("SENT_START"):
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # Fallback
    return [s.strip() for s in text.split('.') if s.strip()]
