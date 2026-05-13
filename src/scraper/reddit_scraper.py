"""
src/scraper/reddit_scraper.py

Fetches a Reddit user's public comment history using PRAW.
Returns a cleaned list of comment strings.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import praw
from praw.exceptions import PRAWException

from config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    COMMENT_LIMIT,
    MIN_COMMENT_LENGTH,
    SUBREDDITS_BLACKLIST,
)

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    username: str
    comments: List[str] = field(default_factory=list)
    total_fetched: int = 0
    total_kept: int = 0
    subreddits_seen: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.comments) > 0

    @property
    def word_count(self) -> int:
        return sum(len(c.split()) for c in self.comments)


class RedditScraper:
    """
    Wraps PRAW to fetch and filter comments for a given username.

    Usage
    -----
    scraper = RedditScraper()
    result  = scraper.fetch("spez", limit=300)
    print(result.comments[:3])
    """

    def __init__(self):
        self._reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            # read-only; no login needed
        )
        logger.info("PRAW client initialised (read-only mode)")

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch(self, username: str, limit: int = COMMENT_LIMIT) -> ScrapeResult:
        """
        Fetch up to `limit` comments for `username`.

        Parameters
        ----------
        username : str
            Reddit username (without u/)
        limit : int
            Maximum number of comments to request from the API.
            PRAW caps at 1000.

        Returns
        -------
        ScrapeResult
        """
        result = ScrapeResult(username=username)

        try:
            redditor = self._reddit.redditor(username)
            comments_raw = list(redditor.comments.new(limit=limit))
        except PRAWException as exc:
            result.error = f"PRAW error: {exc}"
            logger.error(result.error)
            return result
        except Exception as exc:
            result.error = f"Unexpected error: {exc}"
            logger.error(result.error)
            return result

        result.total_fetched = len(comments_raw)
        seen_subreddits = set()

        for comment in comments_raw:
            # Skip deleted/bot content
            author = getattr(comment, "author", None)
            if author is None or str(author) in SUBREDDITS_BLACKLIST:
                continue

            body = comment.body.strip()
            subreddit = str(comment.subreddit)

            # Basic quality filter
            if len(body) < MIN_COMMENT_LENGTH:
                continue
            if body in ("[deleted]", "[removed]"):
                continue

            result.comments.append(body)
            seen_subreddits.add(subreddit)

        result.total_kept = len(result.comments)
        result.subreddits_seen = sorted(seen_subreddits)

        logger.info(
            "Fetched %d comments for u/%s, kept %d after filtering",
            result.total_fetched, username, result.total_kept,
        )
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def merge_comments(comments: List[str], separator: str = " ") -> str:
        """Concatenate all comments into a single string for bulk NLP."""
        return separator.join(comments)
