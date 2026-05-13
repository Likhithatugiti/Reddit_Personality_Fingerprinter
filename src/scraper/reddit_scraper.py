"""
reddit_scraper.py
-----------------
Fetches a Reddit user's comment history using Arctic Shift
(https://arctic-shift.photon-reddit.com) instead of the Reddit API (PRAW).
"""

import time
import logging
from typing import Optional, List
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"
REDDIT_JSON_BASE = "https://www.reddit.com/user/{username}/comments.json"

@dataclass
class ScrapeResult:
    success: bool
    comments: list
    total_kept: int
    subreddits_seen: list
    error: Optional[str] = None

class RedditScraper:
    """
    Fetches public comment history for a Reddit user.

    Primary source  : Arctic Shift API (deep archive, no credentials)
    Fallback source : Reddit JSON endpoint (recent ~1000 comments)
    """

    def __init__(
        self,
        max_comments: int = 500,
        request_timeout: int = 15,
        retry_attempts: int = 3,
    ):
        self.max_comments = max_comments
        self.timeout = request_timeout
        self.retries = retry_attempts
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "PersonalityFingerprinter/2.0 (arctic-shift backend)"}
        )

    # --------------- App-expected interface ---------------
    def fetch(self, username: str, limit: int = 200) -> ScrapeResult:
        """
        Fetch up to *limit* comments for *username*.
        Returns a ScrapeResult object expected by app.py and other modules.
        """
        effective_limit = min(limit, self.max_comments)
        logger.info("Fetching comments for u/%s via Arctic Shift ...", username)
        comments = self._fetch_arctic_shift(username, effective_limit)

        if not comments:
            logger.warning("Arctic Shift returned 0 comments — trying Reddit JSON fallback.")
            comments = self._fetch_reddit_json(username, effective_limit)

        if not comments:
            return ScrapeResult(
                success=False,
                comments=[],
                total_kept=0,
                subreddits_seen=[],
                error=f"No public comments found for u/{username}. The account may be private, suspended, or non-existent.",
            )

        subreddits = sorted({c["subreddit"] for c in comments if c.get("subreddit")})
        return ScrapeResult(
            success=True,
            comments=comments,
            total_kept=len(comments),
            subreddits_seen=subreddits,
        )

    # --------------- Legacy/test interface ---------------
    def fetch_comments(self, username: str) -> list:
        """
        Return a list of comment dicts for *username*.
        Each dict contains at minimum:
            body        : str   – raw comment text
            score       : int   – upvotes
            subreddit   : str   – subreddit name
            created_utc : float – unix timestamp
        """
        logger.info("Fetching comments for u/%s via Arctic Shift …", username)
        comments = self._fetch_arctic_shift(username, self.max_comments)

        if not comments:
            logger.warning(
                "Arctic Shift returned 0 comments for u/%s – trying Reddit JSON fallback.",
                username,
            )
            comments = self._fetch_reddit_json(username, self.max_comments)

        logger.info("Retrieved %d comments for u/%s.", len(comments), username)
        return comments

    # --------------- Arctic Shift (primary) ---------------
    def _fetch_arctic_shift(self, username: str, limit: int) -> list:
        comments: list = []
        after: Optional[str] = None
        page_size = min(100, limit)

        while len(comments) < limit:
            params: dict = {
                "author": username,
                "limit": page_size,
                "sort": "desc",
            }
            if after:
                params["after"] = after

            data = self._get(f"{ARCTIC_SHIFT_BASE}/comments", params)
            if data is None:
                break

            page = data.get("data", [])
            if not page:
                break  # exhausted

            comments.extend(self._normalise_arctic(c) for c in page)

            # Pagination: Arctic Shift returns an `after` cursor
            after = data.get("metadata", {}).get("after")
            if not after:
                break

            time.sleep(0.25)

        return comments[:limit]

    @staticmethod
    def _normalise_arctic(raw: dict) -> dict:
        """Map Arctic Shift field names to our internal schema."""
        return {
            "body": raw.get("body", ""),
            "score": int(raw.get("score", 0)),
            "subreddit": raw.get("subreddit", ""),
            "created_utc": float(raw.get("created_utc", 0)),
            "id": raw.get("id", ""),
            "link_id": raw.get("link_id", ""),
            "permalink": raw.get("permalink", ""),
        }

    # --------------- Reddit JSON fallback ---------------
    def _fetch_reddit_json(self, username: str, limit: int) -> list:
        comments: list = []
        after: Optional[str] = None
        url = REDDIT_JSON_BASE.format(username=username)

        while len(comments) < limit:
            params: dict = {"limit": 100, "raw_json": 1}
            if after:
                params["after"] = after

            data = self._get(url, params)
            if data is None:
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                c = child.get("data", {})
                comments.append(
                    {
                        "body": c.get("body", ""),
                        "score": int(c.get("score", 0)),
                        "subreddit": c.get("subreddit", ""),
                        "created_utc": float(c.get("created_utc", 0)),
                        "id": c.get("id", ""),
                        "link_id": c.get("link_id", ""),
                        "permalink": "https://reddit.com" + c.get("permalink", ""),
                    }
                )

            after = data.get("data", {}).get("after")
            if not after:
                break

            time.sleep(1.0)

        return comments[:limit]

    # --------------- Shared HTTP helper ---------------
    def _get(self, url: str, params: dict) -> Optional[dict]:
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else "?"
                if status == 404:
                    logger.warning("User not found (404). Stopping.")
                    return None
                if status == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited. Waiting %ds …", wait)
                    time.sleep(wait)
                else:
                    logger.error("HTTP %s on attempt %d/%d", status, attempt, self.retries)
            except requests.exceptions.RequestException as exc:
                logger.error("Request error on attempt %d/%d: %s", attempt, self.retries, exc)
                time.sleep(1)
        return None
