"""
reddit_scraper.py
-----------------
Fetches a Reddit user's comment history using Arctic Shift
(https://arctic-shift.photon-reddit.com) instead of the Reddit API (PRAW).

"""

import time
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"

# Fallback: Reddit's own unofficial JSON endpoint (no auth needed)
REDDIT_JSON_BASE = "https://www.reddit.com/user/{username}/comments.json"


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

    # ------------------------------------------------------------------
    # Public interface (same as the original PRAW-based scraper)
    # ------------------------------------------------------------------

    def fetch_comments(self, username: str) -> list[dict]:
        """
        Return a list of comment dicts for *username*.

        Each dict contains at minimum:
            body        : str   – raw comment text
            score       : int   – upvotes
            subreddit   : str   – subreddit name
            created_utc : float – unix timestamp
        """
        logger.info("Fetching comments for u/%s via Arctic Shift …", username)
        comments = self._fetch_arctic_shift(username)

        if not comments:
            logger.warning(
                "Arctic Shift returned 0 comments for u/%s – trying Reddit JSON fallback.",
                username,
            )
            comments = self._fetch_reddit_json(username)

        logger.info("Retrieved %d comments for u/%s.", len(comments), username)
        return comments

    # ------------------------------------------------------------------
    # Arctic Shift (primary)
    # ------------------------------------------------------------------

    def _fetch_arctic_shift(self, username: str) -> list[dict]:
        """
        Paginate through Arctic Shift's /comments endpoint.

        Docs: https://arctic-shift.photon-reddit.com/api/comments?author=<user>
        """
        comments: list[dict] = []
        after: Optional[str] = None
        page_size = min(100, self.max_comments)  # Arctic Shift max per page = 100

        while len(comments) < self.max_comments:
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

            # Pagination: Arctic Shift returns a `after` cursor
            after = data.get("metadata", {}).get("after")
            if not after:
                break

            # polite delay
            time.sleep(0.25)

        return comments[: self.max_comments]

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

    # ------------------------------------------------------------------
    # Reddit JSON fallback (no auth, ~1000 comment limit)
    # ------------------------------------------------------------------

    def _fetch_reddit_json(self, username: str) -> list[dict]:
        """
        Use Reddit's public .json endpoint as a fallback.
        Returns at most ~1000 recent comments (Reddit hard limit).
        """
        comments: list[dict] = []
        after: Optional[str] = None
        url = REDDIT_JSON_BASE.format(username=username)

        while len(comments) < self.max_comments:
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

            time.sleep(1.0)  # Reddit public endpoint needs a gentle touch

        return comments[: self.max_comments]

    # ------------------------------------------------------------------
    # Shared HTTP helper with retry logic
    # ------------------------------------------------------------------

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
