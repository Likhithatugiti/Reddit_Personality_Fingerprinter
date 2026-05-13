"""
tests/test_scraper.py
Tests for the Arctic Shift-based Reddit scraper.
Uses mocked HTTP responses so no network access is required.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.scraper.reddit_scraper import RedditScraper


# ── Fixtures ──────────────────────────────────────────────────────────────

ARCTIC_PAGE_1 = {
    "data": [
        {
            "id": "abc123",
            "body": "This is a test comment.",
            "score": 42,
            "subreddit": "python",
            "created_utc": 1700000000,
            "link_id": "t3_xyz",
            "permalink": "/r/python/comments/xyz/abc123",
        }
    ],
    "metadata": {"after": None},
}

ARCTIC_EMPTY = {"data": [], "metadata": {"after": None}}

REDDIT_JSON_PAGE = {
    "data": {
        "children": [
            {
                "data": {
                    "id": "def456",
                    "body": "Fallback comment.",
                    "score": 10,
                    "subreddit": "learnpython",
                    "created_utc": 1699000000,
                    "link_id": "t3_aaa",
                    "permalink": "/r/learnpython/comments/aaa/def456",
                }
            }
        ],
        "after": None,
    }
}


# ── Tests ─────────────────────────────────────────────────────────────────

class TestRedditScraper:

    def test_fetch_comments_arctic_shift_success(self):
        scraper = RedditScraper(max_comments=10)
        mock_resp = MagicMock()
        mock_resp.json.return_value = ARCTIC_PAGE_1
        mock_resp.raise_for_status.return_value = None

        with patch.object(scraper.session, "get", return_value=mock_resp):
            comments = scraper.fetch_comments("testuser")

        assert len(comments) == 1
        assert comments[0]["body"] == "This is a test comment."
        assert comments[0]["score"] == 42
        assert comments[0]["subreddit"] == "python"

    def test_fallback_to_reddit_json_when_arctic_empty(self):
        scraper = RedditScraper(max_comments=10)

        # Arctic Shift returns empty; Reddit JSON returns data
        responses = [ARCTIC_EMPTY, REDDIT_JSON_PAGE]
        call_count = {"n": 0}

        def side_effect(url, params, timeout):
            mock = MagicMock()
            mock.json.return_value = responses[call_count["n"]]
            mock.raise_for_status.return_value = None
            call_count["n"] += 1
            return mock

        with patch.object(scraper.session, "get", side_effect=side_effect):
            comments = scraper.fetch_comments("testuser")

        assert len(comments) == 1
        assert comments[0]["body"] == "Fallback comment."

    def test_max_comments_respected(self):
        scraper = RedditScraper(max_comments=1)
        # Page has 3 comments but max_comments=1
        page = {
            "data": [
                {"id": str(i), "body": f"Comment {i}", "score": i,
                 "subreddit": "test", "created_utc": 1700000000 + i,
                 "link_id": "t3_x", "permalink": "/r/test"}
                for i in range(3)
            ],
            "metadata": {"after": None},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = page
        mock_resp.raise_for_status.return_value = None

        with patch.object(scraper.session, "get", return_value=mock_resp):
            comments = scraper.fetch_comments("testuser")

        assert len(comments) == 1

    def test_normalise_arctic_fields(self):
        raw = {
            "id": "zzz",
            "body": "Hello world",
            "score": "99",        # sometimes comes as string
            "subreddit": "AskReddit",
            "created_utc": "1700000000",
            "link_id": "t3_qqq",
            "permalink": "/r/AskReddit/comments/qqq/zzz",
        }
        result = RedditScraper._normalise_arctic(raw)
        assert result["body"] == "Hello world"
        assert isinstance(result["score"], int)
        assert isinstance(result["created_utc"], float)

    def test_user_not_found_returns_empty(self):
        import requests as req
        scraper = RedditScraper(max_comments=10)

        http_err = req.exceptions.HTTPError(response=MagicMock(status_code=404))

        with patch.object(scraper.session, "get", side_effect=http_err):
            comments = scraper.fetch_comments("ghost_user_xyz")

        assert comments == []
