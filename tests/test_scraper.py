"""
tests/test_scraper.py

Unit tests for the Reddit scraper.
Uses mock PRAW objects so no real API call is made.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.scraper.reddit_scraper import RedditScraper, ScrapeResult


class FakeComment:
    def __init__(self, body, author="testuser", subreddit="python"):
        self.body       = body
        self.author     = MagicMock()
        self.author.__str__ = lambda self: author
        self.subreddit  = MagicMock()
        self.subreddit.__str__ = lambda self: subreddit


@pytest.fixture
def mock_redditor():
    redditor = MagicMock()
    redditor.comments.new.return_value = [
        FakeComment("This is a normal comment about Python programming."),
        FakeComment("I really enjoy working on open source projects."),
        FakeComment("Maybe we should consider a different approach here."),
        FakeComment("short"),   # too short — should be filtered
        FakeComment("[deleted]"),  # should be filtered
    ]
    return redditor


def test_scrape_result_dataclass():
    result = ScrapeResult(username="testuser", comments=["a", "b"])
    assert result.success
    assert result.word_count == 2


def test_scrape_result_empty():
    result = ScrapeResult(username="testuser")
    assert not result.success


def test_fetch_filters_short_and_deleted(mock_redditor):
    scraper = RedditScraper.__new__(RedditScraper)
    scraper._reddit = MagicMock()
    scraper._reddit.redditor.return_value = mock_redditor

    result = scraper.fetch("testuser", limit=10)

    assert result.total_fetched == 5
    assert result.total_kept == 3   # "short" and "[deleted]" filtered out
    assert all(len(c) >= 15 for c in result.comments)


def test_fetch_handles_praw_exception():
    scraper = RedditScraper.__new__(RedditScraper)
    scraper._reddit = MagicMock()
    scraper._reddit.redditor.side_effect = Exception("Network error")

    result = scraper.fetch("someuser")
    assert not result.success
    assert "error" in result.error.lower() or result.error


def test_merge_comments():
    comments = ["Hello world.", "This is a test."]
    merged   = RedditScraper.merge_comments(comments)
    assert "Hello world." in merged
    assert "This is a test." in merged
