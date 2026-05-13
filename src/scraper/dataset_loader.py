# src/scraper/dataset_loader.py

import sqlite3
import pandas as pd
from src.scraper.reddit_scraper import ScrapeResult
from config import MIN_COMMENT_LENGTH

class DatasetLoader:
    """
    Loads comments from the Kaggle Reddit May 2015 SQLite dataset
    instead of hitting the live Reddit API.
    
    Dataset columns we use: author, body, subreddit
    """

    def __init__(self, db_path: str):
        """
        Parameters
        ----------
        db_path : str  path to the downloaded .sqlite file
                       e.g. 'data/database.sqlite'
        """
        self.db_path = db_path

    def fetch(self, username: str, limit: int = 500) -> ScrapeResult:
        """
        Fetch comments for a given username from the local SQLite file.
        Returns the same ScrapeResult as the live scraper — 
        so the rest of the pipeline works unchanged.
        """
        conn  = sqlite3.connect(self.db_path)
        query = """
            SELECT body, subreddit
            FROM May2015
            WHERE author = ?
            AND body NOT IN ('[deleted]', '[removed]')
            AND length(body) >= ?
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(username, MIN_COMMENT_LENGTH, limit))
        conn.close()

        result = ScrapeResult(
            username=username,
            comments=df['body'].tolist(),
            total_fetched=len(df),
            total_kept=len(df),
            subreddits_seen=df['subreddit'].unique().tolist(),
        )
        return result

    def get_all_users(self, min_comments: int = 50) -> list:
        """
        Returns list of usernames who have at least min_comments.
        Useful for batch processing.
        """
        conn  = sqlite3.connect(self.db_path)
        query = """
            SELECT author, COUNT(*) as comment_count
            FROM May2015
            WHERE body NOT IN ('[deleted]', '[removed]')
            GROUP BY author
            HAVING comment_count >= ?
            ORDER BY comment_count DESC
        """
        df = pd.read_sql_query(query, conn, params=(min_comments,))
        conn.close()
        return df['author'].tolist()
