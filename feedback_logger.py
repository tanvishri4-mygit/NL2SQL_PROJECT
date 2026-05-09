"""
feedback_logger.py
==================
Handles saving user feedback (thumbs up/down) as training data.

Thumbs up  → saves {question, sql} to data/feedback_positive.csv
Thumbs down + correct SQL → verifies SQL against DB → saves to data/feedback_pairs.csv

feedback_pairs.csv is automatically included in the next fine-tune run.
"""

import os
import csv
import sqlite3
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "data" / "spider" / "database" / "chinook_1" / "chinook_1.sqlite"
POSITIVE_CSV = BASE_DIR / "data" / "feedback_positive.csv"
PAIRS_CSV    = BASE_DIR / "data" / "feedback_pairs.csv"

_HEADERS = ["question", "sql", "db_id", "source"]


def _ensure_csv(path: Path):
    """Create CSV with headers if it doesn't exist."""
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_HEADERS)


def verify_sql(sql: str) -> tuple:
    """
    Run SQL against Chinook DB.
    Returns (success: bool, error: str, row_count: int)
    """
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, None, len(rows)
    except Exception as e:
        return False, str(e), 0


def save_positive(question: str, sql: str) -> bool:
    """Save thumbs-up pair to feedback_positive.csv."""
    try:
        _ensure_csv(POSITIVE_CSV)
        with open(POSITIVE_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([question, sql, "chinook_1", "feedback_positive"])
        return True
    except Exception:
        return False


def save_correction(question: str, correct_sql: str) -> tuple:
    """
    Verify and save user-provided correct SQL to feedback_pairs.csv.
    Returns (success: bool, error: str, row_count: int)
    """
    ok, err, count = verify_sql(correct_sql)
    if not ok:
        return False, err, 0

    try:
        _ensure_csv(PAIRS_CSV)
        with open(PAIRS_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([question, correct_sql, "chinook_1", "feedback_correction"])
        return True, None, count
    except Exception as e:
        return False, str(e), 0


def get_feedback_stats() -> dict:
    """Return count of saved feedback pairs."""
    def count_rows(path):
        if not path.exists():
            return 0
        with open(path, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)  # minus header

    return {
        "positive"   : count_rows(POSITIVE_CSV),
        "corrections": count_rows(PAIRS_CSV),
    }
