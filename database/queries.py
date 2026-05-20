import sqlite3
import re
import numpy as np
import pandas as pd

class DatabaseManager:
    """ A class that handles SQLite queries and database management
    """
    def __init__(self, db_path: str):
        if not db_path.endswith(".db"):
            raise ValueError("db_path must point to a .db file")
        self.db_path = db_path

    def create_clusters_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emails (
                    id            INTEGER PRIMARY KEY,  -- auto-assigned when omitted
                    cluster_id    INTEGER,
                    cluster_label TEXT,
                    body          TEXT
                )
            """)

    def save_emails_bulk(self, emails: list[dict]):
        """emails: list of dicts with keys cluster_id, cluster_label, body — id is auto-assigned"""
        if not emails:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO emails (cluster_id, cluster_label, body)
                VALUES (:cluster_id, :cluster_label, :body)
            """, emails)
    
    def get_column_names(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(emails)")
            return [row[1] for row in cursor.fetchall()]

    def get_cluster_labels(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT cluster_label FROM emails")
            return [row[0] for row in cursor.fetchall()]

    def query_emails_by_cluster(self, cluster_label: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM emails WHERE cluster_label = ?",
                (cluster_label,)
            )
            return cursor.fetchall()

    @staticmethod
    def _sanitize(name: str) -> str:
        """ Sanitizes user-input for cohesion and security purposes

        Args:
            name (str): the user-input string to sanitize

        Raises:
            TypeError: if the input is not a string
            ValueError: if the input is empty or starts with a non-alphabetic character
            ValueError: if the input is too long

        Returns:
            str: the sanitized string
        """
        if not isinstance(name, str):
            raise TypeError("Column name must be a string")
        cleaned = re.sub(r"[^\w]", "_", name.strip()).lower()
        if not cleaned or not cleaned[0].isalpha():
            raise ValueError(f"Invalid column name: '{name}'")
        if len(cleaned) > 64:
            raise ValueError("Column name too long (max 64 chars)")
        return cleaned