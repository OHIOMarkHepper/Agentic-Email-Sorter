import sqlite3
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path

    ALLOWED_TABLES = {"emails"}
    ALLOWED_CLUSTER_COLS = {"cluster_label", "category"}

    def _validate_identifier(self, name, allowed):
        if name not in allowed:
            raise ValueError(f"Invalid identifier: '{name}'")

    def load_dataframe(self, table_name="emails"):
        self._validate_identifier(table_name, ALLOWED_TABLES)
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM emails", conn)  # hardcoded now safe
        return df

    def query_emails_by_cluster(self, cluster_label):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM emails WHERE cluster_label = ?",
                (cluster_label,)   # parameterized — never interpolated
            )
            results = cursor.fetchall()
        return results

    def save_dataframe(self, df, table_name="emails", mode="append"):
        if mode not in ("replace", "append", "fail"):
            raise ValueError(f"Invalid mode: {mode}")
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists=mode, index=False)
    