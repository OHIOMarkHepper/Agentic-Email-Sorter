from database.queries import DatabaseManager
import pandas as pd


class queries_loop:
    """ A class that handles a database and its interactions with the user
    """
    def __init__(self, agent, db_manager, db_path):
        self.agent = agent
        self.db_manager = db_manager
        if not db_path.endswith(".db"):
            raise ValueError("db_path must point to a .db file")
        self.db_path = db_path
        self.db_manager.db_path = db_path  # <-- sync the manager's path

    def run(self):
        """run """
        while True:
            self.query_clusters() #
            self.query_columns()
            print(f"Not implemented yet. This will be the loop for querying the database of classified emails and clusters.")
            break

    
    def query_clusters(self):
        """query_clusters: Queries the database for distinct cluster labels and prints them."""
        clusters = self.db_manager.get_cluster_labels()
        print("Clusters found in the database:")
        for cluster in clusters:
            print(f"- Cluster {cluster}")

    def query_columns(self):
        """query_columns: Queries the database for column names and prints them."""
        columns = self.db_manager.get_column_names()
        print("Columns found in the database:")
        for column in columns: 
            print(f"- {column}")

    def query_emails_by_cluster(self, cluster_label):
        """query_emails_by_cluster: Queries the database for emails belonging to a specific cluster.

        Args:
            cluster_label (str): The label of the cluster to query.

        Returns:
            pd.DataFrame: A DataFrame containing the emails in the specified cluster.
        """
        results = self.db_manager.query_emails_by_cluster(cluster_label)
        df = pd.DataFrame(results, columns=self.db_manager.get_column_names())
        print(f"Emails in cluster '{cluster_label}':")
        print(df[["id", "cluster_id", "cluster_label"]])
        return df