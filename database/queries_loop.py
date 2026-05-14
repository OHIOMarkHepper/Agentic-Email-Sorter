from database.queries import DatabaseManager

class queries_loop:
    def __init__(self, agent, db_manager, db_path):
        self.agent = agent
        self.db_manager = db_manager
        if db_path.suffix != ".db":
            raise ValueError("db_path must point to a .db file")
        self.db_path = db_path

    def run(self):
        while True:
            printf("Not implemented yet. This will be the loop for querying the database of classified emails and clusters.")
            break
    
    def query_clusters(self):
        """query_clusters: Queries the database for distinct cluster labels and prints them."""
        clusters = self.agent.db_manager.select_clusters()
        print("Clusters found in the database:")
        for cluster in clusters:
            print(f"- Cluster {cluster}")

    def query_emails_by_cluster(self, cluster_label):
        """query_emails_by_cluster: Queries the database for emails belonging to a specific cluster.

        Args:
            cluster_label (str): The label of the cluster to query.

        Returns:
            pd.DataFrame: A DataFrame containing the emails in the specified cluster.
        """
        results = self.agent.db_manager.query_emails_by_cluster(cluster_label)
        df = pd.DataFrame(results, columns=self.agent.db_manager.get_column_names())
        print(f"Emails in cluster '{cluster_label}':")
        print(df[["id", "sender", "subject", "timestamp"]])
        return df