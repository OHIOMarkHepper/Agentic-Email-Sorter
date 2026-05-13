from cache import CACHE, CLUSTER_STATE
from agent.llm import get_llm_analysis
import numpy as np
import hashlib, json



class EmailAgent:

    def __init__(self, strategy, vectorizer, llm_enabled=True):
        self.strategy = strategy
        self.vectorizer = vectorizer
        self.llm_enabled = llm_enabled
        self.cluster_names = {}

    def train(self, texts):
        self.strategy.fit(texts, self.vectorizer)

        report = self.strategy.summarize()

        if self.llm_enabled:
            cluster_summaries = self._build_summaries(report)
            llm_result = get_llm_analysis(cluster_summaries)
            self.cluster_names = llm_result.get("cluster_names", {})

        self.report = report

    def classify(self, email):
        label = self.strategy.predict([email])[0]

        if isinstance(label, int):  # KMeans case
            label = self.cluster_names.get(label, f"Cluster_{label}")

        return label

    def _build_summaries(self, report):
        return {
            str(k): v for k, v in report.items()
        }


def get_signature(labels, report):
    """Produces a stable hash of the current clustering state."""
    raw = json.dumps({
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "report_keys": sorted(str(k) for k in report.keys())
    }, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()

def should_call_llm(sig):
    """Returns True if the clustering has changed since the last LLM call."""
    return CLUSTER_STATE["last_signature"] != sig

def build_cluster_summaries(report, labels):
    """Builds a JSON-serialisable summary dict for the LLM."""
    unique, counts = np.unique(labels, return_counts=True)
    summaries = {}
    for cluster_id, count in zip(unique, counts):
        entry = dict(report.get(cluster_id, {}))
        entry["size"] = int(count)
        summaries[str(cluster_id)] = entry
    return summaries


def run_agent_loop(filepath, config, load_data_func, vectorizer_fn, clustering_fn, report_fn, iterations=3):
    """run_agent_loop - the main loop of the agent, which loads data, runs clustering, and calls the LLM for analysis

    Args:
        filepath (string): the path to the data file
        config (dict): the configuration for the vectorizer and clustering
        load_data_func (function): the function to load the data
        vectorizer_fn (function): the function to create the vectorizer
        clustering_fn (function): the function to perform clustering
        report_fn (function): the function to generate the report
        iterations (int, optional): Amount of iterations to run. Defaults to 3.

    Returns:
        dict: the final LLM analysis result
    """

    # Load data and detect schema
    df, texts, _ = load_data_func(filepath)

    # Create vectorizer and transform texts
    vectorizer = vectorizer_fn(config)
    X = vectorizer.fit_transform(texts)

    # Cluster data
    model, labels = clustering_fn(X, config)

    # Generate report
    report = report_fn(model, vectorizer.get_feature_names_out())

    # Agent loop with LLM analysis
    for i in range(iterations):
        print(f"\n--- Iteration {i+1} ---")

        sig = get_signature(labels, report)

        # Check if we need to call the LLM (if the clustering changed significantly)
        if should_call_llm(sig):

            cluster_summaries = build_cluster_summaries(report, labels)
            llm_result = get_llm_analysis(cluster_summaries)

            CLUSTER_STATE["last_signature"] = sig
            CLUSTER_STATE["last_llm_result"] = llm_result

            print("LLM ran")
        else:
            llm_result = CLUSTER_STATE["last_llm_result"]
            print("LLM skipped (cached)")

        print("Quality:", llm_result.get("quality"))

    return llm_result