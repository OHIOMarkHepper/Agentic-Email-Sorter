import json
import hashlib
import numpy as np

from cache.cache import CACHE, CLUSTER_STATE, LLM_CACHE
from agent.llm import get_llm_analysis, chat_with_agent
from agent.providers import get_llm_provider
from ml.strategies import KMeansStrategy


class EmailAgent:
    """EmailAgent owns all clustering state and exposes clean methods for
    training, retraining, relabeling, classifying, and retrieving analysis.
    """

    def __init__(self, strategy, vectorizer, config=None, llm_enabled=True):
        self.strategy = strategy
        self.vectorizer = vectorizer
        self.config = config
        self.llm_enabled = llm_enabled
        self.cluster_names: dict = {}
        self.report: dict = {}
        self._analysis: dict = {}   # cached result from the last LLM analysis call
        self._texts: list = []      # kept so retrain() can refit without re-loading data

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train(self, texts: list) -> dict:
        """Fit the strategy, summarize clusters, and run LLM analysis.

        Args:
            texts: The list of email body strings to train on.

        Returns:
            The LLM analysis dict (cluster_names, quality, issues, …).
        """
        self._texts = texts
        self.strategy.fit(texts, self.vectorizer)
        self.report = self.strategy.summarize()

        if self.llm_enabled:
            summaries = self.get_cluster_summaries()
            self._analysis = get_llm_analysis(summaries)
            # Auto-populate cluster_names from LLM result (can be overridden by relabel)
            for k, v in self._analysis.get("cluster_names", {}).items():
                self.cluster_names[k] = v
                try:
                    self.cluster_names[int(k)] = v
                except ValueError:
                    pass

        return self._analysis

    # ------------------------------------------------------------------
    # Retrain
    # ------------------------------------------------------------------

    def retrain(self, k: int) -> dict:
        """Replace the current strategy with a new KMeans model at a given K
        and re-run the full training pipeline.

        Args:
            k: Number of clusters for the new KMeans model.

        Returns:
            Updated LLM analysis dict.
        """
        if not self._texts:
            raise RuntimeError("No training data available. Call train() first.")
        if self.config is None:
            raise RuntimeError("Agent was created without a config; cannot retrain.")

        new_strategy = KMeansStrategy(self.config, forced_k=k)
        self.strategy = new_strategy
        # cluster_names are stale after a retrain — clear them so the LLM re-names
        self.cluster_names = {}
        return self.train(self._texts)

    # ------------------------------------------------------------------
    # Relabel
    # ------------------------------------------------------------------

    def relabel(self, cluster_id: int, new_name: str) -> None:
        """Rename a cluster. Stores both int and str keys for safe lookup.

        Args:
            cluster_id: The integer id of the cluster to rename.
            new_name:   The new human-readable label.
        """
        self.cluster_names[cluster_id] = new_name
        self.cluster_names[str(cluster_id)] = new_name

    # ------------------------------------------------------------------
    # Classify
    # ------------------------------------------------------------------

    def classify(self, email: str) -> str:
        """Classify a single email and return its cluster name.

        Args:
            email: Raw email body text.

        Returns:
            The cluster name string.
        """
        label = self.strategy.predict([email])[0]
        if isinstance(label, int):
            label = self.cluster_names.get(label,
                    self.cluster_names.get(str(label), f"Cluster_{label}"))
        return label

    # ------------------------------------------------------------------
    # Analysis / summaries
    # ------------------------------------------------------------------

    def get_cluster_summaries(self) -> dict:
        """Build a JSON-serialisable summary dict that merges top_words from
        the report with live email counts from the strategy labels.

        Returns:
            Dict keyed by cluster id (int) with 'top_words' and 'size'.
        """
        labels = self.strategy.labels
        unique, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(unique, counts))
        return {
            k: {**v, "size": int(size_map.get(k, 0))}
            for k, v in self.report.items()
        }

    def get_analysis(self) -> dict:
        """Return the cached LLM analysis from the last train/retrain call.

        Returns:
            Dict with keys: cluster_names, quality, issues, suggested_params.
        """
        return self._analysis

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, db_path: str) -> None:
        """Persist classified emails to a SQLite database.

        Args:
            db_path: Path to the .db file to write to.
        """
        from database.queries import DatabaseManager

        db = DatabaseManager(db_path)
        db.create_emails_table()
        db.save_emails_bulk([
            {
                "cluster_id": int(label),
                "cluster_label": self.cluster_names.get(
                    int(label),
                    self.cluster_names.get(str(label), f"Cluster {label}")
                ),
                "body": text,
            }
            for text, label in zip(self._texts, self.strategy.labels)
        ])

    # ------------------------------------------------------------------
    # LLM chat (used by review_clusters_chat in user_loop.py)
    # ------------------------------------------------------------------

    def chat(self, user_message: str, history: list | None = None) -> tuple[str, list]:
        """Send one turn of conversation about the current clusters.

        Args:
            user_message: The user's message text.
            history:      Conversation history list of {role, content} dicts.
                          Pass None to start a fresh conversation.

        Returns:
            (reply_text, updated_history)
        """
        if history is None:
            history = []
        history.append({"role": "user", "content": user_message})
        reply, history = chat_with_agent(
            self.get_cluster_summaries(),
            self.cluster_names,
            history
        )
        return reply, history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_names_from_reply(reply: str) -> dict:
        """Extract a {cluster_id: name} map from the LLM's JSON reply block.
        Tolerates markdown fences and trailing commentary.

        Args:
            reply: Raw LLM response string.

        Returns:
            Dict with both int and str keys for each cluster id found.
        """
        try:
            clean = (reply.strip()
                     .removeprefix("```json")
                     .removeprefix("```")
                     .removesuffix("```")
                     .strip())
            start, end = clean.index("{"), clean.rindex("}") + 1
            parsed = json.loads(clean[start:end])
            name_map = {}
            for k, v in parsed.items():
                try:
                    cid = int(k)
                    name_map[cid] = str(v).strip()
                    name_map[str(cid)] = str(v).strip()
                except ValueError:
                    continue
            return name_map
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  Warning: could not parse cluster names from reply ({e})")
            return {}