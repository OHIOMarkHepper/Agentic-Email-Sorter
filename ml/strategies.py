from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from ml.clustering import ClusteringStrategy
from sklearn.linear_model import LogisticRegression
import numpy as np

class KMeansStrategy(ClusteringStrategy):
    """KMeansStrategy implements a clustering strategy 
       using KMeans and a logistic regression classifier to 
       assign new texts to clusters. The number of clusters 
       is determined by maximizing the silhouette score, 
       but can also be forced to a specific value.
    """
    
    def __init__(self, config, forced_k=None):
            self.config = config
            self.forced_k = forced_k
            self.model = None
            self.vectorizer = None
            self.labels = None
            self.classifier = None

    def fit(self, texts, vectorizer):
        """fit fits the KMeans model to the given texts using the provided vectorizer

        Args:
            texts (Array-Like): the list of texts to cluster
            vectorizer (): A fitted vectorizer to transform the texts into a feature matrix 
        """
        self.vectorizer = vectorizer
        X = vectorizer.fit_transform(texts)

        best_score = -1
        k_values = [self.forced_k] if self.forced_k else [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for k in k_values:
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=42
            )

            labels = model.fit_predict(X)

            score = silhouette_score(
                X,
                labels,
                sample_size=min(2000, X.shape[0])
            )

            if score > best_score:
                best_score = score
                self.model = model
                self.labels = labels
                self.best_k = k

        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )

        self.classifier.fit(X, self.labels)

        print(
            f"KMeans selected K={self.best_k}, "
            f"score={best_score:.4f}"
        )

    def predict(self, texts):
        X = self.vectorizer.transform(texts)

        probs = self.classifier.predict_proba(X)

        labels = []

        for row in probs:
            confidence = np.max(row)

            if confidence < 0.35:
                labels.append(-1)
            else:
                labels.append(np.argmax(row))
        
        if labels[0] == -1:
            return "Unknown / Other"

        return labels

    def summarize(self):
        """summarize generates a summary of the clusters, including the top words for each cluster and the size of each cluster

        Returns:
            Dict: a dictionary containing the top words for each cluster and the size of each cluster
        """
        feature_names = self.vectorizer.get_feature_names_out()

        report = {}
        for i, center in enumerate(self.model.cluster_centers_):
            top_idx = center.argsort()[-10:][::-1]
            report[i] = {
                "top_words": [feature_names[j] for j in top_idx]
            }

        return report

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class UserDefinedStrategy(ClusteringStrategy):

    def __init__(self, categories):
        """
        categories = {
            "Billing": ["invoice", "payment"],
            "Security": ["password reset", "verify account"]
        }
        """
        self.categories = categories

    def fit(self, texts, vectorizer):
        """fit Generates centroids based on user's input

        Args:
            texts (Array-Like): the list of texts to cluster
            vectorizer (): A fitted vectorizer to transform the texts into a feature matrix
        """
        self.vectorizer = vectorizer
        self.vectorizer.fit(texts)

        # build category centroids
        self.centroids = {}

        for name, examples in self.categories.items():
            X = self.vectorizer.transform(examples)
            self.centroids[name] = np.asarray(X.mean(axis=0))

    def predict(self, texts):
        """predict 

        Args:
            texts (Array-Like): the list of texts to predict

        Returns:
            Array-Like: the predicted cluster labels
        """
        X = self.vectorizer.transform(texts)

        centroid_matrix = np.vstack([self.centroids[c] for c in self.centroids])
        sims = cosine_similarity(X, centroid_matrix)

        labels = []
        category_names = list(self.centroids.keys())

        for row in sims:
            idx = np.argmax(row)
            labels.append(category_names[idx])

        return labels

    def summarize(self):
        return {k: {"examples": v} for k, v in self.categories.items()}