from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import numpy as np

from abc import ABC, abstractmethod

class ClusteringStrategy(ABC):
    @abstractmethod
    def fit(self, texts, vectorizer):
        pass

    @abstractmethod
    def predict(self, texts):
        pass

    @abstractmethod
    def summarize(self):
        pass


def select_best_k(X, k_range=(2, 10)):
    best = None
    best_score = -1

    for k in range(k_range[0], k_range[1] + 1):
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init="auto")
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels, sample_size=min(2000, X.shape[0]))

        if score > best_score:
            best_score = score
            best = (model, labels, k, score)

    return best


def get_top_words(model, feature_names, top_n=10):
    clusters = {}

    for i, center in enumerate(model.cluster_centers_):
        top_idx = center.argsort()[-top_n:][::-1]
        clusters[i] = {
            "top_words": [feature_names[j] for j in top_idx]
        }

    return clusters

def assign__clusters(model, X):
    return model.predict(X)
