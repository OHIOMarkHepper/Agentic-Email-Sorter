from agent.clustering import select_best_k
from Email_Agent.cache import CACHE

def automl_train(X, vectorizer):
    """automl_train _summary_

    Args:
        X (array-like): the feature matrix to cluster
        vectorizer (): the fitted vectorizer used to transform the data

    Returns:
        tuple: a tuple of (best_model, labels)
    """

    best_model, labels, best_k, score = select_best_k(X)

    CACHE["vectorizer"] = vectorizer
    CACHE["X"] = X
    CACHE["model"] = best_model
    CACHE["labels"] = labels
    CACHE["best_k"] = best_k

    print(f"Cached model K={best_k}, silhouette={score:.4f}")

    return best_model, labels