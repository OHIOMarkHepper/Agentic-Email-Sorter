from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(config):
    """build_vectorizer create a vectorizer from a configuration

    Args:
        config (dict): the configuration for the vectorizer

    Returns:
        TfidfVectorizer: the created vectorizer
    """

    # For simplicity, we only implement a TF-IDF vectorizer here, but this could be extended to support other types of vectorizers
    TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.90,
        min_df=2,
        max_features=10000,
        sublinear_tf=True,
        strip_accents="unicode"
    )