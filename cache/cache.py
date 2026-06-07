
# cache.py
# holds cached data for vectorizer, model, and other intermediate results to avoid redundant computations

CACHE = {
    "vectorizer": None,
    "X": None,
    "model": None,
    "labels": None,
    "best_k": None,
    "report": None
}

# LLM_CACHE is used to store the last signature and result from the LLM to avoid redundant calls

LLM_CACHE = {
    "analysis": None
}
CLUSTER_STATE = {
    "last_signature": None,
    "last_llm_result": None,
}