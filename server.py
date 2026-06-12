import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from ml.strategies import KMeansStrategy, UserDefinedStrategy
from ml.vectorizer import build_vectorizer
from processing.data import load_data
from config.config import get_default_config
from agent.agent import EmailAgent


app = FastAPI()

# Module-level singleton — holds the trained agent between requests.
agent: EmailAgent | None = None


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class TrainInput(BaseModel):
    filepath: str
    strategy: str                               # "kmeans" or "user_defined"
    categories: Optional[dict[str, list[str]]] = None   # required for user_defined
    use_ai_labels: bool = True

class EmailInput(BaseModel):
    text: str

class RelabelInput(BaseModel):
    new_label: str


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _require_agent() -> EmailAgent:
    """Return the global agent or raise 503 if not yet trained."""
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="No model trained yet. POST to /train first."
        )
    return agent


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/train", status_code=201)
def train_model(body: TrainInput):
    """Load data, build the chosen strategy, and train the agent.

    - strategy "kmeans"       → KMeans auto-selects the best K
    - strategy "user_defined" → requires a categories dict

    Returns 409 if a model is already loaded (restart server to replace).
    Returns 400 for an unrecognised strategy or missing categories.
    Returns 201 with a summary on success.
    """
    global agent

    if agent is not None:
        raise HTTPException(
            status_code=409,
            detail="Model already trained. Restart the server to retrain."
        )

    config = get_default_config()

    # Load data from disk
    try:
        df, texts, schema = load_data(body.filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

    vectorizer = build_vectorizer(config)

    # Build strategy
    if body.strategy == "kmeans":
        strategy = KMeansStrategy(config)
    elif body.strategy == "user_defined":
        if not body.categories:
            raise HTTPException(
                status_code=400,
                detail="categories must be provided for user_defined strategy."
            )
        strategy = UserDefinedStrategy(body.categories)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{body.strategy}'. Use 'kmeans' or 'user_defined'."
        )

    agent = EmailAgent(strategy, vectorizer, config=config, llm_enabled=body.use_ai_labels)
    analysis = agent.train(texts)

    return JSONResponse(
        status_code=201,
        content={
            "message": "Model trained successfully.",
            "clusters": len(agent.cluster_names) // 2,  # int+str keys, so halve
            "quality": analysis.get("quality", "unknown"),
        }
    )


@app.post("/train/retrain", status_code=200)
def retrain_model(k: int):
    """Retrain the existing model with a specific number of clusters K.

    Requires a model to already be trained. Returns 400 if K is invalid.
    """
    a = _require_agent()
    if k < 2:
        raise HTTPException(status_code=400, detail="k must be 2 or greater.")
    try:
        analysis = a.retrain(k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": f"Model retrained with K={k}.",
        "quality": analysis.get("quality", "unknown"),
    }


@app.post("/classify", status_code=200)
def classify_email(body: EmailInput):
    """Classify a single email and return its cluster label.

    Returns 503 if no model is trained yet.
    """
    a = _require_agent()
    label = a.classify(body.text)
    return {"label": label}


@app.get("/clusters", status_code=200)
def get_clusters():
    """Return all cluster ids and their current names.

    Returns 503 if no model is trained yet.
    """
    a = _require_agent()
    # cluster_names stores both int and str keys — return only str keys for JSON safety
    return {
        str(k): v
        for k, v in a.cluster_names.items()
        if isinstance(k, str)
    }


@app.post("/clusters/{id}/relabel", status_code=200)
def relabel_cluster(id: int, body: RelabelInput):
    """Rename a cluster by its integer id.

    Returns 503 if no model is trained, 404 if the cluster id doesn't exist.
    """
    a = _require_agent()

    valid_ids = {k for k in a.cluster_names if isinstance(k, int)}
    if id not in valid_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster {id} not found. Valid ids: {sorted(valid_ids)}"
        )

    a.relabel(id, body.new_label)
    return {"message": f"Cluster {id} relabelled to '{body.new_label}'."}


@app.get("/analysis", status_code=200)
def get_analysis():
    """Return the cached LLM analysis from the last train or retrain call.

    Returns 503 if no model is trained yet.
    Returns 204 if the model was trained with use_ai_labels=False (no analysis available).
    """
    a = _require_agent()
    analysis = a.get_analysis()

    if not analysis:
        # Trained without LLM — no analysis to return
        return JSONResponse(status_code=204, content=None)

    return analysis