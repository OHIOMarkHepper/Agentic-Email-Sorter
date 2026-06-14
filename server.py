import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from agent.agent import EmailAgent
from ml.strategies import KMeansStrategy, UserDefinedStrategy
from ml.vectorizer import build_vectorizer
from processing.data import load_data
from config.config import get_default_config
from database.queries import DatabaseManager


app = FastAPI()

# CORS — allows a local frontend (e.g. localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singleton — holds the trained agent between requests.
agent: EmailAgent | None = None

DEFAULT_DB_PATH = "./emaildata/emails.db"


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class TrainInput(BaseModel):
    filepath: str
    strategy: str                                        # "kmeans" or "user_defined"
    categories: Optional[dict[str, list[str]]] = None   # required for user_defined
    use_ai_labels: bool = True

class EmailInput(BaseModel):
    text: str

class RelabelInput(BaseModel):
    new_label: str

class SaveInput(BaseModel):
    db_path: str = DEFAULT_DB_PATH


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _require_agent() -> EmailAgent:
    """Return the global agent or raise 503 if not yet trained."""
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="No model trained yet. POST to /train first."
        )
    return agent

def _get_db(db_path: str = DEFAULT_DB_PATH) -> DatabaseManager:
    """Return a DatabaseManager for the given path."""
    try:
        return DatabaseManager(db_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------
# Training endpoints
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

    try:
        df, texts, schema = load_data(body.filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

    vectorizer = build_vectorizer(config)

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
    """Retrain the existing model with a specific K.

    Returns 400 if K < 2 or no training data is available.
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


# ------------------------------------------------------------------
# Classification endpoints
# ------------------------------------------------------------------

@app.post("/classify", status_code=200)
def classify_email(body: EmailInput):
    """Classify a single email and return its cluster label."""
    a = _require_agent()
    label = a.classify(body.text)
    return {"label": label}


# ------------------------------------------------------------------
# Cluster endpoints
# ------------------------------------------------------------------

@app.get("/clusters", status_code=200)
def get_clusters():
    """Return all cluster ids and their current names."""
    a = _require_agent()
    return {
        str(k): v
        for k, v in a.cluster_names.items()
        if isinstance(k, str)
    }


@app.post("/clusters/{id}/relabel", status_code=200)
def relabel_cluster(id: int, body: RelabelInput):
    """Rename a cluster by its integer id.

    Returns 404 if the cluster id doesn't exist.
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


# ------------------------------------------------------------------
# Analysis endpoint
# ------------------------------------------------------------------

@app.get("/analysis", status_code=200)
def get_analysis():
    """Return the cached LLM analysis from the last train or retrain call.

    Returns 204 if trained with use_ai_labels=False (no analysis available).
    """
    a = _require_agent()
    analysis = a.get_analysis()

    if not analysis:
        return JSONResponse(status_code=204, content=None)

    return analysis


# ------------------------------------------------------------------
# Database endpoints
# ------------------------------------------------------------------

@app.post("/emails/save", status_code=201)
def save_emails(body: SaveInput):
    """Persist all currently classified emails to a SQLite database.

    Call this once you're happy with your clusters and labels.
    Returns 201 with a count of saved emails on success.
    """
    a = _require_agent()
    try:
        a.save(body.db_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save emails: {e}")

    return JSONResponse(
        status_code=201,
        content={
            "message": f"Emails saved to {body.db_path}.",
            "count": len(a._texts),
        }
    )


@app.get("/emails", status_code=200)
def get_emails(cluster: Optional[str] = None, db_path: str = DEFAULT_DB_PATH):
    """Query saved emails from the database.

    Optional query param:
      ?cluster=Orders+%26+Shipping  → filter by cluster label
      (omit to return all distinct cluster labels instead)

    Returns 404 if the cluster label doesn't exist in the DB.
    """
    db = _get_db(db_path)

    if cluster:
        rows = db.query_emails_by_cluster(cluster)
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No emails found for cluster '{cluster}'."
            )
        columns = db.get_column_names()
        return [dict(zip(columns, row)) for row in rows]

    # No filter — return the distinct cluster labels present in the DB
    labels = db.get_cluster_labels()
    if not labels:
        raise HTTPException(
            status_code=404,
            detail="No emails in the database yet. POST to /emails/save first."
        )
    return {"clusters": labels}


@app.get("/emails/clusters", status_code=200)
def get_db_clusters(db_path: str = DEFAULT_DB_PATH):
    """Return the distinct cluster labels currently stored in the database.

    Useful for populating a frontend dropdown without loading all emails.
    Returns 404 if the database is empty.
    """
    db = _get_db(db_path)
    labels = db.get_cluster_labels()
    if not labels:
        raise HTTPException(
            status_code=404,
            detail="No emails in the database yet. POST to /emails/save first."
        )
    return {"clusters": labels}