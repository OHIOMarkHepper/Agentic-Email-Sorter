from fastapi import FastAPI
from pydantic import BaseModel
from agent.user_loop import user_loop
from agent.agent import EmailAgent
from ml.strategies import UserDefinedStrategy, KMeansStrategy
from llm import get_llm_analysis
from cache.cache import CACHE, LLM_CACHE, CLUSTER_STATE

app = FastAPI()

# Module-level state — the trained agent lives here between requests
agent: EmailAgent | None = None

class EmailInput(BaseModel):
    text: str
class TrainInput(BaseModel):
    filepath: str
    strategy: str  # "kmeans" or "user_defined"
    categories: dict[str, list[str]] | None = None  # only for user_defined
    use_ai_labels: bool = True



@app.post("/classify")
def classify_email(body: EmailInput):
    if agent is None:
        return {"error": "No model trained yet. POST to /train first."}
    label = agent.classify(body.text)
    return {"label": label}

@app.post("/train")
async def train_model(body: TrainInput):
    if agent is not None:
        return {"error": "Model already trained. Restart server to retrain."}
    global agent
    agent = EmailAgent(body.filepath, body.config)
    if body.strategy == "kmeans":
        agent.strategy = KMeansStrategy(agent.vectorizer)
    elif body.strategy == "user_defined":
        if body.categories is None:
            return {"error": "categories must be provided for user_defined strategy"}
        agent.strategy = UserDefinedStrategy(agent.vectorizer, body.categories)
    else:
        return {"error": "Invalid strategy. Must be 'kmeans' or 'user_defined'."}
    agent.train()
    

@app.get("/clusters")
def get_clusters():
    if agent is None:
        return {"error": "No model trained yet. POST to /train first."}
    return agent.cluster_names

@app.post("/clusters/{id}/relabel")
async def relabel_cluster(id: int, new_label: str):
    global agent
    if agent is None:
        return {"error": "No model trained yet. POST to /train first."}
    agent.strategy.relabel_cluster(id, new_label)
    return {"message": f"Cluster {id} re-labelled to {new_label}"}

@app.get("/analysis")
def get_analysis():
    if agent is None:
        return {"error": "No model trained yet. POST to /train first."}
    if LLM_CACHE.analysis is not None:
        return LLM_CACHE.analysis
    


