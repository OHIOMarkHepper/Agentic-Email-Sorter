from fastapi import FastAPI
from pydantic import BaseModel
from agent.user_loop import user_loop
from agent.agent import EmailAgent

app = FastAPI()

# Module-level state — the trained agent lives here between requests
agent: EmailAgent | None = None

class EmailInput(BaseModel):
    text: str

@app.post("/classify")
def classify_email(body: EmailInput):
    if agent is None:
        return {"error": "No model trained yet. POST to /train first."}
    label = agent.classify(body.text)
    return {"label": label}

@app.post("/train")
async def train_model(filepath: str, config: dict):
    global agent

@app.get("/clusters")
def get_clusters():
    if agent is None:
        return {"error": "No model trained yet. POST to /train first."}
    return agent.cluster_names

@app.post("clusters/{id}/relable")
async def relabel_cluster(id: int, new_label: str):
    global agent
    if agent is None:
        return {"error": "No model trained yet. POST to /train first."}
    agent.strategy.relabel_cluster(id, new_label)
    return {"message": f"Cluster {id} re-labelled to {new_label}"}

@app.post("/classify")
def classify_email(body: EmailInput):
    if agent is None:
        return {"error": "No model trained yet. POST to /train first."}
    label = agent.classify(body.text)
    return {"label": label}