import json
import hashlib
from cache.cache import LLM_CACHE
from google import genai
from ml.strategies import KMeansStrategy
import numpy as np
import os

_client = None

def chat_with_agent(cluster_summaries, cluster_names, history=None):
    """Single turn of a conversation about the clustering results."""
    if history is None:
        history = []

    system_context = f"""You are an assistant helping a user review email clustering results.
The KMeans algorithm has grouped emails into the following clusters:

{json.dumps({
    str(k): {
        "name": cluster_names.get(k, cluster_names.get(str(k), f"Cluster {k}")),
        "top_words": v.get("top_words", []),
        "size": v.get("size", "unknown")
    }
    for k, v in cluster_summaries.items()
}, indent=2)}

Help the user understand what each cluster represents, whether the groupings make sense,
and answer any questions they have about the results. Be concise and specific."""

    prompt = system_context + "\n\nConversation so far:\n"
    for turn in history:
        role_label = "User" if turn["role"] == "user" else "Assistant"
        prompt += f"{role_label}: {turn['content']}\n"

    result = call_gemini(prompt)
    history.append({"role": "assistant", "content": result})
    return result, history

def get_client():
    """Returns the shared Gemini client, prompting for a key if not yet set."""
    global _client
    if _client is None:
        key = input("Enter your Gemini API key: ").strip()
        _client = genai.Client(api_key=key)
    return _client

class LLMAnalysis:
    def __init__(self):
        self.cache = LLM_CACHE

    def analyze(self, cluster_summaries):
        key = self._hash_clusters(cluster_summaries)
        if key in self.cache:
            return self.cache[key]
        result = self._call_gemini(cluster_summaries)
        try:
            parsed = json.loads(result)
        except:
            parsed = {
                "cluster_names": {},
                "quality": "unknown",
                "issues": [],
                "suggested_params": {}
            }
        self.cache[key] = parsed
        return parsed

    def _hash_clusters(self, cluster_summaries):
        raw = json.dumps(cluster_summaries, sort_keys=True).encode()
        return hashlib.md5(raw).hexdigest()

    def _call_gemini(self, cluster_summaries):
        return call_gemini(analyze_clusters(cluster_summaries))




def call_gemini(prompt):
    response = get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def hash_clusters(cluster_summaries):
    raw = json.dumps(cluster_summaries, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()


def analyze_clusters(cluster_summaries):
    prompt = f"""
Return JSON only:

{{
  "cluster_names": {{}},
  "quality": "good|ok|bad",
  "issues": [],
  "suggested_params": {{}}
}}

Clusters:
{json.dumps(cluster_summaries, indent=2)}
"""
    return call_gemini(prompt)


def get_llm_analysis(cluster_summaries):
    key = hash_clusters(cluster_summaries)

    if key in LLM_CACHE:
        return LLM_CACHE[key]

    result = analyze_clusters(cluster_summaries)

    try:
        parsed = json.loads(result)
    except:
        parsed = {
            "cluster_names": {},
            "quality": "unknown",
            "issues": [],
            "suggested_params": {}
        }

    LLM_CACHE[key] = parsed
    return parsed


def generate_category_examples(category_name):
    """Generates example phrases for a user-defined category using the LLM."""
    prompt = f"""You are helping configure an email classifier.
Generate 6 short example phrases or keywords that would appear in emails belonging to the category: "{category_name}".
Return JSON only, no markdown, no code fences, no explanation:
{{"examples": ["phrase1", "phrase2", "phrase3", "phrase4", "phrase5", "phrase6"]}}"""
    result = call_gemini(prompt)
    try:
        clean = result.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        return parsed["examples"]
    except Exception as e:
        print(f"  Warning: could not parse LLM response ({e}). Raw response was:\n  {result}")
        return []

def review_clusters_chat(agent, texts, vectorizer, config):
    """Interactive chat loop for reviewing and adjusting KMeans clustering results."""
    print("\n--- Cluster Review ---")
    print("You can now chat with the agent about your clusters.")
    print("Special commands:")
    print("  relabel <id> <new name>  — rename a cluster (e.g. 'relabel 3 Spam')")
    print("  retrain <k>              — retrain with a specific number of clusters")
    print("  done                     — finish review and proceed\n")

    def build_summaries(agent):
        report = agent.report
        labels = agent.cluster_names
        unique, counts = np.unique(labels, return_counts=True)
        size_map = dict(zip(unique, counts))
        return {
            k: {**v, "size": int(size_map.get(k, 0))}
            for k, v in report.items()
        }

    def print_summary(agent):
        for cid, info in build_summaries(agent).items():
            name = agent.cluster_names.get(cid, agent.cluster_names.get(str(cid), f"Cluster {cid}"))
            words = ", ".join(info.get("top_words", [])[:5])
            size = info.get("size", "?")
            print(f"  [{cid}] {name} — {size} emails — top words: {words}")
        print()

    print("Current clusters:")
    print_summary(agent)

    cluster_summaries = build_summaries(agent)
    opening_prompt = "Please name each cluster in a list in the format: <cluster_id>: <name>. Afterwards, provide an overall assessment of the clustering quality and any issues you see with the results."
    history = [{"role": "user", "content": opening_prompt}]
    reply, history = chat_with_agent(cluster_summaries, agent.cluster_names, history)
    print(f"Agent: {reply}\n")


    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # relabel command
        if user_input.lower().startswith("relabel "):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: relabel <cluster_id> <new name>\n")
                continue
            cid_str, new_name = parts[1], parts[2]
            try:
                cid = int(cid_str)
                agent.cluster_names[cid] = new_name
                agent.cluster_names[str(cid)] = new_name
                print(f"  Cluster {cid} renamed to '{new_name}'.\n")
                cluster_summaries = build_summaries(agent)
            except ValueError:
                print(f"  '{cid_str}' is not a valid cluster id.\n")
            continue

        # retrain command
        if user_input.lower().startswith("retrain "):
            parts = user_input.split()
            if len(parts) < 2:
                print("Usage: retrain <k>\n")
                continue
            try:
                new_k = int(parts[1])
                print(f"  Retraining with K={new_k}...")
                new_strategy = KMeansStrategy(config, forced_k=new_k)
                new_strategy.fit(texts, vectorizer)
                agent.strategy = new_strategy
                agent.train(texts)
                cluster_summaries = build_summaries(agent)
                history = []
                print("  Retrain complete. Updated clusters:")
                print_summary(agent)
                opening_prompt = f"The model was retrained with K={new_k}. Please re-evaluate whether the new groupings make sense."
                history = [{"role": "user", "content": opening_prompt}]
                reply, history = chat_with_agent(cluster_summaries, agent.cluster_names, history)
                print(f"Agent: {reply}\n")
            except ValueError:
                print(f"  '{parts[1]}' is not a valid number.\n")
            continue

        if user_input.lower() == "done":
            print("Exiting cluster review.")
            break

        history.append({"role": "user", "content": user_input})
        reply, history = chat_with_agent(cluster_summaries, agent.cluster_names, history)
        print(f"\nAgent: {reply}\n")