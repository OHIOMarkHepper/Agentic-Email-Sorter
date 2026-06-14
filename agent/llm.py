
"""
This module contains the logic for interacting with an LLM provider to generate category examples.
It uses the `genai` library to communicate with the OpenAI API.
"""

import json
import hashlib
from cache.cache import LLM_CACHE
from google import genai
from ml.strategies import KMeansStrategy
from agent.providers import get_llm_provider
from config.config import get_default_config
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

    provider = get_llm_provider(get_default_config())
    result = provider.generate(prompt)
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
    """ LLMAnalysis handles the analysis of clustering results using a language model, with caching to avoid redundant calls.
        It generates summaries of clusters and provides feedback on clustering quality, issues, and suggestions for improvement
    """
    def __init__(self):
        self.cache = LLM_CACHE

    def analyze(self, cluster_summaries):
        key = self._hash_clusters(cluster_summaries)
        if key in self.cache:
            return self.cache[key]
        result = self._provider.generate(cluster_summaries)
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
        provider = get_llm_provider(get_default_config())
        return provider.generate(analyze_clusters(cluster_summaries))


def call_gemini(prompt):
    """Legacy function that uses the genai client to generate content based on a prompt."""
    response = get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response

def hash_clusters(cluster_summaries):
    """ hash_clusters produces a stable hash of the current clustering summaries for caching purposes.

    Args:
        cluster_summaries (dict): A dictionary containing summaries of the clusters, including top words and sizes.

    Returns:
        str: A hash string representing the current state of the cluster summaries.
    """
    raw = json.dumps(cluster_summaries, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()


def analyze_clusters(cluster_summaries):
    """analyze_clusters generates a prompt for the Gemini API to analyze the clustering results and returns the response."""
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
    provider = get_llm_provider(get_default_config())
    response = provider.generate(prompt)
    return response


def get_llm_analysis(cluster_summaries):
    """get_llm_analysis retrieves the analysis of the clustering results from the LLM, using caching to avoid redundant calls.

    Args:
        cluster_summaries (dict): A dictionary containing summaries of the clusters, including top words and sizes.

    Returns:
        dict: The analysis result from the LLM.
    """
    key = hash_clusters(cluster_summaries)

    if key in LLM_CACHE:
        return LLM_CACHE[key]

    result = analyze_clusters(cluster_summaries)

    try: # check for JSON decode error
        parsed = json.loads(result)
    except json.JSONDecodeError as e:
        print(f"Warning: LLM response was not valid JSON ({e}). Raw:\n{result}")
        parsed = {"cluster_names": {}, "quality": "unknown", "issues": [], "suggested_params": {}}

    LLM_CACHE[key] = parsed
    return parsed


def generate_category_examples(category_name):
    """Generates example phrases for a user-defined category using the LLM."""
    prompt = f"""You are helping configure an email classifier.
Generate 6 short example phrases or keywords that would appear in emails belonging to the category: "{category_name}".
Return JSON only, no markdown, no code fences, no explanation:
{{"examples": ["phrase1", "phrase2", "phrase3", "phrase4", "phrase5", "phrase6"]}}"""
    provider = get_llm_provider(get_default_config())
    result = provider.generate(prompt)
    try:
        clean = result.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        return parsed["examples"]
    except Exception as e:
        print(f"  Warning: could not parse LLM response ({e}). Raw response was:\n  {result}")
        return []
