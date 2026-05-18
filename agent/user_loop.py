from agent.agent import EmailAgent
from agent.automl import automl_train
from agent.llm import generate_category_examples, chat_with_agent, get_client, analyze_clusters, get_llm_analysis
from ml.clustering import get_top_words, select_best_k
from ml.vectorizer import build_vectorizer
from ml.strategies import KMeansStrategy, UserDefinedStrategy
from agent.llm import generate_category_examples, chat_with_agent, get_client
from processing.input import get_multiline_input
from processing.data import load_data
from config.config import get_default_config
from database.queries import DatabaseManager
import json
import re
import numpy as np

def classify_loop(agent):
    """Lets the user input emails and see how the agent classifies them."""
    print("\n--- Email Classifier ---")
    print("Enter an email to classify it. Type 'done' to exit.\n")

    while True:
        user_email = get_multiline_input()
        if user_email.lower() == "done":
            print("Exiting classifier.")
            break
        if not user_email:
            continue

        label = agent.classify(user_email)
        print(f"  Classified as: {label}\n")


def review_clusters_chat(agent, texts, vectorizer, config):
    """ A 

    Args:
        agent (agemt): the email agent whose clusters we are reviewing
        texts (list): the original email texts that were clustered
        vectorizer (object): the vectorizer used to transform the texts for clustering
        config (dict): the configuration dictionary for the agent and clustering

    Returns:
        None: this function is for interactive review and does not return anything
    """
    print("\n--- Cluster Review ---")
    print("You can now chat with the agent about your clusters.")
    print("Special commands:")
    print("  relabel <id> <new name>  — rename a cluster (e.g. 'relabel 3 Spam')")
    print("  retrain <k>              — retrain with a specific number of clusters")
    print("  done                     — finish review and proceed\n")

    def build_summaries(agent):
        """ 

        Args:
            agent (_type_): _description_

        Returns:
            _type_: _description_
        """
        report = agent.report
        labels = agent.strategy.labels
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
    
    def get_names_from_reply(reply: str) -> dict:
        """Parse cluster names from a JSON block in the LLM reply."""
        try:
            # strip markdown fences if the LLM wraps the JSON
            clean = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            # extract just the first {...} block in case the LLM adds commentary after
            start, end = clean.index("{"), clean.rindex("}") + 1
            parsed = json.loads(clean[start:end])
            name_map = {}
            for k, v in parsed.items():
                try:
                    cid = int(k)
                    name_map[cid] = str(v).strip()
                    name_map[str(cid)] = str(v).strip()
                except ValueError:
                    continue
            return name_map
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  Warning: could not parse cluster names from reply ({e})")
            return {}
    print("Current clusters:")
    print_summary(agent)

    cluster_summaries = build_summaries(agent)

    # Ask the agent to name the clusters and provide an assessment of the clustering quality
    opening_prompt = "Please name each cluster in a list in the format: JSON only! {<cluster_id>: <name>, ...}. Afterwards, provide an overall assessment of the clustering quality and any issues you see with the results."
    history = [{"role": "user", "content": opening_prompt}]
    reply, history = chat_with_agent(cluster_summaries, agent.cluster_names, history)
    
    # rename clusters based on LLM reply
    import re
    name_map = get_names_from_reply(reply)
    for cid, name in name_map.items():
        agent.cluster_names[cid] = name
        agent.cluster_names[str(cid)] = name
        

    print(f"Agent: {reply}\n")
    
    # Print updated cluster summaries after renaming
    for cid, info in cluster_summaries.items():
        name = agent.cluster_names.get(cid, agent.cluster_names.get(str(cid), f"Cluster {cid}"))
        print(f"Cluster {cid} ({name}):")
        print(f"  Size: {info.get('size', '?')}")
        print(f"  Top words: {', '.join(info.get('top_words', [])[:10])}")


    # Enter the interactive chat loop for cluster review and potential retraining
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        
        # Handle relabel command
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
        
        # Handle retrain command
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
        
        # Check for exit command
        if user_input.lower() == "done":
            print("Exiting cluster review.")
            break
        
        # For any other input, treat it as a general question or comment about the clusters and pass it to the agent
        history.append({"role": "user", "content": user_input})
        reply, history = chat_with_agent(cluster_summaries, agent.cluster_names, history)
        print(f"\nAgent: {reply}\n")


def clustering_fn(X, config):
    """ A function that handles the acquisition of a clustering training result

    Args:
        X (array-like): the feature matrix to cluster
        config (dict): the configuration dictionary for clustering parameters

    Returns:
        model: the fitted clustering model
        labels: the cluster labels for each input
    """
    model, labels, k, score = select_best_k(X)
    return model, labels


def report_fn(model, feature_names):
    """ A function that generates a report of the top words in each cluster for interpretability"""
    return get_top_words(model, feature_names)

def user_loop(filepath, config):
    """ A function that handles the main user loop for training the agent, including data loading, clustering, and interactive review

    Args:
        filepath (str): the path to the email data file to load and train on
        config (dict): the configuration dictionary for the agent and clustering parameters
    """
    config = get_default_config()

    filepath = input("Please enter the path to your data file (e.g., 'data/emails.csv'): ").strip()
    df, texts, schema = load_data(filepath)

    vectorizer = build_vectorizer(config)
    X = vectorizer.fit_transform(texts)

    get_client()  # Ensure Gemini client is initialized before proceeding

    print("Would you like suggested clusters from KMeans or user-defined clusters?")
    print("type 'k' for KMeans or 'u' for user-defined")

    # Train the agent based on user choice of strategy
    if input().lower() == 'k':
        # Train with KMeans strategy and enter review loop
        strategy = KMeansStrategy(config)
        strategy.fit(texts, vectorizer)
        agent = EmailAgent(strategy, vectorizer)
        agent.train(texts)
        review_clusters_chat(agent, texts, vectorizer, config)
        print("\nTraining complete.")

    else:
        # Train with user-defined strategy and enter classify loop
        print("Enter category names one at a time. The LLM will suggest example phrases for each.")
        print("Type 'done' when finished.\n")
        categories = {}
        while True:
            name = input("Category name (or 'done'): ").strip()
            if name.lower() == "done":
                break
            print(f"  Generating examples for '{name}'...")
            examples = generate_category_examples(name)
            if not examples:
                print("  LLM returned no examples. Enter them manually (comma-separated):")
                raw = input("  Examples: ").strip()
                examples = [e.strip() for e in raw.split(",")]
            else:
                print(f"  Suggested examples: {', '.join(examples)}")
                confirm = input("  Accept these? (y to accept / or type replacements, comma-separated): ").strip()
                if confirm.lower() != "y":
                    examples = [e.strip() for e in confirm.split(",")]
            categories[name] = examples
            print()
        strategy = UserDefinedStrategy(categories)
        agent = EmailAgent(strategy, vectorizer)
        agent.train(texts)
        print("\nTraining complete.")
    
    # After training and review, offer to save results to a database
    print("Would you like to save the classified emails to a database? (y/n)")
    if input().lower() == 'y':
        db_path = input("Enter the path for the database (e.g., './email_clusters.db'): ").strip()
        # Save classified emails to the specified database
        dbManager = DatabaseManager(db_path)
        dbManager.create_clusters_table()
        dbManager.save_emails_bulk([
            {
                "cluster_id": int(label),
                "cluster_label": agent.cluster_names.get(int(label), 
                                 agent.cluster_names.get(str(label), f"Cluster {label}")),
                "body": text,
            }
            for text, label in zip(texts, agent.strategy.labels)
        ])
        print(f"Classified emails saved to {db_path}.")
    else:
        print("Classified emails not saved to a database.")

    classify_loop(agent)