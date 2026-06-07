from agent.agent import EmailAgent
from agent.llm import generate_category_examples
from ml.vectorizer import build_vectorizer
from ml.strategies import KMeansStrategy, UserDefinedStrategy
from processing.input import get_multiline_input
from processing.data import load_data
from config.config import get_default_config
from database.queries import DatabaseManager
from agent.providers import get_llm_provider
import json
import re
import numpy as np


def classify_loop(agent: EmailAgent) -> None:
    """Lets the user paste emails and see how the agent classifies them."""
    print("\n--- Email Classifier ---")
    print("Enter an email to classify it. Type 'done' to exit.\n")

    while True:
        user_email = get_multiline_input()
        if user_email.lower() == "done":
            print("Exiting classifier.")
            break
        if not user_email:
            continue
        print(f"  Classified as: {agent.classify(user_email)}\n")


def _print_summary(agent: EmailAgent) -> None:
    """Print a one-line summary for each cluster."""
    for cid, info in agent.get_cluster_summaries().items():
        name = agent.cluster_names.get(cid, agent.cluster_names.get(str(cid), f"Cluster {cid}"))
        words = ", ".join(info.get("top_words", [])[:5])
        size = info.get("size", "?")
        print(f"  [{cid}] {name} — {size} emails — top words: {words}")
    print()


def review_clusters_chat(agent: EmailAgent) -> None:
    """Interactive CLI loop for reviewing, relabeling, and retraining clusters.

    Args:
        agent: A fully trained EmailAgent instance.
    """
    print("\n--- Cluster Review ---")
    print("You can now chat with the agent about your clusters.")
    print("Special commands:")
    print("  relabel <id> <new name>  — rename a cluster (e.g. 'relabel 3 Spam')")
    print("  retrain <k>              — retrain with a specific number of clusters")
    print("  done                     — finish review and proceed\n")

    print("Current clusters:")
    _print_summary(agent)

    # Opening turn: ask the LLM to name clusters and assess quality
    opening = (
        "Please name each cluster in a list in the format: JSON only! "
        "{<cluster_id>: <name>, ...}. Afterwards, provide an overall assessment "
        "of the clustering quality and any issues you see with the results."
    )
    reply, history = agent.chat(opening)

    # Apply any names the LLM returned
    name_map = EmailAgent._parse_names_from_reply(reply)
    for cid, name in name_map.items():
        agent.cluster_names[cid] = name

    print(f"Agent: {reply}\n")

    # Print updated cluster detail after LLM naming
    for cid, info in agent.get_cluster_summaries().items():
        name = agent.cluster_names.get(cid, agent.cluster_names.get(str(cid), f"Cluster {cid}"))
        print(f"Cluster {cid} ({name}):")
        print(f"  Size: {info.get('size', '?')}")
        print(f"  Top words: {', '.join(info.get('top_words', [])[:10])}")
    print()

    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # relabel <id> <new name>
        if user_input.lower().startswith("relabel "):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: relabel <cluster_id> <new name>\n")
                continue
            try:
                agent.relabel(int(parts[1]), parts[2])
                print(f"  Cluster {parts[1]} renamed to '{parts[2]}'.\n")
            except ValueError:
                print(f"  '{parts[1]}' is not a valid cluster id.\n")
            continue

        # retrain <k>
        if user_input.lower().startswith("retrain "):
            parts = user_input.split()
            if len(parts) < 2:
                print("Usage: retrain <k>\n")
                continue
            try:
                new_k = int(parts[1])
                print(f"  Retraining with K={new_k}...")
                agent.retrain(new_k)
                print("  Retrain complete. Updated clusters:")
                _print_summary(agent)
                reply, history = agent.chat(
                    f"The model was retrained with K={new_k}. "
                    "Please re-evaluate whether the new groupings make sense."
                )
                print(f"Agent: {reply}\n")
            except ValueError:
                print(f"  '{parts[1]}' is not a valid number.\n")
            except RuntimeError as e:
                print(f"  Error: {e}\n")
            continue

        # done
        if user_input.lower() == "done":
            print("Exiting cluster review.")
            break

        # general chat
        reply, history = agent.chat(user_input, history)
        print(f"\nAgent: {reply}\n")


def _build_user_defined_strategy(config: dict) -> UserDefinedStrategy:
    """Interactively collect category names and LLM-suggested examples from the user.

    Args:
        config: App config dict (passed to generate_category_examples via provider).

    Returns:
        A UserDefinedStrategy built from the collected categories.
    """
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
            examples = [e.strip() for e in input("  Examples: ").strip().split(",")]
        else:
            print(f"  Suggested examples: {', '.join(examples)}")
            confirm = input("  Accept these? (y to accept / or type replacements, comma-separated): ").strip()
            if confirm.lower() != "y":
                examples = [e.strip() for e in confirm.split(",")]
        categories[name] = examples
        print()
    return UserDefinedStrategy(categories)


def user_loop(filepath: str = None, config: dict = None) -> None:
    """Main CLI entry point: load data, train the agent, review clusters, save.

    Args:
        filepath: Optional path to the data file. Prompts the user if not given.
        config:   Optional config dict. Falls back to get_default_config().
    """
    config = config or get_default_config()

    if not filepath:
        filepath = input("Please enter the path to your data file (e.g., 'data/emails.csv'): ").strip()

    df, texts, schema = load_data(filepath)
    vectorizer = build_vectorizer(config)
    X = vectorizer.fit_transform(texts)

    get_llm_provider(config) # Initialize the LLM provider if enabled

    print("Would you like suggested clusters from KMeans or user-defined clusters?")
    print("type 'k' for KMeans or 'u' for user-defined")

    if input().strip().lower() == 'k':
        strategy = KMeansStrategy(config)
        agent = EmailAgent(strategy, vectorizer, config=config)
    else:
        strategy = _build_user_defined_strategy(config)
        agent = EmailAgent(strategy, vectorizer, config=config)

    agent.train(texts)
    print("\nTraining complete.")

    # KMeans gets an interactive review; user-defined clusters are self-explanatory
    if isinstance(agent.strategy, KMeansStrategy):
        review_clusters_chat(agent)

    # Offer to save
    print("\nWould you like to save the classified emails to a database? (y/n)")
    if input().strip().lower() == 'y':
        db_path = input("Enter the path for the database (e.g., './email_clusters.db'): ").strip()
        agent.save(db_path)
        print(f"Classified emails saved to {db_path}.")
    else:
        print("Classified emails not saved.")

    classify_loop(agent)