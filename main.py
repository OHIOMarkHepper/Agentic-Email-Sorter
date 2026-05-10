from config import get_default_config
from data import load_data
from vectorizer import build_vectorizer
from clustering import select_best_k, get_top_words
from agent import run_agent_loop, EmailAgent
from strategies import KMeansStrategy, UserDefinedStrategy
from llm import generate_category_examples, chat_with_agent, get_client
import numpy as np

def get_multiline_input():
    print("Paste email. Type END on a new line when finished:")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()

def classify_loop(agent):
    """Lets the user input emails and see how the agent classifies them."""
    print("\n--- Email Classifier ---")
    print("Enter an email to classify it. Type 'done' to exit.\n")

    while True:
        email = get_multiline_input()
        if email.lower() == "done":
            print("Exiting classifier.")
            break
        if not email:
            continue

        label = agent.classify(email)
        print(f"  Classified as: {label}\n")

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

    print("Current clusters:")
    print_summary(agent)

    cluster_summaries = build_summaries(agent)
    opening_prompt = "Please name each cluster and explain whether the groupings make sense, based on the top words and sizes shown."
    history = [{"role": "user", "content": opening_prompt}]
    reply, history = chat_with_agent(cluster_summaries, agent.cluster_names, history)
    print(f"Agent: {reply}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

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

def clustering_fn(X, config):
    model, labels, k, score = select_best_k(X)
    return model, labels

def report_fn(model, feature_names):
    return get_top_words(model, feature_names)

def main():
    config = get_default_config()
    vectorizer = build_vectorizer(config)

    filepath = input("Please enter the path to your data file (e.g., 'data/emails.csv'): ").strip()
    df, texts, schema = load_data(filepath)
    vectorizer = build_vectorizer(config)
    X = vectorizer.fit_transform(texts)

    get_client()  # Ensure Gemini client is initialized before proceeding

    X = vectorizer.fit_transform(texts)

    print("Would you like suggested clusters from KMeans or user-defined clusters?")
    print("type 'k' for KMeans or 'u' for user-defined")

    if input().lower() == 'k':
        use_kmeans = True
    else:
        use_kmeans = False

    if use_kmeans:
        strategy = KMeansStrategy(config)
        strategy.fit(texts, vectorizer)
        agent = EmailAgent(strategy, vectorizer)
        agent.train(texts)
        review_clusters_chat(agent, texts, vectorizer, config)
        print("\nTraining complete.")
        classify_loop(agent)
    else:
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
        classify_loop(agent)
    
    classify_loop(agent)

    print("\nFinal result:", result)

if __name__ == "__main__":
    main()
