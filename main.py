from config import get_default_config
from processing.data import load_data
from ml.vectorizer import build_vectorizer
from ml.clustering import select_best_k, get_top_words
from agent.agent import EmailAgent
from ml.strategies import KMeansStrategy, UserDefinedStrategy
from agent.llm import generate_category_examples, chat_with_agent, get_client
from agent.user_loop import classify_loop, review_clusters_chat, user_loop
import numpy as np



def main():
    print("Welcome to the Email Clustering Agent!")
    print("This agent will help you cluster and understand your email data using machine learning and LLM analysis.\n")
    config = get_default_config()
    filepath = config["data_path"]
    print("Would you like to train your model or view your sorted emails? (type 't' or 'v')")
    choice = input().strip().lower()
    if choice == 't':
        user_loop(filepath, config)
    elif choice == 'v':
        print("Loading existing model and classifications...")
        # Here we would load the cached model and classifications if we had a persistent cache
        # For this example, we'll just print a message since we don't have a real cache implementation
        print("No persistent cache implemented. Please run training first.")
    else:
        print("Invalid choice. Please restart and type 't' to train or 'v' to view sorted emails.")

if __name__ == "__main__":
    main()