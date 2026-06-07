
from wandb import agent
from database.queries import DatabaseManager
from processing.data import load_data
from ml.vectorizer import build_vectorizer
from ml.clustering import select_best_k, get_top_words
from agent.agent import EmailAgent
from config.config import get_default_config
from ml.strategies import KMeansStrategy, UserDefinedStrategy
from agent.llm import generate_category_examples, chat_with_agent, get_client
from agent.user_loop import classify_loop, review_clusters_chat, user_loop
import numpy as np



def main():
    print("Welcome to the Email Clustering Agent!")
    print("This agent will help you cluster and understand your email data using machine learning and LLM analysis.\n")
    config = get_default_config()
    filepath = "./emaildata/email_classification_dataset.csv"  # Default path to email data
    print("Would you like to train your model or view your sorted emails? (type 't' or 'v')")
    choice = input().strip().lower()
    if choice == 't':
        user_loop(filepath, config)
    elif choice == 'v':
        from database.queries_loop import queries_loop
        db_manager = DatabaseManager(config["db_path"])
        print("Loading existing model and classifications...")
        #load existing model and classifications
        print("Please enter the path to your existing database (or press Enter to use the default path):")
        db_path = input().strip()
        if not db_path:
            db_path = config["db_path"]
        agent = EmailAgent(None, None)  # Placeholder agent; in a full implementation, you'd load the model and vectorizer here
        query_loop = queries_loop(agent, db_manager, db_path)
        query_loop.run()
    else:
        print("Invalid choice. Please restart and type 't' to train or 'v' to view sorted emails.")

if __name__ == "__main__":
    main()