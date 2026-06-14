
# Holds the configs for various aspects of the application

def get_default_config():
    """get_default_config - get the default configuration for the vectorizer and clustering

    Returns:
        dict: the default configuration
    """

    return {
        "ngram_range": (1, 2),  # length of phrases
        "max_df": 0.8,          # ignore terms that appear in more than 80% of the documents
        "min_df": 3,            # ignore terms that appear in less than 3 documents
        "max_features": 5000,   # maximum number of features to consider
        "k": 10,                # number of top features to select based on chi-squared test
        "threshold": 0.25,      # threshold for selecting features based on chi-squared scores
        "db_path": "./emaildata/test.db", # default path to the SQLite database
        "llm": 
            {
                "Enabled": True,
                "provider": "ollama",
                "model": "qwen3.5:9b",
                "base_url": "http://localhost:11434",
                "api_key": "none"
            }
    }

def get_IMAP_config():
    """get_IMAP_config - get the configuration for the IMAP source
    Returns:
        dict: the configuration
    """

    return {
        "imap_server": "imap.gmail.com",  # server to connect to
        "email_user": "your_email@gmail.com",  # email address to use
        "access_token": "your_access_token"  # OAuth2 token string
    }