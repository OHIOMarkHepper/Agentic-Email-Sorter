import pandas as pd


def detect_schema(df):
    """detect_schema: detects the schema of any given csv file

    Args:
        df (pandas.DataFrame): the dataframe to detect the schema of

    Raises:
        ValueError: if no text column is found in the dataframe

    Returns:
        dict: the detected schema
    """    

    # Normalize column names by stripping whitespace and converting to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Heuristic to find the text column - look for common names like "email", "text", "message", "body", "content"
    text_candidates = ["email", "text", "message", "body", "content"]

    # Check if any of the candidate columns exist in the dataframe
    for c in text_candidates:
        if c in df.columns:
            return {
                "text_col": c,
                "label_col": None,
                "columns": list(df.columns)
            }

    raise ValueError("No text column found")


def load_data(filepath):
    """
    load data from file into a pandas dataframe and detect the schema

    Args:
        filepath (string): the filepath of the csv file to load

    Returns:
        tuple: a tuple of (dataframe, list of texts, schema)
    """
    df = pd.read_csv(filepath)
    schema = detect_schema(df)

    texts = df[schema["text_col"]].astype(str).fillna("").tolist()

    print("Detected schema:", schema)

    return df, texts, schema