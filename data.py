import pandas as pd
import mailbox
import email
import sqlite3
import json
import os
import pathlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from email.utils import parseaddr

@dataclass
class EmailRecord:
    id: str
    thread_id: Optional[str]
    sender: str
    sender_domain: str
    recipients: list[str]
    subject: str
    body: str
    timestamp: Optional[str]
    has_attachment: bool
    labels: list[str]
    raw_source: str


def get_body(msg):
    """get_body: Gets the body part of the email.

    Args:
        msg: The email message object to extract the body from.

    Returns:
        str: The plain-text body of the email message, or empty string if not found.
    """
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode(errors="ignore")

    return ""


def has_attachment(msg):
    """has_attachment: Checks whether the email contains any attachments.

    Args:
        msg: The email message object.

    Returns:
        bool: True if an attachment is found, False otherwise.
    """
    if msg.is_multipart():
        for part in msg.walk():
            disposition = part.get_content_disposition()
            if disposition and "attachment" in disposition.lower():
                return True
    return False


def get_recipients(msg):
    """get_recipients: Extracts all recipient addresses from To, Cc, and Bcc headers.

    Args:
        msg: The email message object.

    Returns:
        list[str]: A deduplicated list of recipient email addresses.
    """
    recipients = []
    for header in ("to", "cc", "bcc"):
        value = msg.get(header, "")
        if value:
            for _, addr in email.utils.getaddresses([value]):
                if addr:
                    recipients.append(addr)
    return list(dict.fromkeys(recipients))  # deduplicate, preserve order


def extract_sender_domain(sender: str) -> str:
    """extract_sender_domain: Parses the domain from a sender address.

    Args:
        sender (str): The raw From header value.

    Returns:
        str: The domain portion of the sender's email, or empty string if unparseable.
    """
    _, addr = parseaddr(sender)
    if "@" in addr:
        return addr.split("@", 1)[1].lower()
    return ""


def detect_schema(df):
    """detect_schema: Detects the schema of any given CSV file.

    Args:
        df (pandas.DataFrame): The dataframe to detect the schema of.

    Raises:
        ValueError: If no text column is found in the dataframe.

    Returns:
        dict: The detected schema with keys 'text_col', 'label_col', and 'columns'.
    """
    # Normalize column names by stripping whitespace and converting to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Heuristic to find the text column
    text_candidates = ["email", "text", "message", "body", "content"]
    for c in text_candidates:
        if c in df.columns:
            return {
                "text_col": c,
                "label_col": None,
                "columns": list(df.columns)
            }

    raise ValueError("No text column found")


def load_data(filepath):
    """load_data: Loads data from a file into a pandas DataFrame and detects the schema.

    For .mbox files, each message is parsed into an EmailRecord and returned as a
    DataFrame. For CSV files, the schema is auto-detected.

    Args:
        filepath (str): The path to the .mbox or .csv file to load.

    Returns:
        tuple: (DataFrame, list of text strings, schema dict)
               For mbox files, texts is the list of email bodies and schema describes
               the mbox columns.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    # Check if file exists
    if not filepath.exists():
        raise ValueError(f"File '{filepath}' does not exist.")
    
    if filepath.suffix.lower() == ".db":
        return load_from_db(filepath)

    # mbox path
    if filepath.suffix.lower() == ".mbox":
        mbox = mailbox.mbox(filepath)
        records = []

        for i, msg in enumerate(mbox):
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i} messages...")
            sender = msg.get("from", "")
            body = get_body(msg)

            record = EmailRecord(
                id=msg.get("message-id", str(i)).strip(),
                thread_id=msg.get("in-reply-to", None),
                sender=sender,
                sender_domain=extract_sender_domain(sender),
                recipients=get_recipients(msg),
                subject=msg.get("subject", ""),
                body=body,
                timestamp=msg.get("date", None),
                has_attachment=has_attachment(msg),
                labels=[],   # mbox has no native labels; extend here if needed
                raw_source=msg.as_string(),
            )
            records.append(record.__dict__)

        df = pd.DataFrame(records)
        texts = df["body"].astype(str).fillna("").tolist()
        schema = {
            "text_col": "body",
            "label_col": None,
            "columns": list(df.columns)
        }

        print("Detected schema (mbox):", schema)
        print("Would you like to create a saved schema for future use? (y/n)")
        if input().lower() == "y":
            save_to_db(df, "./emaildata/emails.db")
        
        return df, texts, schema

    # CSV path
    df = pd.read_csv(filepath)
    schema = detect_schema(df)
    texts = df[schema["text_col"]].astype(str).fillna("").tolist()

    print("Detected schema:", schema)
    return df, texts, schema

def save_to_db(df, db_path):
    """save_to_db: Persists a DataFrame of EmailRecords to a SQLite database.
 
    List fields (recipients, labels) are serialized as JSON strings so they
    can be stored in SQLite text columns and round-tripped cleanly.
 
    Args:
        df (pandas.DataFrame): The DataFrame to save. Must match the EmailRecord schema.
        db_path (str): Path to the .db file to create or overwrite.
 
    Returns:
        str: The resolved path to the saved database file.
    """
    db_df = df.copy()
 
    # Serialize list columns to JSON strings for SQLite storage
    for col in ("recipients", "labels"):
        if col in db_df.columns:
            db_df[col] = db_df[col].apply(
                lambda v: json.dumps(v) if isinstance(v, list) else v
            )
 
    with sqlite3.connect(db_path) as conn:
        db_df.to_sql("emails", conn, if_exists="replace", index=False)
 
    print(f"Saved {len(db_df)} records to {db_path}")
    return db_path

def load_from_db(db_path):
    """load_from_db: Loads EmailRecords from a SQLite database into a DataFrame.
 
    List fields (recipients, labels) are deserialized from their JSON string
    representation back into Python lists.
 
    Args:
        db_path (str): Path to the .db file to read.
 
    Raises:
        FileNotFoundError: If the database file does not exist.
 
    Returns:
        tuple: (DataFrame, list of body text strings, schema dict)
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
 
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM emails", conn)
 
    # Deserialize JSON string columns back to lists
    for col in ("recipients", "labels"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )
 
    texts = df["body"].astype(str).fillna("").tolist()
    schema = {
        "text_col": "body",
        "label_col": None,
        "columns": list(df.columns)
    }
 
    print(f"Loaded {len(df)} records from {db_path}")
    return df, texts, schema
