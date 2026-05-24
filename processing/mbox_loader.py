from processing.base import EmailSource
from processing.data import (
    get_body, has_attachment, get_recipients,
    extract_sender_domain, EmailRecord, save_to_db
)
from google_auth_httplib2 import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import mailbox
import pandas as pd

class MboxSource(EmailSource):
    def __init__(self, mbox_path):
        self.mbox_path = mbox_path

    def get_emails(self):
        """get_emails: Retrieves email data from an mbox file.

        Returns:
            list[EmailRecord]: A list of EmailRecord objects representing the emails in the mbox file.
        """
        # Implementation to read from mbox file and return list of EmailRecord objects
        pass
    
    def _parse_mbox(self):
        
        mbox = mailbox.mbox(self.mbox_path)
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