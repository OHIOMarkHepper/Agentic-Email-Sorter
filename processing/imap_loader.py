"""
@note This module is responsible for loading email data from a CSV file, saving it to a SQLite database, and executing SQL queries. 
      It uses the `pandas` library to handle data manipulation and the `sqlite3` library to interact with the SQLite database. 
      The `DataProcessor` class provides methods for loading email data from a CSV file, saving it to a SQLite database, and executing SQL queries. 
"""

import imaplib
import email as email_lib
import pandas as pd
from processing.base import EmailSource
from processing.data import (
    get_body, has_attachment, get_recipients,
    extract_sender_domain, EmailRecord, save_to_db
)

class IMAPSource(EmailSource):
    def __init__(self, imap_server: str, email_user: str, access_token: str,
                 mailbox: str = "INBOX", max_emails: int = 500):

        self.imap_server = imap_server
        self.email_user = email_user
        self.access_token = access_token  # OAuth2 token string
        self.mailbox_name = mailbox
        self.max_emails = max_emails

    def _connect(self):
        """ Connects to the imap server
        Returns:
            status, data = mail.uid("search")
        """
        mail = imaplib.IMAP4_SSL(self.imap_server, 993)
        # XOAUTH2 string format required by Gmail
        auth_string = f"user={self.email_user}\x01auth=Bearer {self.access_token}\x01\x01"
        mail.authenticate("XOAUTH2", lambda _: auth_string.encode())
        return mail

    def get_emails(self):
        """get_emails Connects to the IMAP server and retrieves emails from the specified mailbox.
            It returns a DataFrame containing the emails, their bodies, and other relevant information. 

        Returns:
            df: DataFrame containing the emails
            texts List[str]: List of email bodies
            schema dict: Schema for the DataFrame
        """
        mail = self._connect()
        mail.select(self.mailbox_name)

        # Fetch all message UIDs — swap "ALL" for "UNSEEN" to only grab new mail
        status, data = mail.uid("search", None, "ALL")
        uids = data[0].split()
        uids = uids[-self.max_emails:]  # take the most recent N

        records = []
        for i, uid in enumerate(uids):
            if i % 100 == 0 and i > 0:
                print(f"Fetched {i}/{len(uids)} emails...")
            status, msg_data = mail.uid("fetch", uid, "(RFC822)")
            if msg_data[0] is None:
                continue
            raw = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw)
            records.append(self._parse_message(msg, i))

        mail.logout()

        df = pd.DataFrame([r.__dict__ for r in records])
        texts = df["body"].astype(str).fillna("").tolist()
        schema = {"text_col": "body", "label_col": None, "columns": list(df.columns)}
        return df, texts, schema

    def _parse_message(self, msg, index: int) -> EmailRecord:
        sender = msg.get("from", "")
        return EmailRecord(
            id=msg.get("message-id", str(index)).strip(),
            thread_id=msg.get("in-reply-to", None),
            sender=sender,
            sender_domain=extract_sender_domain(sender),
            recipients=get_recipients(msg),
            subject=msg.get("subject", ""),
            body=get_body(msg),
            timestamp=msg.get("date", None),
            has_attachment=has_attachment(msg),
            labels=msg.get("x-gmail-labels", "").split(",") if msg.get("x-gmail-labels") else [],
            raw_source=msg.as_string(),
        )