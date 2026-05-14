
class EmailSource:
    def get_emails(self):
        """get_emails: Abstract method to retrieve email data. Must be implemented by subclasses.

        Returns:
            list[EmailRecord]: A list of EmailRecord objects representing the emails.
        """
        pass  