import re


class MetadataExtractor:

    def extract(self, text):
        """
        Extract metadata from document text
        """

        metadata = {}

        # Extract date
        date_match = re.search(r'\b\d{1,2}\s\w+\s\d{4}\b', text)
        metadata["date"] = date_match.group() if date_match else "Unknown"

        # Extract circular number
        circ_match = re.search(r'(Circular\s*No\.?\s*[\w\/-]+)', text, re.IGNORECASE)
        metadata["circular_number"] = circ_match.group() if circ_match else "Unknown"

        # Simple classification
        metadata["doc_type"] = self.classify(text)

        return metadata

    def classify(self, text):
        text = text.lower()

        if "circular" in text:
            return "Circular"
        elif "notification" in text:
            return "Notification"
        elif "guideline" in text:
            return "Guideline"
        elif "advisory" in text:
            return "Advisory"
        else:
            return "Other"