import re
from typing import List, Dict


class SupersessionDetector:

    def detect(self, docs: List[Dict]) -> List[Dict]:
        """
        Detect supersession relationships between documents
        """

        results = []

        for doc in docs:
            text = doc["text"]

            # Patterns to detect supersession
            patterns = [
                r"supersedes\s+(.*)",
                r"in\s+supersession\s+of\s+(.*)",
                r"replaces\s+(.*)",
                r"substitutes\s+(.*)"
            ]

            found = None

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)

                if match:
                    found = match.group(1)
                    break

            results.append({
                "source": doc["source"],
                "circular_number": doc["metadata"].get("circular_number", "Unknown"),
                "date": doc["metadata"].get("date", "Unknown"),
                "supersedes": found
            })

        return results