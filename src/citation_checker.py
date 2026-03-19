from typing import List, Dict


class CitationChecker:
    def verify(self, answer: str, sources: List[Dict]) -> List[Dict]:
        """
        Verify sources using keyword overlap
        """
        verified = []

        answer_words = set(answer.lower().split())

        for src in sources:
            chunk_words = set(src["text"].lower().split())

            overlap = len(answer_words & chunk_words)

            if overlap > 10:
                verified.append({
                    "source": src["source"],
                    "chunk_id": src["chunk_id"]
                })

        return verified