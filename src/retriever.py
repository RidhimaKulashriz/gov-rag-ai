from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


class Retriever:
    def __init__(self, collection_name="gov_docs"):
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to Qdrant
        self.client = QdrantClient(path="./qdrant_storage")

        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query
        """

        # Convert query → embedding
        query_vector = self.model.encode(query).tolist()

        # ✅ NEW QDRANT API (IMPORTANT FIX)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points

        # Format results
        retrieved_docs = []

        for res in results:
            retrieved_docs.append({
                "text": res.payload["text"],
                "source": res.payload["source"],
                "chunk_id": res.payload["chunk_id"],
                "score": res.score
            })

        return retrieved_docs


# 🔥 TEST BLOCK
if __name__ == "__main__":
    retriever = Retriever()

    query = "What is the circular about?"
    results = retriever.retrieve(query)

    print("\n🔍 Query:", query)

    for i, r in enumerate(results):
        print("\n---")
        print(f"Rank {i+1} | Score: {r['score']:.4f}")
        print(f"Source: {r['source']} | Chunk: {r['chunk_id']}")
        print(r["text"][:300])