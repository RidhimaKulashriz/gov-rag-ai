from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid


class Embedder:

    def __init__(self, collection_name="gov_docs"):
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Qdrant local storage
        self.client = QdrantClient(path="qdrant_storage")

        self.collection_name = collection_name

        self._create_collection()

    def _create_collection(self):
        """
        Create Qdrant collection if not exists
        """
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

    def embed_and_store(self, chunked_docs: List[Dict]):
        """
        Generate embeddings and store in Qdrant with metadata
        """
        points = []

        for doc in chunked_docs:
            embedding = self.model.encode(doc["text"]).tolist()

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        "source": doc["source"],
                        "chunk_id": doc["chunk_id"],

                        # 🔥 NEW: metadata added
                        "doc_type": doc.get("metadata", {}).get("doc_type", "Unknown"),
                        "date": doc.get("metadata", {}).get("date", "Unknown"),
                        "circular_number": doc.get("metadata", {}).get("circular_number", "Unknown")
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Stored {len(points)} chunks in Qdrant")


# 🔥 TEST PIPELINE
if __name__ == "__main__":
    from ingestion import PDFIngestor
    from chunking import TextChunker

    # Step 1: Ingest
    ingestor = PDFIngestor(pdf_dir="data/pdfs")
    docs = ingestor.ingest()

    # Step 2: Chunk
    chunker = TextChunker()
    chunked_docs = chunker.process_documents(docs)

    # Step 3: Embed + Store
    embedder = Embedder()
    embedder.embed_and_store(chunked_docs)