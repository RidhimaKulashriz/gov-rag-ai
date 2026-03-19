from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid


class Embedder:   # ✅ CLASS STARTS HERE

    def __init__(self, collection_name="gov_docs"):   # ✅ INSIDE CLASS
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to Qdrant
        self.client = QdrantClient(path="./qdrant_storage")

        self.collection_name = collection_name

        # 🔥 Reset collection safely
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except:
            pass

        self._create_collection()

    def _create_collection(self):
        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

    def embed_and_store(self, chunked_docs: List[Dict]):
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
                        "chunk_id": doc["chunk_id"]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"✅ Stored {len(points)} chunks in Qdrant")


# 🔥 TEST
if __name__ == "__main__":
    from ingestion import PDFIngestor
    from chunking import TextChunker

    ingestor = PDFIngestor(pdf_dir="data/pdfs")
    docs = ingestor.ingest()

    chunker = TextChunker()
    chunked_docs = chunker.process_documents(docs)

    embedder = Embedder()
    embedder.embed_and_store(chunked_docs)