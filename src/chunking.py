from typing import List, Dict


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks with overlap
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Convert full documents into chunked documents
        """
        chunked_docs = []

        for doc in documents:
            chunks = self.chunk_text(doc["text"])

            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "source": doc["source"],
                    "chunk_id": i,
                    "text": chunk
                })

        return chunked_docs


# 🔥 Integration Test
if __name__ == "__main__":
    from ingestion import PDFIngestor

    ingestor = PDFIngestor(pdf_dir="data/pdfs")
    docs = ingestor.ingest()

    chunker = TextChunker(chunk_size=500, overlap=100)
    chunked_docs = chunker.process_documents(docs)

    print(f"\n✅ Total Chunks: {len(chunked_docs)}")

    for c in chunked_docs[:5]:
        print("\n---")
        print(f"Source: {c['source']} | Chunk ID: {c['chunk_id']}")
        print(c["text"][:200])