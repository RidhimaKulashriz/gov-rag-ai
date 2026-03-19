import os
from typing import List, Dict
from pypdf import PdfReader

from metadata_extractor import MetadataExtractor


class PDFIngestor:

    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        self.extractor = MetadataExtractor()

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file
        """
        text = ""

        try:
            reader = PdfReader(file_path)

            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        return text.strip()

    def ingest(self) -> List[Dict]:
        """
        Ingest all PDFs and return structured documents
        """
        documents = []

        for filename in os.listdir(self.pdf_dir):

            if not filename.endswith(".pdf"):
                continue

            file_path = os.path.join(self.pdf_dir, filename)

            print(f"Processing: {filename}")

            text = self.extract_text_from_pdf(file_path)

            if not text:
                print(f"Skipping empty file: {filename}")
                continue

            # 🔥 Extract metadata
            metadata = self.extractor.extract(text)

            # Store structured doc
            documents.append({
                "text": text,
                "source": filename,
                "metadata": metadata
            })

        print(f"\nTotal documents processed: {len(documents)}")

        return documents


# 🔥 TEST
if __name__ == "__main__":
    ingestor = PDFIngestor(pdf_dir="data/pdfs")
    docs = ingestor.ingest()

    for doc in docs:
        print("\n---")
        print("Source:", doc["source"])
        print("Metadata:", doc["metadata"])
        print("Preview:", doc["text"][:200])