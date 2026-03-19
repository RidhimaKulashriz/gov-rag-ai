import os
from typing import List, Dict
from pypdf import PdfReader


class PDFIngestor:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def load_pdfs(self) -> List[str]:
        """
        Get all PDF file paths from directory
        """
        pdf_files = []
        for file in os.listdir(self.pdf_dir):
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(self.pdf_dir, file))
        return pdf_files

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF
        """
        reader = PdfReader(pdf_path)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        return text.strip()

    def ingest(self) -> List[Dict]:
        """
        Main ingestion pipeline
        Returns structured documents
        """
        documents = []
        pdf_files = self.load_pdfs()

        for pdf in pdf_files:
            text = self.extract_text(pdf)

            documents.append({
                "source": os.path.basename(pdf),
                "text": text
            })

        return documents


# 🔥 For testing
if __name__ == "__main__":
    ingestor = PDFIngestor(pdf_dir="data/pdfs")
    docs = ingestor.ingest()

    for doc in docs:
        print(f"\n📄 {doc['source']}")
        print(doc["text"][:500])  # preview first 500 char