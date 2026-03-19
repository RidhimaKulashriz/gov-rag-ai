from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class DocumentSummarizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def summarize(self, text: str) -> str:
        """
        Generate summary using direct model (no pipeline)
        """

        # Limit input
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
             if len(line.strip()) > 30 and "Figure" not in line:
                  clean_lines.append(line)
        clean_text = " ".join(clean_lines)

# Limit size
        clean_text = clean_text[:3000]

        prompt = f"Summarize the following government document:\n\n{text}"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = self.model.generate(
            **inputs,
            max_length=200,
            min_length=50
        )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summary


# 🔥 TEST
if __name__ == "__main__":
    from ingestion import PDFIngestor

    ingestor = PDFIngestor(pdf_dir="data/pdfs")
    docs = ingestor.ingest()

    summarizer = DocumentSummarizer()

    for doc in docs:
        print(f"\n📄 {doc['source']}")
        summary = summarizer.summarize(doc["text"])
        print("📝 Summary:\n", summary)