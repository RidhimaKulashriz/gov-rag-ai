from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import Retriever
from citation_checker import CitationChecker


class RAGPipeline:
    def __init__(self):
        # Initialize retriever
        self.retriever = Retriever()

        # Load FLAN-T5 model correctly (seq2seq)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        # Citation verification module
        self.citation_checker = CitationChecker()

    def generate_answer(self, query: str):
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks
        2. Clean and compress context
        3. Generate answer
        4. Verify citations
        """

        # -------------------------
        # Step 1: Retrieve documents
        # -------------------------
        docs = self.retriever.retrieve(query)

        # -------------------------
        # Step 2: Clean context
        # -------------------------
        clean_chunks = []

        for doc in docs:
            text = doc["text"]

            # remove noisy lines
            lines = text.split("\n")
            lines = [
                l for l in lines
                if len(l.strip()) > 40 and "Figure" not in l
            ]

            cleaned = " ".join(lines)
            clean_chunks.append(cleaned[:300])  # keep context short

        context = " ".join(clean_chunks)

        # -------------------------
        # Step 3: Prompt
        # -------------------------
        prompt = f"""
You are an expert government policy assistant.

Carefully read the context and answer the question clearly.
Do NOT copy text directly. Explain in your own words.

Context:
{context}

Question:
{query}

Give a detailed answer in 3-5 sentences:
"""

        # -------------------------
        # Step 4: Tokenize
        # -------------------------
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # -------------------------
        # Step 5: Generate answer
        # -------------------------
        outputs = self.model.generate(
            **inputs,
            max_length=200
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # -------------------------
        # Step 6: Verify citations
        # -------------------------
        verified = self.citation_checker.verify(answer, docs)

        return answer, docs, verified


# ---------------------------------
# CLI TESTING
# ---------------------------------
if __name__ == "__main__":
    rag = RAGPipeline()

    query = "What is the trauma protocol?"

    answer, sources, verified = rag.generate_answer(query)

    print("\n🧠 ANSWER:\n")
    print(answer)

    print("\n📚 RETRIEVED SOURCES:\n")
    for s in sources:
        print(f"{s['source']} | Chunk {s['chunk_id']}")

    print("\n✅ VERIFIED SOURCES:\n")
    for v in verified:
        print(f"{v['source']} | Chunk {v['chunk_id']}")