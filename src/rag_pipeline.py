from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import Retriever

class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever = None,
        model_name: str = "facebook/bart-large-cnn",
        max_input_length: int = 1024,
        max_output_length: int = 300,
        min_output_length: int = 100
    ):
        self.retriever = retriever if retriever else Retriever()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length

        # Device handling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_prompt(self, question: str, context_docs: List[Dict]) -> str:
        """
        Creates a prompt that highlights differences between sources.
        """
        context_text = "\n".join(
            [f"Source {i+1}: {doc['text']}" for i, doc in enumerate(context_docs)]
        )
        prompt = (
            "Summarize the following documents to answer the question. "
            "Highlight any differences or supersessions between sources.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return prompt

    def answer(self, question: str, top_k: int = 10, include_sources: bool = True) -> str:
        """
        Retrieve relevant docs and generate a summary highlighting differences.
        """
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved_docs:
            return "No relevant documents found."

        prompt = self.generate_prompt(question, retrieved_docs)

        # Tokenize with truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=self.max_output_length,
            min_length=self.min_output_length,
            length_penalty=2.0,
            early_stopping=True,
            bos_token_id=self.model.config.bos_token_id
        )

        answer_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        if include_sources:
            answer_text += "\n\nSources:\n" + "\n".join([f"Source {i+1}" for i in range(len(retrieved_docs))])

        return answer_text


if __name__ == "__main__":
    rag = RAGPipeline()
    question = "What is the trauma protocol?"
    print("\nAnswer:\n", rag.answer(question))