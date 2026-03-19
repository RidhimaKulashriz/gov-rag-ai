# ✅ FIX: Add src to path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Imports
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np


class DocumentGraphBuilder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_graph(self, docs):
        """
        Build similarity graph between documents
        """

        # Step 1: Combine text per document
        doc_texts = {}

        for doc in docs:
            source = doc["source"]

            if source not in doc_texts:
                doc_texts[source] = ""

            doc_texts[source] += " " + doc["text"][:500]

        doc_names = list(doc_texts.keys())
        texts = list(doc_texts.values())

        # Step 2: Generate embeddings
        embeddings = self.model.encode(texts)

        # Step 3: Create graph
        G = nx.Graph()

        # Add nodes
        for name in doc_names:
            G.add_node(name)

        # Step 4: Add edges based on similarity
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):

                sim = self.cosine_similarity(
                    embeddings[i],
                    embeddings[j]
                )

                # 🔥 Adjust threshold if needed
                if sim > 0.5:
                    G.add_edge(doc_names[i], doc_names[j], weight=sim)

        return G

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def visualize(self, G):
        """
        Draw the graph
        """
        plt.figure(figsize=(8, 6))

        pos = nx.spring_layout(G)

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=3000,
            font_size=10
        )

        plt.title("Document Relationship Graph")
        plt.show()


# 🔥 TEST
if __name__ == "__main__":
    from ingestion import PDFIngestor

    # Load documents
    ingestor = PDFIngestor(pdf_dir="data/pdfs")
    docs = ingestor.ingest()

    # Build graph
    graph_builder = DocumentGraphBuilder()
    G = graph_builder.build_graph(docs)

    # Visualize
    graph_builder.visualize(G)