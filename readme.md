

# Government Document Intelligence Platform

This project is an AI powered system designed to analyze and query government documents using Retrieval Augmented Generation. It enables intelligent search, summarization, and question answering over PDF based circulars while ensuring that responses are grounded in source documents.



## Overview

The platform processes government circulars in PDF format and converts them into structured knowledge. It extracts text, divides it into meaningful chunks, generates embeddings, and stores them in a vector database.

At query time, the system retrieves the most relevant document segments and generates accurate answers using only the retrieved context.



## Features

* PDF ingestion and text extraction
* Intelligent document chunking
* Semantic embeddings using Sentence Transformers
* Vector search using Qdrant
* Retrieval Augmented Generation based question answering
* Source citation and verification
* Automatic document summarization
* Document relationship graph using NetworkX
* Timeline extraction from document content



## System Architecture

The system follows a modular pipeline:

1. PDF files are ingested
2. Text is extracted
3. Content is divided into chunks
4. Embeddings are generated
5. Data is stored in a vector database
6. Relevant chunks are retrieved for a query
7. A language model generates the final answer


## Tech Stack

* Python
* Sentence Transformers
* Qdrant
* HuggingFace Transformers
* NetworkX


## Project Structure

```
gov-rag-ai/

data/                # Input PDF files  
src/                 # Core modules (ingestion, chunking, embedding, retriever, etc.)  
chatbot/             # Interface logic (can be adapted to API)  
visualization/       # Graph builder  
requirements.txt  
README.md  
```


## Installation

1. Create a virtual environment
2. Activate the environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```



## Running the System

1. Place PDF files inside the data directory
2. Run ingestion and embedding pipeline
3. Use the RAG pipeline module to query the system programmatically

Example:

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

answer, sources, verified = rag.generate_answer("Your question here")

print(answer)
```



## Deployment

This project can be deployed using a backend API framework and hosted on serverless platforms such as Vercel.

Suggested approach:

* Wrap the RAG pipeline inside an API using FastAPI
* Expose endpoints for query and document ingestion
* Connect a frontend interface if needed
* Deploy using Vercel serverless functions or a compatible backend service



## Key Highlights

* End to end implementation of a Retrieval Augmented Generation pipeline
* Grounded answers with verifiable citations
* Semantic document relationship graph
* Modular and scalable architecture



## Future Improvements

* PDF answer highlighting
* Timeline visualization dashboard
* Multi document comparison
* Scalable cloud deployment with managed vector database



## Use Case

This system can be used by analysts, researchers, and policymakers to efficiently extract insights from large volumes of government documents without manual review.

