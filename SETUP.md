# Government RAG AI

A Retrieval-Augmented Generation (RAG) system for querying government documents using natural language.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Google Gemini API Key

1. Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```

### 3. Run the Application

Since you're already in the project directory, run:

**Bash/Command Prompt:**
```bash
streamlit run chatbot/app.py
```

**PowerShell:**
```powershell
streamlit run chatbot/app.py
```

**If you need to change directory first:**
```powershell
cd "c:/Users/HP/Downloads/Bharatx"; streamlit run chatbot/app.py
```

The app will be available at `http://localhost:8501`

## Features

- 📄 Upload and process PDF documents
- 🔍 Search documents using natural language
- 🤖 AI-powered answers with document citations
- 📊 Document relationship visualization
- ⚡ Fast similarity-based retrieval

## Project Structure

```
.
├── chatbot/
│   └── app.py              # Streamlit web interface
├── src/
│   ├── rag_pipeline.py     # Core RAG Pipeline
│   ├── ingestion.py        # Document ingestion
│   ├── embedding.py        # Embedding generation
│   ├── retriever.py        # Document retrieval
│   ├── chunking.py         # Text chunking
│   └── ...                 # Other modules
├── data/
│   └── pdfs/               # Sample documents
├── cli.py                  # Command-line interface
└── requirements.txt        # Python dependencies
```

## Environment Variables

- `GOOGLE_API_KEY` - Your Google Gemini API key (required)
- `GEMINI_MODEL` - Model to use (default: gemini-pro)
- `QDRANT_URL` - Qdrant server URL (optional)
- `QDRANT_PORT` - Qdrant server port (optional)

## Usage

### Web Interface

1. Upload PDF documents using the sidebar
2. Configure chunk size and overlap settings
3. Click "Process Documents" to create embeddings
4. Ask questions in the chat interface

### Command Line

```bash
python cli.py --help
```

## License

MIT
