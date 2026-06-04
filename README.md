# RAG Paper Evaluation (LLM)

This script compares an answer paper against an answer key using a retrieval-augmented approach and a large language model (Gemini). It extracts text from PDFs, splits the text into chunks, embeds the chunks with a SentenceTransformer, stores embeddings in ChromaDB, and asks Gemini to evaluate each answer.

## Files

- `paper-evaluation-rag-llm.py` — main script that runs the workflow.

## Requirements

- Python 3.8+
- Packages (install via `pip`):
  - `python-dotenv`
  - `pdfplumber`
  - `chromadb`
  - `sentence-transformers`
  - `google-generativeai`
  - `torch`

Example:

```
pip install python-dotenv pdfplumber chromadb sentence-transformers google-generativeai torch
```

## Environment

- Create a `.env` file with your Google API key:

```
GOOGLE_API_KEY=your_key_here
```

## High-level Workflow

1. Force CPU usage and disable CUDA to ensure GPU-safe execution.
2. Configure Gemini with the `GOOGLE_API_KEY` and create a `GenerativeModel` instance.
3. Extract text from input PDFs using `pdfplumber` (`extract_text_from_pdf`).
4. Split the extracted text into chunks. The script provides two chunking helpers:
   - `chunk_text(text, chunk_size=500, overlap=100)` — fixed-size chunks with overlap.
   - `chunk_by_questions(text)` — splits on double newlines (used in the current flow).
5. Load a CPU `SentenceTransformer` embedding model (`all-MiniLM-L6-v2`).
6. Create two ChromaDB collections:
   - `answer_keys_collection` for the answer key chunks.
   - `answer_paper_collection` for the answer paper chunks.
7. Encode and store chunks as embeddings in the respective collections using `store_chunks(DB, chunks)`.
8. For each chunk, build a query containing:
   - the question (chunk content),
   - the matching answer key chunk from `answer_keys_collection`, and
   - the corresponding answer paper chunk from `answer_paper_collection`.
     Then call Gemini to evaluate the paper answer against the key and print a score.

## How to run

From the `RAG-Paper-Evaluation` folder run:

```
python paper-evaluation-rag-llm.py

# The script will prompt for two file paths:
# 1) Answer Key File Path
# 2) Answer Paper File Path
```