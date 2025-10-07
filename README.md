# AI Manual Retriever — Prototype (Streamlit)

Prototype MVP showing how a Windows/Streamlit app can load technical manuals (PDF or text), index their content with embeddings + FAISS, and answer user questions via retrieval + LLM.

## Features
- Load PDFs / TXT files (uploads) or load example Kaggle dataset (if present).
- Chunk documents, compute embeddings (local sentence-transformers or OpenAI embeddings).
- Build FAISS vector index for retrieval.
- Ask questions; retrieve top-k passages; optionally generate final answer using OpenAI LLM.

## Files
- `app.py` — Streamlit app (main).
- `utils/docs_loader.py` — PDF/text extraction helpers.
- `utils/embeddings_faiss.py` — embedding computation + FAISS build/query + save/load.
- `requirements.txt`, `.gitignore`, `README.md`.

## Kaggle dataset (example)
This prototype can optionally scan the following Kaggle paths if you run it in a Kaggle/compatible environment:

- `/kaggle/input/data-retreiver/Data_ret.csv`
- `/kaggle/input/data-retreiver/Structured data-20250319T105519Z-001/Structured data`

If these paths exist in your environment, the app will attempt to read them and include their text in the index.

## Setup (local / Windows / Streamlit Cloud)
1. Clone the repo.
2. Create a Python 3.11 virtual environment and activate it.
3. Install requirements:
