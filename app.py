# app.py
import os
import glob
import streamlit as st
from pathlib import Path
import pandas as pd
from utils.docs_loader import load_text_from_files, extract_text_from_pdf
from utils.embeddings_faiss import (
    build_faiss_index,
    query_faiss,
    save_faiss_index,
    load_faiss_index,
    EMBEDDING_DIM,
    compute_embeddings_batch,
)
import tempfile
import openai
import json

st.set_page_config(page_title="AI Manual Retriever (Prototype)", layout="wide")

# === Sidebar / Config ===
st.sidebar.title("Settings")
embedding_backend = st.sidebar.selectbox(
    "Embeddings backend",
    options=["sentence-transformers (local)", "openai (embeddings)"],
    index=0,
)
use_kaggle_data = st.sidebar.checkbox("Load example Kaggle dataset paths", value=True)
persist_index = st.sidebar.checkbox("Persist FAISS index to /tmp/faiss_index", value=True)

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key (only required for LLM answers or OpenAI embeddings)",
    type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
)
if openai_api_key:
    openai.api_key = openai_api_key

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Prototype notes:\n\n- By default embeddings use `sentence-transformers` locally. "
    "Set OpenAI key to use OpenAI for generation or embeddings.\n- You can upload PDFs or TXT files."
)

# === main area ===
st.title("AI Manual Retriever — Prototype MVP")
st.write(
    """
Upload one or more PDF / text files, or load provided Kaggle dataset sample.
The app creates a vector index of document chunks and answers user queries by retrieving relevant passages.
"""
)

# === Kaggle dataset paths (as provided by user) ===
kaggle_csv_path = "/kaggle/input/data-retreiver/Data_ret.csv"
kaggle_structured_dir = "/kaggle/input/data-retreiver/Structured data-20250319T105519Z-001/Structured data"

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files (you can upload multiple). Or check 'Load Kaggle dataset' in sidebar.",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

all_texts = []  # list of dicts: {"source":..., "text":...}

# Optionally load Kaggle CSV and folder (if present in environment)
if use_kaggle_data:
    st.markdown("### Loading example Kaggle dataset (if available)")
    if Path(kaggle_csv_path).exists():
        try:
            df = pd.read_csv(kaggle_csv_path)
            st.write("Loaded CSV from Kaggle:", kaggle_csv_path)
            st.dataframe(df.head(5))
            # If CSV has text columns, add them as documents
            for i, row in df.iterrows():
                text_cols = [str(v) for v in row.values if isinstance(v, (str, int, float))]
                joined = " \n".join(text_cols)
                if joined.strip():
                    all_texts.append({"source": f"Kaggle CSV row {i}", "text": joined})
        except Exception as e:
            st.error(f"Failed to read Kaggle CSV: {e}")
    else:
        st.info(f"Kaggle CSV path not found: {kaggle_csv_path}")

    if Path(kaggle_structured_dir).exists():
        st.write("Scanning Kaggle structured data folder for text files:", kaggle_structured_dir)
        patterns = ["**/*.txt", "**/*.pdf", "**/*.md"]
        for patt in patterns:
            for p in Path(kaggle_structured_dir).glob(patt):
                st.write("Found:", str(p))
                if p.suffix.lower() == ".pdf":
                    text = extract_text_from_pdf(p)
                    all_texts.append({"source": str(p), "text": text})
                else:
                    try:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                        all_texts.append({"source": str(p), "text": text})
                    except Exception:
                        continue
    else:
        st.info(f"Kaggle structured folder not found: {kaggle_structured_dir}")

# Save uploaded files to a temp folder and load text
if uploaded_files:
    tmp_dir = Path("data/uploaded_files")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    st.write(f"Saving {len(uploaded_files)} uploaded files to `{tmp_dir}` and extracting text...")
    for up in uploaded_files:
        save_path = tmp_dir / up.name
        with open(save_path, "wb") as f:
            f.write(up.getbuffer())
        if save_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(save_path)
        else:
            text = save_path.read_text(encoding="utf-8", errors="ignore")
        all_texts.append({"source": f"uploaded/{up.name}", "text": text})

if not all_texts:
    st.info("No documents found yet. Upload PDFs/TXT or enable Kaggle data in the sidebar.")
    st.stop()

st.success(f"Found {len(all_texts)} documents / rows to index.")

# === Build or load index ===
index_path = Path("/tmp/faiss_index")  # simple persistence point in cloud
if persist_index and index_path.exists() and any(index_path.glob("*")):
    st.info("Loading existing FAISS index from disk...")
    index, metadata = load_faiss_index(str(index_path))
else:
    index, metadata = None, None

if st.button("Build (or rebuild) vector index from loaded documents"):
    # prepare texts and metadata
    documents = []
    for doc in all_texts:
        documents.append({"source": doc["source"], "text": doc["text"]})

    # chunk, embed, build FAISS
    with st.spinner("Building chunks, computing embeddings, and creating FAISS index..."):
        index, metadata = build_faiss_index(documents, embedding_backend=embedding_backend, openai_key=openai_api_key)
        st.success("Index built.")
        if persist_index:
            save_faiss_index(index, metadata, str(index_path))
            st.write("Index saved to", str(index_path))

# If index is still None (not built nor loaded), ask to build
if index is None or metadata is None:
    st.warning("No vector index available. Please press 'Build (or rebuild) vector index...' to create it.")
    st.stop()

st.divider()
st.header("Ask a question")
question = st.text_input("Enter your question about the manuals / data:", "")
top_k = st.slider("How many retrieved passages to use", 1, 10, 3)

if st.button("Get answer") and question.strip():
    with st.spinner("Retrieving relevant passages..."):
        results = query_faiss(index, metadata, question, top_k=top_k, embedding_backend=embedding_backend, openai_key=openai_api_key)
    # results: list of dicts {"source":..., "text":..., "score":...}
    st.subheader("Retrieved Passages")
    for r in results:
        st.markdown(f"**Source:** `{r['source']}` — score {r.get('score', 0):.4f}")
        st.write(r["text"][:2000])
        st.write("---")

    # If OpenAI key present, generate a concise answer using retrieved context
    if openai_api_key:
        st.subheader("AI-generated answer (OpenAI LLM)")
        # build prompt
        retrieved_text = "\n\n---\n\n".join([f"Source: {r['source']}\n\n{r['text']}" for r in results])
        prompt = (
            "You are an assistant that answers the user's question using only the provided retrieved passages. "
            "If the answer is not present, say 'I don't know from the provided documents.'\n\n"
            f"Context:\n{retrieved_text}\n\nQuestion: {question}\n\nAnswer:"
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list() else "gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # fallback to simpler call or model names
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.0,
                )
                answer = resp["choices"][0]["message"]["content"].strip()
            except Exception as e2:
                answer = f"(LLM call failed: {e2})"
        st.write(answer)
    else:
        st.info("Set OpenAI API key in the sidebar to generate an LLM answer. For now, treat retrieved passages above as the answer.")

st.write("\n---\nPrototype built by bundling: simple PDF/text extraction → chunking → embeddings → FAISS retrieval → optional OpenAI LLM.")
