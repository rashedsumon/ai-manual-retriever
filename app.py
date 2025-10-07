import os
import streamlit as st
from pathlib import Path
import pandas as pd
from utils.docs_loader import extract_text_from_pdf
from utils.embeddings_faiss import (
    build_faiss_index,
    query_faiss,
    save_faiss_index,
    load_faiss_index,
)
import openai

# === Streamlit setup ===
st.set_page_config(page_title="AI Manual Retriever (Prototype)", layout="wide")

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
    "Prototype notes:\n\n- Default embeddings use local sentence-transformers.\n"
    "- Upload PDFs or TXT files or enable Kaggle sample data."
)

st.title("AI Manual Retriever — Prototype MVP")
st.write(
    "Upload one or more PDF / text files, or load provided Kaggle dataset sample.\n"
    "The app builds a FAISS vector index and retrieves relevant passages to answer user questions."
)

# === Flexible Kaggle/local paths ===
KAGGLE_CSV = "/kaggle/input/data-retreiver/Data_ret.csv"
KAGGLE_STRUCTURED = "/kaggle/input/data-retreiver/Structured data-20250319T105519Z-001/Structured data"
LOCAL_CSV = "data/Data_ret.csv"
LOCAL_STRUCTURED = "data/Structured data"

# === File uploader ===
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files (or enable Kaggle dataset in sidebar)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

all_texts = []  # store {"source":..., "text":...}

# === Kaggle data loading ===
if use_kaggle_data:
    st.markdown("### Loading example Kaggle dataset (if available)")

    # Try Kaggle or local CSV
    if Path(KAGGLE_CSV).exists():
        csv_path = Path(KAGGLE_CSV)
    elif Path(LOCAL_CSV).exists():
        csv_path = Path(LOCAL_CSV)
    else:
        csv_path = None

    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            st.write("✅ Loaded CSV from:", csv_path)
            st.dataframe(df.head(5))
            for i, row in df.iterrows():
                text_cols = [str(v) for v in row.values if isinstance(v, (str, int, float))]
                joined = " \n".join(text_cols)
                if joined.strip():
                    all_texts.append({"source": f"CSV row {i}", "text": joined})
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        st.warning("CSV not found in Kaggle or local path.")

    # Try Kaggle or local structured folder
    if Path(KAGGLE_STRUCTURED).exists():
        structured_dir = Path(KAGGLE_STRUCTURED)
    elif Path(LOCAL_STRUCTURED).exists():
        structured_dir = Path(LOCAL_STRUCTURED)
    else:
        structured_dir = None

    if structured_dir:
        st.write("Scanning folder for PDFs/TXT:", structured_dir)
        for ext in ("**/*.pdf", "**/*.txt", "**/*.md"):
            for file in structured_dir.glob(ext):
                try:
                    if file.suffix.lower() == ".pdf":
                        text = extract_text_from_pdf(file)
                    else:
                        text = file.read_text(encoding="utf-8", errors="ignore")
                    all_texts.append({"source": str(file), "text": text})
                except Exception as e:
                    st.error(f"Error reading {file}: {e}")
    else:
        st.info("No structured data folder found.")

# === Uploaded files ===
if uploaded_files:
    upload_dir = Path("data/uploaded_files")
    upload_dir.mkdir(parents=True, exist_ok=True)
    st.write(f"Saving {len(uploaded_files)} uploaded files to `{upload_dir}` ...")
    for up in uploaded_files:
        save_path = upload_dir / up.name
        with open(save_path, "wb") as f:
            f.write(up.getbuffer())
        text = extract_text_from_pdf(save_path) if save_path.suffix.lower() == ".pdf" else save_path.read_text(encoding="utf-8", errors="ignore")
        all_texts.append({"source": f"uploaded/{up.name}", "text": text})

# === If no documents found ===
if not all_texts:
    st.info("No documents found. Upload PDFs/TXT or enable Kaggle dataset in sidebar.")
    st.stop()

st.success(f"📚 Loaded {len(all_texts)} document(s) for indexing.")

# === Build/load FAISS index ===
index_path = Path("/tmp/faiss_index")
if persist_index and index_path.exists() and any(index_path.glob("*")):
    st.info("Loading existing FAISS index from /tmp ...")
    index, metadata = load_faiss_index(str(index_path))
else:
    index, metadata = None, None

if st.button("Build / Rebuild FAISS Index"):
    with st.spinner("Building FAISS index..."):
        index, metadata = build_faiss_index(all_texts, embedding_backend=embedding_backend, openai_key=openai_api_key)
        st.success("✅ Index built successfully.")
        if persist_index:
            save_faiss_index(index, metadata, str(index_path))
            st.info(f"Index saved to {index_path}")

if index is None or metadata is None:
    st.warning("No index found. Please click 'Build / Rebuild FAISS Index'.")
    st.stop()

# === Q&A section ===
st.divider()
st.header("Ask a question about your manuals")
question = st.text_input("Enter your question:", "")
top_k = st.slider("Retrieved passages", 1, 10, 3)

if st.button("Get answer") and question.strip():
    with st.spinner("Retrieving relevant passages..."):
        results = query_faiss(index, metadata, question, top_k=top_k, embedding_backend=embedding_backend, openai_key=openai_api_key)
    st.subheader("Retrieved Passages")
    for r in results:
        st.markdown(f"**Source:** `{r['source']}` — score {r.get('score', 0):.4f}")
        st.write(r["text"][:2000])
        st.write("---")

    if openai_api_key:
        st.subheader("AI-generated summary answer")
        context_text = "\n\n---\n\n".join([f"Source: {r['source']}\n\n{r['text']}" for r in results])
        prompt = (
            "You are an assistant that answers the question using only the provided retrieved passages. "
            "If unknown, say 'I don't know from the provided documents.'\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.0,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            answer = f"(LLM call failed: {e})"
        st.write(answer)
    else:
        st.info("Set your OpenAI API key in the sidebar for AI-generated answers.")

st.markdown("---")
st.caption("Prototype: PDF/Text ➜ Embeddings ➜ FAISS ➜ Retrieval ➜ Optional LLM Answer.")
