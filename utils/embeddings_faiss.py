# utils/embeddings_faiss.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Tuple, Dict
from tqdm import tqdm
from utils.docs_loader import extract_text_from_pdf
import math

# embedding dim default for all-MiniLM-L6-v2 is 384
EMBEDDING_DIM = 384

# Simple chunker
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
    text = text.replace("\r", " ")
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# Embedding model loader
_cached_model = None
def get_local_embedding_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _cached_model

def compute_embeddings_batch(texts: List[str], embedding_backend="sentence-transformers (local)", openai_key: str = None):
    """
    Returns numpy array shape (n, dim)
    """
    if embedding_backend.startswith("openai"):
        # use OpenAI embeddings
        import openai
        openai.api_key = openai_key or os.getenv("OPENAI_API_KEY")
        model = "text-embedding-3-small"  # small ada alternative; adjust as desired
        resp = openai.Embedding.create(input=texts, model=model)
        embs = [r["embedding"] for r in resp["data"]]
        return np.array(embs, dtype=np.float32)
    else:
        model = get_local_embedding_model()
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(embs, dtype=np.float32)

def build_faiss_index(documents: List[Dict], embedding_backend="sentence-transformers (local)", openai_key: str = None):
    """
    documents: list of {"source":..., "text":...}
    returns: faiss_index, metadata_list
    metadata_list: list of dicts {"source":..., "text":..., "chunk_id":...}
    """
    texts = []
    metadatas = []
    for d in documents:
        chunks = chunk_text(d["text"])
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": d["source"], "text": c, "chunk_id": i})

    if not texts:
        raise ValueError("No text to index.")

    # compute embeddings in batches to avoid memory issues
    batch_size = 64
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = compute_embeddings_batch(batch, embedding_backend=embedding_backend, openai_key=openai_key)
        embeddings.append(embs)
    embeddings = np.vstack(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, metadatas

def query_faiss(index, metadata, query_text: str, top_k: int = 3, embedding_backend="sentence-transformers (local)", openai_key: str = None):
    q_emb = compute_embeddings_batch([query_text], embedding_backend=embedding_backend, openai_key=openai_key)
    D, I = index.search(q_emb.astype("float32"), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        results.append({"source": meta["source"], "text": meta["text"], "score": float(score)})
    return results

# Simple persistence: save and load using faiss.write_index and json metadata
def save_faiss_index(index, metadata, folder_path: str):
    import json, os
    os.makedirs(folder_path, exist_ok=True)
    faiss.write_index(index, os.path.join(folder_path, "index.faiss"))
    with open(os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_faiss_index(folder_path: str):
    import json, os
    idx_path = os.path.join(folder_path, "index.faiss")
    meta_path = os.path.join(folder_path, "metadata.json")
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index files not found in " + folder_path)
    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata
