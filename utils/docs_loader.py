# utils/docs_loader.py
from pathlib import Path
from typing import List
import io
import pdfplumber

def extract_text_from_pdf(path_or_file):
    """
    Extracts text from a PDF file path or a file-like object.
    """
    text = []
    # pdfplumber handles local path or file-like
    try:
        with pdfplumber.open(path_or_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        # fallback: try PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(path_or_file)
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    text.append(t)
        except Exception:
            return ""
    return "\n\n".join(text)


def load_text_from_files(file_paths):
    """
    Returns list of dicts {"source": path, "text": text}
    """
    docs = []
    for p in file_paths:
        p = Path(p)
        if p.suffix.lower() == ".pdf":
            t = extract_text_from_pdf(p)
        else:
            t = p.read_text(encoding="utf-8", errors="ignore")
        docs.append({"source": str(p), "text": t})
    return docs
