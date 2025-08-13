import os, glob, re
from datetime import datetime
from typing import List
from sec_edgar_downloader import Downloader
from pypdf import PdfReader

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.getenv("CHROMA_DIR", "storage")

def _get_client():
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
    return client

def _embedding_model():
    return SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))

def _chunk(text: str, chunk_size: int = 900, overlap: int = 150):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return chunks

def _read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        txt = ""
        for p in reader.pages:
            txt += p.extract_text() or ""
        return txt
    except Exception:
        return ""

def ensure_sec_index(ticker: str) -> str:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = _get_client()
    coll_name = f"sec_{ticker.lower()}"
    # (re)create collection if missing
    try:
        coll = client.get_collection(coll_name)
    except Exception:
        coll = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})

    # download last 2 filings of each type
    dl = Downloader("edgar_cache")
    forms = ["10-K", "10-Q"]
    paths = []
    for f in forms:
        try:
            res = dl.get(f, ticker, amount=2)
            paths.extend(res)
        except Exception:
            pass

    if not paths:
        return "No new filings downloaded (maybe already cached or ticker invalid)."

    embedder = _embedding_model()
    ids, docs, metas = [], [], []
    for p in paths:
        # Try to capture filing date from path name
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p)
        filing_date = m.group(1) if m else ""
        txt = _read_pdf(p)
        if not txt:
            continue
        chunks = _chunk(txt)
        for i, ch in enumerate(chunks):
            ids.append(f"{os.path.basename(p)}_{i}")
            docs.append(ch)
            metas.append({"source": p, "filing_date": filing_date, "form": os.path.basename(os.path.dirname(p))[:4]})
    if not docs:
        return "Parsed 0 pages; PDF parsing may have failed."
    embs = embedder.encode(docs, show_progress_bar=False).tolist()
    coll.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    return f"Indexed {len(docs)} chunks from {len(paths)} filings."
