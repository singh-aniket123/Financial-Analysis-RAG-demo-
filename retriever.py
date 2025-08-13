import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.getenv("CHROMA_DIR", "storage")

def _client():
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))

def _embedder():
    return SentenceTransformer(os.getenv("EMBED_MODEL","all-MiniLM-L6-v2"))

def retrieve(ticker: str, query: str, k: int = 5):
    coll_name = f"sec_{ticker.lower()}"
    client = _client()
    try:
        coll = client.get_collection(coll_name)
    except Exception:
        return []
    embedder = _embedder()
    q = embedder.encode([query]).tolist()
    res = coll.query(query_embeddings=q, n_results=k, include=["documents","metadatas","distances"])
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "page_content": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })
    return out
