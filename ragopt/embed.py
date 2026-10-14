"""Embedding with persistent caching (documents) and per-query encode-time caching."""
import hashlib
import pickle
import time

import numpy as np

from .config import CACHE_DIR, EMBED_MODEL, QUERY_PREFIX


class Embedder:
    def __init__(self, model_name=EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        import torch
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self._qpath = CACHE_DIR / "queries.pkl"
        CACHE_DIR.mkdir(exist_ok=True, parents=True)
        self._q = pickle.loads(self._qpath.read_bytes()) if self._qpath.exists() else {}

    def embed_documents(self, texts):
        key = hashlib.sha1((self.model_name + "\x00".join(texts)).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"docs_{key}.npy"
        if path.exists():
            return np.load(path)
        E = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=True, convert_to_numpy=True).astype("float32")
        np.save(path, E)
        return E

    def embed_query(self, text):
        """Returns (vector, encode_ms). Unchanged queries are served from cache, but the
        ORIGINAL encode time is reported so latency stays honest."""
        if text not in self._q:
            t = time.perf_counter()
            v = self.model.encode(QUERY_PREFIX + text, normalize_embeddings=True,
                                  convert_to_numpy=True).astype("float32")
            self._q[text] = (v, (time.perf_counter() - t) * 1000)
            self._qpath.write_bytes(pickle.dumps(self._q))
        return self._q[text]
