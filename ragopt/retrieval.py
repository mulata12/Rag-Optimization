"""Index + search. The three optimisations are switches on PipelineConfig."""
from collections import defaultdict

import numpy as np

from .bm25 import BM25
from .chunking import build_chunks
from .config import TOP_K
from .rerank import paper_aware_mmr


def rrf(rank_lists, k):
    """Reciprocal Rank Fusion: score(d)=sum 1/(k+rank). Scale-free, so cosine and BM25
    scores (different units) can be combined without tuning a weight."""
    s = defaultdict(float)
    for lst in rank_lists:
        for r, i in enumerate(lst, 1):
            s[int(i)] += 1.0 / (k + r)
    return sorted(s.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, papers: dict, cfg, embedder, st):
        self.cfg, self.st = cfg, st
        self.chunks = [c for p in papers.values() for c in build_chunks(p, cfg.chunker, st)]
        self.E = embedder.embed_documents([c.embed_text for c in self.chunks])
        self.bm25 = BM25([c.embed_text for c in self.chunks], st.bm25_k1, st.bm25_b) if cfg.hybrid else None

    def search(self, query: str, qvec, k: int = TOP_K):
        """Searches the WHOLE split's corpus (no filtering by target paper)."""
        st = self.st
        dense = self.E @ qvec
        pool = max(st.pool, st.rerank_pool)
        d_top = np.argsort(-dense)[:pool]
        if self.cfg.hybrid:
            b_sc = self.bm25.scores(query)
            b_top = np.argsort(-b_sc)[:pool]
            cands = rrf([d_top, b_top], st.rrf_k)
        else:
            cands = [(int(i), float(dense[i])) for i in d_top]
        if self.cfg.rerank:
            cands = cands[: st.rerank_pool]
            return paper_aware_mmr([i for i, _ in cands], [s for _, s in cands],
                                   self.chunks, self.E, st, k)
        return cands[:k]
