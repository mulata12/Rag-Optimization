"""Retrieval metrics (80%-token-coverage rule) and answer metrics (normalised F1)."""
import re
import string
from collections import Counter

import numpy as np

COVER_THRESHOLD = 0.80


def gold_coverage(q, paper, retrieved, k):
    """Token coverage of each gold paragraph by the union of spans in the top-k chunks
    (only chunks from the TARGET paper count)."""
    masks = {pid: np.zeros(len(paper.par[pid].text), bool) for pid in q.gold}
    for ch in retrieved[:k]:
        if ch["paper_id"] != q.paper_id:
            continue
        for sp in ch["source_spans"]:
            m = masks.get(sp["paragraph_id"])
            if m is not None:
                m[sp["start"]:sp["end"]] = True
    cov = {}
    for pid, m in masks.items():
        toks = paper.par[pid].tokens
        cov[pid] = (sum(bool(m[s:e].all()) for s, e in toks) / len(toks)) if toks else 0.0
    return cov


def score_question(q, paper, retrieved):
    """Recall@k = share of gold paragraphs whose token coverage >= 80% within the top-k.
    MRR@10 = 1/r for the smallest r at which ANY gold paragraph reaches >= 80%."""
    def recall(k):
        cov = gold_coverage(q, paper, retrieved, k)
        return sum(c >= COVER_THRESHOLD for c in cov.values()) / len(cov)
    rr = 0.0
    for r in range(1, len(retrieved) + 1):
        if any(c >= COVER_THRESHOLD for c in gold_coverage(q, paper, retrieved, r).values()):
            rr = 1.0 / r
            break
    return {"recall5": recall(5), "recall10": recall(10), "rr": rr,
            "cov10": gold_coverage(q, paper, retrieved, 10)}


# ---------------------------------------------------------------- answer metrics
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def best_f1(pred, golds):
    return max((answer_f1(pred, g) for g in golds), default=0.0)
