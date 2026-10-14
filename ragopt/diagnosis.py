"""Helpers for the 10-case failure analysis (step B). Tags are *hints* - you must still read
the cases (show_case) and write/confirm the table yourself."""
import numpy as np
import pandas as pd

from .metrics import COVER_THRESHOLD, gold_coverage
from .rerank import span_overlap
from .chunking import Chunk

TAGS = {
    "boundary": ("Gold paragraph only partly covered by retrieved chunks (0 < coverage < 80%).",
                 "Fixed windows cut evidence at arbitrary characters / across sections.",
                 "A: sentence-snapped, section-aware chunks."),
    "wrong_paper": ("Top-1 chunk comes from a different paper than the target.",
                    "Short/generic query ('What dataset do they use?') matches many papers; "
                    "dense vectors blur lexical cues, no paper-level coherence.",
                    "B (lexical signal) and/or C (paper prior)."),
    "repetitive": ("Top-10 contain heavily overlapping chunks (wasted slots).",
                   "Overlapping windows return near-duplicates of the same passage.",
                   "C: span-overlap dedup + MMR."),
    "miss": ("No gold paragraph in top-10, no obvious structural cause.",
             "Vocabulary mismatch between question and paper wording / abstract question.",
             "B (BM25 on rare terms) or query rewriting/HyDE (LLM cost)."),
}


def _overlap_ratio(retrieved):
    cs = [Chunk("", c["paper_id"], c["text"], c["source_spans"], "") for c in retrieved]
    vals = [span_overlap(c, cs[:i]) for i, c in enumerate(cs) if i]
    return float(np.mean(vals)) if vals else 0.0


def tag_case(q, paper, retrieved):
    cov = gold_coverage(q, paper, retrieved, 10)
    tags = []
    if any(0 < c < COVER_THRESHOLD for c in cov.values()):
        tags.append("boundary")
    if retrieved[0]["paper_id"] != q.paper_id:
        tags.append("wrong_paper")
    if _overlap_ratio(retrieved) > 0.25:
        tags.append("repetitive")
    return tags or ["miss"]


def failure_table(exp, name, n=10):
    """Pick n poor dev cases (recall@10 < 1), balanced across failure tags."""
    recs = {r["question_id"]: r for r in exp.records[name]}
    buckets = {t: [] for t in TAGS}
    for row in sorted(exp.per_question(name), key=lambda r: (r["recall10"], r["rr"])):
        if row["recall10"] >= 1.0:
            continue
        q = exp.qmap[row["qid"]]
        tags = tag_case(q, exp.papers[q.paper_id], recs[q.qid]["retrieved"])
        buckets[tags[0]].append((q, row, tags))
    out, i = [], 0
    while len(out) < n and any(buckets.values()):
        t = list(TAGS)[i % len(TAGS)]; i += 1
        if buckets[t]:
            out.append(buckets[t].pop(0))
    rows = []
    for q, row, tags in out:
        obs, cause, fix = TAGS[tags[0]]
        rows.append({"question_id": q.qid, "question": q.text, "recall@10": row["recall10"],
                     "tags": ",".join(tags), "observed_problem": obs,
                     "likely_cause": cause, "proposed_fix": fix})
    return pd.DataFrame(rows)


def show_case(exp, name, qid, n_show=3, width=350):
    q = exp.qmap[qid]
    paper = exp.papers[q.paper_id]
    print(f"Q: {q.text}\nTarget: {paper.title}")
    for pid in q.gold:
        print(f"\n[GOLD {pid}] {paper.par[pid].text[:width]}...")
    r = next(x for x in exp.records[name] if x["question_id"] == qid)
    for c in r["retrieved"][:n_show]:
        ok = "TARGET" if c["paper_id"] == q.paper_id else "OTHER "
        print(f"\n#{c['rank']} [{ok}] {c['chunk_id']}: {c['text'][:width]}...")
