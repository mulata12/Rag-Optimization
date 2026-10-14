"""Runs one pipeline config over one split -> JSONL-ready records + summary tables."""
import statistics as stats
import time

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, TOP_K
from .io_utils import write_jsonl
from .metrics import best_f1, score_question
from .retrieval import Retriever


class Experiment:
    def __init__(self, split, papers, questions, embedder, st):
        self.split, self.papers, self.questions = split, papers, questions
        self.qmap = {q.qid: q for q in questions}
        self.embedder, self.st = embedder, st
        self._retrievers = {}              # one index per (chunker, hybrid)
        self.timing = {}                   # cfg name -> qid -> {"retrieval_ms", "gen_ms"}
        self.records = {}                  # cfg name -> list of records

    def retriever(self, cfg):
        key = (cfg.chunker, cfg.hybrid, cfg.rerank)
        if key not in self._retrievers:
            self._retrievers[key] = Retriever(self.papers, cfg, self.embedder, self.st)
        return self._retrievers[key]

    def run(self, cfg, generator=None, answer_qids=None, write=True):
        R = self.retriever(cfg)
        recs, tim = [], {}
        for q in self.questions:
            qvec, enc_ms = self.embedder.embed_query(q.text)
            t = time.perf_counter()
            hits = R.search(q.text, qvec, TOP_K)
            retr_ms = (time.perf_counter() - t) * 1000 + enc_ms
            retrieved = []
            for rank, (i, sc) in enumerate(hits, 1):
                c = R.chunks[i]
                retrieved.append({"rank": rank, "chunk_id": c.chunk_id, "paper_id": c.paper_id,
                                  "score": round(float(sc), 6), "text": c.text,
                                  "source_spans": c.spans})
            ans = abst = None; calls = 0; gen_ms = 0.0
            if generator is not None and answer_qids and q.qid in answer_qids:
                ans, abst, gen_ms = generator(q.qid, q.text, retrieved); calls = 1
            tim[q.qid] = {"retrieval_ms": retr_ms, "gen_ms": gen_ms}
            recs.append({"question_id": q.qid, "target_paper_id": q.paper_id,
                         "retrieved": retrieved, "answer": ans, "abstained": abst,
                         "latency_ms": round(retr_ms + gen_ms, 2), "llm_calls": calls})
        self.timing[cfg.name], self.records[cfg.name] = tim, recs
        if write:
            write_jsonl(recs, RESULTS_DIR / f"{self.split}_{cfg.name}.jsonl")
        return recs

    # ------------------------------------------------------------ evaluation
    def per_question(self, name):
        rows = []
        for r in self.records[name]:
            q = self.qmap[r["question_id"]]
            if not q.gold:                                   # unanswerable / figure-only evidence
                continue
            s = score_question(q, self.papers[q.paper_id], r["retrieved"])
            rows.append({"qid": q.qid, **s})
        return rows

    def summarize(self, name):
        rows = self.per_question(name)
        tim, recs = self.timing[name], self.records[name]
        no_gen = [tim[r["question_id"]]["retrieval_ms"] for r in recs if r["llm_calls"] == 0]
        return {"config": name, "n_scored": len(rows), "n_questions": len(recs),
                "R@5": np.mean([r["recall5"] for r in rows]),
                "R@10": np.mean([r["recall10"] for r in rows]),
                "MRR@10": np.mean([r["rr"] for r in rows]),
                "retrieval_ms (median)": stats.median(tim[r["question_id"]]["retrieval_ms"] for r in recs),
                "LLM calls/q": np.mean([r["llm_calls"] for r in recs])}

    def table(self, names):
        return pd.DataFrame([self.summarize(n) for n in names]).set_index("config").round(4)

    def answer_metrics(self, name):
        """Answer F1 (answerable only), false-answer rate (gold-unanswerable only),
        and separate median latencies for the two scopes."""
        recs, tim = self.records[name], self.timing[name]
        sel = [r for r in recs if r["llm_calls"] == 1]
        f1s, false_ans, n_unans = [], 0, 0
        for r in sel:
            q = self.qmap[r["question_id"]]
            if q.gold_unanswerable:
                n_unans += 1; false_ans += int(not r["abstained"])
            else:
                f1s.append(0.0 if r["abstained"] else best_f1(r["answer"], q.gold_answers))
        rest = [tim[r["question_id"]]["retrieval_ms"] for r in recs if r["llm_calls"] == 0]
        return {"config": name, "n_selected": len(sel), "answerable": len(f1s),
                "gold_unanswerable": n_unans,
                "Answer F1": float(np.mean(f1s)) if f1s else None,
                "false_answer_rate": (false_ans / n_unans) if n_unans else "undefined (0 unanswerable)",
                "abstain_rate": float(np.mean([r["abstained"] for r in sel])) if sel else None,
                "median_ms retrieval-only (rest)": stats.median(rest) if rest else None,
                "median_ms retrieval+gen (20)": stats.median(r["latency_ms"] for r in sel) if sel else None,
                "LLM calls/q (20)": 1.0}
