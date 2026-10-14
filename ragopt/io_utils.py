"""JSONL writing + a strict validator mirroring every submission rule."""
import json
from pathlib import Path

from .config import MAX_TOTAL_CHARS, TOP_K


def write_jsonl(records, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_jsonl(path, papers, questions, answer_qids=None):
    """Raises AssertionError on the first violated rule; returns record count."""
    expected = {q.qid: q for q in questions}
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qid = r["question_id"]
            assert qid in expected and qid not in seen, f"bad/duplicate qid {qid}"
            seen.add(qid)
            assert set(r) >= {"question_id", "target_paper_id", "retrieved", "answer",
                              "abstained", "latency_ms", "llm_calls"}
            assert "gold" not in r
            ret = r["retrieved"]
            assert len(ret) == TOP_K and [c["rank"] for c in ret] == list(range(1, TOP_K + 1)), qid
            assert len({c["chunk_id"] for c in ret}) == TOP_K, f"{qid}: duplicate chunks"
            assert sum(len(c["text"]) for c in ret) <= MAX_TOTAL_CHARS, f"{qid}: >12000 chars"
            for c in ret:
                for sp in c["source_spans"]:
                    par = papers[c["paper_id"]].par[sp["paragraph_id"]]
                    assert 0 <= sp["start"] < sp["end"] <= len(par.text), f"{qid}: bad span"
            sel = answer_qids is not None and qid in answer_qids
            if sel:
                assert isinstance(r["answer"], str) and isinstance(r["abstained"], bool)
                assert (r["answer"] == "") == r["abstained"], f"{qid}: answer/abstained mismatch"
            else:
                assert r["answer"] is None and r["abstained"] is None, f"{qid}: must be null"
    assert seen == set(expected), "missing questions"
    return len(seen)
