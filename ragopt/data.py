"""Dataset loading: papers -> addressable paragraphs, questions -> gold paragraph IDs."""
import random
import re
from dataclasses import dataclass, field
from functools import cached_property

from .config import PARQUET_URL, SEED

_TOK = re.compile(r"\w+")


@dataclass
class Paragraph:
    pid: str            # "abstract" or "s{section}:p{paragraph}"
    text: str
    section: str

    @cached_property
    def tokens(self):
        return [(m.start(), m.end()) for m in _TOK.finditer(self.text)]


@dataclass
class Paper:
    id: str
    title: str
    paragraphs: list
    par: dict = field(default_factory=dict)      # pid -> Paragraph

    def __post_init__(self):
        self.par = {p.pid: p for p in self.paragraphs}

    def match_evidence(self, ev: str):
        """Map a QASPER evidence string back to a paragraph id (exact, then containment)."""
        norm = lambda s: " ".join(s.split())
        e = norm(ev)
        if not e or e.startswith("FLOAT SELECTED"):
            return None                         # figure/table evidence: not text-retrievable
        for p in self.paragraphs:
            if norm(p.text) == e:
                return p.pid
        for p in self.paragraphs:
            n = norm(p.text)
            if len(e) >= 30 and (e in n or (len(n) >= 30 and n in e)):
                return p.pid
        return None


@dataclass
class Question:
    qid: str
    paper_id: str
    text: str
    gold: list                  # gold paragraph ids (union over annotators)
    gold_unanswerable: bool     # majority of annotators say unanswerable
    gold_answers: list          # reference answer strings (answerable annotations)


@dataclass
class Data:
    dev_papers: dict
    held_papers: dict
    dev_questions: list
    held_questions: list
    answer_qids: set


# ---------- schema normalisation (HF may return list-of-dicts OR dict-of-lists) ----------
def _rows(x):
    if x is None:
        return []
    if isinstance(x, dict):
        keys = list(x)
        n = len(x[keys[0]]) if keys else 0
        return [{k: x[k][i] for k in keys} for i in range(n)]
    return list(x)


def _annotations(ans_item):
    out = []
    if isinstance(ans_item, dict) and "answer" in ans_item:
        out = _rows(ans_item["answer"])
    elif isinstance(ans_item, (list, tuple)):
        for it in ans_item:
            out += _annotations(it)
    return out


def _build_paper(rec) -> Paper:
    pars = [Paragraph("abstract", rec["abstract"] or "", "Abstract")]
    ft = rec["full_text"]
    for si, (sec, plist) in enumerate(zip(ft["section_name"], ft["paragraphs"])):
        for pi, txt in enumerate(plist):
            if txt and txt.strip():
                pars.append(Paragraph(f"s{si}:p{pi}", txt, sec or ""))
    return Paper(rec["id"], rec["title"], [p for p in pars if p.text.strip()])


def _build_questions(rec, paper: Paper):
    out = []
    for q in _rows(rec["qas"]):
        anns = _annotations(q.get("answers"))
        gold, refs, unans = [], [], 0
        for a in anns:
            if a.get("unanswerable"):
                unans += 1
                continue
            for ev in (a.get("evidence") or []):
                pid = paper.match_evidence(ev)
                if pid and pid not in gold:
                    gold.append(pid)
            spans = a.get("extractive_spans") or []
            if spans:
                refs.append(", ".join(spans))
            elif (a.get("free_form_answer") or "").strip():
                refs.append(a["free_form_answer"].strip())
            elif a.get("yes_no") is not None:
                refs.append("yes" if a["yes_no"] else "no")
        out.append(Question(str(q["question_id"]), paper.id, q["question"], gold,
                            bool(anns) and unans * 2 > len(anns), refs))
    return out


def load_data() -> Data:
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"validation": PARQUET_URL}, split="validation")
    paper_ids = sorted(set(ds["id"]))
    random.Random(SEED).shuffle(paper_ids)
    build_ids, held_ids = paper_ids[:80], paper_ids[80:130]
    held = set(held_ids)

    # --- the exact organiser snippet for the 20 answer questions ---
    qids = sorted({str(qid) for paper in ds if paper["id"] in held
                   for qid in (paper["qas"]["question_id"] if isinstance(paper["qas"], dict)
                               else [q["question_id"] for q in paper["qas"]])})
    random.Random(SEED).shuffle(qids)
    answer_qids = set(qids[:20])

    dev_p, held_p, dev_q, held_q = {}, {}, [], []
    wanted = set(build_ids) | held
    for rec in ds:
        if rec["id"] not in wanted:
            continue
        paper = _build_paper(rec)
        qs = _build_questions(rec, paper)
        if rec["id"] in held:
            held_p[paper.id] = paper; held_q += qs
        else:
            dev_p[paper.id] = paper; dev_q += qs
    return Data(dev_p, held_p, dev_q, held_q, answer_qids)
