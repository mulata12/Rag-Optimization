"""Offline sanity test: synthetic papers + hashing embedder (no downloads, no API).
Run:  python -m tests.smoke_test"""
import hashlib, random, sys, tempfile, pathlib
import numpy as np
import ragopt.config as C
from ragopt.config import CONFIGS, Settings
from ragopt.data import Paper, Paragraph, Question
from ragopt.experiment import Experiment
from ragopt.io_utils import validate_jsonl
from ragopt.bm25 import BM25

C.RESULTS_DIR = pathlib.Path(tempfile.mkdtemp())
import ragopt.experiment as E; E.RESULTS_DIR = C.RESULTS_DIR


class HashEmbedder:
    def _v(self, t):
        v = np.zeros(256, "float32")
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return v / (np.linalg.norm(v) + 1e-9)
    def embed_documents(self, texts): return np.stack([self._v(t) for t in texts])
    def embed_query(self, t): return self._v(t), 1.0


rng = random.Random(0)
vocab = [f"w{i}" for i in range(400)]
def sentence(topic): return " ".join(rng.choices(vocab, k=12) + [topic]) + "."
papers, qs = {}, []
for k in range(6):
    topic = f"topic{k}"
    pars = [Paragraph("abstract", " ".join(sentence(topic) for _ in range(4)), "Abstract")]
    for s in range(3):
        for p in range(4):
            pars.append(Paragraph(f"s{s}:p{p}", " ".join(sentence(topic) for _ in range(rng.randint(3, 12))), f"Sec{s}"))
    papers[f"P{k}"] = Paper(f"P{k}", f"Title {topic}", pars)
    gold = [pars[3].pid]
    qs.append(Question(f"q{k}", f"P{k}", f"{topic} " + pars[3].text[:60], gold, False, ["x"]))

exp = Experiment("dev", papers, qs, HashEmbedder(), Settings())
names = ["baseline", "A", "B", "C", "final"]
for n in names:
    exp.run(CONFIGS[n])
    assert validate_jsonl(C.RESULTS_DIR / f"dev_{n}.jsonl", papers, qs) == len(qs)
print(exp.table(names))

# BM25 sanity: query term must rank its own document first
b = BM25(["apple banana", "cherry grape grape", "apple apple pear"])
assert b.scores("grape").argmax() == 1 and b.scores("apple").argmax() == 2
print("smoke test passed")
