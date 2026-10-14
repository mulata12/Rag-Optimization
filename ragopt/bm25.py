"""Hand-written Okapi BM25 (tokenisation, document frequencies, IDF, scoring)."""
import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOP = set("""a an the of in on at to for from by with and or but not is are was were be been being
this that these those it its as we our they their he she his her you your i do does did have has had
can could may might will would should shall which who whom what when where why how than then so such
also into over under between about through during using used use based""".split())


def tokenize(text: str):
    toks = []
    for t in _TOKEN.findall(text.lower()):
        if t in STOP:
            continue
        if len(t) > 4 and t.endswith("ies"):  t = t[:-3] + "y"      # light plural stemming
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"): t = t[:-1]
        toks.append(t)
    return toks


class BM25:
    """score(q,d) = sum_t IDF(t) * tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl))
       IDF(t)     = ln(1 + (N-df+0.5)/(df+0.5))          (always positive)"""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_len = np.zeros(self.N)
        post = defaultdict(list)                       # term -> [(doc, tf)]
        for i, d in enumerate(docs):
            tf = Counter(tokenize(d))
            self.doc_len[i] = sum(tf.values())
            for t, c in tf.items():
                post[t].append((i, c))
        self.avgdl = self.doc_len.mean() if self.N else 0.0
        self.idf = {t: math.log(1 + (self.N - len(v) + 0.5) / (len(v) + 0.5)) for t, v in post.items()}
        self.docs_idx = {t: np.array([i for i, _ in v]) for t, v in post.items()}
        self.docs_tf = {t: np.array([c for _, c in v], float) for t, v in post.items()}

    def scores(self, query: str):
        s = np.zeros(self.N)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            idx, tf = self.docs_idx[t], self.docs_tf[t]
            norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
            s[idx] += self.idf[t] * tf * (self.k1 + 1) / norm
        return s
