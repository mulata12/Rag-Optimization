import json
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# RAG Optimization on QASPER — retrieval-first lab
**Pipeline:** load data → naive baseline → diagnose 10 failures → 3 independent optimizations → dev ablation → held-out comparison → validate submission files.

| Task requirement | Where it is handled |
|---|---|
| 80/50 split + 20 answer questions (seed 42) | §1 `ragopt/data.py` (organiser snippet copied verbatim) |
| Baseline: overlapping char chunks, dense, top-10, cached embeddings | §2 `chunking.fixed_chunks`, `embed.Embedder` |
| Search ALL papers, score only target-paper evidence | `retrieval.Retriever.search` / `metrics.gold_coverage` |
| 10-case failure analysis | §3 `diagnosis.py` |
| ≥3 optimizations over ≥2 stages, tested independently | §4 (A=Ingestion, B=Retrieval, C=Selection) |
| R@5 / R@10 / MRR@10, latency, LLM calls | `experiment.summarize` |
| 20 answers only on held-out, F1 + false-answer rate | §7 `generate.py`, `experiment.answer_metrics` |
| 7 JSONL files, strict format | `io_utils.validate_jsonl` (§8) |
""")

md("## 0. Setup")
code("""
import os, sys, random
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath("."))
%load_ext autoreload
%autoreload 2

from ragopt.config import CONFIGS, Settings, RESULTS_DIR, SEED
from ragopt.data import load_data
from ragopt.embed import Embedder
from ragopt.experiment import Experiment
from ragopt.io_utils import validate_jsonl
from ragopt import diagnosis

random.seed(SEED); np.random.seed(SEED)
SET = Settings()           # every hyper-parameter is in ragopt/config.py
pd.set_option("display.max_colwidth", 120, "display.width", 200)
""")

md("""## 1. Data
80 development papers (debug / pick optimizations) and 50 held-out papers (final comparison only).
Gold evidence = paragraph(s) named by annotators (union across annotators), mapped back to paragraph IDs like `s2:p4`.
Questions with no text evidence (unanswerable, or figure/table-only evidence) cannot be scored for retrieval; we report how many are skipped.
""")
code("""
data = load_data()
print("dev papers:", len(data.dev_papers), "| dev questions:", len(data.dev_questions),
      "| scored (have text gold):", sum(bool(q.gold) for q in data.dev_questions))
print("held papers:", len(data.held_papers), "| held questions:", len(data.held_questions),
      "| scored:", sum(bool(q.gold) for q in data.held_questions))
sel = [q for q in data.held_questions if q.qid in data.answer_qids]
print("selected answer questions:", len(sel), "| gold-unanswerable among them:",
      sum(q.gold_unanswerable for q in sel))
""")

md("""## 2. Baseline (naive RAG)
* **Chunking:** 800-char windows, 160 overlap, over the concatenated paper (blind to sentences/sections).
* **Embedding:** `BAAI/bge-small-en-v1.5`, L2-normalised, so inner product = cosine. Queries get BGE's retrieval instruction prefix.
* **Search:** brute-force inner product over *all* dev chunks (exact search; FAISS `IndexFlatIP` would return identical results), top-10.
* **Caching:** document matrices are saved as `.npy` keyed by a hash of (model + texts); each query's vector and its original encode time are cached, and the stored encode time is added back to the latency so numbers stay honest.
""")
code("""
embedder = Embedder()
dev = Experiment("dev", data.dev_papers, data.dev_questions, embedder, SET)
dev.run(CONFIGS["baseline"])
dev.table(["baseline"])
""")

md("""## 3. Diagnosis — inspect ≥10 poor development cases
`failure_table` picks poor cases (Recall@10 < 1) balanced across four *automatically detected* symptoms: **boundary** (gold paragraph partially covered), **wrong_paper** (top-1 from another paper), **repetitive** (overlapping chunks wasted top-10 slots), **miss**.
The pre-filled text is a hint only. Run `show_case` on every row, read it, and edit `observed_problem / likely_cause / proposed_fix` to what you actually see before submitting.
""")
code("""
fails = diagnosis.failure_table(dev, "baseline", n=10)
fails
""")
code("""
for qid in fails.question_id[:3]:        # read ALL 10 yourself; start with these
    print("="*110); diagnosis.show_case(dev, "baseline", qid)
""")
code("""
fails.to_csv(RESULTS_DIR / "failure_table.csv", index=False)   # edit/confirm the text, then re-save
fails.tags.value_counts()
""")

md("""## 4. Three independent optimizations (each applied on top of the baseline *alone*)

### A — Ingestion: sentence-snapped, section-aware chunks + contextual headings
*Theory.* Embedding a chunk compresses it into one vector; if a fixed window cuts an evidence passage in half, neither half is a good representation (lost semantics, and under the 80%-coverage rule neither half counts). Packing whole **sentences** inside a **section** keeps semantic units intact. Prefixing the *embedded* text with `Title | Section` ("contextual headings", the idea behind contextual retrieval) disambiguates pronoun-heavy sentences such as "The model achieves…". The returned `text`/`source_spans` still point to the untouched original paragraph. No overlap ⇒ no near-duplicates.

### B — Retrieval: dense + BM25 with Reciprocal Rank Fusion (own BM25)
*Theory.* Dense retrievers capture paraphrase but blur rare, exact tokens (acronyms, dataset/model names, numbers). BM25 scores `IDF(t)·tf(k1+1)/(tf+k1(1-b+b·|d|/avgdl))`: IDF up-weights rare terms, `k1` saturates repeated terms, `b` normalises length. The two rankers fail differently, so fusing them helps. **RRF** (`Σ 1/(60+rank)`) fuses *ranks*, which avoids calibrating cosine against BM25 magnitudes.

### C — Selection: paper-aware MMR + overlap de-duplication
*Theory.* QASPER evidence for a question lives in **one paper**, but we search a pool of papers. If several chunks of one paper score high, that paper is probably the target → add a **paper prior** (mean of its top-3 normalised chunk scores × λ). Then pick the final 10 with **MMR**: `argmax λ·rel − (1−λ)·max cos(d, chosen)`, and skip any chunk whose characters are >50% already shown. This removes the near-duplicates created by overlapping windows and spends slots on new evidence.
""")
code("""
for name in ["A", "B", "C"]:
    dev.run(CONFIGS[name])
ablation = dev.table(["baseline", "A", "B", "C"]); ablation
""")

md("""## 5. Choose the final combination — from DEV results only
Look at the table above. Default `final` = A + B + C. If an ingredient *hurts* on dev, switch it off in `ragopt/config.py` (`CONFIGS["final"]`) and note it as a failed trial in your reflection. **Never look at held-out scores to decide this.**
""")
code("""
dev.run(CONFIGS["final"])
dev_table = dev.table(["baseline", "A", "B", "C", "final"])
dev_table.to_csv(RESULTS_DIR / "dev_ablation.csv"); dev_table
""")
code("""
# Optional extra ablation (helps the reflection): pairwise / leave-one-out combinations
from ragopt.config import PipelineConfig
extra = {"A+B": PipelineConfig("A+B", "structure", True, False),
         "A+C": PipelineConfig("A+C", "structure", False, True),
         "B+C": PipelineConfig("B+C", "fixed", True, True)}
for c in extra.values(): dev.run(c, write=False)
dev.table(list(extra))
""")

md("""## 6. Failure re-check
Re-run the diagnosis on the final system to see which symptoms disappeared and which remain.
""")
code("""
final_fails = diagnosis.failure_table(dev, "final", n=10)
print("baseline tags:", dict(fails.tags.str.split(",").str[0].value_counts()))
print("final tags   :", dict(final_fails.tags.str.split(",").str[0].value_counts()))
""")

md("""## 7. Held-out evaluation (baseline vs final only) + answers for the 20 fixed questions
Set `GEMINI_API_KEY` first. Generation uses the **same prompt** for both systems, so only the retrieved context differs. Cost: 20 calls per system.
Metrics: normalised token **Answer F1** on gold-answerable questions (abstaining = 0), **false-answer rate** = share of gold-unanswerable questions that were answered (exploratory with n=20).
""")
code("""
from ragopt.generate import Generator
held = Experiment("heldout", data.held_papers, data.held_questions, embedder, SET)
for name in ["baseline", "final"]:
    held.run(CONFIGS[name], generator=Generator(f"heldout_{name}"), answer_qids=data.answer_qids)
held_table = held.table(["baseline", "final"]); held_table.to_csv(RESULTS_DIR / "heldout_retrieval.csv"); held_table
""")
code("""
ans = pd.DataFrame([held.answer_metrics(n) for n in ["baseline", "final"]]).set_index("config")
ans.to_csv(RESULTS_DIR / "heldout_answers.csv"); ans.T
""")

md("## 8. Validate every submission file against the rules")
code("""
files = {"dev_baseline": (data.dev_papers, data.dev_questions, None), "dev_A": (data.dev_papers, data.dev_questions, None),
         "dev_B": (data.dev_papers, data.dev_questions, None), "dev_C": (data.dev_papers, data.dev_questions, None),
         "dev_final": (data.dev_papers, data.dev_questions, None),
         "heldout_baseline": (data.held_papers, data.held_questions, data.answer_qids),
         "heldout_final": (data.held_papers, data.held_questions, data.answer_qids)}
for f, (p, q, a) in files.items():
    print(f"{f}.jsonl OK ->", validate_jsonl(RESULTS_DIR / f"{f}.jsonl", p, q, a), "records")
""")

md("""## 9. Reflection (fill in with *your* numbers)
**What helped / why** — link each gain to the failure tag it targeted (§3 ↔ §4).
**What failed** — e.g. a λ or chunk size you tried on dev and dropped; negative results are valued.
**Quality vs cost** — all three optimizations use **0 LLM calls**; extra cost is a one-off BM25 index build (seconds) and ≈ +10 ms/query for MMR. Generation cost is identical across systems (20 calls each).
**Caveats** — 20 answer questions ⇒ F1 differences are noisy; the false-answer rate is exploratory (report the count of gold-unanswerable cases). MRR here is *cumulative*: first rank at which the union of top-r chunks covers ≥80% of any gold paragraph.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("RAG_Optimization.ipynb", "w"), indent=1)
