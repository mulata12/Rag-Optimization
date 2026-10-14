"""Optimisation C: paper-aware, diversity-aware selection of the final top-k."""
from collections import defaultdict

import numpy as np


def span_overlap(chunk, others):
    """Fraction of `chunk`'s characters already shown by `others` (same paragraphs)."""
    seen = defaultdict(list)
    for o in others:
        for sp in o.spans:
            seen[sp["paragraph_id"]].append((sp["start"], sp["end"]))
    total = covered = 0
    for sp in chunk.spans:
        total += sp["end"] - sp["start"]
        ivs = sorted(seen.get(sp["paragraph_id"], []))
        cur = sp["start"]
        for a, b in ivs:
            a, b = max(a, cur), min(b, sp["end"])
            if a < b:
                covered += b - a; cur = b
    return covered / total if total else 0.0


def paper_aware_mmr(cand_idx, cand_scores, chunks, E, st, k):
    """1) min-max normalise candidate scores;
       2) add a PAPER PRIOR: papers whose several chunks score high are more likely the
          paper the question is about (QASPER evidence is clustered in one paper);
       3) greedy MMR:  argmax  lam*boosted - (1-lam)*max cosine to already chosen;
       4) skip chunks whose text is mostly already shown (overlapping windows)."""
    s = np.asarray(cand_scores, float)
    rng = s.max() - s.min()
    s = (s - s.min()) / rng if rng > 0 else np.ones_like(s)
    per_paper = defaultdict(list)
    for j, i in enumerate(cand_idx):
        per_paper[chunks[i].paper_id].append(s[j])
    prior = {p: float(np.mean(sorted(v, reverse=True)[:st.paper_top_m])) for p, v in per_paper.items()}
    boosted = np.array([s[j] + st.paper_lambda * prior[chunks[i].paper_id] for j, i in enumerate(cand_idx)])

    selected, skipped, remaining = [], [], list(range(len(cand_idx)))
    while remaining and len(selected) < k:
        best, best_val = None, -1e9
        for j in remaining:
            div = max((float(E[cand_idx[j]] @ E[cand_idx[t]]) for t in selected), default=0.0)
            val = st.mmr_lambda * boosted[j] - (1 - st.mmr_lambda) * div
            if val > best_val:
                best, best_val = j, val
        remaining.remove(best)
        if selected and span_overlap(chunks[cand_idx[best]],
                                     [chunks[cand_idx[t]] for t in selected]) > st.dup_overlap:
            skipped.append(best); continue
        selected.append(best)
    selected += skipped[: k - len(selected)]           # back-fill so we always return k
    return [(cand_idx[j], float(boosted[j])) for j in selected]
