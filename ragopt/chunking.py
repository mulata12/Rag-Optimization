"""Chunkers. Every chunk carries `spans` pointing back into ORIGINAL paragraphs,
which is what the grader uses to decide if gold evidence was covered."""
import re
from dataclasses import dataclass

from .data import Paper


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    text: str            # what is returned / shown to the LLM
    spans: list          # [{"paragraph_id", "start", "end"}]
    embed_text: str      # what is embedded/indexed (may add headings)
    section: str = ""


# ---------------------------------------------------------------- baseline
def fixed_chunks(paper: Paper, size: int, overlap: int):
    """Overlapping fixed-length character windows over the whole paper (blind to structure)."""
    pieces, buf, pos = [], [], 0              # buf: (pid, global_start, global_end)
    for p in paper.paragraphs:
        buf.append((p.pid, pos, pos + len(p.text)))
        pieces.append(p.text)
        pos += len(p.text) + 1                # +1 for the "\n" separator
    full = "\n".join(pieces)
    n, step, chunks, start, idx = len(full), size - overlap, [], 0, 0
    while start < n:
        end = min(start + size, n)
        if start > 0 and end == n and n - start <= overlap:
            break                              # tail already fully covered by previous window
        spans = []
        for pid, gs, ge in buf:
            lo, hi = max(gs, start), min(ge, end)
            if lo < hi:
                spans.append({"paragraph_id": pid, "start": lo - gs, "end": hi - gs})
        text = full[start:end]
        chunks.append(Chunk(f"{paper.id}:fc{idx}", paper.id, text, spans, text))
        idx += 1
        if end == n:
            break
        start += step
    return chunks


# ---------------------------------------------------- optimisation A (ingestion)
_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9(\["“])')


def _sentence_spans(text):
    out, start = [], 0
    for m in _SENT.finditer(text):
        out.append((start, m.start())); start = m.end()
    out.append((start, len(text)))
    return [(s, e) for s, e in out if e > s]


def _split_long(text, s, e, max_chars):
    """Hard-split an over-long sentence at whitespace (rare: tables, long enumerations)."""
    parts = []
    while e - s > max_chars:
        cut = text.rfind(" ", s + max_chars // 2, s + max_chars)
        cut = cut if cut != -1 else s + max_chars
        parts.append((s, cut)); s = cut + 1 if text[cut:cut + 1] == " " else cut
    parts.append((s, e))
    return parts


def structure_chunks(paper: Paper, max_chars: int, min_chars: int):
    """Sentence-snapped, section-aware packing + contextual heading for the embedder.

    * boundaries only fall between sentences (never mid-sentence);
    * chunks never cross a section boundary;
    * a paragraph break is used as a split point once the chunk is 'big enough';
    * the *embedded* text is prefixed with 'Title | Section' (contextual headings),
      while `text`/`spans` stay faithful to the original paper.
    """
    chunks, idx = [], 0

    def flush(cur, section):
        nonlocal idx
        if not cur:
            return
        merged = []                            # merge consecutive sentences of same paragraph
        for pid, s, e in cur:
            if merged and merged[-1][0] == pid and s - merged[-1][2] <= 1:
                merged[-1][2] = e
            else:
                merged.append([pid, s, e])
        text = "\n".join(paper.par[pid].text[s:e] for pid, s, e in merged)
        spans = [{"paragraph_id": pid, "start": s, "end": e} for pid, s, e in merged]
        embed = f"{paper.title} | {section}\n{text}"
        chunks.append(Chunk(f"{paper.id}:sc{idx}", paper.id, text, spans, embed, section))
        idx += 1

    cur, cur_len, cur_sec = [], 0, None
    for p in paper.paragraphs:
        if cur_sec is not None and p.section != cur_sec:
            flush(cur, cur_sec); cur, cur_len = [], 0
        cur_sec = p.section
        for s, e in _sentence_spans(p.text):
            for ss, ee in _split_long(p.text, s, e, max_chars):
                L = ee - ss
                new_par = cur and cur[-1][0] != p.pid
                if cur and (cur_len + L + 1 > max_chars or (new_par and cur_len >= min_chars)):
                    flush(cur, cur_sec); cur, cur_len = [], 0
                cur.append((p.pid, ss, ee)); cur_len += L + 1
    flush(cur, cur_sec)
    return chunks


def build_chunks(paper, chunker, st):
    if chunker == "fixed":
        return fixed_chunks(paper, st.fixed_size, st.fixed_overlap)
    if chunker == "structure":
        return structure_chunks(paper, st.struct_max, st.struct_min)
    raise ValueError(chunker)
