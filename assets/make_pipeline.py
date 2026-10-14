"""Regenerates assets/pipeline.png  ->  python assets/make_pipeline.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(21, 7.4))
ax.set_xlim(0, 21); ax.set_ylim(0, 7.4); ax.axis("off")
C = {"base": ("#EEF1F5", "#6B7785"), "A": ("#DCEBFF", "#2F6FDE"), "B": ("#DDF4E4", "#2E9E5B"),
     "C": ("#FFE9D2", "#E07B1F"), "out": ("#EADFF8", "#7B4BC4")}

def box(x, y, w, h, title, lines, kind, tag=None, gap=0.45):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=2.2))
    ax.text(x + w/2, y + h - 0.38, title, ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1b1f24")
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 0.95 - i*gap, ln, ha="center", va="center", fontsize=10.8, color="#2b3138")
    if tag:
        ax.add_patch(FancyBboxPatch((x + 0.12, y + h - 0.02), 0.62, 0.42, boxstyle="round,pad=0.02,rounding_size=0.1", fc=ec, ec=ec))
        ax.text(x + 0.43, y + h + 0.19, tag, ha="center", va="center", fontsize=12, fontweight="bold", color="white")

def arrow(x1, y1, x2, y2, color="#4a5560"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=color))

ymid, H = 3.7, 3.2
y0 = ymid - H/2
ax.text(10.5, 7.05, "Retrieval-first RAG pipeline on QASPER  (search runs over ALL papers in the split)",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#1b1f24")
lx = 0.3
for k, lab in [("base", "Baseline"), ("A", "A  Ingestion"), ("B", "B  Retrieval"), ("C", "C  Selection"), ("out", "Output")]:
    fc, ec = C[k]
    ax.add_patch(FancyBboxPatch((lx, 6.15), 0.38, 0.38, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fc, ec=ec, lw=2))
    ax.text(lx + 0.52, 6.34, lab, va="center", fontsize=11, color="#2b3138"); lx += 1.9

box(0.2, y0, 2.2, H, "QASPER", ["papers", "(80 dev /", "50 held-out)", "+ questions"], "base")
box(3.0, y0 - 0.35, 3.9, H + 0.7, "Chunking", ["Baseline: fixed 800-char", "windows, 160 overlap", "", "A: sentence-snapped,", "section-aware chunks", "+ \"Title | Section\" heading"], "A", "A")
box(7.5, y0, 2.3, H, "Embedding", ["BAAI/bge-small", "-en-v1.5", "L2-normalised", "(cached)"], "base")
box(10.4, y0 + 1.95, 2.9, 1.35, "Dense top-100", ["cosine / inner product"], "base")
box(10.4, y0 - 0.15, 2.9, 1.4, "BM25 top-100", ["hand-written, raw question"], "B", "B")
box(13.9, y0 + 0.55, 2.3, 2.1, "RRF fusion", ["score = Σ 1/(60+rank)", "(baseline: dense only)"], "B")
box(16.8, y0 - 0.35, 2.55, H + 0.7, "Selection", ["paper prior", "+ MMR diversity", "+ overlap de-dup", "→ exactly 10 chunks"], "C", "C")

arrow(2.4, ymid, 3.0, ymid); arrow(6.9, ymid, 7.5, ymid)
arrow(9.8, ymid, 10.4, y0 + 2.6); arrow(9.8, ymid, 10.4, y0 + 0.55)
arrow(13.3, y0 + 2.6, 13.9, ymid + 0.4); arrow(13.3, y0 + 0.55, 13.9, ymid - 0.4)
arrow(16.2, ymid, 16.8, ymid)

ax.add_patch(FancyBboxPatch((10.9, 5.95), 1.9, 0.6, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec="#2F6FDE", lw=2))
ax.text(11.85, 6.25, "Question", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#2F6FDE")
arrow(11.85, 5.95, 11.85, y0 + 3.3 + 0.05, "#2F6FDE")
ax.text(12.0, 5.55, "query embedding", fontsize=10.5, color="#2F6FDE", va="center")

box(16.8, 0.1, 2.55, 1.25, "Top-10 chunks", ["ranked, ≤ 12,000 chars"], "out")
arrow(18.07, y0 - 0.35, 18.07, 1.35)
box(13.9, 0.1, 2.3, 1.25, "Gemini answer", ["20 fixed questions"], "out")
arrow(16.8, 0.72, 16.2, 0.72)
ax.text(15.05, 1.6, "temperature 0", ha="center", fontsize=10, style="italic", color="#555")
ax.text(0.3, 0.45, "0 LLM calls at retrieval time  ·  each stage is an independent switch (baseline = A, B, C off)",
        fontsize=11.5, color="#444", style="italic")
fig.savefig("assets/pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
