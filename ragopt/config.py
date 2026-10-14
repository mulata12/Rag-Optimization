"""All knobs live here so every experiment is reproducible from one place."""
from dataclasses import dataclass
from pathlib import Path

SEED = 42
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE retrieval models expect this instruction on the QUERY side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GEN_MODEL = "gemini-3.5-flash-lite"
PARQUET_URL = ("hf://datasets/allenai/qasper@0fbdb8edab3c45c9df13ed1e1cbb4d64e96dbd46/"
               "qasper/validation-00000-of-00001.parquet")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

TOP_K = 10
MAX_TOTAL_CHARS = 12_000          # task limit for the ten chunks together


@dataclass(frozen=True)
class Settings:
    # --- baseline fixed-length chunker ---
    fixed_size: int = 800
    fixed_overlap: int = 160
    # --- optimisation A: structure-aware chunker ---
    struct_max: int = 900
    struct_min: int = 250
    # --- optimisation B: BM25 + RRF ---
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    pool: int = 100                 # candidates taken from each ranker
    rrf_k: int = 60
    # --- optimisation C: paper-aware MMR re-ranking ---
    rerank_pool: int = 50
    paper_top_m: int = 3            # chunks per paper that form its "prior"
    paper_lambda: float = 0.30      # weight of the paper prior
    mmr_lambda: float = 0.75        # 1.0 = pure relevance, lower = more diversity
    dup_overlap: float = 0.50       # drop a chunk if >50% of its chars already shown


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    chunker: str = "fixed"          # "fixed" | "structure"           (A switches this)
    hybrid: bool = False            # dense + BM25 fusion             (B)
    rerank: bool = False            # paper prior + MMR + dedup       (C)


CONFIGS = {
    "baseline": PipelineConfig("baseline"),
    "A": PipelineConfig("A", chunker="structure"),
    "B": PipelineConfig("B", hybrid=True),
    "C": PipelineConfig("C", rerank=True),
    # Final = combination chosen from DEV results (edit here if dev says otherwise).
    "final": PipelineConfig("final", chunker="structure", hybrid=True, rerank=True),
}
