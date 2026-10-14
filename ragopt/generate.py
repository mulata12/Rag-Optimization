"""Answer generation (Gemini, temperature 0). Same prompt for baseline and final so that
only the *retrieved context* differs between systems (controlled comparison)."""
import json
import os
import time

from .config import CACHE_DIR, GEN_MODEL

PROMPT = """You answer questions about a scientific paper using ONLY the numbered excerpts below.
The excerpts are ranked by relevance and may include text from other papers.
Rules:
- Answer in one short sentence or phrase. For yes/no questions answer "Yes" or "No".
- If the excerpts do not contain enough information, reply exactly: UNANSWERABLE

Excerpts:
{context}

Question: {question}
Answer:"""


class Generator:
    def __init__(self, tag):
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.path = CACHE_DIR / f"gen_{tag}.json"
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}

    def __call__(self, qid, question, retrieved):
        """-> (answer, abstained, generation_ms). Cached calls re-report original latency."""
        if qid not in self.cache:
            ctx = "\n\n".join(f"[{c['rank']}] {c['text']}" for c in retrieved)
            t = time.perf_counter()
            for attempt in range(4):
                try:
                    r = self.client.models.generate_content(
                        model=GEN_MODEL, contents=PROMPT.format(context=ctx, question=question),
                        config=self._types.GenerateContentConfig(temperature=0, max_output_tokens=200))
                    text = (r.text or "").strip(); break
                except Exception:
                    time.sleep(2 ** attempt); text = ""
            self.cache[qid] = {"text": text, "ms": (time.perf_counter() - t) * 1000}
            self.path.write_text(json.dumps(self.cache))
        c = self.cache[qid]
        abstained = c["text"].upper().startswith("UNANSWERABLE") or not c["text"]
        return ("" if abstained else c["text"]), abstained, c["ms"]
