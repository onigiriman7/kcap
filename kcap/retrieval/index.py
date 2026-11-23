"""Hybrid BM25 + (optional) embedding index over (id, text) items."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")
_embedder_cache: dict = {}


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / max(1, len(sa | sb))


def _embedder(name: str):
    if name not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        _embedder_cache[name] = SentenceTransformer(name)
    return _embedder_cache[name]


class HybridIndex:
    def __init__(self, items: list[tuple[str, str]], embeddings: bool = False,
                 embedding_model: str = "all-MiniLM-L6-v2", bm25_weight: float = 0.5):
        self.ids = [i for i, _ in items]
        self.texts = [t for _, t in items]
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in self.texts]) if items else None
        self.w = bm25_weight
        self.emb_model = embedding_model if embeddings else None
        self.emb = None
        if self.emb_model and items:
            self.emb = _embedder(self.emb_model).encode(self.texts, normalize_embeddings=True)

    @classmethod
    def from_cfg(cls, items, cfg: dict):
        r = cfg.get("retrieval", {})
        return cls(items, r.get("embeddings", False), r.get("embedding_model", "all-MiniLM-L6-v2"),
                   r.get("bm25_weight", 0.5))

    def search(self, query: str, k: int = 5, exclude: set[str] | None = None) -> list[tuple[str, str, float]]:
        if not self.ids:
            return []
        s = np.asarray(self.bm25.get_scores(tokenize(query) or ["_"]), dtype=float)
        if s.max() > 0:
            s = s / s.max()
        if self.emb is not None:
            q = _embedder(self.emb_model).encode([query], normalize_embeddings=True)[0]
            s = self.w * s + (1 - self.w) * (self.emb @ q)
        order = np.argsort(-s)
        out = []
        for i in order:
            if exclude and self.ids[i] in exclude:
                continue
            out.append((self.ids[i], self.texts[i], float(s[i])))
            if len(out) >= k:
                break
        return out

    def similarity(self, a: str, b: str) -> float:
        if self.emb_model:
            m = _embedder(self.emb_model)
            ea, eb = m.encode([a, b], normalize_embeddings=True)
            return float(ea @ eb)
        return jaccard(a, b)
