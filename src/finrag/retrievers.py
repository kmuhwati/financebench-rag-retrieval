"""Sparse, dense and hybrid retrieval over a table of text units.

Every retriever returns, for a query and an optional set of allowed documents,
a list of (unit_index, score) sorted best-first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BM25:
    name = "BM25"

    def __init__(self, units: pd.DataFrame):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.units = units
        tokens = bm25s.tokenize(units.text.tolist(), stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.index = bm25s.BM25(k1=1.2, b=0.75)
        self.index.index(tokens, show_progress=False)

    def scores(self, query: str) -> np.ndarray:
        q = self._bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, show_progress=False)
        ids = q.ids[0]
        vocab_inv = {v: k for k, v in q.vocab.items()}
        terms = [vocab_inv[i] for i in ids]
        return self.index.get_scores(terms)


class Dense:
    """WordLlama static embeddings (256-d): CPU-fast, no GPU and no API calls.

    Static embeddings are a deliberately cheap dense baseline; swapping in a
    transformer encoder only changes ``_encode``.
    """

    name = "Dense (WordLlama)"

    def __init__(self, units: pd.DataFrame, dim: int = 256):
        from wordllama import WordLlama

        self.model = WordLlama.load(dim=dim)
        self.units = units
        self.emb = self._encode(units.text.tolist())

    def _encode(self, texts):
        e = np.asarray(self.model.embed(texts, norm=True, batch_size=256), dtype=np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)

    def scores(self, query: str) -> np.ndarray:
        return self.emb @ self._encode([query])[0]


def top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> np.ndarray:
    s = scores if allowed is None else np.where(allowed, scores, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    if k == 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx], kind="stable")]


def rrf(rankings: list[np.ndarray], k: int = 60, depth: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal rank fusion (Cormack et al., 2009): robust to incomparable score scales."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking[:depth]):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (k + r + 1)
    order = sorted(fused, key=fused.get, reverse=True)
    return np.array(order, dtype=int), np.array([fused[i] for i in order])
