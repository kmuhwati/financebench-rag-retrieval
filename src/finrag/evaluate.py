"""Page-level retrieval metrics against FinanceBench's annotated evidence pages."""
from __future__ import annotations

import numpy as np
import pandas as pd

KS = (1, 3, 5, 10, 20)


def first_hit_rank(ranking: np.ndarray, unit_doc: np.ndarray, unit_page: np.ndarray, gold: set) -> float:
    """1-based rank of the first retrieved unit lying on a gold (doc, page); inf if none."""
    for r, idx in enumerate(ranking):
        if (unit_doc[idx], int(unit_page[idx])) in gold:
            return r + 1
    return np.inf


def summarise(ranks: pd.Series) -> dict:
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in KS}
    out["mrr@20"] = float(np.where(ranks <= 20, 1.0 / ranks, 0.0).mean())
    return out


def doc_hit(ranking: np.ndarray, unit_doc: np.ndarray, gold_doc: str, k: int = 5) -> bool:
    return gold_doc in set(unit_doc[ranking[:k]])
