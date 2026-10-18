"""Learning-to-rank reranker built from cheap, interpretable signals.

A cross-encoder would be the usual choice; this project runs CPU-only with no
model downloads, so the reranker learns how to weigh lexical and semantic
scores alongside finance-specific cues: does the page carry the statement the
question asks about, the fiscal year it names, and the numbers it needs?
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

STATEMENTS = {
    "cash_flow": (r"cash ?flow", r"statements? of cash flows?|cash flows? from operating"),
    "balance_sheet": (r"balance sheet|statement of financial position", r"balance sheets?|statements? of financial position|total liabilities and"),
    "income": (r"income statement|statement of (?:income|operations|earnings)|p&l",
               r"statements? of (?:income|operations|earnings)|income statements?"),
}
_NUM = re.compile(r"^\(?\$?-?\d[\d,]*\.?\d*\)?%?$")
_STOP = set("the a an of in for to and or is was what were did does how much by on as at be from with that this it its which are their".split())

FEATURES = [
    "bm25", "bm25_rank", "dense", "dense_rank", "rrf", "route_doc_score", "is_10k", "is_quarterly", "is_8k",
    "page_pos", "num_density", "statement_match", "statement_asked", "year_hits", "term_coverage", "length",
]


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in _STOP}


def candidate_features(question: str, years: list[int], cand: pd.DataFrame) -> pd.DataFrame:
    """``cand`` holds candidate units with retrieval scores; returns the feature frame."""
    q_lower = question.lower()
    q_terms = _terms(question)
    asked = {k for k, (q_pat, _) in STATEMENTS.items() if re.search(q_pat, q_lower)}
    feats = cand.copy()
    texts = cand.text.str.lower()
    feats["statement_asked"] = float(bool(asked))
    feats["statement_match"] = [
        float(any(re.search(STATEMENTS[k][1], t) for k in asked)) if asked else 0.0 for t in texts
    ]
    feats["year_hits"] = [sum(str(y) in t for y in years) for t in texts]
    feats["term_coverage"] = [len(q_terms & _terms(t)) / max(len(q_terms), 1) for t in texts]
    toks = cand.text.str.split()
    feats["num_density"] = [sum(bool(_NUM.match(w)) for w in ws) / max(len(ws), 1) for ws in toks]
    feats["length"] = toks.str.len()
    return feats


def grouped_cv_rerank(frame: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> pd.Series:
    """Out-of-fold LambdaMART scores; folds are grouped by *company* so the model
    never sees questions about the same filings it is evaluated on."""
    from lightgbm import LGBMRanker
    from sklearn.model_selection import GroupKFold

    frame = frame.sort_values(["qid", "rrf"], ascending=[True, False])
    out = pd.Series(np.nan, index=frame.index)
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, te_idx in gkf.split(frame, groups=frame["company"]):
        tr, te = frame.iloc[tr_idx], frame.iloc[te_idx]
        model = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1)
        model.fit(tr[FEATURES], tr["label"], group=tr.groupby("qid", sort=False).size().to_numpy())
        out.loc[te.index] = model.predict(te[FEATURES])
    return out


def fit_full(frame: pd.DataFrame, seed: int = 0):
    from lightgbm import LGBMRanker

    frame = frame.sort_values(["qid", "rrf"], ascending=[True, False])
    model = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15,
                       min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=seed, verbose=-1)
    model.fit(frame[FEATURES], frame["label"], group=frame.groupby("qid", sort=False).size().to_numpy())
    return model
