"""The production retrieval path: route -> hybrid candidates -> learned rerank.

Mirrors the evaluated "routed + LTR" configuration in scripts/run_retrieval_eval.py,
using the reranker trained there (data/processed/ltr_model.txt).
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import corpus, rerank, router
from .retrievers import BM25, Dense, rrf, top_k

ROOT = Path(__file__).resolve().parents[2]


class FinancialRetriever:
    def __init__(self, ltr_path: Path = ROOT / "data" / "processed" / "ltr_model.txt", depth: int = 200, cand: int = 150):
        pages = corpus.extract_pages()
        info = corpus.load_doc_info()
        self.docs = info[info.doc_name.isin(pages.doc_name.unique())].reset_index(drop=True)
        self.doc_meta = self.docs.set_index("doc_name")
        self.companies = sorted(self.docs.company.unique())
        self.n_pages = pages.groupby("doc_name").n_pages.first()
        self.units = corpus.as_units(pages, "chunk")
        self.bm25, self.dense = BM25(self.units), Dense(self.units)
        self.ltr = lgb.Booster(model_file=str(ltr_path))
        self.depth, self.cand = depth, cand

    def search(self, question: str, k: int = 5) -> list[dict]:
        route = router.parse(question, self.companies)
        rs = router.doc_scores(route, self.docs)
        u_doc = self.units.doc_name.to_numpy()
        mask = np.isin(u_doc, rs.dropna().index) if rs.notna().any() else None
        s_b, s_d = self.bm25.scores(question), self.dense.scores(question)
        r_b, r_d = top_k(s_b, self.depth, mask), top_k(s_d, self.depth, mask)
        r_h, rrf_s = rrf([r_b, r_d], depth=self.depth)
        idx = np.unique(np.concatenate([r_b[: self.cand], r_d[: self.cand]]))
        rrf_map, rank_b, rank_d = dict(zip(r_h, rrf_s)), {v: i for i, v in enumerate(r_b)}, {v: i for i, v in enumerate(r_d)}
        cand = self.units.iloc[idx][["doc_name", "page", "text"]].copy()
        cand["bm25"], cand["dense"] = s_b[idx], s_d[idx]
        cand["bm25_rank"] = [rank_b.get(i, self.depth) for i in idx]
        cand["dense_rank"] = [rank_d.get(i, self.depth) for i in idx]
        cand["rrf"] = [rrf_map.get(i, 0.0) for i in idx]
        cand["route_doc_score"] = rs.reindex(cand.doc_name).fillna(-5.0).to_numpy()
        dtype = self.doc_meta.loc[cand.doc_name, "doc_type"].str.lower().to_numpy()
        cand["is_10k"] = (dtype == "10k").astype(float)
        cand["is_quarterly"] = np.isin(dtype, ["10q", "earnings"]).astype(float)
        cand["is_8k"] = (dtype == "8k").astype(float)
        cand["page_pos"] = cand.page.to_numpy() / self.n_pages.loc[cand.doc_name].to_numpy()
        cand = rerank.candidate_features(question, route.years, cand)
        cand["score"] = self.ltr.predict(cand[rerank.FEATURES])
        top = cand.sort_values("score", ascending=False).head(k)
        return [{"doc_name": r.doc_name, "page": int(r.page), "score": float(r.score), "text": r.text} for r in top.itertuples()]
