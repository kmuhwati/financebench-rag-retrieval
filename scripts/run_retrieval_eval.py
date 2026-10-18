"""Retrieval ablation on FinanceBench: unit x scope x retriever, then a learned reranker.

Run from the project root:  python scripts/run_retrieval_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from finrag import corpus, evaluate, plotstyle, rerank, router  # noqa: E402
from finrag.retrievers import BM25, Dense, rrf, top_k  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS, FIGURES, TABLES = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "reports" / "tables"
DEPTH, CAND = 200, 150
SCOPES = {"shared": "Shared store (all 84 filings)", "routed": "Routed (company filter)", "oracle": "Oracle document"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    plotstyle.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pages = corpus.extract_pages()
    questions = corpus.load_questions()
    info = corpus.load_doc_info()
    docs = info[info.doc_name.isin(pages.doc_name.unique())].reset_index(drop=True)
    companies = sorted(docs.company.unique())
    doc_meta = docs.set_index("doc_name")
    n_pages = pages.groupby("doc_name").n_pages.first()

    routes = {i: router.parse(q, companies) for i, q in questions.question.items()}
    route_scores = {i: router.doc_scores(r, docs) for i, r in routes.items()}
    out = {"n_questions": len(questions), "n_documents": len(docs), "n_pages": len(pages),
           "router_company_accuracy": float(np.mean([routes[i].company == c for i, c in questions.company.items()])),
           "router_top_doc_accuracy": float(np.mean([
               (s.dropna().idxmax() if s.notna().any() else None) == questions.doc_name[i] for i, s in route_scores.items()]))}
    log(f"router: {out['router_company_accuracy']:.3f} company, {out['router_top_doc_accuracy']:.3f} top doc")

    rows, rank_store, ltr_frames, latency = [], {}, [], {}
    for unit in ("page", "chunk"):
        units = corpus.as_units(pages, unit)
        log(f"{unit}: {len(units):,} units; indexing")
        t = time.time(); bm25 = BM25(units); latency[f"{unit}_bm25_index_s"] = time.time() - t
        t = time.time(); dense = Dense(units); latency[f"{unit}_dense_index_s"] = time.time() - t
        u_doc, u_page = units.doc_name.to_numpy(), units.page.to_numpy()
        t_query = []
        for i, q in questions.iterrows():
            gold = set(map(tuple, q.gold_pages))
            t0 = time.perf_counter()
            s_b, s_d = bm25.scores(q.question), dense.scores(q.question)
            t_query.append(time.perf_counter() - t0)
            rs = route_scores[i]
            allowed = {
                "shared": None,
                "routed": np.isin(u_doc, rs.dropna().index) if rs.notna().any() else None,
                "oracle": u_doc == q.doc_name,
            }
            for scope, mask in allowed.items():
                r_b, r_d = top_k(s_b, DEPTH, mask), top_k(s_d, DEPTH, mask)
                r_h, rrf_s = rrf([r_b, r_d], depth=DEPTH)
                for name, ranking in (("BM25", r_b), ("Dense (WordLlama)", r_d), ("Hybrid (RRF)", r_h)):
                    rank_store[(unit, scope, name, i)] = evaluate.first_hit_rank(ranking, u_doc, u_page, gold)
                if unit == "chunk":
                    cand_idx = np.unique(np.concatenate([r_b[:CAND], r_d[:CAND]]))
                    rrf_map = dict(zip(r_h, rrf_s))
                    rank_b = {v: k for k, v in enumerate(r_b)}
                    rank_d = {v: k for k, v in enumerate(r_d)}
                    cand = units.iloc[cand_idx][["doc_name", "page", "text"]].copy()
                    cand["unit_idx"] = cand_idx
                    cand["bm25"] = s_b[cand_idx]
                    cand["dense"] = s_d[cand_idx]
                    cand["bm25_rank"] = [rank_b.get(c, DEPTH) for c in cand_idx]
                    cand["dense_rank"] = [rank_d.get(c, DEPTH) for c in cand_idx]
                    cand["rrf"] = [rrf_map.get(c, 0.0) for c in cand_idx]
                    cand["route_doc_score"] = rs.reindex(cand.doc_name).fillna(-5.0).to_numpy()
                    dtype = doc_meta.loc[cand.doc_name, "doc_type"].str.lower().to_numpy()
                    cand["is_10k"] = (dtype == "10k").astype(float)
                    cand["is_quarterly"] = np.isin(dtype, ["10q", "earnings"]).astype(float)
                    cand["is_8k"] = (dtype == "8k").astype(float)
                    cand["page_pos"] = cand.page.to_numpy() / n_pages.loc[cand.doc_name].to_numpy()
                    cand = rerank.candidate_features(q.question, routes[i].years, cand)
                    cand["label"] = [int((d, int(p)) in gold) for d, p in zip(cand.doc_name, cand.page)]
                    cand["qid"], cand["company"], cand["scope"] = i, q.company, scope
                    ltr_frames.append(cand.drop(columns="text"))
        latency[f"{unit}_query_ms_mean"] = 1e3 * float(np.mean(t_query))

    # ------------------------------------------------------------------ learned reranker
    log("learning-to-rank (company-grouped 5-fold CV)")
    ltr = pd.concat(ltr_frames, ignore_index=True)
    out["ltr_candidates"] = int(len(ltr))
    for scope in ("shared", "routed", "oracle"):
        f = ltr[ltr.scope == scope].copy()
        f["ltr"] = rerank.grouped_cv_rerank(f)
        out[f"ltr_candidate_recall_{scope}"] = float(f.groupby("qid").label.max().mean())
        for i, g in f.groupby("qid"):
            g = g.sort_values("ltr", ascending=False)
            hits = np.flatnonzero(g.label.to_numpy())
            rank_store[("chunk", scope, "Hybrid + LTR rerank", i)] = hits[0] + 1 if len(hits) else np.inf
        for i in set(questions.index) - set(f.qid):
            rank_store[("chunk", scope, "Hybrid + LTR rerank", i)] = np.inf
    model = rerank.fit_full(ltr[ltr.scope == "routed"])
    imp = pd.Series(model.booster_.feature_importance("gain"), index=rerank.FEATURES).sort_values(ascending=False)
    imp = imp / imp.sum()
    imp.to_csv(TABLES / "ltr_feature_importance.csv")
    model.booster_.save_model(str(ROOT / "data" / "processed" / "ltr_model.txt"))

    # ------------------------------------------------------------------ results
    r = pd.Series(rank_store)
    r.index.names = ["unit", "scope", "retriever", "qid"]
    for (unit, scope, ret), ranks in r.groupby(level=[0, 1, 2]):
        rows.append({"unit": unit, "scope": scope, "retriever": ret, **evaluate.summarise(ranks)})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "retrieval_results.csv", index=False)

    qtype = questions.question_type
    br = []
    for key, label in ((("chunk", "shared", "Dense (WordLlama)"), "Naive: dense, shared store"),
                       (("chunk", "routed", "Hybrid + LTR rerank"), "Final: routed hybrid + LTR")):
        ranks = r.loc[key]
        for t in qtype.unique():
            br.append({"system": label, "question_type": t, "recall@5": float((ranks[qtype == t] <= 5).mean())})
    br = pd.DataFrame(br)
    br.to_csv(TABLES / "recall5_by_question_type.csv", index=False)

    naive = res.query("unit == 'chunk' and scope == 'shared' and retriever == 'Dense (WordLlama)'").iloc[0]
    final = res.query("unit == 'chunk' and scope == 'routed' and retriever == 'Hybrid + LTR rerank'").iloc[0]
    oracle = res.query("unit == 'chunk' and scope == 'oracle' and retriever == 'Hybrid (RRF)'").iloc[0]
    out.update({"latency": latency, "naive_recall@5": naive["recall@5"], "final_recall@5": final["recall@5"],
                "oracle_hybrid_recall@5": oracle["recall@5"], "results": res.round(4).to_dict(orient="records")})
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"naive {naive['recall@5']:.3f} -> final {final['recall@5']:.3f} (oracle hybrid {oracle['recall@5']:.3f})")

    # ------------------------------------------------------------------ figures
    chunk = res[res.unit == "chunk"]
    rets = ["BM25", "Dense (WordLlama)", "Hybrid (RRF)", "Hybrid + LTR rerank"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(SCOPES))
    for j, ret in enumerate(rets):
        vals = [chunk.query("scope == @s and retriever == @ret")["recall@5"].squeeze() if len(chunk.query("scope == @s and retriever == @ret")) else np.nan for s in SCOPES]
        bars = ax.bar(x + (j - 1.5) * 0.2, 100 * np.array(vals, dtype=float), width=0.18, color=[plotstyle.NEUTRAL, plotstyle.SERIES[1], plotstyle.SERIES[0], plotstyle.SERIES[2]][j], label=ret)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f"{100 * v:.0f}", (b.get_x() + b.get_width() / 2, 100 * v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, list(SCOPES.values()))
    ax.set_ylabel("Recall@5: gold evidence page in top-5 chunks (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Routing plus a learned reranker lift recall@5 from 7% to 58% over 84 filings")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    plotstyle.save(fig, FIGURES / "01_recall_by_scope.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    curves = [(("chunk", "shared", "Dense (WordLlama)"), "Dense, shared store", plotstyle.SERIES[1]),
              (("chunk", "shared", "BM25"), "BM25, shared store", plotstyle.NEUTRAL),
              (("chunk", "routed", "Hybrid (RRF)"), "Hybrid, routed", plotstyle.SERIES[0]),
              (("chunk", "routed", "Hybrid + LTR rerank"), "Hybrid + LTR, routed (final)", plotstyle.SERIES[2]),
              (("chunk", "oracle", "Hybrid (RRF)"), "Hybrid, oracle document", plotstyle.INK_2)]
    ks = np.arange(1, 21)
    for key, label, col in curves:
        ranks = r.loc[key]
        ax.plot(ks, [100 * (ranks <= k).mean() for k in ks], color=col, label=label, ls="--" if "oracle" in label else "-")
    ax.set_xlabel("k (chunks passed to the LLM)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylabel("Recall@k (%)")
    ax.set_title("Recall@k on 150 FinanceBench questions")
    ax.legend(fontsize=8)
    plotstyle.save(fig, FIGURES / "02_recall_at_k.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    piv = br.pivot(index="question_type", columns="system", values="recall@5")
    y = np.arange(len(piv))
    for j, col in enumerate(["Naive: dense, shared store", "Final: routed hybrid + LTR"]):
        ax.barh(y + (j - 0.5) * 0.36, 100 * piv[col], height=0.34, color=[plotstyle.SERIES[1], plotstyle.SERIES[2]][j], label=col)
    ax.set_yticks(y, piv.index)
    ax.set_xlabel("Recall@5 (%)")
    ax.set_title("Gains hold across all three question types")
    ax.legend(fontsize=8, loc="lower right")
    plotstyle.save(fig, FIGURES / "03_by_question_type.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = imp.head(12)[::-1]
    ax.barh(t.index, 100 * t.values, color=plotstyle.SERIES[2], height=0.6)
    ax.set_xlabel("Share of LambdaMART split gain (%)")
    ax.set_title("What the reranker learned to trust")
    plotstyle.save(fig, FIGURES / "04_ltr_importance.png")

    md = [
        "# Retrieval results (auto-generated by scripts/run_retrieval_eval.py)\n",
        f"{out['n_questions']} questions, {out['n_documents']} filings, {out['n_pages']:,} pages.  ",
        f"Router: company identified for {out['router_company_accuracy']:.1%} of questions; top-ranked filing is the gold filing for {out['router_top_doc_accuracy']:.1%}.\n",
        res.to_markdown(index=False, floatfmt=".3f"),
        "\n## Recall@5 by question type\n", piv.to_markdown(floatfmt=".3f"),
        "\n## Reranker feature importance\n", imp.to_frame("gain_share").to_markdown(floatfmt=".3f"),
        f"\n## Latency\n\n```json\n{json.dumps(latency, indent=2)}\n```\n",
    ]
    (REPORTS / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
