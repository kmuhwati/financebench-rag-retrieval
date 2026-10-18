"""Unit tests for routing, fusion, chunking, features and metrics (no PDFs needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag import corpus, evaluate, rerank, router  # noqa: E402
from finrag.retrievers import rrf, top_k  # noqa: E402

COMPANIES = ["3M", "American Express", "Johnson & Johnson", "Pfizer", "PepsiCo", "Best Buy"]
DOCS = pd.DataFrame({
    "doc_name": ["PFIZER_2021_10K", "Pfizer_2023Q2_10Q", "3M_2018_10K", "3M_2022_10K"],
    "company": ["Pfizer", "Pfizer", "3M", "3M"],
    "doc_type": ["10k", "10q", "10k", "10k"],
    "doc_period": [2021, 2023, 2018, 2022],
})


def test_router_handles_aliases_and_periods():
    r = router.parse("Does AMEX have an improving operating margin profile as of 2022?", COMPANIES)
    assert r.company == "American Express" and r.years == [2022]
    r = router.parse("How did JnJ's US sales growth compare in FY2022?", COMPANIES)
    assert r.company == "Johnson & Johnson"
    r = router.parse("For Pfizer, which region had the biggest drop in Q22023 revenues?", COMPANIES)
    assert r.company == "Pfizer" and r.quarter == 2 and 2023 in r.years
    r = router.parse("Did Pfizer grow its PPNE between FY20 and FY21?", COMPANIES)
    assert r.years == [2020, 2021]


def test_doc_scores_prefer_matching_period_and_type():
    s = router.doc_scores(router.parse("What was 3M's FY2018 capital expenditure?", COMPANIES), DOCS)
    assert s.dropna().idxmax() == "3M_2018_10K"
    assert s[["PFIZER_2021_10K", "Pfizer_2023Q2_10Q"]].isna().all()
    s = router.doc_scores(router.parse("Pfizer revenue in Q2 of FY2023?", COMPANIES), DOCS)
    assert s.dropna().idxmax() == "Pfizer_2023Q2_10Q"


def test_no_company_means_no_filter():
    s = router.doc_scores(router.parse("What drove SG&A in FY2023?", COMPANIES), DOCS)
    assert (s == 0).all()


def test_rrf_rewards_agreement():
    order, _ = rrf([np.array([1, 2, 3]), np.array([2, 1, 4])])
    assert set(order[:2]) == {1, 2} and order[-1] in (3, 4)


def test_top_k_respects_mask():
    scores = np.array([5.0, 4.0, 3.0, 2.0])
    assert list(top_k(scores, 2, np.array([False, True, True, False]))) == [1, 2]


def test_chunks_never_span_pages_and_cover_the_tail():
    pages = pd.DataFrame({"doc_name": ["D", "D"], "page": [0, 1], "text": [" ".join(f"w{i}" for i in range(450)), "short page text here ok"]})
    ch = corpus.chunk_pages(pages, words=300, stride=200)
    assert set(ch.page) == {0, 1}
    assert ch[ch.page == 0].text.str.contains("w449").any()


def test_first_hit_rank_and_summary():
    docs, pgs = np.array(["A", "A", "B"]), np.array([1, 2, 3])
    assert evaluate.first_hit_rank(np.array([0, 2, 1]), docs, pgs, {("B", 3)}) == 2
    assert evaluate.first_hit_rank(np.array([0]), docs, pgs, {("B", 3)}) == np.inf
    s = evaluate.summarise(pd.Series([1, 2, np.inf, 6]))
    assert s["recall@1"] == 0.25 and s["recall@5"] == 0.5


def test_statement_feature_fires_on_matching_page():
    cand = pd.DataFrame({"text": ["Consolidated Statement of Cash Flows Years ended 2018 1,577 (1,373)", "Risk factors narrative text"]})
    f = rerank.candidate_features("What is FY2018 capex? Use the cash flow statement.", [2018], cand)
    assert f.statement_match.tolist() == [1.0, 0.0]
    assert f.year_hits.tolist() == [1, 0]
    assert f.num_density.iloc[0] > f.num_density.iloc[1]
