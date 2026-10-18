"""PDF -> pages -> chunks, with provenance (document, zero-indexed page) kept on every unit."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PAGES_PARQUET = PROCESSED / "pages.parquet"

_WS = re.compile(r"\s+")


def load_questions() -> pd.DataFrame:
    q = pd.read_json(RAW / "financebench_open_source.jsonl", lines=True)
    q["gold_pages"] = q.evidence.apply(lambda ev: sorted({(e["doc_name"], int(e["evidence_page_num"])) for e in ev}))
    return q


def load_doc_info() -> pd.DataFrame:
    rows = [json.loads(line) for line in (RAW / "financebench_document_information.jsonl").read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(rows)


def clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def extract_pages(pdf_dir: Path = RAW / "pdfs", cache: bool = True) -> pd.DataFrame:
    if cache and PAGES_PARQUET.exists():
        return pd.read_parquet(PAGES_PARQUET)
    import pymupdf

    rows = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            for i, page in enumerate(doc):
                rows.append({"doc_name": pdf.stem, "page": i, "n_pages": len(doc), "text": clean(page.get_text())})
    pages = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pages.to_parquet(PAGES_PARQUET, index=False)
    return pages


def chunk_pages(pages: pd.DataFrame, words: int = 300, stride: int = 200) -> pd.DataFrame:
    """Overlapping word windows inside each page (a chunk never spans two pages,
    so every chunk maps to exactly one citable page)."""
    rows = []
    for doc, page, text in pages[["doc_name", "page", "text"]].itertuples(index=False):
        toks = text.split(" ")
        if len(toks) < 5:
            continue
        starts = range(0, max(len(toks) - words, 0) + 1, stride)
        for j, s in enumerate(starts):
            rows.append({"doc_name": doc, "page": page, "chunk": j, "text": " ".join(toks[s:s + words])})
        if len(toks) > words and (len(toks) - words) % stride:
            rows.append({"doc_name": doc, "page": page, "chunk": len(starts), "text": " ".join(toks[-words:])})
    return pd.DataFrame(rows)


def as_units(pages: pd.DataFrame, unit: str) -> pd.DataFrame:
    if unit == "page":
        return pages[pages.text.str.len() > 20][["doc_name", "page", "text"]].reset_index(drop=True)
    if unit == "chunk":
        return chunk_pages(pages).reset_index(drop=True)
    raise ValueError(unit)
