"""Download the FinanceBench open-source sample and only the PDFs its 150 questions use.

Source: https://github.com/patronus-ai/financebench (CC-BY-NC-4.0).
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    for name in ("financebench_open_source.jsonl", "financebench_document_information.jsonl"):
        fetch(f"{BASE}/data/{name}", RAW / name)
    questions = [json.loads(line) for line in (RAW / "financebench_open_source.jsonl").read_text(encoding="utf-8").splitlines()]
    docs = sorted({q["doc_name"] for q in questions})
    for i, doc in enumerate(docs, 1):
        fetch(f"{BASE}/pdfs/{doc}.pdf", RAW / "pdfs" / f"{doc}.pdf")
        print(f"[{i}/{len(docs)}] {doc}")
    print(f"{len(questions)} questions, {len(docs)} filings in {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
