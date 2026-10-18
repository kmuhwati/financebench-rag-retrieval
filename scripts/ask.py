"""Ask a question about the indexed filings and see the evidence pages retrieved.

    python scripts/ask.py "What was 3M's FY2018 capital expenditure?"
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.pipeline import FinancialRetriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    hits = FinancialRetriever().search(args.question, k=args.k)
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] {h['doc_name']}  page {h['page'] + 1}  (score {h['score']:.2f})")
        print(textwrap.indent(textwrap.shorten(h["text"], 400), "    "))


if __name__ == "__main__":
    main()
