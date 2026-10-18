"""Metadata routing: read the company and fiscal period out of the question.

In a corpus of many filings that all talk about "revenue" and "operating
margin", the hardest part of retrieval is picking the right *document*.
Questions almost always name the company and the period, so a deterministic
parser recovers most of that information before any embedding is computed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Aliases seen in financial writing; keys are FinanceBench company names.
ALIASES = {
    "American Express": ["amex"],
    "Johnson & Johnson": ["jnj", "j&j", "johnson and johnson"],
    "PepsiCo": ["pepsi"],
    "Coca-Cola": ["coca cola", "coke"],
    "MGM Resorts": ["mgm"],
    "AES Corporation": ["aes"],
    "JPMorgan": ["jp morgan", "jpm", "jpmorgan chase"],
    "Activision Blizzard": ["activision"],
    "CVS Health": ["cvs"],
    "Block": ["square"],
    "Kraft Heinz": ["kraft"],
    "Ulta Beauty": ["ulta"],
    "American Water Works": ["american water"],
    "Paypal": ["paypal"],
}

_YEAR = re.compile(r"(?:fy\s?'?|fiscal(?: year)?\s?)(\d{2,4})(?!\d)|(?<!\d)(20\d{2})(?!\d)|(?<![a-z])q[1-4]\s?(20\d{2})", re.I)
_QUARTER = re.compile(r"(?<![a-z])q([1-4])(?![0-9]{1}\b)", re.I)


@dataclass
class Route:
    company: str | None
    years: list[int]
    quarter: int | None


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9& ]", " ", s.lower())


def parse(question: str, companies: list[str]) -> Route:
    text = f" {_normalise(question)} "
    best, best_len = None, 0
    for c in companies:
        for alias in [c.lower(), *ALIASES.get(c, [])]:
            a = _normalise(alias).strip()
            if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", text) and len(a) > best_len:
                best, best_len = c, len(a)
    years = []
    for m in _YEAR.finditer(question):
        y = int(m.group(1) or m.group(2) or m.group(3))
        years.append(y + 2000 if y < 100 else y)
    q = _QUARTER.search(question)
    return Route(best, sorted(set(years)), int(q.group(1)) if q else None)


def doc_scores(route: Route, docs: pd.DataFrame) -> pd.Series:
    """Score each document for a route: company match dominates, then period fit.

    Returns NaN for documents of other companies (filtered out) when a company
    was found; otherwise every document gets 0 (no filter).
    """
    if route.company is None:
        return pd.Series(0.0, index=docs.doc_name)
    s = pd.Series(float("nan"), index=docs.doc_name)
    mine = docs[docs.company == route.company]
    for _, d in mine.iterrows():
        score = 1.0
        period = int(d.doc_period)
        if route.years:
            latest = max(route.years)
            score += 2.0 if period == latest else (1.0 if period in route.years else -0.5 * min(abs(period - latest), 3))
        is_quarterly = d.doc_type.lower() in ("10q", "earnings")
        if route.quarter is not None:
            score += 1.0 if is_quarterly and f"Q{route.quarter}" in d.doc_name else -0.5
        elif is_quarterly or d.doc_type.lower() == "8k":
            score -= 0.25  # annual questions default to the 10-K
        s[d.doc_name] = score
    return s
