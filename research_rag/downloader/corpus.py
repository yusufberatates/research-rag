"""Registry of what is already in the corpus, for download-time dedup.

Builds the set of known paper ids and DOIs by scanning the per-topic
``raw/*.meta.json`` sidecars and ``extracted/*.json`` records, so the
downloader can skip anything already fetched (even under a different topic or
reached via a different query / snowball path).
"""
from __future__ import annotations

import json

from research_rag.config import PAPERS_DIR


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None


def known_ids() -> tuple[set[str], set[str]]:
    """Return (known_paper_ids, known_dois) currently in the corpus."""
    pids: set[str] = set()
    dois: set[str] = set()
    if not PAPERS_DIR.exists():
        return pids, dois

    for meta_path in PAPERS_DIR.glob("*/raw/*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("paper_id"):
            pids.add(meta["paper_id"])
        nd = normalize_doi(meta.get("doi"))
        if nd:
            dois.add(nd)

    for ex_path in PAPERS_DIR.glob("*/extracted/*.json"):
        try:
            rec = json.loads(ex_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("paper_id"):
            pids.add(rec["paper_id"])
        nd = normalize_doi(rec.get("doi"))
        if nd:
            dois.add(nd)

    return pids, dois
