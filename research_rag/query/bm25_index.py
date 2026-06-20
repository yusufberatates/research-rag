"""Lightweight BM25 index over paper titles/abstracts/summaries.

Supplements vector search by catching exact keyword matches that embeddings
miss: abbreviations ("OPA", "GHZ"), author names, specific model numbers.
Pure Python via rank_bm25 — no GPU, ~0 extra RAM beyond the text itself.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from research_rag.storage import all_extracted_json_paths

_corpus: "_BM25Corpus | None" = None

_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _BM25Corpus:
    def __init__(self) -> None:
        self._paper_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._build()

    def _build(self) -> None:
        docs: list[list[str]] = []
        for path in all_extracted_json_paths():
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pid = rec.get("paper_id")
            if not pid:
                continue
            text = " ".join(filter(None, [
                rec.get("title", ""),
                rec.get("abstract", ""),
                rec.get("summary", ""),
            ]))
            self._paper_ids.append(pid)
            docs.append(_tokenize(text))
        if docs:
            self._bm25 = BM25Okapi(docs)

    def _search_one(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._bm25 or not self._paper_ids:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            (self._paper_ids[i], float(scores[i]))
            for i in ranked[:top_k]
            if scores[i] > 0.0
        ]

    def search(self, queries: list[str], top_k: int = 8) -> list[str]:
        """Multi-query BM25 with RRF merge. Returns top paper_ids."""
        rrf: dict[str, float] = defaultdict(float)
        fetch = top_k * 2
        for q in queries:
            for rank, (pid, _) in enumerate(self._search_one(q, fetch)):
                rrf[pid] += 1.0 / (_RRF_K + rank + 1)
        return sorted(rrf, key=lambda p: rrf[p], reverse=True)[:top_k]

    @property
    def size(self) -> int:
        return len(self._paper_ids)


def get_bm25() -> _BM25Corpus:
    """Lazy singleton — built once per process from disk."""
    global _corpus
    if _corpus is None:
        _corpus = _BM25Corpus()
    return _corpus


def reset_bm25() -> None:
    """Drop cached index so the next get_bm25() call rebuilds from disk."""
    global _corpus
    _corpus = None
