"""Downloader orchestration: search arXiv + Semantic Scholar for a topic,
then fetch each PDF under strict security constraints, in parallel.

Security invariants enforced here (see also research_rag.security):
  * Every URL is checked against the domain whitelist before any request.
  * Only responses with Content-Type: application/pdf are saved.
  * Downloads are capped in size to avoid disk-fill abuse.
  * Downloaded bytes are never executed, opened by an external program, or
    interpreted as anything other than opaque PDF data.
  * Every saved file is checksummed and appended to an audit log.

Scale features:
  * Up to ``DOWNLOAD_WORKERS`` PDFs fetch concurrently, gated by a shared
    arXiv rate limiter so the process stays within ~3 req/s.
  * Candidates already in the corpus (by paper_id or DOI) are skipped.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import requests

from research_rag.config import (
    ALLOWED_CONTENT_TYPES,
    DOWNLOAD_WORKERS,
    MAX_PDF_BYTES,
)
from research_rag.security import assert_domain_allowed, log_download
from research_rag.storage import raw_pdf_path

from . import arxiv_client, semantic_scholar_client
from .corpus import known_ids, normalize_doi
from .rate_limit import arxiv_limiter


def _download_pdf(url: str, dest: Path) -> bool:
    """Download a single PDF to ``dest``. Returns True on success.

    Refuses to save anything that is not declared as application/pdf, and
    enforces a hard size cap while streaming so a malicious/huge response
    can't exhaust disk space.
    """
    assert_domain_allowed(url)
    arxiv_limiter.acquire()  # whitelisted PDFs are arxiv.org-hosted
    with requests.get(url, stream=True, timeout=60, headers={"Accept": "application/pdf"}) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type not in ALLOWED_CONTENT_TYPES:
            return False

        total = 0
        tmp_path = dest.with_suffix(".part")
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    f.close()
                    tmp_path.unlink(missing_ok=True)
                    return False
                f.write(chunk)

        with open(tmp_path, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            tmp_path.unlink(missing_ok=True)
            return False

        tmp_path.replace(dest)
        return True


def _search_candidates(query: str, max_results: int, use_arxiv: bool, use_s2: bool) -> list[dict]:
    candidates: list[dict] = []
    if use_arxiv:
        try:
            for p in arxiv_client.search(query, max_results=max_results):
                candidates.append(asdict(p))
        except requests.RequestException as exc:
            print(f"Warning: arXiv search failed ({exc}); continuing without it.")
    if use_s2:
        try:
            for p in semantic_scholar_client.search(query, max_results=max_results):
                candidates.append(asdict(p))
        except requests.RequestException as exc:
            print(f"Warning: Semantic Scholar search failed ({exc}); continuing without it.")
    return candidates


def _dedup(candidates: list[dict]) -> list[dict]:
    """Drop candidates already in the corpus (by paper_id or DOI) and any
    duplicates within this batch."""
    known_pids, known_dois = known_ids()
    seen_pids: set[str] = set()
    seen_dois: set[str] = set()
    out: list[dict] = []
    for c in candidates:
        pid = c.get("paper_id")
        nd = normalize_doi(c.get("doi"))
        if pid in known_pids or pid in seen_pids:
            continue
        if nd and (nd in known_dois or nd in seen_dois):
            continue
        seen_pids.add(pid)
        if nd:
            seen_dois.add(nd)
        out.append(c)
    return out


def _fetch_one(paper: dict, topic: str, tier: int | None) -> dict | None:
    """Download + persist a single candidate. Returns its metadata dict on
    success, else None. Safe to run in a worker thread."""
    dest = raw_pdf_path(topic, paper["paper_id"])
    if not dest.exists():
        try:
            if not _download_pdf(paper["pdf_url"], dest):
                return None
        except requests.RequestException:
            return None

    log_download(
        source=paper["source"],
        url=paper["pdf_url"],
        topic=topic,
        file_path=dest,
        paper_id=paper["paper_id"],
    )
    meta = {**paper, "pdf_path": str(dest), "topic": topic}
    if tier is not None:
        meta["tier"] = tier
    meta_path = dest.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def download_candidates(
    candidates: list[dict],
    topic: str,
    tier: int | None = None,
    workers: int = DOWNLOAD_WORKERS,
) -> list[dict]:
    """Download an already-built, already-deduped list of candidates in
    parallel. Reused by both topic search and snowball expansion."""
    downloaded: list[dict] = []
    if not candidates:
        return downloaded
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_fetch_one, c, topic, tier): c for c in candidates}
        for fut in as_completed(futures):
            meta = fut.result()
            if meta is not None:
                downloaded.append(meta)
    return downloaded


def _relevance_filter(candidates: list[dict]) -> list[dict]:
    """Drop candidates the LLM judges off-topic for the taxonomy, using only
    their title/abstract (so junk is filtered before any PDF is fetched)."""
    from research_rag.classifier.relevance import judge_relevance

    kept: list[dict] = []
    for c in candidates:
        relevant, _ = judge_relevance(c.get("title", ""), c.get("abstract", ""))
        if relevant:
            kept.append(c)
        else:
            print(f"  gate: rejected (off-topic) {c.get('title', '')[:70]}")
    return kept


def download_topic(
    topic: str,
    query: str | None = None,
    max_results: int = 10,
    use_arxiv: bool = True,
    use_semantic_scholar: bool = True,
    tier: int | None = None,
    relevance_gate: bool = False,
) -> list[dict]:
    """Search and download papers for ``topic`` (parallel, deduped).

    When ``relevance_gate`` is set, each candidate is LLM-judged against the
    taxonomy on its title/abstract and off-topic ones are dropped before
    download -- so noise never reaches extract/classify/embed.

    Returns a list of metadata dicts for every paper that was successfully
    downloaded, each with a ``pdf_path`` pointing at the saved file.
    """
    query = query or topic
    candidates = _search_candidates(query, max_results, use_arxiv, use_semantic_scholar)
    candidates = _dedup(candidates)
    if relevance_gate:
        candidates = _relevance_filter(candidates)
    return download_candidates(candidates, topic, tier=tier)
