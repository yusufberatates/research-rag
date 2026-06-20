"""Thin client for the Semantic Scholar Graph API.

Papers returned by Semantic Scholar often link to open-access PDFs hosted
on third-party domains (publisher sites, institutional repos, etc.). Since
the downloader only trusts arxiv.org / semanticscholar.org / pubmed, any
result whose PDF URL is not on the whitelist is dropped rather than
silently fetched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from research_rag.config import (
    S2_BACKOFF_BASE_SECONDS,
    S2_BACKOFF_ENABLED,
    S2_BACKOFF_MAX_SECONDS,
    S2_MAX_RETRIES,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS,
)
from research_rag.security import assert_domain_allowed, is_domain_allowed

from .rate_limit import s2_limiter

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
FIELDS = "title,abstract,year,authors,openAccessPdf,externalIds"

# Status codes worth retrying with backoff (rate limit + transient server errors).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _headers() -> dict:
    """Authenticated header when an API key is configured, else keyless."""
    return {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}


def _get_with_backoff(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """GET with optional exponential backoff on 429/5xx.

    When ``S2_BACKOFF_ENABLED`` is off, behaves like a plain single request
    (raising for status), preserving the previous fail-fast behaviour.
    """
    attempts = S2_MAX_RETRIES if S2_BACKOFF_ENABLED else 0
    for attempt in range(attempts + 1):
        s2_limiter.acquire()
        resp = requests.get(url, params=params, timeout=timeout, headers=_headers())
        if resp.status_code not in _RETRYABLE_STATUS or attempt == attempts:
            resp.raise_for_status()
            return resp
        # Honour a Retry-After header if present, else exponential backoff.
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        else:
            delay = S2_BACKOFF_BASE_SECONDS * (2 ** attempt)
        delay = min(delay, S2_BACKOFF_MAX_SECONDS)
        # Tag each retry with auth status so a 429 storm is never ambiguous:
        # [keyed] storms = S2 throttling this endpoint (the /paper/search
        # endpoint does this even with a valid key); [KEYLESS] storms = the key
        # isn't loaded in this process. Endpoint helps tell search from refs.
        auth = "keyed" if SEMANTIC_SCHOLAR_API_KEY else "KEYLESS"
        endpoint = "search" if "/search" in url else ("refs" if "/references" in url else "paper")
        print(
            f"Semantic Scholar returned {resp.status_code} [{auth}/{endpoint}]; "
            f"retrying in {delay:.0f}s (attempt {attempt + 1}/{attempts})."
        )
        time.sleep(delay)
    # Unreachable, but keeps type checkers happy.
    raise RuntimeError("unreachable")


@dataclass
class SemanticScholarPaper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    pdf_url: str
    doi: str | None = None
    source: str = "semanticscholar"


def _paper_from_item(item: dict) -> SemanticScholarPaper | None:
    """Build a SemanticScholarPaper from a Graph API paper object, or None if
    it has no open-access PDF on a whitelisted domain."""
    if not item:
        return None
    oa = item.get("openAccessPdf") or {}
    pdf_url = oa.get("url") or ""
    if not pdf_url or not is_domain_allowed(pdf_url):
        return None
    external = item.get("externalIds") or {}
    authors = [a.get("name", "") for a in item.get("authors", []) or []]
    return SemanticScholarPaper(
        paper_id=f"s2_{item.get('paperId')}",
        title=(item.get("title") or "").strip(),
        authors=authors,
        abstract=(item.get("abstract") or "").strip(),
        year=item.get("year"),
        pdf_url=pdf_url,
        doi=(external.get("DOI") or external.get("doi")),
    )


def search(query: str, max_results: int = 20) -> list[SemanticScholarPaper]:
    assert_domain_allowed(SEARCH_URL)
    params = {"query": query, "limit": max_results, "fields": FIELDS}
    resp = _get_with_backoff(SEARCH_URL, params=params, timeout=30)
    time.sleep(SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS)

    data = resp.json()
    papers: list[SemanticScholarPaper] = []
    for item in data.get("data", []):
        paper = _paper_from_item(item)
        if paper is not None:
            papers.append(paper)
    return papers


def get_references(paper_ref: str, limit: int = 50) -> list[SemanticScholarPaper]:
    """Return the papers cited by ``paper_ref`` (its reference list).

    ``paper_ref`` may be an arXiv id formatted as ``ARXIV:2407.21701`` or a
    Semantic Scholar paper id. Only references with an open-access PDF on a
    whitelisted domain are returned; the rest are dropped (snowball can only
    ingest what it is allowed to fetch).
    """
    url = f"{PAPER_URL}/{paper_ref}/references"
    assert_domain_allowed(url)
    params = {"fields": FIELDS, "limit": limit}
    try:
        resp = _get_with_backoff(url, params=params, timeout=30)
    except requests.RequestException:
        return []
    time.sleep(SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS)

    out: list[SemanticScholarPaper] = []
    for row in resp.json().get("data", []):
        paper = _paper_from_item(row.get("citedPaper") or {})
        if paper is not None:
            out.append(paper)
    return out


def arxiv_ref(paper_id: str) -> str | None:
    """Convert an internal arxiv_<id> paper_id into an 'ARXIV:<id>' ref for
    the Graph API, or None if it is not an arXiv id.

    Kept for callers that only handle arXiv ids; new code should prefer
    ``paper_ref`` which also resolves Semantic-Scholar-sourced ids."""
    if paper_id.startswith("arxiv_"):
        return "ARXIV:" + paper_id[len("arxiv_"):].split("v")[0]
    return None


def paper_ref(paper_id: str) -> str | None:
    """Convert any internal paper_id into a Semantic Scholar Graph API paper
    reference usable in ``/paper/{ref}/references``, or None if its id type
    cannot be resolved.

    The Graph API accepts several id forms; we map our two internal prefixes:

      * ``arxiv_<id>``   -> ``ARXIV:<id>``   (trailing version suffix stripped)
      * ``s2_<paperId>`` -> ``<paperId>``    (the raw Semantic Scholar paperId,
                                              which the API takes directly)

    Without the ``s2_`` case, every Semantic-Scholar-sourced paper -- which
    includes basically everything snowball itself discovers -- is a dead end
    for further citation expansion. Any other id shape returns None so the
    caller can log and skip it rather than fetch the wrong reference list.
    """
    if paper_id.startswith("arxiv_"):
        return "ARXIV:" + paper_id[len("arxiv_"):].split("v")[0]
    if paper_id.startswith("s2_"):
        return paper_id[len("s2_"):].strip() or None
    return None
