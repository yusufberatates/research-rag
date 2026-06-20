"""Thin client for the arXiv API (https://export.arxiv.org/api/query).

Only ever talks to export.arxiv.org / arxiv.org, both on the domain
whitelist. Returns plain dataclasses describing each paper; downloading
the PDF bytes is handled separately in downloader.py so the security
checks live in one place.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

from research_rag.security import assert_domain_allowed

from .rate_limit import arxiv_limiter

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


@dataclass
class ArxivPaper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    pdf_url: str
    doi: str | None = None
    source: str = "arxiv"


def search(query: str, max_results: int = 20) -> list[ArxivPaper]:
    assert_domain_allowed(ARXIV_API_URL)
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    arxiv_limiter.acquire()
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    papers: list[ArxivPaper] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id_full = entry.findtext(f"{ATOM_NS}id", default="")
        arxiv_id = arxiv_id_full.rsplit("/", 1)[-1]
        title = (entry.findtext(f"{ATOM_NS}title", default="") or "").strip().replace("\n", " ")
        abstract = (entry.findtext(f"{ATOM_NS}summary", default="") or "").strip().replace("\n", " ")
        published = entry.findtext(f"{ATOM_NS}published", default="")
        year = int(published[:4]) if published[:4].isdigit() else None

        authors = [
            (a.findtext(f"{ATOM_NS}name", default="") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]

        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        if not pdf_url:
            continue

        doi = entry.findtext(f"{ARXIV_NS}doi") or None

        papers.append(
            ArxivPaper(
                paper_id=f"arxiv_{arxiv_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                pdf_url=pdf_url,
                doi=doi,
            )
        )
    return papers
