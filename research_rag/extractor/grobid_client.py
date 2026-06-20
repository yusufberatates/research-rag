"""Optional GROBID integration for higher-quality metadata extraction.

GROBID (https://github.com/kermitt2/grobid) requires running a separate
Java service and is not started by this project. It is disabled by
default (config.GROBID_ENABLED). When enabled and reachable, it is used
to extract title/authors/abstract more reliably than heuristics; on any
failure (service down, timeout, parse error) callers should fall back to
pymupdf/API-derived metadata -- this module never raises.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from research_rag.config import GROBID_URL

TEI_NS = "{http://www.tei-c.org/ns/1.0}"


def extract_metadata(pdf_path: Path) -> dict | None:
    """Returns a dict with title/authors/abstract or None if GROBID is
    unavailable or extraction failed."""
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{GROBID_URL}/api/processHeaderDocument",
                files={"input": f},
                timeout=60,
            )
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.content)

        title = root.findtext(f".//{TEI_NS}titleStmt/{TEI_NS}title")
        abstract = root.findtext(f".//{TEI_NS}abstract/{TEI_NS}p")

        authors = []
        for pers in root.findall(f".//{TEI_NS}sourceDesc//{TEI_NS}persName"):
            forename = pers.findtext(f"{TEI_NS}forename") or ""
            surname = pers.findtext(f"{TEI_NS}surname") or ""
            name = f"{forename} {surname}".strip()
            if name:
                authors.append(name)

        return {
            "title": (title or "").strip() or None,
            "abstract": (abstract or "").strip() or None,
            "authors": authors or None,
        }
    except (requests.RequestException, ET.ParseError):
        return None
