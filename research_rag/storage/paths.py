"""Filesystem layout helpers.

Papers are organized as::

    data/papers/<topic_slug>/raw/<paper_id>.pdf
    data/papers/<topic_slug>/extracted/<paper_id>.json
"""
from __future__ import annotations

import re
from pathlib import Path

from research_rag.config import PAPERS_DIR


def topic_slug(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return slug or "untitled_topic"


def raw_dir(topic: str) -> Path:
    d = PAPERS_DIR / topic_slug(topic) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extracted_dir(topic: str) -> Path:
    d = PAPERS_DIR / topic_slug(topic) / "extracted"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_pdf_path(topic: str, paper_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)
    return raw_dir(topic) / f"{safe_id}.pdf"


def extracted_json_path(topic: str, paper_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)
    return extracted_dir(topic) / f"{safe_id}.json"


def all_extracted_json_paths() -> list[Path]:
    """Every extracted record across all topics (papers/*/extracted/*.json)."""
    if not PAPERS_DIR.exists():
        return []
    return sorted(PAPERS_DIR.glob("*/extracted/*.json"))
