"""Extract text + metadata from a downloaded PDF.

Pipeline per page: try the embedded text layer first (pymupdf); if it
looks too sparse to be real text, treat the page as scanned and fall back
to OCR. Metadata preference order: GROBID (if enabled & reachable) >
the API metadata saved by the downloader (title/authors/year/abstract
from arXiv/Semantic Scholar) > pymupdf's own PDF metadata dict.

The PDF is only ever opened as data with pymupdf -- never executed or
handed to an external viewer/program ("sandboxed extraction").
"""
from __future__ import annotations

import json
from pathlib import Path

import fitz  # pymupdf

from research_rag.config import GROBID_ENABLED, OCR_MIN_CHARS_PER_PAGE
from research_rag.storage import extracted_json_path

from .ocr_fallback import ocr_page


def _load_sidecar_metadata(pdf_path: Path) -> dict:
    meta_path = pdf_path.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def extract_paper(pdf_path: Path, topic: str, paper_id: str) -> dict:
    doc = fitz.open(pdf_path)  # parses PDF structure only, never executes content
    try:
        pages_text: list[str] = []
        ocr_used = False
        for page in doc:
            text = page.get_text("text")
            if len(text.strip()) < OCR_MIN_CHARS_PER_PAGE:
                ocr_text = ocr_page(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    ocr_used = True
            pages_text.append(text)

        pdf_meta = doc.metadata or {}
    finally:
        doc.close()

    full_text = "\n\n".join(pages_text)
    sidecar = _load_sidecar_metadata(pdf_path)

    grobid_meta = None
    if GROBID_ENABLED:
        from . import grobid_client

        grobid_meta = grobid_client.extract_metadata(pdf_path)

    title = (
        (grobid_meta or {}).get("title")
        or sidecar.get("title")
        or pdf_meta.get("title")
        or pdf_path.stem
    )
    authors = (
        (grobid_meta or {}).get("authors")
        or sidecar.get("authors")
        or ([pdf_meta["author"]] if pdf_meta.get("author") else [])
    )
    abstract = (grobid_meta or {}).get("abstract") or sidecar.get("abstract") or ""
    year = sidecar.get("year")

    record = {
        "paper_id": paper_id,
        "topic": topic,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "source": sidecar.get("source", "unknown"),
        "doi": sidecar.get("doi"),
        "tier": sidecar.get("tier"),
        "num_pages": len(pages_text),
        "ocr_used": ocr_used,
        "pages": pages_text,
        "full_text": full_text,
        "pdf_path": str(pdf_path),
    }

    out_path = extracted_json_path(topic, paper_id)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record
