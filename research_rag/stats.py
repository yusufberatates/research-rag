"""Corpus statistics for the ``pipeline_stats`` CLI command.

Reports corpus size, on-disk usage, taxonomy tree shape, papers-per-tier, and
how much of the corpus has actually been embedded into the vector store.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from research_rag.classifier import load_taxonomy
from research_rag.config import (
    DATA_DIR,
    PAPERS_DIR,
    QDRANT_COLLECTION,
    VECTORSTORE_DIR,
)
from research_rag.storage import all_extracted_json_paths


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _indexed_paper_ids() -> tuple[set[str], bool]:
    """Distinct paper_ids present in the vector store, and whether the
    collection exists. Scrolls all points (metadata may be nested)."""
    from research_rag.indexing.vector_store import get_qdrant_client

    client = get_qdrant_client()
    cols = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in cols:
        return set(), False

    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            pid = payload.get("paper_id") or (payload.get("metadata") or {}).get("paper_id")
            if pid:
                ids.add(pid)
        if offset is None:
            break
    return ids, True


def gather_stats() -> dict:
    extracted_paths = all_extracted_json_paths()
    raw_pdfs = list(PAPERS_DIR.glob("*/raw/*.pdf")) if PAPERS_DIR.exists() else []

    records = []
    for p in extracted_paths:
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    extracted_ids = {r.get("paper_id") for r in records if r.get("paper_id")}
    classified = [r for r in records if r.get("field")]
    tiers = Counter(r.get("tier") for r in classified)
    by_field = Counter(r.get("field") for r in classified)

    taxonomy = load_taxonomy()
    fields = taxonomy.get("fields", {})
    n_sub = sum(len(f.get("subfields", {})) for f in fields.values())
    n_ss = sum(
        len(s.get("subsubfields", {}))
        for f in fields.values()
        for s in f.get("subfields", {}).values()
    )
    # papers per top-level field straight from the tree (authoritative)
    tree_field_papers = {}
    for fname, fdata in fields.items():
        count = sum(
            len(ss.get("paper_ids", []))
            for s in fdata.get("subfields", {}).values()
            for ss in s.get("subsubfields", {}).values()
        )
        tree_field_papers[fname] = count

    indexed_ids, collection_exists = _indexed_paper_ids()

    return {
        "raw_pdfs": len(raw_pdfs),
        "extracted": len(records),
        "classified": len(classified),
        "tiers": dict(sorted((k, v) for k, v in tiers.items() if k is not None)),
        "by_field": dict(by_field),
        "tree": {"fields": len(fields), "subfields": n_sub, "subsubfields": n_ss},
        "tree_field_papers": tree_field_papers,
        "disk": {
            "papers": _dir_size(PAPERS_DIR),
            "vectorstore": _dir_size(VECTORSTORE_DIR),
            "data_total": _dir_size(DATA_DIR),
        },
        "embedding": {
            "collection_exists": collection_exists,
            "indexed_papers": len(indexed_ids & extracted_ids),
            "total_papers": len(extracted_ids),
            "complete": collection_exists and extracted_ids and extracted_ids <= indexed_ids,
        },
    }


def print_stats() -> None:
    s = gather_stats()
    print("=== Corpus ===")
    print(f"  raw PDFs:           {s['raw_pdfs']}")
    print(f"  extracted records:  {s['extracted']}")
    print(f"  classified:         {s['classified']}")

    print("\n=== Papers per tier ===")
    if s["tiers"]:
        for tier, n in s["tiers"].items():
            print(f"  tier {tier}: {n}")
    else:
        print("  (none classified yet)")

    print("\n=== Tree shape ===")
    t = s["tree"]
    print(f"  {t['fields']} fields / {t['subfields']} subfields / {t['subsubfields']} sub-subfields")
    print("  papers per top-level field:")
    for fname, count in sorted(s["tree_field_papers"].items(), key=lambda kv: -kv[1]):
        marker = "" if count else "  (empty)"
        print(f"    {fname}: {count}{marker}")

    print("\n=== Disk usage ===")
    d = s["disk"]
    print(f"  papers:      {_human(d['papers'])}")
    print(f"  vectorstore: {_human(d['vectorstore'])}")
    print(f"  data total:  {_human(d['data_total'])}")

    print("\n=== Embedding completion ===")
    e = s["embedding"]
    if not e["collection_exists"]:
        print("  vector store collection does not exist yet (run 'index').")
    else:
        status = "COMPLETE" if e["complete"] else "incomplete"
        print(f"  {e['indexed_papers']}/{e['total_papers']} papers embedded  [{status}]")
