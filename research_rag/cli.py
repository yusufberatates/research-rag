"""Command-line interface tying every component together.

Subcommands:
  download <topic> [--query Q] [--max-results N]
  extract  <topic>
  classify <topic>
  index    <topic>
  pipeline <topic> [--query Q] [--max-results N]   (download+extract+classify+index)
  query    "<question>"
  taxonomy                          show the current field tree
  consolidate_taxonomy              merge semantically similar sibling nodes
  reset_taxonomy                    overwrite taxonomy with the seeded 7 fields
  rebuild_taxonomy                  re-attach classified records to the tree
  reclassify                        re-run classification over the whole corpus
  reset_index                       drop the local vector store collection
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_rag.classifier import (
    attach_existing,
    classify_paper,
    consolidate_taxonomy,
    load_taxonomy,
    paper_paths,
    reset_to_seed,
    summarize_paper,
)
from research_rag.downloader import download_topic
from research_rag.extractor import extract_paper
from research_rag.indexing import add_paper_to_index
from research_rag.ingest import NightlyController, snowball
from research_rag.query import answer_question
from research_rag.storage import all_extracted_json_paths, extracted_dir, raw_dir


def _iter_raw_pdfs(topic: str) -> list[Path]:
    return sorted(raw_dir(topic).glob("*.pdf"))


def _iter_extracted_records(topic: str) -> list[dict]:
    records = []
    for p in sorted(extracted_dir(topic).glob("*.json")):
        records.append(json.loads(p.read_text(encoding="utf-8")))
    return records


def cmd_download(args: argparse.Namespace) -> None:
    results = download_topic(
        args.topic,
        query=args.query,
        max_results=args.max_results,
        tier=getattr(args, "tier", None),
        relevance_gate=getattr(args, "gate", False),
    )
    print(f"Downloaded {len(results)} papers for topic '{args.topic}'.")
    for r in results:
        print(f"  - {r['title']} [{r['source']}]")


def cmd_extract(args: argparse.Namespace) -> None:
    pdfs = _iter_raw_pdfs(args.topic)
    if not pdfs:
        print(f"No raw PDFs found for topic '{args.topic}'. Run 'download' first.")
        return
    for pdf_path in pdfs:
        paper_id = pdf_path.stem
        record = extract_paper(pdf_path, args.topic, paper_id)
        print(f"Extracted: {record['title']} ({record['num_pages']} pages, ocr={record['ocr_used']})")


def cmd_classify(args: argparse.Namespace) -> None:
    records = _iter_extracted_records(args.topic)
    if not records:
        print(f"No extracted papers found for topic '{args.topic}'. Run 'extract' first.")
        return
    for record in records:
        # Resumable: skip papers already classified (so a re-run continues).
        # Still re-attach to the taxonomy tree, which may be out of sync with
        # the record (e.g. after reset_taxonomy) -- cheap no-op when present.
        if record.get("field") and record.get("tier") is not None:
            reattached = attach_existing(record)
            note = " (re-attached to taxonomy)" if reattached else ""
            print(f"Already classified: {record['title']} (skip){note}")
            continue
        summary = summarize_paper(record)
        result = classify_paper(record, summary)
        record["summary"] = summary
        record["field"] = result["field"]
        record["subfield"] = result["subfield"]
        record["subsubfield"] = result["subsubfield"]
        # An explicit tier set at download time wins; otherwise derive from field.
        if record.get("tier") is None:
            record["tier"] = result["tier"]
        out_path = extracted_dir(args.topic) / f"{record['paper_id']}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Classified: {record['title']} -> "
            f"{result['field']} / {result['subfield']} / {result['subsubfield']} "
            f"(tier {record['tier']})"
        )


def cmd_index(args: argparse.Namespace) -> None:
    records = _iter_extracted_records(args.topic)
    if not records:
        print(f"No extracted papers found for topic '{args.topic}'. Run 'extract'/'classify' first.")
        return
    # Resumable: skip papers already embedded (so a re-run continues).
    from research_rag.stats import _indexed_paper_ids

    already_indexed, _ = _indexed_paper_ids()
    skipped = 0
    for record in records:
        if record["paper_id"] in already_indexed:
            print(f"Already indexed: {record['title']} (skip)")
            continue
        # Never invent a field for an unclassified record. A silent fallback to
        # supporting_quantum_optics/uncategorized tags chunks with a path that
        # exists in NO taxonomy node, so the hierarchical router can never
        # retrieve the paper -- it looks "indexed" but is invisible to queries.
        # Skip it loudly so classification failures surface instead of hiding.
        field = record.get("field")
        if not field:
            skipped += 1
            print(f"Not classified, skipping (run 'classify' first): {record['title']}")
            continue
        subfield = record.get("subfield") or "uncategorized"
        subsubfield = record.get("subsubfield") or "uncategorized"
        add_paper_to_index(record, field, subfield, subsubfield)
        print(f"Indexed: {record['title']} ({field} / {subfield} / {subsubfield})")
    if skipped:
        print(
            f"Skipped {skipped} unclassified paper(s). Run "
            f"'classify {args.topic}', then re-run 'index {args.topic}'."
        )


def cmd_pipeline(args: argparse.Namespace) -> None:
    cmd_download(args)
    cmd_extract(args)
    cmd_classify(args)
    cmd_index(args)


def cmd_query(args: argparse.Namespace) -> None:
    result = answer_question(args.question)
    print("\n=== Answer ===")
    print(result["answer"])
    print(f"\n(routing: {result['routing']})")
    print("\n=== Citations ===")
    if not result["citations"]:
        print("(no sources retrieved)")
    for c in result["citations"]:
        path = " / ".join(
            x for x in (c.get("field"), c.get("subfield"), c.get("subsubfield")) if x
        )
        print(f"  - {c['title']} — {c['authors']} ({c['year']})  [{path}]")


def cmd_taxonomy(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy()
    fields = taxonomy.get("fields", {})
    if not fields:
        print("(taxonomy is empty)")
        return
    for field, fdata in fields.items():
        subs = fdata.get("subfields", {})
        n_papers = sum(
            len(ss.get("paper_ids", []))
            for s in subs.values()
            for ss in s.get("subsubfields", {}).values()
        )
        marker = "" if n_papers else "  (empty)"
        print(f"- {field}  [{n_papers} papers]{marker}")
        for subfield, sdata in subs.items():
            print(f"    - {subfield}")
            for subsub, ssdata in sdata.get("subsubfields", {}).items():
                pids = ssdata.get("paper_ids", [])
                print(f"        - {subsub}: {', '.join(pids) if pids else '(no papers)'}")


def cmd_consolidate(args: argparse.Namespace) -> None:
    result = consolidate_taxonomy()
    print("Consolidated taxonomy.")
    print(f"  before: {result['before']}")
    print(f"  after:  {result['after']}")

    # Re-tag papers whose field/subfield/sub-subfield path changed as a result
    # of the merge, both in their extracted records and in the vector index.
    paths = paper_paths()
    retagged = 0
    for json_path in all_extracted_json_paths():
        record = json.loads(json_path.read_text(encoding="utf-8"))
        pid = record.get("paper_id")
        if pid not in paths:
            continue
        field, subfield, subsubfield = paths[pid]
        current = (record.get("field"), record.get("subfield"), record.get("subsubfield"))
        if current == (field, subfield, subsubfield):
            continue
        record["field"], record["subfield"], record["subsubfield"] = (
            field,
            subfield,
            subsubfield,
        )
        json_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        add_paper_to_index(record, field, subfield, subsubfield)  # upsert: re-tags chunks
        retagged += 1
        print(f"  re-tagged: {record.get('title', pid)} -> {field} / {subfield} / {subsubfield}")
    print(f"Re-tagged {retagged} paper(s).")


def cmd_pipeline_stats(args: argparse.Namespace) -> None:
    from research_rag.stats import print_stats

    print_stats()


def cmd_reset_taxonomy(args: argparse.Namespace) -> None:
    reset_to_seed()
    print("Taxonomy reset to the 7 seeded top-level fields.")


def cmd_reclassify(args: argparse.Namespace) -> None:
    """Rebuild classifications for the whole corpus from a fresh seeded
    taxonomy, using the improved classifier.

    Overwrites each record's field/subfield/subsubfield/tier (reusing the
    stored summary, so PDFs are never re-read) and re-tags the vector index
    ONLY for papers whose path actually changed -- so unchanged papers are not
    needlessly re-embedded. Destructive to existing classifications.

    Run this alone: it writes both the taxonomy and the vector store, which
    must not be touched by a concurrent pipeline run."""
    reset_to_seed()
    print("Taxonomy reset to seed; re-classifying with the improved classifier.")
    records = []
    for json_path in all_extracted_json_paths():
        try:
            records.append((json_path, json.loads(json_path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    total = len(records)
    retagged = 0
    for i, (json_path, record) in enumerate(records, 1):
        old = (record.get("field"), record.get("subfield"), record.get("subsubfield"))
        summary = record.get("summary") or summarize_paper(record)
        result = classify_paper(record, summary)
        new = (result["field"], result["subfield"], result["subsubfield"])
        record["summary"] = summary
        record["field"], record["subfield"], record["subsubfield"] = new
        record["tier"] = result["tier"]
        json_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        changed = new != old
        if changed:
            add_paper_to_index(record, *new)  # re-tag chunks (embeds if missing)
            retagged += 1
        print(
            f"[{i}/{total}] {' / '.join(new)}"
            f"{'  (re-tagged)' if changed else ''}  <- {record.get('title', '?')[:70]}"
        )
    print(f"Re-classified {total} record(s); re-tagged {retagged} changed paper(s).")


def cmd_rebuild_taxonomy(args: argparse.Namespace) -> None:
    """Repopulate the taxonomy tree from already-classified extracted records.

    Repairs a tree that has fallen out of sync with the corpus (e.g. after a
    reset_taxonomy while records kept their classification) without
    re-downloading, re-extracting, or re-running the LLM classifier."""
    seen = changed = 0
    for json_path in all_extracted_json_paths():
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not record.get("field"):
            continue
        seen += 1
        if attach_existing(record):
            changed += 1
            print(f"  attached: {record.get('title', record.get('paper_id'))}")
    print(f"Re-attached {changed}/{seen} classified paper(s) to the taxonomy tree.")


def cmd_snowball(args: argparse.Namespace) -> None:
    nc = NightlyController("snowball", enabled=args.nightly)
    result = snowball(
        tier=args.tier,
        max_papers=args.max_papers,
        dry_run=args.dry_run,
        refs_per_paper=args.refs,
        nightly=nc,
    )
    if result.get("interrupted"):
        print(f"Interrupted after ingesting {result['ingested']} papers (checkpoint saved).")
        return
    if args.dry_run:
        print(f"Dry run: {result['would_ingest']} paper(s) would be ingested (tier<={args.tier}).")
        for d in result["decisions"]:
            print(f"  - {d['title'][:75]} -> {d['field']} (tier {d['tier']})")
    else:
        print(f"Snowball ingested {result['ingested']} paper(s) (tier<={args.tier}).")


def cmd_healthcheck(args: argparse.Namespace) -> None:
    """Verify Ollama is reachable and the required models are present, and
    report the Semantic Scholar key status.

    Exits non-zero only on the Ollama check (the LLM is required); the S2 key is
    optional, so its status is reported but never fatal."""
    from research_rag.config import SEMANTIC_SCHOLAR_API_KEY
    from research_rag.llm.ollama_client import check_health

    ok, msg = check_health()
    print(msg)
    # Authoritative S2 key status: read via the SAME loader the downloader and
    # snowball use (env var OR data/s2_api_key.txt, BOM-tolerant, stripped), so
    # it can never contradict what actually happens at runtime.
    if SEMANTIC_SCHOLAR_API_KEY:
        print(
            f"Semantic Scholar key: detected (len {len(SEMANTIC_SCHOLAR_API_KEY)}); "
            "S2 calls will be authenticated."
        )
    else:
        print(
            "Semantic Scholar key: NOT found (checked env SEMANTIC_SCHOLAR_API_KEY "
            "and data/s2_api_key.txt); S2 will run keyless (slower, more 429s)."
        )
    if not ok:
        raise SystemExit(2)


def cmd_reset_index(args: argparse.Namespace) -> None:
    from research_rag.config import QDRANT_COLLECTION
    from research_rag.indexing.vector_store import get_qdrant_client

    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        print(f"Dropped vector store collection '{QDRANT_COLLECTION}'.")
    else:
        print("No vector store collection to drop.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research_rag", description="Local offline research paper RAG system")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="Search and download papers for a topic")
    p_download.add_argument("topic", help="Topic name (used as the storage folder)")
    p_download.add_argument("--query", default=None, help="Search query (defaults to topic name)")
    p_download.add_argument("--max-results", type=int, default=10)
    p_download.add_argument("--tier", type=int, choices=(1, 2, 3), default=None,
                            help="Force a tier (1 core / 2 supporting / 3 adjacent) for this batch")
    p_download.add_argument("--gate", action="store_true",
                            help="LLM relevance-gate candidates against the taxonomy before downloading")
    p_download.set_defaults(func=cmd_download)

    p_extract = sub.add_parser("extract", help="Extract text/metadata from downloaded PDFs")
    p_extract.add_argument("topic")
    p_extract.set_defaults(func=cmd_extract)

    p_classify = sub.add_parser("classify", help="Summarize and classify extracted papers")
    p_classify.add_argument("topic")
    p_classify.set_defaults(func=cmd_classify)

    p_index = sub.add_parser("index", help="Add classified papers to the vector index")
    p_index.add_argument("topic")
    p_index.set_defaults(func=cmd_index)

    p_pipeline = sub.add_parser("pipeline", help="Run download+extract+classify+index for a topic")
    p_pipeline.add_argument("topic")
    p_pipeline.add_argument("--query", default=None)
    p_pipeline.add_argument("--max-results", type=int, default=10)
    p_pipeline.add_argument("--tier", type=int, choices=(1, 2, 3), default=None,
                            help="Force a tier for this batch (default: derive from field)")
    p_pipeline.add_argument("--gate", action="store_true",
                            help="LLM relevance-gate candidates against the taxonomy before downloading")
    p_pipeline.set_defaults(func=cmd_pipeline)

    p_query = sub.add_parser("query", help="Ask a question over the indexed corpus")
    p_query.add_argument("question")
    p_query.set_defaults(func=cmd_query)

    p_tax = sub.add_parser("taxonomy", help="Print the current field tree")
    p_tax.set_defaults(func=cmd_taxonomy)

    p_stats = sub.add_parser(
        "pipeline_stats", help="Show corpus size, disk usage, tree shape, tiers, embedding status"
    )
    p_stats.set_defaults(func=cmd_pipeline_stats)

    p_cons = sub.add_parser("consolidate_taxonomy", help="Merge semantically similar sibling nodes")
    p_cons.set_defaults(func=cmd_consolidate)

    p_rt = sub.add_parser("reset_taxonomy", help="Reset taxonomy to the 7 seeded fields")
    p_rt.set_defaults(func=cmd_reset_taxonomy)

    p_rbt = sub.add_parser(
        "rebuild_taxonomy",
        help="Repopulate the taxonomy tree from already-classified records (no re-download/LLM)",
    )
    p_rbt.set_defaults(func=cmd_rebuild_taxonomy)

    p_rc = sub.add_parser(
        "reclassify",
        help="Re-run classification over the whole corpus from a fresh taxonomy (improved classifier)",
    )
    p_rc.set_defaults(func=cmd_reclassify)

    p_ri = sub.add_parser("reset_index", help="Drop the local vector store collection")
    p_ri.set_defaults(func=cmd_reset_index)

    p_health = sub.add_parser(
        "healthcheck", help="Verify Ollama is reachable and required models are present"
    )
    p_health.set_defaults(func=cmd_healthcheck)

    p_snow = sub.add_parser(
        "snowball", help="Expand the corpus by following citations (LLM relevance-judged)"
    )
    p_snow.add_argument("--tier", type=int, choices=(1, 2, 3), default=1,
                        help="Ingest papers up to this tier (1 core / 2 +supporting / 3 +adjacent)")
    p_snow.add_argument("--max-papers", type=int, default=2000,
                        help="Stop after ingesting this many papers; 0 = unlimited "
                             "(run until the citation queue empties or you press Ctrl+C)")
    p_snow.add_argument("--refs", type=int, default=50,
                        help="References to pull per paper")
    p_snow.add_argument("--dry-run", action="store_true",
                        help="Judge relevance and report, but download/ingest nothing")
    p_snow.add_argument("--nightly", action="store_true",
                        help="Batch checkpoint, resumable, CPU-throttle 09:00-22:00, log + notify")
    p_snow.set_defaults(func=cmd_snowball)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
