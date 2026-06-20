"""Snowball expansion: grow the corpus by following citations.

Starting from the papers already in the corpus, fetch each one's reference
list (via Semantic Scholar), ask the local LLM whether each referenced paper
is relevant to the quantum-radar taxonomy (and which top-level field it fits),
ingest the relevant ones that have a whitelisted open-access PDF, then recurse
into their references -- breadth-first -- until a paper budget is hit.

Relevance + tier are judged against the *tree descriptors*, so the expansion
stays on-topic without a human in the loop. Honour ``--dry-run`` to see what
would be ingested without downloading anything.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import asdict

from research_rag.classifier import classify_paper, summarize_paper
from research_rag.classifier.relevance import judge_relevance
from research_rag.classifier.seed_taxonomy import tier_for_field
from research_rag.config import SEMANTIC_SCHOLAR_API_KEY, SNOWBALL_REFS_PER_PAPER
from research_rag.downloader import download_candidates, semantic_scholar_client
from research_rag.downloader.corpus import known_ids
from research_rag.extractor import extract_paper
from research_rag.indexing import add_paper_to_index
from research_rag.llm import generate
from research_rag.storage import all_extracted_json_paths, extracted_json_path, raw_pdf_path

from .nightly import NightlyController

SNOWBALL_TOPIC = "snowball"


def _ingest_candidate(cand_dict: dict, tier: int) -> dict | None:
    """Download + extract + classify + index one candidate. Returns its
    classification result, or None if it could not be fetched."""
    metas = download_candidates([cand_dict], topic=SNOWBALL_TOPIC, tier=tier)
    if not metas:
        return None
    paper_id = metas[0]["paper_id"]
    pdf_path = raw_pdf_path(SNOWBALL_TOPIC, paper_id)
    record = extract_paper(pdf_path, SNOWBALL_TOPIC, paper_id)
    summary = summarize_paper(record)
    result = classify_paper(record, summary)
    record["summary"] = summary
    record["field"] = result["field"]
    record["subfield"] = result["subfield"]
    record["subsubfield"] = result["subsubfield"]
    # Tier derives from the authoritative classified field, NOT the provisional
    # gate-time tier carried in via the download sidecar. A candidate judged
    # relevant-but-unmapped is gated at the run's --tier and would otherwise be
    # stored with that tier (e.g. a tier-2/3 paper permanently labeled tier 1).
    record["tier"] = result["tier"]
    extracted_json_path(SNOWBALL_TOPIC, paper_id).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    add_paper_to_index(record, result["field"], result["subfield"], result["subsubfield"])
    return result


def _corpus_seed_ids() -> list[str]:
    ids = []
    for p in all_extracted_json_paths():
        try:
            ids.append(json.loads(p.read_text(encoding="utf-8"))["paper_id"])
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return ids


def snowball(
    tier: int,
    max_papers: int,
    dry_run: bool = False,
    refs_per_paper: int = SNOWBALL_REFS_PER_PAPER,
    nightly: NightlyController | None = None,
) -> dict:
    """Citation-driven corpus expansion. See module docstring."""
    nc = nightly or NightlyController("snowball", enabled=False)
    known_pids, _ = known_ids()
    seeds = _corpus_seed_ids()

    ckpt = nc.load_checkpoint() if (nc.enabled and not dry_run) else None
    if ckpt:
        queue = deque(ckpt.get("queue", []))
        seen = set(ckpt.get("seen", []))
        ingested = int(ckpt.get("ingested", 0))
        nc.log(f"Resuming: {ingested} ingested, {len(queue)} queued.")
    else:
        queue = deque(seeds)
        seen = set(known_pids) | set(seeds)
        ingested = 0

    would_ingest = 0
    decisions: list[dict] = []
    processed = 0

    # max_papers <= 0 means "run until the citation queue empties or the user
    # stops with Ctrl+C" -- i.e. limitless ingestion.
    unlimited = max_papers <= 0
    cap = "unlimited" if unlimited else max_papers

    nc.log(
        f"Snowball start: tier<={tier}, max_papers={cap}, "
        f"dry_run={dry_run}, seeds={len(seeds)}."
    )
    if not SEMANTIC_SCHOLAR_API_KEY:
        nc.log(
            "WARNING: no Semantic Scholar API key set -- reference fetching uses "
            "the throttled keyless pool and will be slow / 429-heavy. Set "
            "SEMANTIC_SCHOLAR_API_KEY (or data/s2_api_key.txt) for reliable snowball."
        )
    try:
        while queue and (unlimited or ingested < max_papers):
            pid = queue.popleft()
            ref = semantic_scholar_client.paper_ref(pid)
            if not ref:
                # Distinguish "id type we can't follow" from "no references":
                # without this log an unattended run silently snowballs from
                # only a subset of seeds/candidates and no one ever knows.
                nc.log(f"skip: cannot resolve a Semantic Scholar ref for id {pid!r}")
                continue
            t0 = time.monotonic()
            refs = semantic_scholar_client.get_references(ref, limit=refs_per_paper)
            for cand in refs:
                if not unlimited and ingested >= max_papers:
                    break
                if cand.paper_id in seen:
                    continue
                seen.add(cand.paper_id)
                relevant, field = judge_relevance(cand.title, cand.abstract)
                if not relevant:
                    continue
                # Unknown field -> assume in-scope for this run; classify_paper
                # assigns the authoritative field/tier after ingest.
                ctier = tier_for_field(field) if field else tier
                if ctier > tier:
                    continue
                if dry_run:
                    would_ingest += 1
                    decisions.append({"title": cand.title, "field": field, "tier": ctier})
                    nc.log(f"[dry-run] would ingest: {cand.title[:70]} -> {field} (tier {ctier})")
                    continue
                result = _ingest_candidate(asdict(cand), ctier)
                if result is not None:
                    ingested += 1
                    queue.append(cand.paper_id)
                    nc.log(
                        f"Ingested {ingested}/{cap}: {cand.title[:70]} -> "
                        f"{result['field']} (tier {ctier})"
                    )
            processed += 1
            nc.throttle(time.monotonic() - t0)
            if not dry_run and processed % nc.batch_size == 0:
                nc.save_checkpoint(
                    {"tier": tier, "queue": list(queue), "seen": list(seen), "ingested": ingested}
                )
                nc.log(f"Checkpoint: {ingested} ingested, {len(queue)} queued.")
    except KeyboardInterrupt:
        if not dry_run:
            nc.save_checkpoint(
                {"tier": tier, "queue": list(queue), "seen": list(seen), "ingested": ingested}
            )
            nc.log("Interrupted; checkpoint saved (resume with the same command).")
        return {"ingested": ingested, "would_ingest": would_ingest, "interrupted": True}
    except Exception as exc:
        # A backend error that outlived the LLM/S2 client retries (e.g. Ollama
        # stays down, disk full) must NOT vaporize a multi-hour unattended run:
        # checkpoint what we have, log it, and stop cleanly so a plain re-run
        # resumes from here instead of starting over with a raw traceback.
        if not dry_run:
            nc.save_checkpoint(
                {"tier": tier, "queue": list(queue), "seen": list(seen), "ingested": ingested}
            )
        nc.log(
            f"Stopped on {type(exc).__name__}: {exc}. Checkpoint saved; "
            "resume with the same command once the cause is cleared."
        )
        return {"ingested": ingested, "would_ingest": would_ingest, "interrupted": True}

    if not dry_run:
        nc.clear_checkpoint()
        nc.notify("Snowball complete", f"Tier {tier}: ingested {ingested} papers.")
    nc.log(f"Snowball done: ingested={ingested}, would_ingest={would_ingest}.")
    return {
        "ingested": ingested,
        "would_ingest": would_ingest,
        "tier": tier,
        "dry_run": dry_run,
        "decisions": decisions[:25],
    }
