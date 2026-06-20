"""LLM relevance judgement against the taxonomy tree.

Shared by snowball expansion and the optional download-time relevance gate:
given a paper's title/abstract, decide whether it belongs in the corpus and,
if so, which top-level field it best fits. Judged against the *current* tree
descriptors so the decision tracks the taxonomy as it grows.
"""
from __future__ import annotations

import re

from research_rag.classifier.taxonomy import load_taxonomy
from research_rag.llm import generate


def _truncate(text: str, limit: int = 220) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _field_options(fields: dict) -> str:
    return "\n".join(
        f"- {name}: {_truncate(f.get('descriptor', ''))}" for name, f in fields.items()
    )


def judge_relevance(title: str, abstract: str) -> tuple[bool, str | None]:
    """Return (relevant, best_field_name). ``best_field_name`` is an existing
    top-level field, or None when the paper is off-topic."""
    fields = load_taxonomy().get("fields", {})
    prompt = (
        "You are curating a research corpus on quantum radar, quantum "
        "illumination, quantum sensing, and the ENABLING technologies they "
        "depend on -- e.g. parametric amplifiers (JPA/JTWPA), squeezed and "
        "entangled microwave/optical sources, single-photon and homodyne "
        "detectors, Rydberg/atomic RF receivers, microwave-optical "
        "transducers, superconducting circuits, and detection/estimation "
        "theory.\n\n"
        "Top-level fields:\n"
        f"{_field_options(fields)}\n\n"
        f"Candidate paper title: {title}\n"
        f"Candidate abstract: {_truncate(abstract, 600)}\n\n"
        "KEEP the paper if it concerns any of the above topics OR a component, "
        "device, method, or theory such systems rely on. Mark it off-topic "
        "ONLY if it is clearly from an unrelated area (e.g. pure mathematics, "
        "biology, cosmology, finance, unrelated machine learning).\n"
        "Respond in EXACTLY this format:\n"
        "RELEVANT: <yes|no>\n"
        "FIELD: <best-fitting field name from the list, or NONE>"
    )
    raw = generate(prompt)
    if not re.search(r"RELEVANT:\s*yes", raw, re.IGNORECASE):
        return False, None

    # Relevant: keep it. Field is best-effort -- a paper the model judged
    # relevant is NOT dropped just because its field guess didn't map cleanly.
    m = re.search(r"FIELD:\s*(.+)", raw)
    choice = (m.group(1).strip().lower() if m else "")
    if choice and choice != "none":
        cn = re.sub(r"[^a-z0-9]+", "_", choice).strip("_")
        for name in fields:
            if cn == name or cn in name or name in cn:
                return True, name
    return True, None
