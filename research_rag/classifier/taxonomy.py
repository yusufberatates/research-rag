"""Four-level hierarchical taxonomy: Main Field -> Subfield -> Sub-subfield ->
Papers.

Shape of data/taxonomy/taxonomy.json::

    {
      "fields": {
        "<main field>": {
          "descriptor": "2-3 paragraphs",
          "subfields": {
            "<subfield>": {
              "descriptor": "2-3 paragraphs",
              "subsubfields": {
                "<sub-subfield>": {
                  "descriptor": "2-3 paragraphs",
                  "paper_ids": [...]
                }
              }
            }
          }
        }
      }
    }

Every non-leaf node (field, subfield, sub-subfield) carries a descriptor;
papers attach at the sub-subfield (leaf) level via ``paper_ids``.

Classification is top-down: the LLM first picks a main field (preferring the
seven seeded ones), then a subfield within it, then a sub-subfield -- creating
a new node only when nothing existing fits. The taxonomy never requires
re-reading papers: each choice is made against short slices of the existing
node descriptors plus the new paper's summary.
"""
from __future__ import annotations

import json
import re
import threading

from research_rag.config import TAXONOMY_PATH
from research_rag.llm import generate

from .seed_taxonomy import seeded_tree, tier_for_field

_lock = threading.Lock()

DEFAULT_FIELD = "supporting_quantum_optics"
DEFAULT_SUBFIELD = "uncategorized"
DEFAULT_SUBSUBFIELD = "uncategorized"

# How much of each descriptor to show the LLM when choosing among siblings.
_CHOICE_DESCRIPTOR_CHARS = 320

_PLACEHOLDER_VALUES = {"", "none", "n/a", "na", "unknown", "null", "new"}

# A real research sub-area name is a few words. The small local model sometimes
# ignores "reply with ONLY the name" and emits a whole sentence, which would
# otherwise be slugged into a 90+ char garbage node (observed in the wild:
# "quantum_estimation_newtons_ninth_law_is_not_relevant_here..."). Cap names
# minted for NEW nodes so one rambling reply can't pollute the tree.
_MAX_NEW_NAME_WORDS = 6
_MAX_NEW_NAME_CHARS = 64


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_taxonomy() -> dict:
    if TAXONOMY_PATH.exists():
        data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
        if data.get("fields"):
            return data
    return seeded_tree()


def save_taxonomy(taxonomy: dict) -> None:
    TAXONOMY_PATH.write_text(
        json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def ensure_seeded() -> dict:
    """Make sure the on-disk taxonomy at least contains the seeded fields."""
    taxonomy = load_taxonomy()
    save_taxonomy(taxonomy)
    return taxonomy


def reset_to_seed() -> dict:
    """Overwrite the taxonomy with a fresh seeded tree (destructive)."""
    taxonomy = seeded_tree()
    save_taxonomy(taxonomy)
    return taxonomy


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "uncategorized"


def _sane_new_name(raw: str) -> str:
    """Slug a model-proposed NEW node name, clamped to a sane length.

    Used only when MINTING a new node (never for matching existing ones, so
    `_match_existing` still sees the full text). Keeps at most the first few
    words / chars so a verbose, instruction-ignoring local-model reply can't
    create an absurdly long leaf name."""
    slug = _slug(raw)
    words = slug.split("_")
    if len(words) > _MAX_NEW_NAME_WORDS:
        slug = "_".join(words[:_MAX_NEW_NAME_WORDS])
    if len(slug) > _MAX_NEW_NAME_CHARS:
        slug = slug[:_MAX_NEW_NAME_CHARS].rstrip("_")
    return slug or "uncategorized"


def _truncate(text: str, limit: int = _CHOICE_DESCRIPTOR_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _options_block(nodes: dict[str, dict]) -> str:
    if not nodes:
        return "(none yet)"
    return "\n".join(
        f"- {name}: {_truncate(node.get('descriptor', ''))}"
        for name, node in nodes.items()
    )


def _match_existing(choice: str, nodes: dict[str, dict]) -> str | None:
    """Map a free-text LLM choice onto an existing node name, if any."""
    c = _slug(choice)
    if c in nodes:
        return c
    for name in nodes:
        if c == _slug(name) or c in _slug(name) or _slug(name) in c:
            return name
    return None


def _choose_node(
    level_label: str,
    nodes: dict[str, dict],
    title: str,
    summary: str,
    *,
    allow_new: bool,
    prefer_existing: bool = False,
) -> tuple[str, bool]:
    """Ask the LLM to pick an existing sibling node or propose a new one.

    Returns (node_name, is_new).

    ``prefer_existing`` biases the choice strongly toward reusing one of the
    existing options. It is used at the main-field level, where the seven
    seeded fields are meant to be the stable top-level structure and new
    top-level fields should be a last resort; subfields/sub-subfields instead
    favour accuracy (create a new well-named category rather than over-reuse).
    """
    new_clause = (
        "If none of the existing options is a genuine fit, reply exactly "
        "'NEW: <short_snake_case_name>'. Name it as a GENERAL research "
        "sub-area (e.g. 'squeezed_light_sources'), never after this one "
        "paper's specific title, device, or method."
        if allow_new
        else "You MUST pick one of the names above; do not invent a new one."
    )
    if prefer_existing:
        reuse_clause = (
            "Strongly prefer reusing one of the existing options: pick the "
            "closest existing one unless the paper genuinely does not belong "
            "under any of them. Only create a new category as a last resort."
        )
    else:
        reuse_clause = (
            "Reuse an existing option only when the paper genuinely belongs to "
            "it; if it is merely loosely or tangentially related, create a new, "
            "well-named category instead."
        )
    prompt = (
        f"Existing {level_label} options:\n{_options_block(nodes)}\n\n"
        f"New paper title: {title}\n"
        f"New paper summary: {summary}\n\n"
        f"Choose the single best-fitting {level_label} for this paper. "
        f"{reuse_clause} {new_clause}\n"
        "Reply with ONLY the chosen name (or the 'NEW: ...' form), nothing else."
    )
    raw = generate(prompt).strip()

    m = re.search(r"NEW\s*:\s*(.+)", raw, re.IGNORECASE)
    if m and allow_new:
        name = _sane_new_name(m.group(1))
        if name not in _PLACEHOLDER_VALUES:
            existing = _match_existing(name, nodes)
            return (existing, False) if existing else (name, True)

    # Plain choice: try to map onto an existing node.
    cleaned = raw.splitlines()[0] if raw else ""
    existing = _match_existing(cleaned, nodes)
    if existing:
        return existing, False
    if allow_new:
        new_name = _sane_new_name(cleaned)
        if new_name not in _PLACEHOLDER_VALUES:
            return new_name, True
    # Fall back to first existing option, or a default leaf name.
    if nodes:
        return next(iter(nodes)), False
    return (DEFAULT_SUBFIELD if level_label == "subfield" else DEFAULT_SUBSUBFIELD), True


def _generate_descriptor(name: str, level_label: str, summary: str) -> str:
    prompt = (
        f"Write a 2-3 paragraph descriptor for a research {level_label} called "
        f"'{name.replace('_', ' ')}', suitable for routing future papers to it. "
        f"Base it on this representative paper summary:\n{summary}\n\n"
        "Describe what work belongs in this area. Respond with only the "
        "descriptor text."
    )
    return generate(prompt).strip() or name.replace("_", " ")


# --------------------------------------------------------------------------- #
# Classification (top-down)
# --------------------------------------------------------------------------- #
def classify_paper(record: dict, summary: str) -> dict:
    """Assign a paper to field -> subfield -> sub-subfield, creating nodes as
    needed, persist the taxonomy, and return the assigned path."""
    title = record.get("title", "")
    paper_id = record["paper_id"]

    with _lock:
        taxonomy = load_taxonomy()
        fields = taxonomy.setdefault("fields", {})

        # Level 1: main field. The seven seeded fields are a fixed top-level
        # structure: every paper must map to one of them (they include broad
        # catch-alls -- supporting_quantum_optics, critical_assessment_literature
        # -- so this is not lossy). New structure grows only below this level.
        # A small local model ignores soft "prefer existing" wording and
        # invents spurious fields, so we forbid new ones here outright.
        field, _ = _choose_node(
            "main field", fields, title, summary, allow_new=False, prefer_existing=True
        )
        fdata = fields.setdefault(field, {"descriptor": "", "subfields": {}})
        if not fdata.get("descriptor"):
            fdata["descriptor"] = _generate_descriptor(field, "field", summary)
        subfields = fdata.setdefault("subfields", {})

        # Level 2: subfield.
        subfield, sub_is_new = _choose_node(
            "subfield", subfields, title, summary, allow_new=True
        )
        sdata = subfields.setdefault(subfield, {"descriptor": "", "subsubfields": {}})
        if sub_is_new or not sdata.get("descriptor"):
            sdata["descriptor"] = _generate_descriptor(subfield, "subfield", summary)
        subsubfields = sdata.setdefault("subsubfields", {})

        # Level 3: sub-subfield (leaf, holds papers).
        subsub, ss_is_new = _choose_node(
            "sub-subfield", subsubfields, title, summary, allow_new=True
        )
        # A leaf that merely repeats its parent subfield name (the degenerate
        # "superconducting_qubits / superconducting_qubits" path) adds no
        # information; fold it into a generic "general" leaf until enough
        # papers arrive to justify a finer split.
        if _slug(subsub) == _slug(subfield):
            existing = _match_existing("general", subsubfields)
            subsub = existing or "general"
            ss_is_new = existing is None
        ssdata = subsubfields.setdefault(subsub, {"descriptor": "", "paper_ids": []})
        if ss_is_new or not ssdata.get("descriptor"):
            ssdata["descriptor"] = _generate_descriptor(
                subsub, "sub-subfield", summary
            )

        if paper_id not in ssdata["paper_ids"]:
            ssdata["paper_ids"].append(paper_id)

        save_taxonomy(taxonomy)

    return {
        "field": field,
        "subfield": subfield,
        "subsubfield": subsub,
        "tier": tier_for_field(field),
    }


# --------------------------------------------------------------------------- #
# Re-attachment: keep the tree in sync with already-classified records
# --------------------------------------------------------------------------- #
def attach_existing(record: dict) -> bool:
    """Ensure an already-classified paper is present in the taxonomy tree.

    A paper's field/subfield/sub-subfield are stored on its extracted record,
    but the tree can fall out of sync with those records -- most importantly
    after ``reset_taxonomy`` wipes the tree while the records keep their
    classification, so a resumed ``classify`` run skips them and never
    re-registers their ``paper_ids``. (That desync makes ``pipeline_stats``
    report every field empty and breaks query routing, since the descriptors
    those papers should populate no longer exist.)

    This re-creates any missing node along the stored path -- generating a
    descriptor from the record's stored summary, never re-reading the paper --
    and appends the paper_id at the leaf. It does NOT re-run the LLM
    classification choice, so it is cheap and deterministic. Returns True if
    the tree changed.
    """
    field = (record.get("field") or "").strip()
    paper_id = record.get("paper_id")
    if not field or not paper_id:
        return False
    subfield = (record.get("subfield") or DEFAULT_SUBFIELD).strip() or DEFAULT_SUBFIELD
    subsub = (record.get("subsubfield") or DEFAULT_SUBSUBFIELD).strip() or DEFAULT_SUBSUBFIELD
    summary = record.get("summary", "")

    with _lock:
        taxonomy = load_taxonomy()
        fields = taxonomy.setdefault("fields", {})
        changed = False

        fdata = fields.get(field)
        if fdata is None:
            fdata, changed = fields.setdefault(field, {"descriptor": "", "subfields": {}}), True
        if not fdata.get("descriptor"):
            fdata["descriptor"], changed = _generate_descriptor(field, "field", summary), True
        subfields = fdata.setdefault("subfields", {})

        sdata = subfields.get(subfield)
        if sdata is None:
            sdata, changed = subfields.setdefault(subfield, {"descriptor": "", "subsubfields": {}}), True
        if not sdata.get("descriptor"):
            sdata["descriptor"], changed = _generate_descriptor(subfield, "subfield", summary), True
        subsubfields = sdata.setdefault("subsubfields", {})

        ssdata = subsubfields.get(subsub)
        if ssdata is None:
            ssdata, changed = subsubfields.setdefault(subsub, {"descriptor": "", "paper_ids": []}), True
        if not ssdata.get("descriptor"):
            ssdata["descriptor"], changed = _generate_descriptor(subsub, "sub-subfield", summary), True

        paper_ids = ssdata.setdefault("paper_ids", [])
        if paper_id not in paper_ids:
            paper_ids.append(paper_id)
            changed = True

        if changed:
            save_taxonomy(taxonomy)
    return changed


# --------------------------------------------------------------------------- #
# Consolidation: merge semantically similar sibling nodes
# --------------------------------------------------------------------------- #
def _child_key(level: int) -> str | None:
    """Name of the child container for a node at the given level.

    level 0 = main fields, 1 = subfields, 2 = sub-subfields (leaves)."""
    return {0: "subfields", 1: "subsubfields"}.get(level)


def _merge_descriptors_text(name: str, level_label: str, texts: list[str]) -> str:
    """Regenerate one unified descriptor covering several merged nodes."""
    joined = "\n\n---\n\n".join(t for t in texts if t)
    prompt = (
        f"The following descriptions of overlapping research {level_label}s are "
        f"being merged into a single one called '{name.replace('_', ' ')}':\n\n"
        f"{joined}\n\n"
        "Write a single unified 2-3 paragraph descriptor that accurately covers "
        "everything above, suitable for routing future papers. Respond with only "
        "the descriptor text."
    )
    return generate(prompt).strip() or texts[0] if texts else name.replace("_", " ")


def _ask_merge_groups(level_label: str, nodes: dict[str, dict]) -> list[list[str]]:
    """Ask the LLM which sibling node names should be merged together.

    Returns a list of groups; each group is a list of >=2 existing names that
    should collapse into one. Names not in any group are left untouched.
    """
    if len(nodes) < 2:
        return []
    prompt = (
        f"Here are sibling {level_label} categories in a research taxonomy, "
        "with descriptors:\n"
        f"{_options_block(nodes)}\n\n"
        "Some of these may be semantically redundant or near-duplicates that "
        "should be merged into one category. Identify groups that should "
        "merge. Respond with one group per line in the form:\n"
        "  group: name_a, name_b\n"
        "Only include groups with two or more names that genuinely overlap. "
        "If nothing should merge, respond with exactly: NONE"
    )
    raw = generate(prompt).strip()
    if raw.upper().startswith("NONE"):
        return []

    groups: list[list[str]] = []
    valid = set(nodes)
    for line in raw.splitlines():
        line = re.sub(r"^\s*group\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        members = []
        for part in re.split(r"[,;]", line):
            match = _match_existing(part, nodes)
            if match and match in valid and match not in members:
                members.append(match)
        if len(members) >= 2:
            groups.append(members)
    return groups


def _merge_into(target: dict, source: dict, level: int) -> None:
    """Merge ``source`` node into ``target`` node in place."""
    ckey = _child_key(level)
    if ckey:  # non-leaf: merge child containers recursively
        tchildren = target.setdefault(ckey, {})
        for cname, cnode in (source.get(ckey) or {}).items():
            if cname in tchildren:
                _merge_into(tchildren[cname], cnode, level + 1)
            else:
                tchildren[cname] = cnode
    else:  # leaf: merge paper_ids
        tids = target.setdefault("paper_ids", [])
        for pid in source.get("paper_ids", []):
            if pid not in tids:
                tids.append(pid)


def _node_has_papers(node: dict, level: int) -> bool:
    ckey = _child_key(level)
    if not ckey:  # leaf
        return bool(node.get("paper_ids"))
    return any(
        _node_has_papers(child, level + 1) for child in (node.get(ckey) or {}).values()
    )


def _consolidate_level(nodes: dict[str, dict], level: int) -> dict[str, dict]:
    """Recursively consolidate a dict of sibling nodes (bottom-up)."""
    ckey = _child_key(level)
    # First consolidate children of each node.
    if ckey:
        for node in nodes.values():
            if node.get(ckey):
                node[ckey] = _consolidate_level(node[ckey], level + 1)

    level_label = {0: "main field", 1: "subfield", 2: "sub-subfield"}[level]
    # Only consider merging siblings that actually contain papers; empty
    # scaffolding (e.g. unused seeded top-level fields) is left intact.
    candidates = {n: node for n, node in nodes.items() if _node_has_papers(node, level)}
    groups = _ask_merge_groups(level_label, candidates) if len(candidates) >= 2 else []
    if not groups:
        return nodes

    merged_away: set[str] = set()
    for group in groups:
        group = [n for n in group if n not in merged_away]
        if len(group) < 2:
            continue
        canonical = group[0]
        descriptor_texts = [nodes[canonical].get("descriptor", "")]
        for other in group[1:]:
            if other in nodes and canonical in nodes:
                descriptor_texts.append(nodes[other].get("descriptor", ""))
                _merge_into(nodes[canonical], nodes[other], level)
                merged_away.add(other)
        # Regenerate the surviving node's descriptor to cover what it absorbed.
        nodes[canonical]["descriptor"] = _merge_descriptors_text(
            canonical, level_label, descriptor_texts
        )
    for name in merged_away:
        nodes.pop(name, None)
    return nodes


def consolidate_taxonomy() -> dict:
    """Merge semantically similar sibling nodes throughout the tree and
    persist the result. Returns a summary of what changed."""
    with _lock:
        taxonomy = load_taxonomy()
        before = _count_nodes(taxonomy)
        taxonomy["fields"] = _consolidate_level(taxonomy.get("fields", {}), 0)
        save_taxonomy(taxonomy)
        after = _count_nodes(taxonomy)
    return {"before": before, "after": after}


def paper_paths(taxonomy: dict | None = None) -> dict[str, tuple[str, str, str]]:
    """Map each paper_id to its authoritative (field, subfield, subsubfield)
    path in the (possibly just-consolidated) taxonomy."""
    taxonomy = taxonomy or load_taxonomy()
    out: dict[str, tuple[str, str, str]] = {}
    for field, fdata in taxonomy.get("fields", {}).items():
        for subfield, sdata in (fdata.get("subfields") or {}).items():
            for subsub, ssdata in (sdata.get("subsubfields") or {}).items():
                for pid in ssdata.get("paper_ids", []):
                    out[pid] = (field, subfield, subsub)
    return out


def _count_nodes(taxonomy: dict) -> dict:
    fields = taxonomy.get("fields", {})
    n_sub = n_ss = 0
    for f in fields.values():
        subs = f.get("subfields", {})
        n_sub += len(subs)
        for s in subs.values():
            n_ss += len(s.get("subsubfields", {}))
    return {"fields": len(fields), "subfields": n_sub, "subsubfields": n_ss}
