"""Top-down hierarchical query routing with LlamaIndex RouterQueryEngine.

Retrieval enhancements:
  1. Hybrid BM25 + vector — keyword search catches exact terms embeddings miss.
  2. Query expansion — LLM rephrases question in 2 ways before BM25 retrieval.
     Disable with QUERY_EXPANSION=0.
  3. Multi-branch routing — flat vector search runs alongside the hierarchical
     router so cross-field papers are not missed.
  4. Router cache — the nested engine tree is rebuilt only when taxonomy.json
     changes on disk (mtime check), not on every query call.
  5. Inline citations — synthesis uses [1][2] markers so every claim is
     traceable to a specific paper in the citation list.
  6. Verbose routing suppressed by default — set VERBOSE_ROUTING=1 to see
     the per-level selector decisions (useful for debugging routing failures).
"""
from __future__ import annotations

import os
from collections import defaultdict

from llama_index.core import Settings
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.ollama import Ollama

from research_rag.classifier.taxonomy import load_taxonomy
from research_rag.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, TAXONOMY_PATH
from research_rag.indexing import get_index
from research_rag.llm import generate

TOP_K = 6           # vector chunks from the hierarchical router
TOP_K_FLAT = 4      # extra chunks from the flat multi-branch search
TOP_K_BM25 = 3      # extra papers surfaced by BM25 (2 chunks each)
TOP_K_FINAL = 9     # max chunks passed to the final synthesizer
_MAX_DESC_CHARS = 500
_RRF_K = 60
_MIN_RELEVANCE_SCORE = 0.30  # below this, no relevant papers found

_QUERY_EXPANSION  = os.environ.get("QUERY_EXPANSION",  "1") == "1"
_VERBOSE_ROUTING  = os.environ.get("VERBOSE_ROUTING",  "0") == "1"

_WINDOW_POSTPROCESSOR = MetadataReplacementPostProcessor(target_metadata_key="window")
_TIER_WEIGHTS: dict[int, float] = {1: 1.3, 2: 1.0, 3: 0.9}

_llm: Ollama | None = None

# Router cache: rebuilt only when taxonomy.json mtime changes.
_router_cache: dict = {"engine": None, "mtime": -1.0}


# --------------------------------------------------------------------------- #
# LLM singleton
# --------------------------------------------------------------------------- #
def get_llm() -> Ollama:
    global _llm
    if _llm is None:
        _llm = Ollama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=600.0,
        )
        Settings.llm = _llm
    return _llm


# --------------------------------------------------------------------------- #
# Query expansion
# --------------------------------------------------------------------------- #
def _expand_query(question: str) -> list[str]:
    """Return [original, alt1, alt2] via one LLM call."""
    if not _QUERY_EXPANSION:
        return [question]
    try:
        raw = generate(
            f"Rephrase the following research question in 2 different ways to "
            f"improve search recall. Use different terminology where possible.\n"
            f"Question: {question}\n"
            "Reply with exactly 2 lines — one rephrasing per line, no labels.",
            system="Output only the 2 rephrased questions, nothing else.",
        )
        alts = [
            ln.strip().strip('"').strip("'").lstrip("12. )-")
            for ln in raw.splitlines()
            if ln.strip()
        ][:2]
        return [question] + [a for a in alts if a and a.lower() != question.lower()]
    except Exception:
        return [question]


# --------------------------------------------------------------------------- #
# Multi-turn: condense follow-up against conversation history
# --------------------------------------------------------------------------- #
def _condense_question(question: str, history: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up as a standalone question (last 3 turns only)."""
    if not history:
        return question
    history_text = "\n".join(
        f"Q: {q}\nA: {a[:300]}{'...' if len(a) > 300 else ''}"
        for q, a in history[-3:]
    )
    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Rewrite the follow-up as a fully self-contained question that can be "
        "understood without the conversation above. "
        "Reply with ONLY the rewritten question, nothing else."
    )
    try:
        condensed = generate(prompt).strip().strip('"').strip("'")
        return condensed if condensed else question
    except Exception:
        return question


# --------------------------------------------------------------------------- #
# Inline-citation synthesis  (replaces llama-index's default synthesizer)
# --------------------------------------------------------------------------- #
def _synthesize_with_citations(question: str, nodes: list) -> str:
    """Generate an answer that cites sources inline as [1], [2], etc.

    Builds a numbered context block from the merged node list and calls
    generate() directly so we fully control the prompt format.
    """
    parts: list[str] = []
    for i, node in enumerate(nodes, 1):
        meta = node.node.metadata
        title = meta.get("title", "Unknown")
        year  = meta.get("year", "")
        snippet = node.node.get_content()[:500].strip().replace("\n", " ")
        parts.append(f"[{i}] {title} ({year})\n{snippet}")

    context = "\n\n".join(parts)
    prompt = (
        "You are a research assistant. Using ONLY the numbered paper excerpts "
        "below, answer the question concisely and accurately. "
        "Cite sources inline as [1], [2], etc. after each claim.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (with inline citations):"
    )
    try:
        return generate(prompt).strip()
    except Exception as exc:
        return f"(synthesis failed: {exc})"


# --------------------------------------------------------------------------- #
# BM25 supplementary retrieval
# --------------------------------------------------------------------------- #
def _bm25_augment(queries: list[str], known_ids: set[str]) -> list:
    """BM25-search the corpus, fetch vector chunks for newly surfaced papers."""
    from research_rag.query.bm25_index import get_bm25
    bm25 = get_bm25()
    if bm25.size == 0:
        return []
    candidate_ids = [
        pid for pid in bm25.search(queries, top_k=TOP_K_BM25 + len(known_ids))
        if pid not in known_ids
    ][:TOP_K_BM25]
    if not candidate_ids:
        return []
    index = get_index()
    extra: list = []
    for pid in candidate_ids:
        try:
            retriever = index.as_retriever(
                similarity_top_k=2,
                filters=MetadataFilters(
                    filters=[ExactMatchFilter(key="paper_id", value=pid)]
                ),
            )
            extra.extend(retriever.retrieve(queries[0]))
        except Exception:
            pass
    return extra


# --------------------------------------------------------------------------- #
# Flat multi-branch retrieval
# --------------------------------------------------------------------------- #
def _flat_retrieve(question: str) -> list:
    """Flat vector search over the whole corpus — catches cross-field papers."""
    try:
        nodes = get_index().as_retriever(similarity_top_k=TOP_K_FLAT).retrieve(question)
        return _WINDOW_POSTPROCESSOR.postprocess_nodes(nodes)
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# RRF merge with tier weighting
# --------------------------------------------------------------------------- #
def _tier_weight(node) -> float:
    tier = node.node.metadata.get("tier")
    return _TIER_WEIGHTS.get(int(tier), 1.0) if tier is not None else 1.0


def _rrf_merge(lists_of_nodes: list[list], top_k: int) -> list:
    """Reciprocal Rank Fusion, deduplicated by node_id, tier-weighted."""
    scores: dict[str, float] = defaultdict(float)
    nodes_by_id: dict[str, object] = {}
    for node_list in lists_of_nodes:
        for rank, node in enumerate(node_list):
            nid = node.node.node_id
            scores[nid] += 1.0 / (_RRF_K + rank + 1)
            nodes_by_id.setdefault(nid, node)
    for nid, node in nodes_by_id.items():
        scores[nid] *= _tier_weight(node)
    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [nodes_by_id[k] for k in ranked[:top_k]]


# --------------------------------------------------------------------------- #
# Router construction with mtime-based cache
# --------------------------------------------------------------------------- #
def _taxonomy_mtime() -> float:
    try:
        return TAXONOMY_PATH.stat().st_mtime
    except OSError:
        return 0.0


def reset_router_cache() -> None:
    """Force a full router rebuild on the next query (e.g. after reclassify)."""
    _router_cache["engine"] = None
    _router_cache["mtime"] = -1.0


def _desc(text: str) -> str:
    return (text or "").strip().replace("\n", " ")[:_MAX_DESC_CHARS]


def _leaf_filters(field: str, subfield: str, subsub: str) -> MetadataFilters:
    return MetadataFilters(filters=[
        ExactMatchFilter(key="field",       value=field),
        ExactMatchFilter(key="subfield",    value=subfield),
        ExactMatchFilter(key="subsubfield", value=subsub),
    ])


def _engine_for_tools(tools: list[QueryEngineTool]):
    if len(tools) == 1:
        return tools[0].query_engine
    return RouterQueryEngine.from_defaults(
        selector=LLMMultiSelector.from_defaults(llm=get_llm()),
        query_engine_tools=tools,
        llm=get_llm(),
        select_multi=True,
        verbose=_VERBOSE_ROUTING,
    )


def _build_router():
    """Return the cached nested router, rebuilding only if taxonomy changed."""
    mtime = _taxonomy_mtime()
    if _router_cache["engine"] is not None and _router_cache["mtime"] == mtime:
        return _router_cache["engine"]

    index = get_index()
    get_llm()
    taxonomy = load_taxonomy()

    field_tools: list[QueryEngineTool] = []
    for field, fdata in taxonomy.get("fields", {}).items():
        sub_tools: list[QueryEngineTool] = []
        for subfield, sdata in (fdata.get("subfields") or {}).items():
            leaf_tools: list[QueryEngineTool] = []
            for subsub, ssdata in (sdata.get("subsubfields") or {}).items():
                if not ssdata.get("paper_ids"):
                    continue
                qe = index.as_query_engine(
                    similarity_top_k=TOP_K,
                    vector_store_query_mode=VectorStoreQueryMode.MMR,
                    vector_store_kwargs={"mmr_threshold": 0.2},
                    filters=_leaf_filters(field, subfield, subsub),
                    node_postprocessors=[_WINDOW_POSTPROCESSOR],
                    llm=get_llm(),
                )
                leaf_tools.append(QueryEngineTool.from_defaults(
                    query_engine=qe,
                    name=f"{field}__{subfield}__{subsub}"[:64],
                    description=_desc(ssdata.get("descriptor", subsub)),
                ))
            if not leaf_tools:
                continue
            sub_tools.append(QueryEngineTool.from_defaults(
                query_engine=_engine_for_tools(leaf_tools),
                name=f"{field}__{subfield}"[:64],
                description=_desc(sdata.get("descriptor", subfield)),
            ))
        if not sub_tools:
            continue
        field_tools.append(QueryEngineTool.from_defaults(
            query_engine=_engine_for_tools(sub_tools),
            name=field[:64],
            description=_desc(fdata.get("descriptor", field)),
        ))

    engine = _engine_for_tools(field_tools) if field_tools else None
    _router_cache["engine"] = engine
    _router_cache["mtime"] = mtime
    return engine


# --------------------------------------------------------------------------- #
# Citations helpers
# --------------------------------------------------------------------------- #
def _paper_url(paper_id: str) -> str | None:
    if paper_id.startswith("arxiv_"):
        base = paper_id[len("arxiv_"):].split("v")[0]
        return f"https://arxiv.org/abs/{base}"
    if paper_id.startswith("s2_"):
        return f"https://www.semanticscholar.org/paper/{paper_id[3:]}"
    return None


def _citations_from_nodes(nodes) -> list[dict]:
    citations: list[dict] = []
    seen: set[str] = set()
    for n in nodes:
        meta = n.node.metadata
        pid  = meta.get("paper_id")
        if pid and pid not in seen:
            seen.add(pid)
            entry: dict = {
                "title":       meta.get("title", "Unknown"),
                "authors":     meta.get("authors", ""),
                "year":        meta.get("year"),
                "field":       meta.get("field"),
                "subfield":    meta.get("subfield"),
                "subsubfield": meta.get("subsubfield"),
            }
            url = _paper_url(pid)
            if url:
                entry["url"] = url
            citations.append(entry)
    return citations


def _best_score(nodes: list) -> float:
    scores = [n.score for n in nodes if n.score is not None]
    return max(scores, default=0.0)


def _flat_answer(question: str) -> dict:
    """Last-resort flat search with inline-citation synthesis."""
    try:
        nodes = get_index().as_retriever(similarity_top_k=TOP_K).retrieve(question)
    except Exception:
        nodes = []
    if not nodes:
        return {"answer": "(no papers indexed yet)", "routing": "flat (fallback)", "citations": []}
    return {
        "answer":    _synthesize_with_citations(question, nodes),
        "routing":   "flat (fallback)",
        "citations": _citations_from_nodes(nodes),
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def answer_question(
    question: str,
    history: list[tuple[str, str]] | None = None,
) -> dict:
    """Answer via hierarchical routing + BM25 + flat search, merged with RRF,
    synthesised with inline [N] citations. Router is cached between calls."""
    engine = _build_router()
    if engine is None:
        return {"answer": "(no papers indexed yet)", "routing": "none", "citations": []}

    # Multi-turn: rewrite follow-up as standalone before routing
    if history:
        question = _condense_question(question, history)

    # Query expansion for BM25 recall
    queries = _expand_query(question)

    # Hierarchical routing — we want source_nodes; the router's text is discarded
    router_nodes: list = []
    router_answer: str | None = None  # kept as emergency fallback only
    try:
        resp = engine.query(question)
        router_nodes  = getattr(resp, "source_nodes", []) or []
        router_answer = str(resp)
    except Exception as exc:
        print(f"Router failed ({type(exc).__name__}: {exc}); continuing with other sources.")

    # Flat multi-branch search
    flat_nodes = _flat_retrieve(question)

    # BM25 supplementation for papers not yet covered
    covered_ids = {
        n.node.metadata.get("paper_id")
        for n in router_nodes + flat_nodes
        if hasattr(n, "node")
    }
    bm25_nodes = _bm25_augment(queries, covered_ids)

    # RRF merge (tier-weighted)
    all_nodes = _rrf_merge([router_nodes, flat_nodes, bm25_nodes], TOP_K_FINAL)

    if not all_nodes:
        return _flat_answer(question) if not router_answer else {
            "answer": router_answer, "routing": "hierarchical", "citations": [],
        }

    # Relevance gate
    if _best_score(all_nodes) < _MIN_RELEVANCE_SCORE:
        return {
            "answer": (
                "No sufficiently relevant papers were found in the corpus for "
                "this question. Try a different phrasing or expand the corpus "
                "with `pipeline` on a related topic."
            ),
            "routing":   "none (below relevance threshold)",
            "citations": [],
        }

    # Routing label
    router_ids = {n.node.metadata.get("paper_id") for n in router_nodes if hasattr(n, "node")}
    flat_ids   = {n.node.metadata.get("paper_id") for n in flat_nodes   if hasattr(n, "node")}
    label_parts = []
    if router_nodes:               label_parts.append("hierarchical")
    if flat_ids - router_ids:      label_parts.append("flat")
    if bm25_nodes:                 label_parts.append("bm25")
    routing = "+".join(label_parts) if label_parts else "flat (fallback)"

    # Synthesise with inline citations — always use our controlled prompt
    try:
        answer = _synthesize_with_citations(question, all_nodes)
    except Exception as exc:
        print(f"Citation synthesis failed ({exc}); falling back to router answer.")
        answer = router_answer or "(synthesis failed)"

    return {
        "answer":    answer,
        "routing":   routing,
        "citations": _citations_from_nodes(all_nodes),
    }
