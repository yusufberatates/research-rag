"""Top-down hierarchical query routing with LlamaIndex RouterQueryEngine.

A nested tree of query engines mirrors the taxonomy:

    RouterQueryEngine(main fields)
        -> RouterQueryEngine(subfields of chosen field)
            -> RouterQueryEngine(sub-subfields of chosen subfield)
                -> vector query engine filtered to that leaf

At each level an LLM selector reads the node descriptors and picks 1-3
branches; an actual vector search only happens at the leaves. Only nodes that
contain papers are turned into tools, so empty seeded branches are skipped.

If routing fails (e.g. the local model emits an unparseable selection) or
returns nothing, we fall back to a flat vector search over the whole corpus
so a query always returns an answer.
"""
from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.ollama import Ollama

from research_rag.classifier.taxonomy import load_taxonomy
from research_rag.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
from research_rag.indexing import get_index

TOP_K = 6
_MAX_DESC_CHARS = 500

_llm: Ollama | None = None


def get_llm() -> Ollama:
    """Local Ollama LLM used for selection + answer synthesis."""
    global _llm
    if _llm is None:
        _llm = Ollama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            # CPU-only inference + cold model load can be slow; be generous.
            request_timeout=600.0,
        )
        Settings.llm = _llm
    return _llm


def _desc(text: str) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:_MAX_DESC_CHARS]


def _leaf_filters(field: str, subfield: str, subsub: str) -> MetadataFilters:
    return MetadataFilters(
        filters=[
            ExactMatchFilter(key="field", value=field),
            ExactMatchFilter(key="subfield", value=subfield),
            ExactMatchFilter(key="subsubfield", value=subsub),
        ]
    )


def _engine_for_tools(tools: list[QueryEngineTool]):
    """A single tool needs no router; >1 gets a multi-select RouterQueryEngine."""
    if len(tools) == 1:
        return tools[0].query_engine
    return RouterQueryEngine.from_defaults(
        selector=LLMMultiSelector.from_defaults(llm=get_llm()),
        query_engine_tools=tools,
        llm=get_llm(),
        select_multi=True,
        verbose=True,
    )


def _build_router():
    """Build the nested router over populated taxonomy nodes. Returns a query
    engine, or None if the corpus is empty."""
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
                    filters=_leaf_filters(field, subfield, subsub),
                    llm=get_llm(),
                )
                leaf_tools.append(
                    QueryEngineTool.from_defaults(
                        query_engine=qe,
                        name=f"{field}__{subfield}__{subsub}"[:64],
                        description=_desc(ssdata.get("descriptor", subsub)),
                    )
                )
            if not leaf_tools:
                continue
            sub_tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=_engine_for_tools(leaf_tools),
                    name=f"{field}__{subfield}"[:64],
                    description=_desc(sdata.get("descriptor", subfield)),
                )
            )
        if not sub_tools:
            continue
        field_tools.append(
            QueryEngineTool.from_defaults(
                query_engine=_engine_for_tools(sub_tools),
                name=field[:64],
                description=_desc(fdata.get("descriptor", field)),
            )
        )

    if not field_tools:
        return None
    return _engine_for_tools(field_tools)


def _citations_from_nodes(nodes) -> list[dict]:
    citations: list[dict] = []
    seen: set[str] = set()
    for n in nodes:
        meta = n.node.metadata
        pid = meta.get("paper_id")
        if pid and pid not in seen:
            seen.add(pid)
            citations.append(
                {
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year"),
                    "field": meta.get("field"),
                    "subfield": meta.get("subfield"),
                    "subsubfield": meta.get("subsubfield"),
                }
            )
    return citations


def _flat_answer(question: str) -> dict:
    index = get_index()
    qe = index.as_query_engine(similarity_top_k=TOP_K, llm=get_llm())
    resp = qe.query(question)
    return {
        "answer": str(resp),
        "routing": "flat (fallback)",
        "citations": _citations_from_nodes(getattr(resp, "source_nodes", []) or []),
    }


def answer_question(question: str) -> dict:
    """Answer a question via hierarchical routing, falling back to a flat
    search if routing fails or finds nothing."""
    engine = _build_router()
    if engine is None:
        return {"answer": "(no papers indexed yet)", "routing": "none", "citations": []}

    try:
        resp = engine.query(question)
        nodes = getattr(resp, "source_nodes", []) or []
        if not nodes:
            return _flat_answer(question)
        return {
            "answer": str(resp),
            "routing": "hierarchical",
            "citations": _citations_from_nodes(nodes),
        }
    except Exception as exc:  # selection/parse failure on a small local model
        print(f"Router query failed ({type(exc).__name__}: {exc}); using flat search.")
        return _flat_answer(question)
