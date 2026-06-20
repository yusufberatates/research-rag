"""Hierarchical vector index over the paper corpus.

"Hierarchical" here means every chunk (node) carries field/subfield
metadata mirroring the classifier's taxonomy. A single Qdrant collection
is used, and retrieval narrows the search by filtering on
field/subfield metadata first (see research_rag.query.query_engine) --
this avoids the overhead of one collection per subfield while still
giving field -> subfield -> chunk routing.
"""
from __future__ import annotations

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from .embeddings import get_embedding_model
from .vector_store import get_vector_store

Settings.embed_model = get_embedding_model()
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

_index: VectorStoreIndex | None = None


def load_or_create_index() -> VectorStoreIndex:
    global _index
    if _index is None:
        storage_context = StorageContext.from_defaults(vector_store=get_vector_store())
        _index = VectorStoreIndex.from_vector_store(
            vector_store=storage_context.vector_store,
            embed_model=Settings.embed_model,
        )
    return _index


def get_index() -> VectorStoreIndex:
    return load_or_create_index()


def add_paper_to_index(
    extracted_record: dict,
    field: str,
    subfield: str,
    subsubfield: str = "",
) -> None:
    """Embed and insert one paper's full text into the index, tagging
    every resulting chunk with hierarchy + citation metadata.

    Idempotent: any chunks previously indexed for this ``paper_id`` are
    removed first (keyed on the document id), so re-running ``index`` never
    accumulates duplicate chunks.
    """
    index = load_or_create_index()
    paper_id = extracted_record["paper_id"]

    metadata = {
        "paper_id": paper_id,
        "title": extracted_record.get("title", ""),
        "authors": ", ".join(extracted_record.get("authors") or []),
        "year": extracted_record.get("year"),
        "topic": extracted_record.get("topic", ""),
        "field": field,
        "subfield": subfield,
        "subsubfield": subsubfield,
        "tier": extracted_record.get("tier"),
    }

    # Upsert: drop any existing nodes for this paper before re-inserting.
    try:
        index.delete_ref_doc(paper_id, delete_from_docstore=True)
    except Exception:
        # Nothing indexed yet for this paper (or backend has no record) -- fine.
        pass

    document = Document(
        text=extracted_record.get("full_text", ""),
        metadata=metadata,
        doc_id=paper_id,
    )
    index.insert(document)
