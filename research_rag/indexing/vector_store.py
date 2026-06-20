"""Local (on-disk) Qdrant vector store wiring.

Runs fully embedded -- no Qdrant server process required -- by pointing
QdrantClient at a local directory.
"""
from __future__ import annotations

import atexit

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from research_rag.config import QDRANT_COLLECTION, QDRANT_PATH

_client: QdrantClient | None = None


def _close_client() -> None:
    """Close the local Qdrant client deterministically at exit.

    The embedded (on-disk) client flushes and releases its directory lock
    on close. Relying on QdrantClient.__del__ during interpreter shutdown
    raises a noisy ImportError ("sys.meta_path is None"), so we close it via
    atexit while the interpreter is still fully alive.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
        atexit.register(_close_client)
    return _client


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(client=get_qdrant_client(), collection_name=QDRANT_COLLECTION)
