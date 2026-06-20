"""Ollama embedding model wiring for llama-index (nomic-embed-text)."""
from __future__ import annotations

from llama_index.embeddings.ollama import OllamaEmbedding

from research_rag.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

_embed_model: OllamaEmbedding | None = None


def get_embedding_model() -> OllamaEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = OllamaEmbedding(
            model_name=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    return _embed_model
