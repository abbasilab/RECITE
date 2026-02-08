"""
RAG package: LlamaIndex-backed retrieval-augmented generation.

Importable and callable from the benchmark (and optionally from a standalone RAG server).
Document-level index caching: repeated queries on the same document reuse the cached index.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from recite.rag.query import (
    DEFAULT_LLM_CONTEXT_WINDOW,
    DEFAULT_MAX_QUERY_TOKENS_FOR_EMBED,
    _doc_id,
    get_cached_hf_embedding,
    query_with_rag,
)


@dataclass
class RAGConfig:
    """Configuration for RAG (embed API and optional overrides)."""

    embed_base_url: str
    embed_model: str
    persist_dir: Optional[Path] = None
    embed_api_key: Optional[str] = None
    embed_api_version: Optional[str] = None
    similarity_top_k: Optional[int] = None
    max_query_tokens_for_embed: Optional[int] = None
    llm_context_window: Optional[int] = None

    def to_query_kwargs(
        self,
        llm_base_url: str,
        llm_model: str,
        system_prompt: str,
        user_prompt: str,
        document: Optional[str],
        persist_dir_override: Optional[Path] = None,
    ) -> dict:
        """Build kwargs for query_with_rag from this config."""
        persist = persist_dir_override or self.persist_dir
        if persist is None:
            raise ValueError("persist_dir must be set (RAGConfig or override)")
        kwargs: dict = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "document": document,
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "embed_base_url": self.embed_base_url,
            "embed_model": self.embed_model,
            "persist_dir": Path(persist),
            "embed_api_key": self.embed_api_key,
            "embed_api_version": self.embed_api_version,
        }
        if self.similarity_top_k is not None:
            kwargs["similarity_top_k"] = self.similarity_top_k
        if self.max_query_tokens_for_embed is not None:
            kwargs["max_query_tokens_for_embed"] = self.max_query_tokens_for_embed
        if self.llm_context_window is not None:
            kwargs["llm_context_window"] = self.llm_context_window
        return kwargs


__all__ = [
    "RAGConfig",
    "query_with_rag",
    "_doc_id",
    "get_cached_hf_embedding",
    "DEFAULT_LLM_CONTEXT_WINDOW",
    "DEFAULT_MAX_QUERY_TOKENS_FOR_EMBED",
]
