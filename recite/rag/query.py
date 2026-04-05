"""
LlamaIndex-backed RAG: query with optional long document (evidence).
When document is provided, index (or load from cache), retrieve, then call LLM.
When document is empty, call LLM directly with system + user prompt.
"""

import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tiktoken
from loguru import logger
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None  # type: ignore[misc, assignment]

# Cache HuggingFaceEmbedding by (model_name, device) so we create once per process and reuse.
# Avoids "Cannot copy out of meta tensor" when multiple threads instantiate SentenceTransformer concurrently.
_hf_embed_cache: Dict[Tuple[str, str], Any] = {}
_hf_embed_cache_lock = threading.Lock()

_DEFAULT_HF_MODEL_KWARGS = {"low_cpu_mem_usage": False, "device_map": None}


def _get_huggingface_embedding(
    model_name: str,
    device: str,
    model_kwargs: Optional[dict] = None,
) -> Any:
    """Get or create a HuggingFaceEmbedding; shared across threads to avoid concurrent meta-tensor load."""
    if HuggingFaceEmbedding is None:
        raise RuntimeError("HuggingFaceEmbedding not available")
    key = (model_name.strip(), device.strip())
    kwargs = model_kwargs if model_kwargs is not None else _DEFAULT_HF_MODEL_KWARGS
    with _hf_embed_cache_lock:
        if key not in _hf_embed_cache:
            _hf_embed_cache[key] = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                model_kwargs=kwargs,
            )
        return _hf_embed_cache[key]


def get_cached_hf_embedding(model_name: str, device: str) -> Any:
    """Public getter for tests: same cached HuggingFaceEmbedding used by query_with_rag."""
    return _get_huggingface_embedding(model_name, device)


class _AzureOpenAILikeEmbedding(OpenAILikeEmbedding):
    """OpenAILikeEmbedding configured for Azure/UCSF: api-version query and api-key header.

    Azure expects the api-key in the 'api-key' header and requires ?api-version= in the URL.
    The standard OpenAI client sends Authorization: Bearer and does not add api-version,
    which causes 401 Invalid Client. This subclass injects default_query and default_headers.
    """

    def _get_credential_kwargs(self, is_async: bool = False):
        kwargs = super()._get_credential_kwargs(is_async=is_async)
        kwargs = dict(kwargs)
        if getattr(self, "api_version", None):
            kwargs["default_query"] = {"api-version": self.api_version}
        # Azure/UCSF expect 'api-key' header; OpenAI client uses Authorization by default
        if getattr(self, "api_key", None):
            headers = dict(kwargs.get("default_headers") or {})
            headers["api-key"] = self.api_key
            kwargs["default_headers"] = headers
        return kwargs


# Embedding API token limit: indexing is chunked (LlamaIndex splits the document and
# embeds each chunk separately, so indexing stays under the limit). At retrieval time
# the *query* string (system + user prompt) is embedded once; that single call can
# exceed the API limit (e.g. 8192), so we truncate the query before embedding.
DEFAULT_MAX_QUERY_TOKENS_FOR_EMBED = 8192

# LLM context window: OpenAILike defaults to ~3900; if prompt + chunks exceed that,
# LlamaIndex raises "available context size was not non-negative". Set to match serve model (e.g. 16384).
DEFAULT_LLM_CONTEXT_WINDOW = 16384


class _VersaLLMAdapter(CustomLLM):
    """LlamaIndex LLM that delegates to UCSF Versa API (chat completions). Subclasses CustomLLM so resolve_llm(llm) passes isinstance(llm, LLM)."""

    def __init__(self, model: str, **kwargs: Any):
        super().__init__(**kwargs)
        from recite.llmapis import UCSFVersaAPI
        self._model = model
        self._api = UCSFVersaAPI(model=model, system_prompt="")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self._model,
            context_window=128000,
            is_chat_model=True,
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # Prompt is typically "[System]\n...\n\n[User]\n..."; pass as user with optional system
        system_prompt = ""
        user_prompt = prompt
        if "[System]" in prompt and "[User]" in prompt:
            parts = prompt.split("[User]", 1)
            if len(parts) == 2:
                system_part = parts[0].replace("[System]", "").strip()
                system_prompt = system_part
                user_prompt = parts[1].strip()
        text = self._api(user_prompt, system_prompt=system_prompt or None)
        out = text if isinstance(text, str) else (str(text) if text is not None else "")
        return CompletionResponse(text=out)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        # Versa API is non-streaming; yield single CompletionResponse
        response = self.complete(prompt, formatted=formatted, **kwargs)
        yield response

_tiktoken_encoding: Optional[tiktoken.Encoding] = None


def _get_encoding() -> tiktoken.Encoding:
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoding


def _truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens (encoding: cl100k_base)."""
    if max_tokens <= 0:
        return text
    enc = _get_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def _doc_id(document: str) -> str:
    """Stable doc_id from document content (sha256 hex)."""
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


# Default max tokens for evidence when no_rag=True (truncate, no retrieval).
DEFAULT_NO_RAG_MAX_TOKENS = 4096

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking_tokens(text: str) -> str:
    """Strip <think>...</think> blocks from model output (e.g. Qwen3 reasoning mode)."""
    return _THINK_RE.sub("", text).strip()


def build_index_for_document(
    document: str,
    persist_dir: Path,
    embed_model: Any,
) -> bool:
    """Build and persist a vector index for a single document. Idempotent if index exists.
    Each document gets its own index (persist_dir / doc_id /); at query time we load only
    that index so retrieval returns chunks from this document only, never from others."""
    if not document or not document.strip():
        return False
    doc_id = _doc_id(document)
    index_dir = persist_dir / doc_id
    index_dir.mkdir(parents=True, exist_ok=True)
    if (index_dir / "docstore.json").exists():
        logger.debug("RAG index already exists for doc_id={}, skipping", doc_id[:16])
        return False
    logger.info("RAG: building index for doc_id={}", doc_id[:16])
    doc = Document(text=document, id_=doc_id)
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        [doc],
        storage_context=storage_context,
        embed_model=embed_model,
    )
    storage_context.persist(persist_dir=str(index_dir))
    return True


def query_with_rag(
    system_prompt: str,
    user_prompt: str,
    document: Optional[str],
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    persist_dir: Path,
    embed_api_key: Optional[str] = None,
    embed_api_version: Optional[str] = None,
    similarity_top_k: Optional[int] = None,
    max_query_tokens_for_embed: Optional[int] = None,
    llm_context_window: Optional[int] = None,
    ucsf_versa_model: Optional[str] = None,
    no_rag: bool = False,
    no_rag_max_tokens: Optional[int] = None,
    embed_local_model: Optional[str] = None,
    embed_device_index: Optional[str] = None,
    embed_device_query: Optional[str] = None,
) -> str:
    """
    Run RAG or direct LLM call.

    If document is None or empty: call LLM once with system_prompt + user_prompt.
    If no_rag=True and document non-empty: truncate document to no_rag_max_tokens and pass in prompt (no retrieval).
    If document is non-empty and embed configured: get or build index (by doc_id), run query_engine.query()
    with system + user prompt, return response.

    Args:
        system_prompt: System instructions for the LLM.
        user_prompt: User question or prompt.
        document: Optional long text (evidence). If provided, index/load and RAG (or truncate when no_rag).
        llm_base_url: Base URL for LLM (e.g. http://localhost:8000/v1).
        llm_model: Model name for LLM.
        embed_base_url: Base URL for embedding API (required when document is provided and not no_rag).
        embed_model: Model name for embeddings.
        persist_dir: Directory for LlamaIndex cache (per-doc index under persist_dir / doc_id).
        embed_api_key: Optional API key for embedding endpoint.
        embed_api_version: Optional API version for Azure/UCSF (e.g. 2024-10-21). Required for Azure-style endpoints.
        similarity_top_k: Optional number of chunks to retrieve (LlamaIndex similarity_top_k). If None, uses default (2).
        max_query_tokens_for_embed: Max tokens for the query string sent to the embedding API (avoids 400 when query exceeds model limit, e.g. 8192). If None, uses DEFAULT_MAX_QUERY_TOKENS_FOR_EMBED.
        llm_context_window: Context window size (tokens) for the LLM. If None, uses DEFAULT_LLM_CONTEXT_WINDOW. Prevents LlamaIndex "available context size not non-negative" when prompt + chunks exceed OpenAILike default (~3900).
        ucsf_versa_model: If set, use UCSF Versa API for chat completions (ignores llm_base_url/llm_model). Requires UCSF_API_KEY, UCSF_API_VER, UCSF_RESOURCE_ENDPOINT.
        no_rag: If True, do not use retrieval; pass evidence in prompt truncated to no_rag_max_tokens.
        no_rag_max_tokens: Max tokens for evidence when no_rag=True. If None, uses DEFAULT_NO_RAG_MAX_TOKENS.

    Returns:
        LLM response text.
    """
    if ucsf_versa_model:
        llm = _VersaLLMAdapter(model=ucsf_versa_model)
    else:
        context_window = llm_context_window if llm_context_window is not None else DEFAULT_LLM_CONTEXT_WINDOW
        # Disable thinking mode for Qwen3 models (generates <think> tokens that waste throughput)
        additional_kwargs = {}
        if llm_model and "qwen3" in llm_model.lower():
            additional_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        llm = OpenAILike(
            model=llm_model,
            api_base=llm_base_url,
            api_key="fake",
            is_chat_model=True,
            context_window=context_window,
            timeout=600.0,
            additional_kwargs=additional_kwargs,
        )

    # Some models (e.g. Gemma) don't support a separate system role via the
    # OpenAI-compatible API.  Detect these and merge system prompt into user.
    _no_system_role = bool(llm_model and "gemma" in llm_model.lower()) and not ucsf_versa_model

    def _build_messages(user_content: str) -> list:
        msgs = []
        if system_prompt and system_prompt.strip():
            if _no_system_role:
                user_content = f"{system_prompt.strip()}\n\n{user_content}"
            else:
                msgs.append(ChatMessage(role="system", content=system_prompt))
        msgs.append(ChatMessage(role="user", content=user_content))
        return msgs

    if not document or not document.strip():
        # Direct LLM call: no RAG
        logger.debug("RAG: document empty, using direct LLM path")
        response = llm.chat(_build_messages(user_prompt))
        return _strip_thinking_tokens(response.message.content or "")

    # No-RAG sweep: pass evidence in prompt, truncated to token limit (no retrieval)
    if no_rag:
        max_tok = no_rag_max_tokens if no_rag_max_tokens is not None else DEFAULT_NO_RAG_MAX_TOKENS
        evidence = _truncate_to_token_limit(document, max_tok)
        if len(evidence) < len(document):
            logger.info("RAG: no_rag=True; truncated evidence from %s to %s tokens", len(_get_encoding().encode(document)), max_tok)
        else:
            logger.debug("RAG: no_rag=True; evidence within {} tokens", max_tok)
        user_with_evidence = f"{user_prompt}\n\nSupporting evidence:\n{evidence}"
        response = llm.chat(_build_messages(user_with_evidence))
        return _strip_thinking_tokens(response.message.content or "")

    # Fallback: document provided but no embed configured (API or local) -> pass full document in prompt
    use_local_embed = bool(
        embed_local_model and embed_local_model.strip() and HuggingFaceEmbedding is not None
    )
    use_api_embed = bool(
        embed_base_url and embed_base_url.strip() and embed_model and embed_model.strip()
    )
    if not (use_api_embed or use_local_embed):
        logger.info(
            "RAG: embed config missing (embed_base_url/embed_model or embed_local_model); passing full document in prompt (no retrieval)"
        )
        user_with_evidence = f"{user_prompt}\n\nSupporting evidence:\n{document}"
        response = llm.chat(_build_messages(user_with_evidence))
        return _strip_thinking_tokens(response.message.content or "")

    # RAG path: embed config present (API or local).
    # One index per document (doc_id = hash(document)); retrieval is scoped to this document only—
    # we load/build only the index for this document, so no chunks from other documents can appear.
    doc_id = _doc_id(document)
    index_dir = persist_dir / doc_id
    index_dir.mkdir(parents=True, exist_ok=True)

    # Check if index already exists (already indexed)
    docstore_path = index_dir / "docstore.json"
    if docstore_path.exists():
        logger.info("RAG: index hit for doc_id={}", doc_id[:16])
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        index = load_index_from_storage(storage_context)
        # For query we need an embed model (query embedding only; doc vectors already in index)
        if use_local_embed:
            query_device = (embed_device_query or "cpu").strip()
            embed = _get_huggingface_embedding(embed_local_model.strip(), query_device)
        else:
            embed_cls = _AzureOpenAILikeEmbedding if embed_api_version else OpenAILikeEmbedding
            embed = embed_cls(
                model_name=embed_model,
                api_base=embed_base_url,
                api_key=embed_api_key or "fake",
                api_version=embed_api_version,
            )
    else:
        logger.info("RAG: index miss for doc_id={}, building index", doc_id[:16])
        if use_local_embed:
            index_device = (embed_device_index or "cuda:0").strip()
            embed = _get_huggingface_embedding(embed_local_model.strip(), index_device)
        else:
            embed_cls = _AzureOpenAILikeEmbedding if embed_api_version else OpenAILikeEmbedding
            embed = embed_cls(
                model_name=embed_model,
                api_base=embed_base_url,
                api_key=embed_api_key or "fake",
                api_version=embed_api_version,
            )
        doc = Document(text=document, id_=doc_id)
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(
            [doc],
            storage_context=storage_context,
            embed_model=embed,
        )
        storage_context.persist(persist_dir=str(index_dir))

    logger.debug("RAG: query start doc_id={}", doc_id[:16])
    query_kwargs: dict = {"llm": llm, "response_mode": "compact"}
    if similarity_top_k is not None:
        query_kwargs["similarity_top_k"] = similarity_top_k
    # When we have a local embedder (or built with one), pass it for query embedding
    if embed is not None:
        query_kwargs["embed_model"] = embed
    query_engine = index.as_query_engine(**query_kwargs)
    # Include system prompt in the query so the LLM sees it with retrieved context
    full_query = f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}" if system_prompt.strip() else user_prompt
    max_tokens = max_query_tokens_for_embed if max_query_tokens_for_embed is not None else DEFAULT_MAX_QUERY_TOKENS_FOR_EMBED
    query_for_embed = _truncate_to_token_limit(full_query, max_tokens)
    if len(query_for_embed) < len(full_query):
        logger.debug("RAG: truncated query for embedding from {} to {} tokens", len(_get_encoding().encode(full_query)), max_tokens)
    # Bottleneck: embed query -> retrieve chunks -> call LLM; LLM generation dominates (often 10–60s)
    logger.info(
        "RAG: embed + retrieve + LLM generate (doc_id={})... may take 10–60s depending on model",
        doc_id[:16],
    )
    _t0 = time.perf_counter()
    response = query_engine.query(query_for_embed)
    _elapsed = time.perf_counter() - _t0
    out = (response.response if hasattr(response, "response") else str(response)) or ""
    out = _strip_thinking_tokens(out)
    logger.info("RAG: LLM done in %.1fs (response_len=%s)", _elapsed, len(out))
    logger.debug("RAG: query end doc_id={} response_len={}", doc_id[:16], len(out))
    return out
