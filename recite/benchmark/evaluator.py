"""Benchmark runner and evaluator."""

import asyncio
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import httpx
import pandas as pd
from loguru import logger

from recite.utils.path_loader import get_project_root


_COLUMN_ALIASES = {
    "source_text": "source_text",
    "reference_text": "reference_text",
    "source_version": "source_version",
    "target_version": "target_version",
    "instance_id": "instance_id",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {old: new for old, new in _COLUMN_ALIASES.items() if old in df.columns and new not in df.columns}
    return df.rename(columns=rename) if rename else df


_HF_CACHE_DIR_LOGGED = False


def _get_hf_cache_dir() -> Path:
    global _HF_CACHE_DIR_LOGGED
    raw = os.environ.get("HF_HOME") or os.environ.get("HF_CACHE")
    if raw:
        path = Path(raw).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
    else:
        path = get_project_root() / "data" / "hf_cache"
        path.mkdir(parents=True, exist_ok=True)
    if not _HF_CACHE_DIR_LOGGED:
        logger.info("HF model cache (checkpoints): {}", path)
        _HF_CACHE_DIR_LOGGED = True
    return path

_PYTHON_GPU_MODEL_CACHE: Dict[Tuple[str, Any], Any] = {}
_JUDGE_API_CACHE: Dict[Tuple[str, str], Any] = {}
_PYTHON_GPU_GENERATE_LOG_COUNT = 0


def _get_python_gpu_model(
    model_name: str,
    device: str = "cuda",
    gpu_ids: Optional[List[int]] = None,
) -> Tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        raise ImportError(
            "python_gpu models require transformers and torch. Install with: pip install transformers torch"
        ) from e

    if gpu_ids is not None and len(gpu_ids) > 0:
        cache_key: Any = tuple(sorted(gpu_ids))
    else:
        cache_key = device

    key = (model_name, cache_key)
    if key in _PYTHON_GPU_MODEL_CACHE:
        return _PYTHON_GPU_MODEL_CACHE[key]

    cache_dir = _get_hf_cache_dir()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )

    if gpu_ids is not None and len(gpu_ids) > 0:
        if not torch.cuda.is_available():
            logger.warning("gpu_ids specified but CUDA not available; falling back to CPU.")
            device = "cpu"
        elif len(gpu_ids) == 1:
            target_device = f"cuda:{gpu_ids[0]}"
            logger.info(
                f"Loading python_gpu model {model_name} on {target_device} (cache: {cache_dir})..."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=cache_dir,
                low_cpu_mem_usage=False,
            )
            model = model.to(target_device)
            logger.info(f"Model loaded on {target_device} (requested gpu_ids={gpu_ids}).")
            _PYTHON_GPU_MODEL_CACHE[key] = (model, tokenizer)
            return _PYTHON_GPU_MODEL_CACHE[key]
        else:
            try:
                import accelerate
            except ImportError:
                raise ImportError(
                    "gpu_ids (multi-GPU) requires accelerate. Install with: pip install accelerate"
                ) from None
            n_gpus = torch.cuda.device_count()
            max_memory = {}
            for i in range(n_gpus):
                if i in gpu_ids:
                    props = torch.cuda.get_device_properties(i)
                    mem_gb = int(props.total_memory / (1024**3) * 0.9)
                    max_memory[i] = f"{mem_gb}GiB"
                else:
                    max_memory[i] = "0GiB"
            logger.info(
                f"Loading python_gpu model {model_name} on GPUs {gpu_ids} (cache: {cache_dir})..."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            actual_gpus = set(model.hf_device_map.values()) if hasattr(model, "hf_device_map") else set()
            logger.info(f"Model loaded on GPUs: {sorted(actual_gpus)} (requested {gpu_ids}).")
            _PYTHON_GPU_MODEL_CACHE[key] = (model, tokenizer)
            return _PYTHON_GPU_MODEL_CACHE[key]

    want_gpu = device.startswith("cuda") or device == "auto"
    if want_gpu and device != "auto" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; using CPU (slower).")
        device = "cpu"
    if device == "auto" and not torch.cuda.is_available():
        logger.warning("device=auto requested but no CUDA; using CPU.")
        device = "cpu"

    logger.info(f"Loading python_gpu model {model_name} on {device} (cache: {cache_dir})...")

    if device == "auto":
        try:
            import accelerate
        except ImportError:
            raise ImportError(
                "device=auto (multi-GPU) requires accelerate. Install with: pip install accelerate"
            ) from None
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        ndev = len(set(model.hf_device_map.values())) if hasattr(model, "hf_device_map") else 1
        logger.info("Model loaded across {} GPU(s) (device_map=auto).", ndev)
        _PYTHON_GPU_MODEL_CACHE[key] = (model, tokenizer)
        return _PYTHON_GPU_MODEL_CACHE[key]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if want_gpu else torch.float32,
        trust_remote_code=True,
        cache_dir=cache_dir,
        low_cpu_mem_usage=False,
    )
    if device != "cpu":
        devices_to_try: List[str] = (
            [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if device == "cuda"
            else [device]
        )
        oom_err: Optional[RuntimeError] = None
        for d in devices_to_try:
            try:
                model = model.to(d)
                device = d
                if len(devices_to_try) > 1:
                    logger.info(f"Using GPU {d} for {model_name}.")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or (
                    hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError)
                ):
                    oom_err = e
                    if len(devices_to_try) > 1:
                        logger.debug("OOM on %s, trying next GPU.", d)
                    continue
                raise
        else:
            if oom_err:
                logger.warning(
                    "CUDA OOM on all tried devices; using CPU (slower)."
                )
            device = "cpu"
            key = (model_name, device)
            if key in _PYTHON_GPU_MODEL_CACHE:
                return _PYTHON_GPU_MODEL_CACHE[key]
    _PYTHON_GPU_MODEL_CACHE[key] = (model, tokenizer)
    return _PYTHON_GPU_MODEL_CACHE[key]


def clear_python_gpu_cache() -> None:
    _PYTHON_GPU_MODEL_CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


async def _vllm_endpoint_call(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    max_context: int = 8192,
    temperature: float = 0,
    timeout: float = 300.0,
) -> str:
    messages = []
    if "gemma" in model.lower():
        combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        messages.append({"role": "user", "content": combined})
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

    est_input_tokens = sum(len(m["content"]) for m in messages) // 4
    effective_max_tokens = min(max_tokens, max(256, max_context - est_input_tokens - 100))

    for attempt in range(3):
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": effective_max_tokens,
            }
            if "qwen3" in model.lower() or "Qwen3" in model:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            resp = await client.post(
                f"{endpoint}/chat/completions",
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _strip_thinking_tags(content)
            elif resp.status_code == 400:
                error = resp.json().get("error", {}).get("message", resp.text[:200])
                if "maximum context length" in error and attempt < 2:
                    user_prompt = user_prompt[:len(user_prompt) // 2]
                    messages[-1]["content"] = user_prompt
                    est_input_tokens = sum(len(m["content"]) for m in messages) // 4
                    effective_max_tokens = min(max_tokens, max(256, max_context - est_input_tokens - 100))
                    continue
                return f"[ERROR_400] {error}"
            elif resp.status_code in (429, 503):
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                resp.raise_for_status()
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError,
                httpx.RemoteProtocolError, httpx.ReadError) as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return f"[ERROR_TIMEOUT] {e}"
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return f"[ERROR] {type(e).__name__}: {e}"
    return "[ERROR] Max retries exceeded"


def _vllm_endpoint_predict_sync(
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    max_context: int = 8192,
    timeout: float = 300.0,
) -> str:
    async def _inner():
        async with httpx.AsyncClient() as client:
            return await _vllm_endpoint_call(
                client, endpoint, model, system_prompt, user_prompt,
                max_tokens=max_tokens, max_context=max_context, timeout=timeout,
            )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _inner()).result()
    return asyncio.run(_inner())


async def _vllm_endpoint_predict_batch(
    endpoint: str,
    model: str,
    samples: List[Dict[str, Any]],
    prompts_obj: "BenchmarkPrompts",
    tokenizer: Any,
    no_rag_max_tokens: int,
    max_concurrent: int = 16,
    max_tokens: int = 2048,
    max_context: int = 8192,
    timeout: float = 300.0,
    prompt_suffix: str = "",
    checkpoint_callback: Optional[Callable[[int], None]] = None,
    save_every: int = 50,
) -> List[str]:
    """Run async concurrent predictions against a vLLM endpoint."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(samples)
    completed = 0

    async with httpx.AsyncClient() as client:
        async def process_one(idx: int, sample: dict):
            nonlocal completed
            async with semaphore:
                source_text = str(sample.get("source_text", "") or "")
                evidence = str(sample.get("evidence", "") or "")
                source_version = sample.get("source_version")
                target_version = sample.get("target_version")
                try:
                    vf = int(float(source_version)) if source_version is not None and not pd.isna(source_version) else None
                except (ValueError, TypeError):
                    vf = None
                try:
                    vt = int(float(target_version)) if target_version is not None and not pd.isna(target_version) else None
                except (ValueError, TypeError):
                    vt = None

                has_document = bool(evidence.strip())
                user_prompt = _format_model_prompt(
                    source_text, prompts_obj.model_prompt,
                    has_document=has_document, source_version=vf, target_version=vt,
                )
                system_prompt = prompts_obj.model_prompt.get("system", "")

                if evidence.strip():
                    ev = evidence.strip()
                    if tokenizer is not None:
                        enc = tokenizer.encode(ev)
                        if len(enc) > no_rag_max_tokens:
                            ev = tokenizer.decode(enc[:no_rag_max_tokens])
                    else:
                        max_chars = no_rag_max_tokens * 4
                        if len(ev) > max_chars:
                            ev = ev[:max_chars]
                    user_prompt += f"\n\nSupporting evidence:\n{ev}"

                if prompt_suffix:
                    user_prompt += f"\n\n{prompt_suffix}"

                try:
                    prediction = await _vllm_endpoint_call(
                        client, endpoint, model, system_prompt, user_prompt,
                        max_tokens=max_tokens, max_context=max_context, timeout=timeout,
                    )
                except Exception as e:
                    prediction = f"[ERROR] {type(e).__name__}: {e}"
                results[idx] = prediction

        tasks = [process_one(i, s) for i, s in enumerate(samples)]
        for coro in asyncio.as_completed(tasks):
            await coro
            completed += 1
            if checkpoint_callback and completed % save_every == 0:
                checkpoint_callback(completed)

    return [r if r is not None else "[ERROR] No result" for r in results]


@dataclass
class PredictionRecord:
    """Single prediction record."""
    id: int
    instance_id: str
    source_version: int
    target_version: int
    source_text: str
    evidence: str
    reference_text: str  # Ground truth
    prediction: str
    timestamp: str
    metadata: Optional[Dict] = None


@dataclass
class BenchmarkPrompts:
    """Benchmark prompts from config."""
    model_prompt: Dict[str, str]
    judge_prompt: Dict[str, str]
    multi_stage_prompts: Optional[Dict[str, Dict[str, str]]] = None
    judge_prompt_batched: Optional[Dict[str, str]] = None

    @classmethod
    def load(cls, config_path: Path) -> "BenchmarkPrompts":
        """Load from JSON config file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Benchmark prompts config not found: {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        multi_stage = data.get("multi_stage_prompts")
        if multi_stage is None and all(
            data.get(k) for k in ("step1_system", "step1_user_template", "step2_system", "step2_user_template")
        ):
            multi_stage = {
                "step1_system": data["step1_system"],
                "step1_user_template": data["step1_user_template"],
                "step2_system": data["step2_system"],
                "step2_user_template": data["step2_user_template"],
            }
        judge_batched = data.get("judge_prompt_batched")
        return cls(
            model_prompt=data["model_prompt"],
            judge_prompt=data["judge_prompt"],
            multi_stage_prompts=multi_stage if isinstance(multi_stage, dict) else None,
            judge_prompt_batched=judge_batched if isinstance(judge_batched, dict) else None,
        )


BENCHMARK_PROMPTS_PATH = get_project_root() / "config" / "benchmark_prompts.json"


def load_benchmark_prompts(prompts_path: Optional[Path] = None) -> BenchmarkPrompts:
    if prompts_path is None:
        prompts_path = BENCHMARK_PROMPTS_PATH
    return BenchmarkPrompts.load(prompts_path)


def _format_model_prompt(
    source_text: str,
    templates: Dict[str, str],
    has_document: bool = False,
    source_version: Optional[int] = None,
    target_version: Optional[int] = None,
) -> str:
    kwargs: Dict[str, Any] = {"source_text": source_text}
    if source_version is not None:
        kwargs["source_version"] = source_version
    if target_version is not None:
        kwargs["target_version"] = target_version
    if has_document and "user_template_rag" in templates:
        return templates["user_template_rag"].format(**kwargs)
    return templates["user_template"].format(**kwargs)


def _format_judge_prompt(ground_truth: str, prediction: str, templates: Dict[str, str]) -> str:
    return templates["user_template"].format(
        ground_truth=ground_truth,
        prediction=prediction,
    )


def _parse_judge_scores(response: str, score_scale: str = "0-4") -> Dict[str, float]:
    if not response or not isinstance(response, str):
        logger.warning(f"Invalid response type for parsing: {type(response)}")
        max_score = 4.0 if score_scale == "0-4" else 10.0
        return {
            "binary_score": 0.0,
            "ordinal_score": max_score / 2.0,
        }
    
    score_str = response.strip()
    
    if score_scale == "0-4":
        max_ordinal = 4.0
        ordinal_pattern = r'\b([0-4])\b'
    elif score_scale == "1-10":
        max_ordinal = 10.0
        ordinal_pattern = r'\b(10|[1-9])\b'
    else:
        logger.warning(f"Unknown score_scale '{score_scale}', defaulting to 0-4")
        max_ordinal = 4.0
        ordinal_pattern = r'\b([0-4])\b'
    
    # Try to parse "binary,ordinal" format (preferred)
    # Pattern: two numbers separated by comma (with optional spaces)
    comma_match = re.search(r'\b([01])\s*,\s*(\d+)\b', score_str)
    if comma_match:
        try:
            binary = float(comma_match.group(1))
            ordinal = float(comma_match.group(2))
            binary = max(0.0, min(1.0, binary))  # Clamp binary to 0-1
            ordinal = max(0.0, min(max_ordinal, ordinal))  # Clamp ordinal to valid range
            return {
                "binary_score": binary,
                "ordinal_score": ordinal,
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse comma-separated scores: {e}")
    
    # Try to find two numbers (binary first, then ordinal)
    # Look for 0 or 1, then ordinal score
    binary_match = re.search(r'\b([01])\b', score_str)
    ordinal_match = re.search(ordinal_pattern, score_str)
    
    if binary_match and ordinal_match:
        try:
            binary = float(binary_match.group(1))
            ordinal = float(ordinal_match.group(1))
            binary = max(0.0, min(1.0, binary))
            ordinal = max(0.0, min(max_ordinal, ordinal))
            return {
                "binary_score": binary,
                "ordinal_score": ordinal,
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse binary and ordinal scores: {e}")
    
    # Fallback: try to extract just ordinal score (backward compatibility)
    ordinal_match = re.search(ordinal_pattern, score_str)
    if ordinal_match:
        try:
            ordinal = float(ordinal_match.group(1))
            ordinal = max(0.0, min(max_ordinal, ordinal))
            # Infer binary from ordinal: >= 2 (for 0-4) or >= 5 (for 1-10) = acceptable
            binary_threshold = 2.0 if score_scale == "0-4" else 5.0
            binary = 1.0 if ordinal >= binary_threshold else 0.0
            logger.info(f"Extracted only ordinal score {ordinal}, inferred binary {binary}")
            return {
                "binary_score": binary,
                "ordinal_score": ordinal,
            }
        except (ValueError, TypeError):
            pass
    
    # Last resort: use defaults
    logger.warning(f"Could not parse judge scores from response (scale {score_scale}): {response[:100]}")
    default_ordinal = max_ordinal / 2.0
    return {
        "binary_score": 0.0,
        "ordinal_score": default_ordinal,
    }


def call_model_with_retry(
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_retries: int = 2,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    timeout: float = 120.0,
    wait_for_revive_seconds: int = 0,
) -> str:
    """
    Call LLM endpoint with retry logic and exponential backoff.
    On 503 or RequestError, optionally wait for server ready (for self-reviving wrapper) then retry.

    Args:
        endpoint: API endpoint URL (e.g., "http://localhost:8000/v1")
        model: Model name/ID
        prompt: User prompt
        system_prompt: Optional system prompt
        max_retries: Maximum number of retry attempts (default 2 = 3 attempts total)
        base_delay: Base delay in seconds for exponential backoff (default 0.5)
        max_delay: Cap delay in seconds (default 5.0)
        timeout: Request timeout in seconds
        wait_for_revive_seconds: If > 0, on 503/RequestError wait for server ready before retry (capped at 60s).

    Returns:
        Model response text

    Raises:
        httpx.HTTPStatusError: For non-retryable errors (e.g., 400)
        httpx.RequestError: For network errors after all retries
    """
    def _maybe_wait_revive() -> bool:
        if wait_for_revive_seconds <= 0:
            return False
        try:
            import time as _time
            wait_secs = min(wait_for_revive_seconds, 60)
            deadline = _time.monotonic() + wait_secs
            while _time.monotonic() < deadline:
                try:
                    r = httpx.get(f"{endpoint}/models", timeout=5)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                _time.sleep(2)
            return False
        except Exception:
            return False

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    last_error = None
    
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(max_retries):
            try:
                resp = client.post(
                    f"{endpoint}/chat/completions",
                    json={"model": model, "messages": messages, "temperature": 0},
                    timeout=timeout,
                )
                
                # Success
                if resp.status_code == 200:
                    result = resp.json()
                    if "choices" not in result or not result["choices"]:
                        raise ValueError("Empty response from LLM")
                    return result["choices"][0]["message"]["content"]
                
                # 400 Bad Request - don't retry
                elif resp.status_code == 400:
                    error_text = resp.text[:500]
                    logger.error(f"Bad request (400) from LLM endpoint: {error_text}")
                    resp.raise_for_status()
                
                # 429 or 503 - retry with backoff (optionally wait for revive first)
                elif resp.status_code in (429, 503):
                    if attempt < max_retries - 1:
                        if _maybe_wait_revive():
                            continue
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Server busy ({resp.status_code}), retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        resp.raise_for_status()
                
                # Other errors
                else:
                    resp.raise_for_status()
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 400:
                    # Don't retry 400 errors
                    raise
                if attempt < max_retries - 1 and e.response.status_code >= 500:
                    if _maybe_wait_revive():
                        continue
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Server error ({e.response.status_code}), retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                raise
                
            except httpx.RequestError as e:
                last_error = e
                if attempt < max_retries - 1:
                    if _maybe_wait_revive():
                        continue
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Request error, retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                raise

        # All retries exhausted
        if last_error:
            raise last_error
        raise RuntimeError("Failed to call model after all retries")


def _query_with_rag_retry(
    system_prompt: str,
    user_prompt: str,
    document: Optional[str],
    llm_base_url: str,
    llm_model: str,
    rag_config: Dict[str, Any],
    persist_dir: Path,
    max_retries: int = 2,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    wait_for_revive_seconds: int = 0,
    azure_openai_model: Optional[str] = None,
) -> str:
    """Call query_with_rag with retry logic."""
    raise NotImplementedError("RAG mode not available in this build")

    kwargs: Dict[str, Any] = {
        "system_prompt": system_prompt or "",
        "user_prompt": user_prompt,
        "document": document if document is not None else "",
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "embed_base_url": rag_config["embed_base_url"],
        "embed_model": rag_config["embed_model"],
        "persist_dir": persist_dir,
        "embed_api_key": rag_config.get("embed_api_key"),
        "embed_api_version": rag_config.get("embed_api_version"),
    }
    if rag_config.get("similarity_top_k") is not None:
        kwargs["similarity_top_k"] = rag_config["similarity_top_k"]
    if rag_config.get("no_rag") is not None:
        kwargs["no_rag"] = rag_config["no_rag"]
    if rag_config.get("no_rag_max_tokens") is not None:
        kwargs["no_rag_max_tokens"] = rag_config["no_rag_max_tokens"]
    if azure_openai_model is not None:
        kwargs["azure_openai_model"] = azure_openai_model
    if rag_config.get("embed_local_model"):
        kwargs["embed_local_model"] = rag_config["embed_local_model"]
        kwargs["embed_device_index"] = rag_config.get("embed_device_index", "cuda:0")
        kwargs["embed_device_query"] = rag_config.get("embed_device_query", "cpu")

    last_error = None
    for attempt in range(max_retries):
        try:
            return query_with_rag(**kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                if wait_for_revive_seconds > 0 and not azure_openai_model:
                    try:
                        import time as _time
                        wait_secs = min(wait_for_revive_seconds, 60)
                        deadline = _time.monotonic() + wait_secs
                        revived = False
                        while _time.monotonic() < deadline:
                            try:
                                r = httpx.get(f"{llm_base_url}/models", timeout=5)
                                if r.status_code == 200:
                                    revived = True
                                    break
                            except Exception:
                                pass
                            _time.sleep(2)
                        if revived:
                            continue
                    except Exception:
                        pass
                delay = min(base_delay * (2 ** attempt), max_delay)
                tb_lines = traceback.format_exc().strip().split("\n")
                tb_tail = "\n".join(tb_lines[-4:]) if len(tb_lines) >= 4 else traceback.format_exc()
                logger.warning(
                    "RAG call failed, retrying in {}s (attempt {}/{}): {}: {}\n{}",
                    delay,
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    e,
                    tb_tail,
                )
                time.sleep(delay)
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Failed RAG call after all retries")


def _is_truncation_retryable_error(e: BaseException) -> bool:
    """True if error is likely due to context length or OOM and retrying with shorter evidence may help."""
    msg = str(e).lower()
    return (
        "out of memory" in msg
        or "oom" in msg
        or "exceed" in msg
        or "maximum length" in msg
        or "token" in msg
        or "length" in msg
        or "size" in msg
    )


def _call_model_with_truncation_retry(
    model_callable: Callable[..., str],
    model: Any,
    source_text: str,
    evidence: str,
    source_version: Optional[int],
    target_version: Optional[int],
) -> str:
    """Call model_callable; on OOM or context-length errors, retry with progressively shorter evidence."""
    evidence_cur = evidence or ""
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            if isinstance(model, dict):
                return model_callable(
                    source_text, evidence_cur,
                    source_version=source_version, target_version=target_version,
                )
            return model_callable(source_text, evidence_cur)
        except Exception as e:
            last_error = e
            if attempt < 2 and _is_truncation_retryable_error(e):
                # Retry with evidence truncated (by character; roughly halves each time)
                new_len = max(0, len(evidence_cur) // 2)
                evidence_cur = evidence_cur[:new_len]
                logger.warning(
                    "Sample failed (likely OOM or context length), retrying with truncated evidence (attempt %s/3): %s",
                    attempt + 2,
                    e,
                )
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Model call failed after truncation retries")


def default_evaluator(ground_truth: str, prediction: str) -> Dict[str, float]:
    """
    Default evaluator using string similarity, BLEU, and ROUGE metrics.
    
    Args:
        ground_truth: Target text
        prediction: Model prediction
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Binary correctness (exact match)
    metrics["binary_correct"] = 1.0 if ground_truth.strip() == prediction.strip() else 0.0
    
    # Normalized edit distance (Levenshtein)
    edit_distance = _levenshtein_distance(ground_truth, prediction)
    max_len = max(len(ground_truth), len(prediction))
    metrics["edit_distance"] = edit_distance
    metrics["normalized_edit_distance"] = edit_distance / max_len if max_len > 0 else 1.0
    metrics["edit_similarity"] = 1.0 - metrics["normalized_edit_distance"]
    
    # BLEU score
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu(
            [ground_truth.split()],
            prediction.split(),
            smoothing_function=smooth,
        )
        metrics["bleu"] = float(bleu)
    except Exception as e:
        logger.warning(f"Failed to compute BLEU: {e}")
        metrics["bleu"] = 0.0
    
    # ROUGE-L (simplified - longest common subsequence)
    rouge_l = _rouge_l(ground_truth, prediction)
    metrics["rouge_l"] = rouge_l
    
    return metrics


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _rouge_l(reference: str, candidate: str) -> float:
    """
    Compute ROUGE-L score (longest common subsequence based).
    
    Simplified implementation focusing on F1 score.
    """
    ref_words = reference.split()
    cand_words = candidate.split()
    
    if not ref_words or not cand_words:
        return 0.0
    
    # Compute LCS length
    lcs_len = _lcs_length(ref_words, cand_words)
    
    if lcs_len == 0:
        return 0.0
    
    precision = lcs_len / len(cand_words)
    recall = lcs_len / len(ref_words)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def llm_judge_evaluator(
    ground_truth: str,
    prediction: str,
    endpoint: str,
    model: str,
    prompts: Optional[BenchmarkPrompts] = None,
    max_retries: int = 3,
    wait_for_revive_seconds: int = 0,
) -> Dict[str, float]:
    """LLM-as-judge evaluator (endpoint-based)."""
    if prompts is None:
        prompts = load_benchmark_prompts()
    
    user_prompt = _format_judge_prompt(ground_truth, prediction, prompts.judge_prompt)
    system_prompt = prompts.judge_prompt.get("system", "")
    score_scale = prompts.judge_prompt.get("score_scale", "0-4")

    try:
        response = call_model_with_retry(
            endpoint=endpoint,
            model=model,
            prompt=user_prompt,
            system_prompt=system_prompt if system_prompt else None,
            max_retries=max_retries,
            wait_for_revive_seconds=wait_for_revive_seconds,
        )

        scores = _parse_judge_scores(response, score_scale)
        binary_score = scores["binary_score"]
        ordinal_score = scores["ordinal_score"]
        max_score = 4.0 if score_scale == "0-4" else 10.0

        return {
            "llm_judge_binary": binary_score,
            "llm_judge_score": ordinal_score,
            "llm_judge_normalized": ordinal_score / max_score,
            "llm_judge_raw_response": response,
        }

    except Exception as e:
        logger.error(f"LLM judge evaluation failed: {e}")
        return {
            "llm_judge_binary": 0.0,
            "llm_judge_score": 0.0,
            "llm_judge_normalized": 0.0,
            "llm_judge_raw_response": None,
        }


def azure_openai_judge_evaluator(
    ground_truth: str,
    prediction: str,
    model: str,
    prompts: Optional[BenchmarkPrompts] = None,
) -> Dict[str, float]:
    """Judge evaluator via AzureOpenAIAPI."""
    from recite.llmapis import AzureOpenAIAPI
    
    if prompts is None:
        prompts = load_benchmark_prompts()
    
    user_prompt = _format_judge_prompt(ground_truth, prediction, prompts.judge_prompt)
    system_prompt = prompts.judge_prompt.get("system", "")
    score_scale = prompts.judge_prompt.get("score_scale", "0-4")

    try:
        cache_key = (model, system_prompt)
        if cache_key not in _JUDGE_API_CACHE:
            _JUDGE_API_CACHE[cache_key] = AzureOpenAIAPI(model=model, system_prompt=system_prompt)
        judge_api = _JUDGE_API_CACHE[cache_key]
        response = judge_api(user_prompt, system_prompt=system_prompt)

        scores = _parse_judge_scores(response, score_scale)
        binary_score = scores["binary_score"]
        ordinal_score = scores["ordinal_score"]
        max_score = 4.0 if score_scale == "0-4" else 10.0

        return {
            "llm_judge_binary": binary_score,
            "llm_judge_score": ordinal_score,
            "llm_judge_normalized": ordinal_score / max_score,
            "llm_judge_raw_response": response,
        }

    except Exception as e:
        logger.error(f"Judge evaluation failed: {e}")
        return {
            "llm_judge_binary": 0.0,
            "llm_judge_score": 0.0,
            "llm_judge_normalized": 0.0,
            "llm_judge_raw_response": None,
        }


def _format_batched_judge_prompt(
    samples: List[Dict[str, Any]],
    templates: Dict[str, str],
) -> str:
    """Format batched judge prompt: N pairs of (ground_truth, prediction).
    samples: list of dicts with 'reference_text' (ground_truth) and 'prediction'.
    """
    pairs_parts = []
    for i, s in enumerate(samples, 1):
        gt = (s.get("reference_text") or "").strip()
        pred = (s.get("prediction") or "").strip()
        pairs_parts.append(f"=== Pair {i} ===\nTarget:\n{gt}\n\nPrediction:\n{pred}")
    pairs_text = "\n\n".join(pairs_parts)
    n = len(samples)
    user_tpl = templates.get("user_template", "Evaluate the following {n} prediction pairs.\n\n{pairs}\n\nReturn JSON: {\"1\": [binary, ordinal], ...}")
    return user_tpl.format(n=n, pairs=pairs_text)


def _parse_batched_judge_response(response: str, n: int, score_scale: str = "0-4") -> List[Dict[str, float]]:
    """Parse batched judge JSON response into list of {binary_score, ordinal_score} dicts.
    Response expected format: {"1": [binary, ordinal], "2": [binary, ordinal], ...}
    """
    if not response or not isinstance(response, str):
        max_score = 4.0 if score_scale == "0-4" else 10.0
        return [{"binary_score": 0.0, "ordinal_score": max_score / 2.0} for _ in range(n)]
    text = response.strip()
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse batched judge JSON: {e}")
        max_score = 4.0 if score_scale == "0-4" else 10.0
        return [{"binary_score": 0.0, "ordinal_score": max_score / 2.0} for _ in range(n)]
    max_ordinal = 4.0 if score_scale == "0-4" else 10.0
    results = []
    for i in range(1, n + 1):
        key = str(i)
        if key not in data:
            results.append({"binary_score": 0.0, "ordinal_score": max_ordinal / 2.0})
            continue
        val = data[key]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                binary = max(0.0, min(1.0, float(val[0])))
                ordinal = max(0.0, min(max_ordinal, float(val[1])))
                results.append({"binary_score": binary, "ordinal_score": ordinal})
            except (TypeError, ValueError):
                results.append({"binary_score": 0.0, "ordinal_score": max_ordinal / 2.0})
        else:
            results.append({"binary_score": 0.0, "ordinal_score": max_ordinal / 2.0})
    return results


def batched_scorer(
    samples: List[Dict[str, Any]],
    model: str,
    prompts: Optional[BenchmarkPrompts] = None,
    batch_size: int = 10,
) -> List[Dict[str, float]]:
    """Evaluate multiple samples in batched API calls."""
    from recite.llmapis import AzureOpenAIAPI

    if prompts is None:
        prompts = load_benchmark_prompts()
    templates = prompts.judge_prompt_batched or prompts.judge_prompt
    score_scale = templates.get("score_scale", "0-4")
    system_prompt = templates.get("system", "")
    max_score = 4.0 if score_scale == "0-4" else 10.0

    all_metrics = []
    cache_key = (model, system_prompt)
    if cache_key not in _JUDGE_API_CACHE:
        _JUDGE_API_CACHE[cache_key] = AzureOpenAIAPI(model=model, system_prompt=system_prompt)
    judge_api = _JUDGE_API_CACHE[cache_key]
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        user_prompt = _format_batched_judge_prompt(batch, templates)
        try:
            response = judge_api(user_prompt, system_prompt=system_prompt)
        except Exception as e:
            logger.error(f"Batched judge API call failed: {e}")
            response = ""
        raw_response = response
        scores_list = _parse_batched_judge_response(response, len(batch), score_scale)
        for s in scores_list:
            binary = s["binary_score"]
            ordinal = s["ordinal_score"]
            all_metrics.append({
                "llm_judge_binary": binary,
                "llm_judge_score": ordinal,
                "llm_judge_normalized": ordinal / max_score,
                "llm_judge_raw_response": raw_response,
            })
    return all_metrics


def run_single_sample(
    sample_row: Dict[str, Any],
    model: Union[Callable[[str, str], str], Dict[str, Any]],
    rag_config: Optional[Dict[str, Any]],
    evaluator_type: str = "default",
    evaluator_config: Optional[Dict[str, Any]] = None,
    prompts_path: Optional[Path] = None,
    split_name: str = "",
    multi_stage: bool = False,
    wait_for_revive_seconds: int = 0,
    max_retries: int = 2,
    max_delay: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Run one sample through the model and evaluators. Returns None on failure."""
    prompts = load_benchmark_prompts(prompts_path)
    use_multi_stage = multi_stage and prompts.multi_stage_prompts is not None

    def _v(key: str, default: Any = None) -> Any:
        val = sample_row.get(key, default)
        if val is None or (hasattr(pd, "isna") and pd.isna(val)):
            return default
        if key in ("id", "source_version", "target_version", "year") and val is not None:
            try:
                return int(float(val))
            except (TypeError, ValueError):
                return default
        if key == "quality_score" and val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        return val

    try:
        if callable(model):
            model_callable = model
            is_endpoint = False
            use_multi_stage = False
        elif isinstance(model, dict) and model.get("api_type") == "azure_openai" and "model" in model:
            if rag_config is None:
                raise ValueError("rag_config required for azure_openai model")
            persist_dir_raw = rag_config.get("persist_dir")
            if persist_dir_raw is None:
                raise ValueError("rag_config must include persist_dir")
            persist_dir = Path(persist_dir_raw)
            if not persist_dir.is_absolute():
                persist_dir = get_project_root() / persist_dir
            rag_cfg = dict(rag_config)
            if model.get("top_k") is not None:
                rag_cfg["similarity_top_k"] = model["top_k"]
            elif rag_cfg.get("similarity_top_k") is None and rag_cfg.get("top_k") is not None:
                rag_cfg["similarity_top_k"] = rag_cfg["top_k"]
            azure_openai_model = model["model"]
            endpoint, llm_model = "n/a", azure_openai_model

            def _call(
                source_text: str,
                evidence: str,
                source_version: Optional[int] = None,
                target_version: Optional[int] = None,
            ) -> str:
                has_document = evidence is not None and bool(evidence.strip())
                user_prompt = _format_model_prompt(
                    source_text, prompts.model_prompt,
                    has_document=has_document, source_version=source_version, target_version=target_version,
                )
                system_prompt = prompts.model_prompt.get("system", "")
                return _query_with_rag_retry(
                    system_prompt=system_prompt or "",
                    user_prompt=user_prompt,
                    document=evidence if evidence is not None else "",
                    llm_base_url=endpoint,
                    llm_model=llm_model,
                    rag_config=rag_cfg,
                    persist_dir=persist_dir,
                    max_retries=max_retries,
                    base_delay=0.5,
                    max_delay=max_delay,
                    wait_for_revive_seconds=wait_for_revive_seconds,
                    azure_openai_model=azure_openai_model,
                )
            model_callable = _call
            is_endpoint = True
        elif isinstance(model, dict) and "endpoint" in model and "model" in model:
            if rag_config is None:
                raise ValueError("rag_config required for endpoint-based model")
            persist_dir_raw = rag_config.get("persist_dir")
            if persist_dir_raw is None:
                raise ValueError("rag_config must include persist_dir")
            persist_dir = Path(persist_dir_raw)
            if not persist_dir.is_absolute():
                persist_dir = get_project_root() / persist_dir
            rag_cfg = dict(rag_config)
            if model.get("top_k") is not None:
                rag_cfg["similarity_top_k"] = model["top_k"]
            elif rag_cfg.get("similarity_top_k") is None and rag_cfg.get("top_k") is not None:
                rag_cfg["similarity_top_k"] = rag_cfg["top_k"]
            endpoint, llm_model = model["endpoint"], model["model"]

            def _call(
                source_text: str,
                evidence: str,
                source_version: Optional[int] = None,
                target_version: Optional[int] = None,
            ) -> str:
                has_document = evidence is not None and bool(evidence.strip())
                user_prompt = _format_model_prompt(
                    source_text, prompts.model_prompt,
                    has_document=has_document, source_version=source_version, target_version=target_version,
                )
                system_prompt = prompts.model_prompt.get("system", "")
                return _query_with_rag_retry(
                    system_prompt=system_prompt or "",
                    user_prompt=user_prompt,
                    document=evidence if evidence is not None else "",
                    llm_base_url=endpoint,
                    llm_model=llm_model,
                    rag_config=rag_cfg,
                    persist_dir=persist_dir,
                    max_retries=max_retries,
                    base_delay=0.5,
                    max_delay=max_delay,
                    wait_for_revive_seconds=wait_for_revive_seconds,
                )
            model_callable = _call
            is_endpoint = True
        elif isinstance(model, dict) and model.get("api_type") == "python_gpu" and "model" in model:
            import torch
            model_name = model["model"]
            device = model.get("device", "cuda")
            gpu_ids = model.get("gpu_ids")  # List[int] for multi-GPU subset, e.g. [0,1]
            hf_model, tokenizer = _get_python_gpu_model(model_name, device, gpu_ids=gpu_ids)
            context_window = model.get("context_window")
            ctx_tokens = int(context_window) if context_window is not None else 131072
            no_rag_max_tokens = model.get("no_rag_max_tokens")
            if no_rag_max_tokens is None and context_window is not None:
                no_rag_max_tokens = max(0, ctx_tokens - 4096)
            if no_rag_max_tokens is None or no_rag_max_tokens <= 0:
                no_rag_max_tokens = 512
            no_rag_max_tokens = min(no_rag_max_tokens, max(256, ctx_tokens - 2048 - 512))

            def _call(
                source_text: str,
                evidence: str,
                source_version: Optional[int] = None,
                target_version: Optional[int] = None,
            ) -> str:
                global _PYTHON_GPU_GENERATE_LOG_COUNT
                has_document = evidence is not None and bool(evidence.strip())
                user_prompt = _format_model_prompt(
                    source_text, prompts.model_prompt,
                    has_document=has_document, source_version=source_version, target_version=target_version,
                )
                system_prompt = prompts.model_prompt.get("system", "")
                if evidence and evidence.strip():
                    ev = evidence.strip()
                    enc = tokenizer.encode(ev)
                    if len(enc) > no_rag_max_tokens:
                        ev = tokenizer.decode(enc[:no_rag_max_tokens])
                    user_content = f"{user_prompt}\n\nSupporting evidence:\n{ev}"
                else:
                    user_content = user_prompt
                system_prompt = (system_prompt or "").strip()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                max_len = min(
                    tokenizer.model_max_length or 131072,
                    ctx_tokens,
                )
                def _apply_template_ids(msgs):
                    """Apply chat template and return input_ids tensor."""
                    result = tokenizer.apply_chat_template(
                        msgs,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_len,
                    )
                    if hasattr(result, "keys") and "input_ids" in result:
                        return result["input_ids"]
                    return result

                try:
                    prompt_ids = _apply_template_ids(messages)
                except Exception as template_err:
                    if "system" in str(template_err).lower() or "TemplateError" in type(template_err).__name__:
                        messages_user_only = [
                            {"role": "user", "content": f"{system_prompt}\n\n{user_content}".strip() if system_prompt else user_content},
                        ]
                        prompt_ids = _apply_template_ids(messages_user_only)
                    else:
                        logger.warning(f"python_gpu generate failed: {template_err}")
                        raise
                seq_len = prompt_ids.shape[1]
                if seq_len > max_len:
                    prompt_ids = prompt_ids[:, -max_len:]
                try:
                    prompt_ids = prompt_ids.to(hf_model.device)
                    attention_mask = prompt_ids.new_ones(prompt_ids.shape, dtype=torch.long)
                    n = _PYTHON_GPU_GENERATE_LOG_COUNT
                    if n < 8:
                        logger.info("python_gpu: generate start (prompt_len={}, call#={})", prompt_ids.shape[1], n + 1)
                    out = hf_model.generate(
                        prompt_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=2048,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                    )
                    _PYTHON_GPU_GENERATE_LOG_COUNT = n + 1
                    if n < 8:
                        logger.info("python_gpu: generate done (out_len={}, call#={})", out.shape[1], n + 1)
                    gen = out[0][prompt_ids.shape[1]:]
                    return tokenizer.decode(gen, skip_special_tokens=True).strip()
                except Exception as e:
                    logger.warning(f"python_gpu generate failed: {e}")
                    raise
            model_callable = _call
            is_endpoint = True
        elif isinstance(model, dict) and model.get("api_type") == "vllm_endpoint" and "model" in model:
            vllm_model_name = model["model"]
            vllm_endpoint = model["endpoint"]
            vllm_max_tokens = int(model.get("max_tokens", 2048))
            vllm_max_context = int(model.get("context_window", 8192))
            vllm_timeout = float(model.get("timeout", 300.0))
            vllm_prompt_suffix = model.get("prompt_suffix", "")
            context_window = model.get("context_window")
            ctx_tokens = int(context_window) if context_window is not None else 8192
            no_rag_max_tokens = model.get("no_rag_max_tokens")
            if no_rag_max_tokens is None and context_window is not None:
                no_rag_max_tokens = max(0, ctx_tokens - 4096)
            if no_rag_max_tokens is None or no_rag_max_tokens <= 0:
                no_rag_max_tokens = 512
            no_rag_max_tokens = min(no_rag_max_tokens, max(256, ctx_tokens - 2048 - 512))
            vllm_tokenizer = None
            try:
                from transformers import AutoTokenizer
                vllm_tokenizer = AutoTokenizer.from_pretrained(vllm_model_name)
            except Exception:
                logger.debug("vllm_endpoint: tokenizer not available for {}, using char-based truncation", vllm_model_name)

            def _call(
                source_text: str,
                evidence: str,
                source_version: Optional[int] = None,
                target_version: Optional[int] = None,
            ) -> str:
                has_document = evidence is not None and bool(evidence.strip())
                user_prompt = _format_model_prompt(
                    source_text, prompts.model_prompt,
                    has_document=has_document, source_version=source_version, target_version=target_version,
                )
                system_prompt = prompts.model_prompt.get("system", "")
                if evidence and evidence.strip():
                    ev = evidence.strip()
                    if vllm_tokenizer is not None:
                        enc = vllm_tokenizer.encode(ev)
                        if len(enc) > no_rag_max_tokens:
                            ev = vllm_tokenizer.decode(enc[:no_rag_max_tokens])
                    else:
                        max_chars = no_rag_max_tokens * 4
                        if len(ev) > max_chars:
                            ev = ev[:max_chars]
                    user_prompt += f"\n\nSupporting evidence:\n{ev}"
                if vllm_prompt_suffix:
                    user_prompt += f"\n\n{vllm_prompt_suffix}"
                return _vllm_endpoint_predict_sync(
                    vllm_endpoint, vllm_model_name, system_prompt, user_prompt,
                    max_tokens=vllm_max_tokens, max_context=vllm_max_context,
                    timeout=vllm_timeout,
                )
            model_callable = _call
            is_endpoint = True
        else:
            raise ValueError("model must be a callable or dict with api_type azure_openai, endpoint/model, python_gpu, or vllm_endpoint")

        # Get prediction
        source_text = sample_row.get("source_text") or ""
        evidence = sample_row.get("evidence")
        if evidence is None or (hasattr(evidence, "__float__") and pd.isna(evidence)):
            evidence = ""
        else:
            evidence = str(evidence)
        source_version = _v("source_version", 0)
        target_version = _v("target_version", 0)

        if use_multi_stage and is_endpoint and prompts.multi_stage_prompts:
            tsp = prompts.multi_stage_prompts
            step1_system = tsp.get("step1_system", "")
            step1_user = tsp["step1_user_template"]
            schema_text = _query_with_rag_retry(
                system_prompt=step1_system,
                user_prompt=step1_user,
                document=evidence,
                llm_base_url=endpoint,
                llm_model=llm_model,
                rag_config=rag_cfg,
                persist_dir=persist_dir,
                max_retries=max_retries,
                base_delay=0.5,
                max_delay=max_delay,
                wait_for_revive_seconds=wait_for_revive_seconds,
                azure_openai_model=model.get("model") if isinstance(model, dict) and model.get("api_type") == "azure_openai" else None,
            )
            step2_user = tsp["step2_user_template"].format(
                schema=schema_text, source_version=source_version, target_version=target_version, source_text=source_text,
            )
            step2_system = tsp.get("step2_system", "")
            prediction = _query_with_rag_retry(
                system_prompt=step2_system,
                user_prompt=step2_user,
                document=evidence,
                llm_base_url=endpoint,
                llm_model=llm_model,
                rag_config=rag_cfg,
                persist_dir=persist_dir,
                max_retries=max_retries,
                base_delay=0.5,
                max_delay=max_delay,
                wait_for_revive_seconds=wait_for_revive_seconds,
                azure_openai_model=model.get("model") if isinstance(model, dict) and model.get("api_type") == "azure_openai" else None,
            )
        else:
            prediction = _call_model_with_truncation_retry(
                model_callable,
                model,
                source_text,
                evidence,
                source_version,
                target_version,
            )

        reference_text = sample_row.get("reference_text") or ""
        default_metrics = default_evaluator(reference_text, prediction)
        if evaluator_type == "llm_judge" and evaluator_config:
            api_type = evaluator_config.get("api_type", "endpoint")
            if api_type == "azure_openai" and "model" in evaluator_config:
                llm_judge_metrics = azure_openai_judge_evaluator(
                    reference_text, prediction,
                    model=evaluator_config["model"],
                    prompts=prompts,
                )
            elif api_type == "endpoint" and "endpoint" in evaluator_config and "model" in evaluator_config:
                llm_judge_metrics = llm_judge_evaluator(
                    reference_text, prediction,
                    endpoint=evaluator_config["endpoint"],
                    model=evaluator_config["model"],
                    prompts=prompts,
                    wait_for_revive_seconds=wait_for_revive_seconds,
                )
            else:
                llm_judge_metrics = {}
            combined_metrics = {**default_metrics, **llm_judge_metrics}
        else:
            combined_metrics = default_metrics

        predicted_at = datetime.now().isoformat()
        result = {
            "id": _v("id", 0),
            "split_name": split_name,
            "instance_id": str(sample_row.get("instance_id", "")),
            "source_version": source_version,
            "target_version": target_version,
            "source_text": source_text,
            "evidence": evidence,
            "reference_text": reference_text,
            "prediction": prediction,
            "quality_score": _v("quality_score"),
            "year": _v("year"),
            "study_type": sample_row.get("study_type"),
            "predicted_at": predicted_at,
            **combined_metrics,
        }
        return result
    except Exception as e:
        sid = sample_row.get("id")
        tb_lines = traceback.format_exc().strip().split("\n")
        # Last 2–3 lines usually show the failing line (e.g. assert in a library)
        tb_tail = "\n".join(tb_lines[-4:]) if len(tb_lines) >= 4 else traceback.format_exc()
        logger.warning(
            "run_single_sample failed for id={}: {}: {}\n{}",
            sid,
            type(e).__name__,
            e,
            tb_tail,
        )
        logger.debug("run_single_sample full traceback for id={}", sid, exc_info=True)
        return None


def run_benchmark(
    model: Union[Callable[[str, str], str], Dict[str, str]],
    parquet_paths: Dict[str, Path],
    output_dir: Path,
    evaluator_type: str = "default",
    evaluator_config: Optional[Dict] = None,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    prompts_path: Optional[Path] = None,
    multi_stage: bool = False,
    rag_config: Optional[Dict[str, Any]] = None,
    wait_for_revive_seconds: int = 0,
    done_sample_ids: Optional[Dict[str, Set[int]]] = None,
    max_concurrent_requests: int = 1,
) -> Dict[str, Any]:
    """Run benchmark evaluation on parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_benchmark_prompts(prompts_path)

    use_multi_stage = multi_stage and prompts.multi_stage_prompts is not None

    if callable(model):
        model_callable = model
        is_endpoint = False
        if use_multi_stage:
            logger.warning("multi_stage=True requires endpoint-based model; using single-step for callable")
            use_multi_stage = False
    elif isinstance(model, dict) and model.get("api_type") == "azure_openai" and "model" in model:
        azure_openai_model = model["model"]
        endpoint = "n/a"
        llm_model = azure_openai_model
        is_endpoint = True
        if rag_config is None:
            raise ValueError(
                "rag_config required for azure_openai model (embed_base_url, embed_model, persist_dir)."
            )
        persist_dir_raw = rag_config.get("persist_dir")
        if persist_dir_raw is None:
            raise ValueError("rag_config must include persist_dir")
        persist_dir = Path(persist_dir_raw)
        if not persist_dir.is_absolute():
            persist_dir = get_project_root() / persist_dir
        rag_cfg = dict(rag_config)
        if model.get("top_k") is not None:
            rag_cfg["similarity_top_k"] = model["top_k"]
        elif rag_cfg.get("similarity_top_k") is None and rag_cfg.get("top_k") is not None:
            rag_cfg["similarity_top_k"] = rag_cfg["top_k"]

        def model_callable(
            source_text: str,
            evidence: str,
            source_version: Optional[int] = None,
            target_version: Optional[int] = None,
        ) -> str:
            has_document = evidence is not None and bool(evidence.strip())
            user_prompt = _format_model_prompt(
                source_text,
                prompts.model_prompt,
                has_document=has_document,
                source_version=source_version,
                target_version=target_version,
            )
            system_prompt = prompts.model_prompt.get("system", "")
            return _query_with_rag_retry(
                system_prompt=system_prompt or "",
                user_prompt=user_prompt,
                document=evidence if evidence is not None else "",
                llm_base_url=endpoint,
                llm_model=llm_model,
                rag_config=rag_cfg,
                persist_dir=persist_dir,
                wait_for_revive_seconds=wait_for_revive_seconds,
                azure_openai_model=azure_openai_model,
            )
    elif isinstance(model, dict) and model.get("api_type") == "python_gpu" and "model" in model:
        import torch
        model_name = model["model"]
        device = model.get("device", "cuda")
        gpu_ids = model.get("gpu_ids")
        gpus = model.get("gpus")
        if gpu_ids is None and gpus is not None and int(gpus) > 1:
            gpu_ids = list(range(int(gpus)))
        hf_model, tokenizer = _get_python_gpu_model(model_name, device, gpu_ids=gpu_ids)
        context_window = model.get("context_window")
        ctx_tokens = int(context_window) if context_window is not None else 131072
        no_rag_max_tokens = model.get("no_rag_max_tokens")
        if no_rag_max_tokens is None and context_window is not None:
            no_rag_max_tokens = max(0, ctx_tokens - 4096)
        if no_rag_max_tokens is None or no_rag_max_tokens <= 0:
            no_rag_max_tokens = 512
        no_rag_max_tokens = min(no_rag_max_tokens, max(256, ctx_tokens - 2048 - 512))
        is_endpoint = False

        def model_callable(
            source_text: str,
            evidence: str,
            source_version: Optional[int] = None,
            target_version: Optional[int] = None,
        ) -> str:
            global _PYTHON_GPU_GENERATE_LOG_COUNT
            has_document = evidence is not None and bool(evidence.strip())
            user_prompt = _format_model_prompt(
                source_text, prompts.model_prompt,
                has_document=has_document, source_version=source_version, target_version=target_version,
            )
            system_prompt = prompts.model_prompt.get("system", "")
            if evidence and evidence.strip():
                ev = evidence.strip()
                enc = tokenizer.encode(ev)
                if len(enc) > no_rag_max_tokens:
                    ev = tokenizer.decode(enc[:no_rag_max_tokens])
                user_content = f"{user_prompt}\n\nSupporting evidence:\n{ev}"
            else:
                user_content = user_prompt
            system_prompt = (system_prompt or "").strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            max_len = min(
                tokenizer.model_max_length or 131072,
                ctx_tokens,
            )
            def _apply_template_ids(msgs):
                result = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors="pt",
                    truncation=True, max_length=max_len,
                )
                if hasattr(result, "keys") and "input_ids" in result:
                    return result["input_ids"]
                return result

            try:
                prompt_ids = _apply_template_ids(messages)
            except Exception as template_err:
                if "system" in str(template_err).lower() or "TemplateError" in type(template_err).__name__:
                    messages_user_only = [
                        {"role": "user", "content": f"{system_prompt}\n\n{user_content}".strip() if system_prompt else user_content},
                    ]
                    prompt_ids = _apply_template_ids(messages_user_only)
                else:
                    raise
            seq_len = prompt_ids.shape[1]
            if seq_len > max_len:
                prompt_ids = prompt_ids[:, -max_len:]
            prompt_ids = prompt_ids.to(hf_model.device)
            attention_mask = prompt_ids.new_ones(prompt_ids.shape, dtype=torch.long)
            n = _PYTHON_GPU_GENERATE_LOG_COUNT
            if n < 8:
                logger.info("python_gpu: generate start (prompt_len={}, call#={})", prompt_ids.shape[1], n + 1)
            out = hf_model.generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
            _PYTHON_GPU_GENERATE_LOG_COUNT = n + 1
            if n < 8:
                logger.info("python_gpu: generate done (out_len={}, call#={})", out.shape[1], n + 1)
            gen = out[0][prompt_ids.shape[1]:]
            return tokenizer.decode(gen, skip_special_tokens=True).strip()
    elif isinstance(model, dict) and model.get("api_type") == "vllm_endpoint" and "model" in model:
        vllm_model_name = model["model"]
        vllm_endpoint = model["endpoint"]
        vllm_max_tokens = int(model.get("max_tokens", 2048))
        vllm_max_context = int(model.get("context_window", 8192))
        vllm_timeout = float(model.get("timeout", 300.0))
        vllm_prompt_suffix = model.get("prompt_suffix", "")
        vllm_max_concurrent = int(model.get("max_concurrent", 16))
        vllm_save_every = int(model.get("save_every", 50))
        context_window = model.get("context_window")
        ctx_tokens = int(context_window) if context_window is not None else 8192
        no_rag_max_tokens = model.get("no_rag_max_tokens")
        if no_rag_max_tokens is None and context_window is not None:
            no_rag_max_tokens = max(0, ctx_tokens - 4096)
        if no_rag_max_tokens is None or no_rag_max_tokens <= 0:
            no_rag_max_tokens = 512
        no_rag_max_tokens = min(no_rag_max_tokens, max(256, ctx_tokens - 2048 - 512))
        vllm_tokenizer = None
        try:
            from transformers import AutoTokenizer
            vllm_tokenizer = AutoTokenizer.from_pretrained(vllm_model_name)
            logger.info("vllm_endpoint: tokenizer loaded ({})", vllm_model_name)
        except Exception:
            logger.info("vllm_endpoint: tokenizer not available for {}, using char-based truncation", vllm_model_name)
        logger.info("vllm_endpoint: model={}, endpoint={}, ctx={}, evidence_budget={}, concurrent={}",
                     vllm_model_name, vllm_endpoint, ctx_tokens, no_rag_max_tokens, vllm_max_concurrent)
        is_endpoint = False

        def model_callable(
            source_text: str,
            evidence: str,
            source_version: Optional[int] = None,
            target_version: Optional[int] = None,
        ) -> str:
            has_document = evidence is not None and bool(evidence.strip())
            user_prompt = _format_model_prompt(
                source_text, prompts.model_prompt,
                has_document=has_document, source_version=source_version, target_version=target_version,
            )
            system_prompt = prompts.model_prompt.get("system", "")
            if evidence and evidence.strip():
                ev = evidence.strip()
                if vllm_tokenizer is not None:
                    enc = vllm_tokenizer.encode(ev)
                    if len(enc) > no_rag_max_tokens:
                        ev = vllm_tokenizer.decode(enc[:no_rag_max_tokens])
                else:
                    max_chars = no_rag_max_tokens * 4
                    if len(ev) > max_chars:
                        ev = ev[:max_chars]
                user_prompt += f"\n\nSupporting evidence:\n{ev}"
            if vllm_prompt_suffix:
                user_prompt += f"\n\n{vllm_prompt_suffix}"
            return _vllm_endpoint_predict_sync(
                vllm_endpoint, vllm_model_name, system_prompt, user_prompt,
                max_tokens=vllm_max_tokens, max_context=vllm_max_context,
                timeout=vllm_timeout,
            )
    elif isinstance(model, dict) and "endpoint" in model and "model" in model:
        endpoint = model["endpoint"]
        llm_model = model["model"]
        is_endpoint = True
        if rag_config is None:
            raise ValueError(
                "rag_config required for endpoint-based model (embed_base_url, embed_model, persist_dir)."
            )
        persist_dir_raw = rag_config.get("persist_dir")
        if persist_dir_raw is None:
            raise ValueError("rag_config must include persist_dir")
        persist_dir = Path(persist_dir_raw)
        if not persist_dir.is_absolute():
            persist_dir = get_project_root() / persist_dir
        rag_cfg = dict(rag_config)
        if model.get("top_k") is not None:
            rag_cfg["similarity_top_k"] = model["top_k"]
        elif rag_cfg.get("similarity_top_k") is None and rag_cfg.get("top_k") is not None:
            rag_cfg["similarity_top_k"] = rag_cfg["top_k"]

        def model_callable(
            source_text: str,
            evidence: str,
            source_version: Optional[int] = None,
            target_version: Optional[int] = None,
        ) -> str:
            has_document = evidence is not None and bool(evidence.strip())
            user_prompt = _format_model_prompt(
                source_text,
                prompts.model_prompt,
                has_document=has_document,
                source_version=source_version,
                target_version=target_version,
            )
            system_prompt = prompts.model_prompt.get("system", "")
            return _query_with_rag_retry(
                system_prompt=system_prompt or "",
                user_prompt=user_prompt,
                document=evidence if evidence is not None else "",
                llm_base_url=endpoint,
                llm_model=llm_model,
                rag_config=rag_cfg,
                persist_dir=persist_dir,
                wait_for_revive_seconds=wait_for_revive_seconds,
            )
    else:
        raise ValueError("model must be either a callable or a dict with 'endpoint'/'model' keys, or api_type 'python_gpu'/'azure_openai'/'vllm_endpoint'")
    
    splits = {}
    required_columns = ["source_text", "evidence", "reference_text", "id", "instance_id", "source_version", "target_version"]

    for split_name, path in parquet_paths.items():
        if path.exists():
            df = _normalize_columns(pd.read_parquet(path))

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Parquet file {path} missing required columns: {missing_cols}")
            
            if num_samples is not None:
                if num_samples < 0:
                    raise ValueError(f"num_samples must be non-negative, got {num_samples}")
                original_len = len(df)
                df = df.head(num_samples)
                logger.info(f"Limiting {split_name} split to first {len(df)} samples (from {original_len})")
            
            if done_sample_ids:
                done = done_sample_ids.get(split_name, set())
                if done:
                    df = df[~df["id"].astype(int).isin(done)].copy()
                    logger.info(f"Loaded {len(df)} remaining samples from {split_name} split ({len(done)} already done)")
            splits[split_name] = df
            if len(df) > 0:
                logger.info(f"Loaded {len(df)} samples from {split_name} split")
        else:
            logger.warning(f"Parquet file not found: {path}, skipping {split_name} split")
    
    if not splits:
        raise ValueError("No valid parquet files found")
    
    default_eval_fn = default_evaluator
    llm_judge_eval_fn = None
    if evaluator_type == "llm_judge":
        if not evaluator_config:
            raise ValueError("llm_judge evaluator requires evaluator_config")
        
        api_type = evaluator_config.get("api_type", "endpoint")

        if api_type == "azure_openai":
            if "model" not in evaluator_config:
                raise ValueError("azure_openai judge requires 'model' in evaluator_config")
            judge_model = evaluator_config["model"]
            llm_judge_eval_fn = lambda gt, pred: azure_openai_judge_evaluator(
                gt, pred,
                model=judge_model,
                prompts=prompts,
            )
        elif api_type == "endpoint":
            if "endpoint" not in evaluator_config or "model" not in evaluator_config:
                raise ValueError("endpoint judge requires 'endpoint' and 'model' in evaluator_config")
            llm_judge_eval_fn = lambda gt, pred: llm_judge_evaluator(
                gt, pred,
                endpoint=evaluator_config["endpoint"],
                model=evaluator_config["model"],
                prompts=prompts,
                wait_for_revive_seconds=wait_for_revive_seconds,
            )
        else:
            raise ValueError(f"Unknown api_type '{api_type}' in evaluator_config. Use 'azure_openai' or 'endpoint'")
    elif evaluator_type != "default":
        raise ValueError(f"Unknown evaluator_type: {evaluator_type}. Use 'default' or 'llm_judge'")
    
    all_results = {}
    all_predictions = []

    for split_name, df in splits.items():
        if len(df) == 0 and done_sample_ids:
            existing_results = _load_existing_results(output_dir, split_name)
            if existing_results:
                results_df = pd.DataFrame(existing_results)
                all_results[split_name] = {
                    "count": len(existing_results),
                    "metrics": {
                        metric: {
                            "mean": float(results_df[metric].mean()),
                            "std": float(results_df[metric].std()),
                            "min": float(results_df[metric].min()),
                            "max": float(results_df[metric].max()),
                        }
                        for metric in results_df.columns
                        if metric not in [
                            "id", "instance_id", "llm_judge_raw_response",
                            "source_version", "target_version", "source_text", "evidence", "reference_text", "prediction",
                        ]
                    },
                }
            logger.info(f"Split {split_name} already complete ({len(existing_results)} samples), skipped.")
            continue

        logger.info(f"Evaluating {split_name} split ({len(df)} samples)...")
        
        predictions = []
        results = []
        
        total_in_split = len(df)
        done_this_run: Set[int] = set(done_sample_ids.get(split_name, set())) if done_sample_ids else set()

        def _run_one_row(row_index: int) -> Tuple[PredictionRecord, Dict[str, Any]]:
            row = df.iloc[row_index]
            if use_multi_stage and is_endpoint:
                tsp = prompts.multi_stage_prompts
                evidence_str = row["evidence"] if row["evidence"] is not None else ""
                step1_user = tsp["step1_user_template"]
                step1_system = tsp.get("step1_system", "")
                schema_text = _query_with_rag_retry(
                    system_prompt=step1_system,
                    user_prompt=step1_user,
                    document=evidence_str,
                    llm_base_url=endpoint,
                    llm_model=llm_model,
                    rag_config=rag_cfg,
                    persist_dir=persist_dir,
                    wait_for_revive_seconds=wait_for_revive_seconds,
                )
                step2_user = tsp["step2_user_template"].format(
                    schema=schema_text,
                    source_version=int(row["source_version"]),
                    target_version=int(row["target_version"]),
                    source_text=row["source_text"],
                )
                step2_system = tsp.get("step2_system", "")
                prediction = _query_with_rag_retry(
                    system_prompt=step2_system,
                    user_prompt=step2_user,
                    document=evidence_str,
                    llm_base_url=endpoint,
                    llm_model=llm_model,
                    rag_config=rag_cfg,
                    persist_dir=persist_dir,
                    wait_for_revive_seconds=wait_for_revive_seconds,
                )
            else:
                if isinstance(model, dict):
                    prediction = model_callable(
                        row["source_text"],
                        row["evidence"],
                        source_version=int(row["source_version"]),
                        target_version=int(row["target_version"]),
                    )
                else:
                    prediction = model_callable(row["source_text"], row["evidence"])
            metadata = {
                "quality_score": float(row.get("quality_score", 0)) if pd.notna(row.get("quality_score")) else None,
                "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else None,
                "study_type": row.get("study_type"),
            } if "quality_score" in row else {}
            pred_record = PredictionRecord(
                id=int(row["id"]),
                instance_id=row["instance_id"],
                source_version=int(row["source_version"]),
                target_version=int(row["target_version"]),
                source_text=row["source_text"],
                evidence=row["evidence"],
                reference_text=row["reference_text"],
                prediction=prediction,
                timestamp=datetime.now().isoformat(),
                metadata=metadata if metadata else None,
            )
            default_metrics = default_eval_fn(row["reference_text"], prediction)
            if llm_judge_eval_fn is not None:
                llm_judge_metrics = llm_judge_eval_fn(row["reference_text"], prediction)
                combined_metrics = {**default_metrics, **llm_judge_metrics}
            else:
                combined_metrics = default_metrics
            result_dict = {
                "id": int(row["id"]),
                "instance_id": row["instance_id"],
                "source_version": int(row["source_version"]),
                "target_version": int(row["target_version"]),
                "source_text": row["source_text"],
                "evidence": row["evidence"],
                "reference_text": row["reference_text"],
                "prediction": prediction,
                **combined_metrics,
            }
            return (pred_record, result_dict)

        if max_concurrent_requests <= 1:
            for sample_idx, (idx, row) in enumerate(df.iterrows()):
                try:
                    logger.debug(
                        "Starting %s sample %s/%s",
                        split_name, sample_idx + 1, total_in_split,
                    )
                    if use_multi_stage and is_endpoint:
                        tsp = prompts.multi_stage_prompts
                        evidence_str = row["evidence"] if row["evidence"] is not None else ""
                        step1_user = tsp["step1_user_template"]
                        step1_system = tsp.get("step1_system", "")
                        logger.debug(
                            "RAG step1 (schema) for %s sample %s/%s",
                            split_name, sample_idx + 1, total_in_split,
                        )
                        schema_text = _query_with_rag_retry(
                            system_prompt=step1_system,
                            user_prompt=step1_user,
                            document=evidence_str,
                            llm_base_url=endpoint,
                            llm_model=llm_model,
                            rag_config=rag_cfg,
                            persist_dir=persist_dir,
                            wait_for_revive_seconds=wait_for_revive_seconds,
                        )
                        step2_user = tsp["step2_user_template"].format(
                            schema=schema_text,
                            source_version=int(row["source_version"]),
                            target_version=int(row["target_version"]),
                            source_text=row["source_text"],
                        )
                        step2_system = tsp.get("step2_system", "")
                        logger.debug(
                            "RAG step2 (amended EC) for %s sample %s/%s",
                            split_name, sample_idx + 1, total_in_split,
                        )
                        prediction = _query_with_rag_retry(
                            system_prompt=step2_system,
                            user_prompt=step2_user,
                            document=evidence_str,
                            llm_base_url=endpoint,
                            llm_model=llm_model,
                            rag_config=rag_cfg,
                            persist_dir=persist_dir,
                            wait_for_revive_seconds=wait_for_revive_seconds,
                        )
                    else:
                        logger.debug(
                            "LLM/RAG single step for %s sample %s/%s",
                            split_name, sample_idx + 1, total_in_split,
                        )
                        if isinstance(model, dict):
                            prediction = model_callable(
                                row["source_text"],
                                row["evidence"],
                                source_version=int(row["source_version"]),
                                target_version=int(row["target_version"]),
                            )
                        else:
                            prediction = model_callable(row["source_text"], row["evidence"])
                    logger.debug(
                        "Sample %s/%s done",
                        sample_idx + 1, total_in_split,
                    )
                    metadata = {
                        "quality_score": float(row.get("quality_score", 0)) if pd.notna(row.get("quality_score")) else None,
                        "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else None,
                        "study_type": row.get("study_type"),
                    } if "quality_score" in row else {}
                    pred_record = PredictionRecord(
                        id=int(row["id"]),
                        instance_id=row["instance_id"],
                        source_version=int(row["source_version"]),
                        target_version=int(row["target_version"]),
                        source_text=row["source_text"],
                        evidence=row["evidence"],
                        reference_text=row["reference_text"],
                        prediction=prediction,
                        timestamp=datetime.now().isoformat(),
                        metadata=metadata if metadata else None,
                    )
                    predictions.append(pred_record)
                    default_metrics = default_eval_fn(row["reference_text"], prediction)
                    if llm_judge_eval_fn is not None:
                        llm_judge_metrics = llm_judge_eval_fn(row["reference_text"], prediction)
                        combined_metrics = {**default_metrics, **llm_judge_metrics}
                    else:
                        combined_metrics = default_metrics
                    results.append({
                        "id": int(row["id"]),
                        "instance_id": row["instance_id"],
                        "source_version": int(row["source_version"]),
                        "target_version": int(row["target_version"]),
                        "source_text": row["source_text"],
                        "evidence": row["evidence"],
                        "reference_text": row["reference_text"],
                        "prediction": prediction,
                        **combined_metrics,
                    })
                    if len(predictions) % batch_size == 0:
                        _save_predictions_checkpoint(predictions, output_dir, split_name)
                except Exception as e:
                    logger.error(
                        "Error processing sample id={} instance_id={}: {}",
                        row.get("id", idx),
                        row.get("instance_id", "?"),
                        e,
                    )
                    logger.debug("Traceback:\n{}", traceback.format_exc())
                    continue
        else:
            for chunk_start in range(0, len(df), max_concurrent_requests):
                chunk_end = min(chunk_start + max_concurrent_requests, len(df))
                chunk_indices = list(range(chunk_start, chunk_end))
                chunk_ids_done = {int(df.iloc[i]["id"]) for i in chunk_indices if int(df.iloc[i]["id"]) in done_this_run}
                existing_preds: Dict[int, PredictionRecord] = {}
                existing_res: Dict[int, Dict[str, Any]] = {}
                if chunk_ids_done:
                    existing_preds = _load_existing_predictions_by_ids(output_dir, split_name, chunk_ids_done)
                    existing_res = _load_existing_results_by_ids(output_dir, split_name, chunk_ids_done)
                todo_indices = [i for i in chunk_indices if int(df.iloc[i]["id"]) not in done_this_run]
                results_by_index: Dict[int, Tuple[PredictionRecord, Dict[str, Any]]] = {}
                if todo_indices:
                    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
                        futures = {executor.submit(_run_one_row, i): i for i in todo_indices}
                        for fut in as_completed(futures):
                            i = futures[fut]
                            try:
                                pred_record, result_dict = fut.result()
                                results_by_index[i] = (pred_record, result_dict)
                            except Exception as e:
                                logger.error("Error processing sample index {}: {}", i, e)
                chunk_predictions = []
                chunk_results = []
                for i in chunk_indices:
                    sid = int(df.iloc[i]["id"])
                    if sid in existing_preds:
                        chunk_predictions.append(existing_preds[sid])
                        chunk_results.append(existing_res[sid])
                    elif i in results_by_index:
                        chunk_predictions.append(results_by_index[i][0])
                        chunk_results.append(results_by_index[i][1])
                    else:
                        continue
                predictions.extend(chunk_predictions)
                results.extend(chunk_results)
                if chunk_start == 0 and not done_this_run:
                    _save_predictions(chunk_predictions, output_dir, split_name)
                    _save_results(chunk_results, output_dir, split_name)
                else:
                    _append_predictions_chunk(chunk_predictions, output_dir, split_name)
                    _append_results_chunk(chunk_results, output_dir, split_name)
                done_this_run.update(int(df.iloc[i]["id"]) for i in chunk_indices)

        if done_sample_ids:
            existing_preds = _load_existing_predictions(output_dir, split_name)
            existing_res = _load_existing_results(output_dir, split_name)
            by_id_existing = {int(r["id"]): r for r in existing_res}
            by_id_pred_existing = {p.id: p for p in existing_preds}
            for r in results:
                by_id_existing[int(r["id"])] = r
            for p in predictions:
                by_id_pred_existing[p.id] = p
            merged_results = [by_id_existing[sid] for sid in sorted(by_id_existing)]
            merged_predictions = [by_id_pred_existing[sid] for sid in sorted(by_id_pred_existing)]
            _save_predictions(merged_predictions, output_dir, split_name)
            _save_results(merged_results, output_dir, split_name)
            all_predictions.extend(merged_predictions)
            if merged_results:
                results_df = pd.DataFrame(merged_results)
                all_results[split_name] = {
                    "count": len(merged_results),
                    "metrics": {
                        metric: {
                            "mean": float(results_df[metric].mean()),
                            "std": float(results_df[metric].std()),
                            "min": float(results_df[metric].min()),
                            "max": float(results_df[metric].max()),
                        }
                        for metric in results_df.columns
                        if metric not in [
                            "id", "instance_id", "llm_judge_raw_response",
                            "source_version", "target_version", "source_text", "evidence", "reference_text", "prediction",
                        ]
                    },
                }
        else:
            _save_predictions(predictions, output_dir, split_name)
            all_predictions.extend(predictions)
            if results:
                _save_results(results, output_dir, split_name)
                results_df = pd.DataFrame(results)
                all_results[split_name] = {
                    "count": len(results),
                    "metrics": {
                        metric: {
                            "mean": float(results_df[metric].mean()),
                            "std": float(results_df[metric].std()),
                            "min": float(results_df[metric].min()),
                            "max": float(results_df[metric].max()),
                        }
                        for metric in results_df.columns
                        if metric not in [
                            "id", "instance_id", "llm_judge_raw_response",
                            "source_version", "target_version", "source_text", "evidence", "reference_text", "prediction",
                        ]
                    },
                }
    
    total_count = sum(all_results[s]["count"] for s in all_results) if all_results else len(all_predictions)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "evaluator_type": evaluator_type,
        "total_predictions": total_count,
        "splits": all_results,
    }
    
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    
    return summary


def _load_existing_predictions(output_dir: Path, split_name: str) -> List[PredictionRecord]:
    """Load existing predictions from JSONL if present. Return empty list on error or missing."""
    path = output_dir / f"predictions_{split_name}.jsonl"
    if not path.exists():
        return []
    out: List[PredictionRecord] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out.append(PredictionRecord(
                    id=int(row["id"]),
                    instance_id=row.get("instance_id", ""),
                    source_version=int(row.get("source_version", 0)),
                    target_version=int(row.get("target_version", 0)),
                    source_text=row.get("source_text", "") or "",
                    evidence=row.get("evidence", "") or "",
                    reference_text=row.get("reference_text", "") or "",
                    prediction=row.get("prediction", "") or "",
                    timestamp=row.get("timestamp", ""),
                    metadata=row.get("metadata"),
                ))
    except Exception:
        return []
    return out


def _load_existing_results(output_dir: Path, split_name: str) -> List[Dict[str, Any]]:
    """Load existing results from JSONL if present. Return empty list on error or missing."""
    path = output_dir / f"results_{split_name}.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except Exception:
        return []
    return out


def _load_existing_predictions_by_ids(
    output_dir: Path, split_name: str, ids: Set[int]
) -> Dict[int, PredictionRecord]:
    """Load existing predictions for given ids. Returns dict id -> PredictionRecord."""
    all_preds = _load_existing_predictions(output_dir, split_name)
    return {p.id: p for p in all_preds if p.id in ids}


def _load_existing_results_by_ids(
    output_dir: Path, split_name: str, ids: Set[int]
) -> Dict[int, Dict[str, Any]]:
    """Load existing results for given ids. Returns dict id -> result row."""
    all_res = _load_existing_results(output_dir, split_name)
    return {int(r["id"]): r for r in all_res if int(r["id"]) in ids}


def _append_predictions_chunk(
    predictions: List[PredictionRecord], output_dir: Path, split_name: str
) -> None:
    """Append prediction records to predictions JSONL file."""
    path = output_dir / f"predictions_{split_name}.jsonl"
    with open(path, "a") as f:
        for pred in predictions:
            f.write(json.dumps({
                "id": pred.id,
                "instance_id": pred.instance_id,
                "source_version": pred.source_version,
                "target_version": pred.target_version,
                "source_text": pred.source_text,
                "evidence": pred.evidence,
                "reference_text": pred.reference_text,
                "prediction": pred.prediction,
                "timestamp": pred.timestamp,
                "metadata": pred.metadata,
            }) + "\n")


def _append_results_chunk(
    results: List[Dict[str, Any]], output_dir: Path, split_name: str
) -> None:
    """Append result rows to results JSONL file."""
    path = output_dir / f"results_{split_name}.jsonl"
    with open(path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def _save_predictions(predictions: List[PredictionRecord], output_dir: Path, split_name: str):
    """Save predictions to JSONL file."""
    predictions_path = output_dir / f"predictions_{split_name}.jsonl"
    with open(predictions_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps({
                "id": pred.id,
                "instance_id": pred.instance_id,
                "source_version": pred.source_version,
                "target_version": pred.target_version,
                "source_text": pred.source_text,
                "evidence": pred.evidence,
                "reference_text": pred.reference_text,
                "prediction": pred.prediction,
                "timestamp": pred.timestamp,
                "metadata": pred.metadata,
            }) + "\n")
    
    logger.info(f"Saved {len(predictions)} predictions to {predictions_path}")


def _save_results(results: List[Dict[str, Any]], output_dir: Path, split_name: str):
    results_path = output_dir / f"results_{split_name}.jsonl"
    with open(results_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Saved {len(results)} evaluation results to {results_path}")


def _save_predictions_checkpoint(predictions: List[PredictionRecord], output_dir: Path, split_name: str):
    """Save predictions checkpoint (for resumability)."""
    checkpoint_path = output_dir / f"predictions_{split_name}_checkpoint.jsonl"
    with open(checkpoint_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps({
                "id": pred.id,
                "instance_id": pred.instance_id,
                "source_version": pred.source_version,
                "target_version": pred.target_version,
                "source_text": pred.source_text,
                "evidence": pred.evidence,
                "reference_text": pred.reference_text,
                "prediction": pred.prediction,
                "timestamp": pred.timestamp,
                "metadata": pred.metadata,
            }) + "\n")
