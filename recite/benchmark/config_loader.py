"""
Load benchmark YAML config and expand to experiment specs with content-addressed config_id.

Used by both the CLI and the recite orchestrator.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from recite.benchmark.results_db import compute_config_fingerprint
from recite.utils.path_loader import get_project_root

_RESERVED_PROMPT_TOKENS = 4096


def load_benchmark_config(config_path: Path) -> Dict[str, Any]:
    """Load benchmark YAML config. Path may be relative to project root."""
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_prompts_snapshot(prompts_file: Path, project_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load prompts JSON for fingerprint/snapshot. Returns None if file missing."""
    root = project_root or get_project_root()
    path = Path(prompts_file) if not isinstance(prompts_file, Path) else prompts_file
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize model id for use as directory name."""
    return re.sub(r"[^\w\-]", "_", model_id).strip("_") or "model"


def _resolve_path(path_val: Any, project_root: Path) -> Path:
    if path_val is None:
        return project_root / "data" / "benchmark_predictions"
    p = Path(path_val) if not isinstance(path_val, Path) else path_val
    if not p.is_absolute():
        p = project_root / p
    return p


def _normalize_top_k_list(config_top_k: Optional[Any]) -> List[int]:
    """Normalize config top_k to list of ints."""
    if config_top_k is None:
        return [2]
    if isinstance(config_top_k, list):
        return [int(x) for x in config_top_k]
    return [int(config_top_k)]


def _effective_no_rag_max_tokens(model_config: Dict[str, Any], reserved: int = _RESERVED_PROMPT_TOKENS) -> Optional[int]:
    """Per-model no_rag_max_tokens from context_window or explicit."""
    if not model_config:
        return None
    explicit = model_config.get("no_rag_max_tokens")
    if explicit is not None:
        return int(explicit)
    ctx = model_config.get("context_window")
    if ctx is not None:
        return max(0, int(ctx) - reserved)
    return None


def get_experiment_specs(
    config_path: Path,
    parquet_paths: Optional[Dict[str, Path]] = None,
    project_root: Optional[Path] = None,
    include_test: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load config and expand to list of experiment specs (model × top_k × no_rag/topk).
    Each spec has config_id (content-addressed fingerprint) and all fields needed for
    ensure_config and run_single_sample.

    Args:
        config_path: Path to benchmarks YAML.
        parquet_paths: If provided, use these; else resolve from config parquet_paths.
        project_root: For resolving relative paths. Default: get_project_root().
        include_test: If True, include test split in parquet_paths.

    Returns:
        List of spec dicts with config_id, model_id, model, top_k, no_rag, parquet_paths,
        rag_config, evaluator_type, evaluator_config, prompts_file, prompts_snapshot,
        two_step, batch_size, num_samples, wait_for_revive_seconds, config_path, run_started_at.
    """
    root = project_root or get_project_root()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    bench_config = load_benchmark_config(config_path)

    # Parquet paths (optionally filtered by splits_to_run)
    if parquet_paths is not None:
        paths = {k: Path(v) for k, v in parquet_paths.items()}
    else:
        pp = bench_config.get("parquet_paths") or {}
        default_dir = root / "data" / "benchmark_splits"
        all_paths = {
            "benchmark": _resolve_path(pp.get("benchmark", default_dir / "benchmark.parquet"), root),
            "train": _resolve_path(pp.get("train", default_dir / "train.parquet"), root),
            "val": _resolve_path(pp.get("val", default_dir / "val.parquet"), root),
            "test": _resolve_path(pp.get("test", default_dir / "test.parquet"), root),
        }
        for key in pp:
            if key not in all_paths:
                all_paths[key] = _resolve_path(pp[key], root)
        splits_to_run = bench_config.get("splits_to_run")
        if splits_to_run is not None:
            paths = {k: all_paths[k] for k in splits_to_run if k in all_paths}
        else:
            paths = all_paths
        if include_test and "test" not in paths and "test" in all_paths:
            paths["test"] = all_paths["test"]
    parquet_paths_str = {k: str(v.resolve()) for k, v in paths.items()}

    # Prompts
    prompts_file = bench_config.get("prompts_file", "config/benchmark_prompts.json")
    prompts_file_path = _resolve_path(prompts_file, root)
    prompts_snapshot = load_prompts_snapshot(prompts_file_path, root)

    # RAG config
    rag_config: Optional[Dict[str, Any]] = None
    if bench_config.get("rag"):
        r = bench_config["rag"]
        rag_config = {
            "embed_base_url": r.get("embed_base_url") or os.environ.get("RAG_EMBED_URL", ""),
            "embed_model": r.get("embed_model") or os.environ.get("RAG_EMBED_MODEL", ""),
            "persist_dir": r.get("persist_dir", "data/llamaindex_cache"),
            "embed_api_key": r.get("embed_api_key") or os.environ.get("RAG_EMBED_API_KEY") or os.environ.get("UCSF_API_KEY"),
            "embed_api_version": r.get("embed_api_version") or os.environ.get("UCSF_API_VER") or os.environ.get("API_VERSION"),
        }
        if r.get("embed_local_model"):
            rag_config["embed_local_model"] = r.get("embed_local_model")
            rag_config["embed_device_index"] = r.get("embed_device_index", "cuda:0")
            rag_config["embed_device_query"] = r.get("embed_device_query", "cpu")
        r_top_k = r.get("top_k")
        if r_top_k is not None and not isinstance(r_top_k, list):
            rag_config["similarity_top_k"] = r_top_k
        ucsf_endpoint = (os.environ.get("UCSF_RESOURCE_ENDPOINT") or "").strip().rstrip("/")
        if ucsf_endpoint and not rag_config["embed_base_url"] and rag_config["embed_model"]:
            rag_config["embed_base_url"] = f"{ucsf_endpoint}/openai/deployments/{rag_config['embed_model']}"

    evaluator_type = bench_config.get("evaluator_type", "default")
    evaluator_config: Optional[Dict[str, Any]] = None
    if evaluator_type == "llm_judge":
        judge_api_type = bench_config.get("judge_api_type", "ucsf_versa")
        judge_model_type = bench_config.get("judge_model_type", "4o")
        model_type_map = {
            "4o": "gpt-4o-2024-08-06",
            "4o-mini": "gpt-4o-mini-2024-07-18",
            "4.5-preview": "gpt-4.5-preview",
            "4-turbo": "gpt-4-turbo-128k",
        }
        judge_model = model_type_map.get(judge_model_type, judge_model_type)
        evaluator_config = {"api_type": judge_api_type, "model": judge_model}

    two_step = bench_config.get("two_step", False)
    batch_size = int(bench_config.get("batch_size", 10))
    num_samples = bench_config.get("num_samples")
    wait_for_revive_seconds = int(bench_config.get("wait_for_revive_seconds", 0))
    top_k_list = _normalize_top_k_list(bench_config.get("top_k"))
    sweep_no_rag = bool((bench_config.get("rag") or {}).get("sweep_no_rag", False))
    sweep_rag = bool((bench_config.get("rag") or {}).get("sweep_rag", True))
    prompt_version = bench_config.get("prompt_version")

    run_started_at = datetime.now(timezone.utc).isoformat()
    specs: List[Dict[str, Any]] = []

    models_list = bench_config.get("models") or []
    for m in models_list:
        api_type = m.get("api_type", "endpoint")
        mod = m.get("model")
        if api_type == "ucsf_versa":
            if not mod:
                logger.warning(f"Skipping model entry missing model (ucsf_versa): {m}")
                continue
            model_id = _sanitize_model_id(m.get("id", mod))
            model = {"api_type": "ucsf_versa", "model": mod}
            if m.get("context_window") is not None:
                model["context_window"] = int(m["context_window"])
            if m.get("no_rag_max_tokens") is not None:
                model["no_rag_max_tokens"] = int(m["no_rag_max_tokens"])
            two_step_val = m.get("two_step", two_step)
        elif api_type == "python_gpu":
            if not mod:
                logger.warning(f"Skipping model entry missing model (python_gpu): {m}")
                continue
            model_id = _sanitize_model_id(m.get("id", mod))
            model = {"api_type": "python_gpu", "model": mod}
            if m.get("context_window") is not None:
                model["context_window"] = int(m["context_window"])
            if m.get("no_rag_max_tokens") is not None:
                model["no_rag_max_tokens"] = int(m["no_rag_max_tokens"])
            model["device"] = m.get("device", "cuda")
            # Per-model GPU count (for multi-GPU models); default 1
            model["gpus"] = int(m.get("gpus", 1))
            two_step_val = m.get("two_step", two_step)
        else:
            ep = m.get("endpoint")
            if not ep or not mod:
                logger.warning(f"Skipping model entry missing endpoint or model: {m}")
                continue
            model_id = _sanitize_model_id(m.get("id", mod))
            model = {"endpoint": ep, "model": mod}
            two_step_val = m.get("two_step", two_step)

        no_rag_max_tokens = _effective_no_rag_max_tokens(m)

        for k in top_k_list:
            if sweep_no_rag:
                spec_no_rag = {
                    "model_id": model_id,
                    "model": dict(model),
                    "top_k": k,
                    "no_rag": True,
                    "parquet_paths": parquet_paths_str,
                    "prompts_file": str(prompts_file_path),
                    "prompts_snapshot": prompts_snapshot,
                    "evaluator_type": evaluator_type,
                    "evaluator_config": evaluator_config,
                    "two_step": two_step_val,
                    "batch_size": batch_size,
                    "num_samples": num_samples,
                    "wait_for_revive_seconds": wait_for_revive_seconds,
                    "config_path": str(config_path),
                    "run_started_at": run_started_at,
                    "prompt_version": prompt_version,
                }
                if rag_config is not None:
                    spec_no_rag["rag_config"] = {**rag_config, "no_rag": True, "similarity_top_k": k}
                    if no_rag_max_tokens is not None:
                        spec_no_rag["rag_config"]["no_rag_max_tokens"] = no_rag_max_tokens
                else:
                    spec_no_rag["rag_config"] = {"no_rag": True, "similarity_top_k": k}
                spec_no_rag["config_id"] = compute_config_fingerprint(spec_no_rag)
                specs.append(spec_no_rag)

            if sweep_rag:
                spec_rag = {
                    "model_id": model_id,
                    "model": dict(model),
                    "top_k": k,
                    "no_rag": False,
                    "parquet_paths": parquet_paths_str,
                    "prompts_file": str(prompts_file_path),
                    "prompts_snapshot": prompts_snapshot,
                    "evaluator_type": evaluator_type,
                    "evaluator_config": evaluator_config,
                    "two_step": two_step_val,
                    "batch_size": batch_size,
                    "num_samples": num_samples,
                    "wait_for_revive_seconds": wait_for_revive_seconds,
                    "config_path": str(config_path),
                    "run_started_at": run_started_at,
                    "prompt_version": prompt_version,
                }
                if rag_config is not None:
                    spec_rag["rag_config"] = {**rag_config, "no_rag": False, "similarity_top_k": k}
                    spec_rag["model"]["top_k"] = k
                else:
                    spec_rag["rag_config"] = {"no_rag": False, "similarity_top_k": k}
                spec_rag["config_id"] = compute_config_fingerprint(spec_rag)
                specs.append(spec_rag)

    return specs
