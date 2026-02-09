"""
Shared CLI helpers: model presets, GPU/TP validation.
"""
import os
import re
import subprocess
from pathlib import Path

import typer
import yaml
from loguru import logger

from recite.utils.path_loader import get_project_root


def load_model_presets() -> dict:
    """Load model presets from config/model_presets.yaml"""
    config_path = get_project_root() / "config" / "model_presets.yaml"

    if not config_path.exists():
        logger.warning(f"Model presets config not found at {config_path}, using defaults")
        return get_default_presets()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cache_config = config.get("cache", {})
    local_cache = Path(cache_config.get("local", "~/.cache/huggingface")).expanduser()
    cluster_cache = Path(cache_config.get("cluster", "~/.cache/huggingface")).expanduser()

    presets = {}
    for preset_name, preset_config in config.get("presets", {}).items():
        preset = preset_config.copy()
        cache_ref = preset.get("cache_dir")
        if cache_ref == "local":
            preset["cache_dir"] = local_cache
        elif cache_ref == "cluster":
            preset["cache_dir"] = cluster_cache
        elif cache_ref:
            preset["cache_dir"] = Path(cache_ref).expanduser()
        else:
            preset["cache_dir"] = local_cache
        presets[preset_name] = preset
    return presets


def get_default_presets() -> dict:
    """Fallback default presets if config file is missing"""
    local_cache = Path.home() / ".cache" / "huggingface"
    cluster_cache = Path.home() / ".cache" / "huggingface"
    return {
        "local": {
            "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "tensor_parallel": 1,
            "cache_dir": local_cache,
            "max_model_len": 4096,
            "gpu_mem": 0.9,
            "description": "Single GPU (~20GB VRAM)",
        },
        "cluster-72b": {
            "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
            "tensor_parallel": 2,
            "cache_dir": cluster_cache,
            "description": "2x GPUs (~40GB VRAM) - better quality",
        },
    }


MODEL_PRESETS = load_model_presets()


def get_available_gpu_count() -> int:
    """Get the number of available CUDA GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                check=True,
            )
            return len(result.stdout.strip().split("\n"))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return 0


def get_gpu_vram_gb(gpu_index: int = 0) -> float | None:
    """Get VRAM capacity of a GPU in GB."""
    try:
        import torch
        if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(gpu_index)
            return props.total_memory / (1024**3)
    except (ImportError, AttributeError):
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip()) / 1024
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return None


def estimate_model_vram_gb(model_name: str) -> float | None:
    """Estimate model VRAM requirements in GB."""
    model_lower = model_name.lower()
    if "awq" in model_lower:
        bytes_per_param = 0.5
    elif "fp8" in model_lower or "fp-8" in model_lower:
        bytes_per_param = 1.0
    elif "int4" in model_lower or "q4" in model_lower:
        bytes_per_param = 0.5
    elif "int8" in model_lower or "q8" in model_lower:
        bytes_per_param = 1.0
    else:
        bytes_per_param = 2.0
    param_match = re.search(r"(\d+(?:\.\d+)?)[bB]", model_name)
    if not param_match:
        return None
    params_b = float(param_match.group(1))
    return (params_b * bytes_per_param * 1.2) + 2.0


def get_model_attention_heads(model_name: str, cache_dir: Path) -> int | None:
    """Load model config and get number of attention heads."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, cache_dir=str(cache_dir))
        return (
            getattr(config, "num_attention_heads", None)
            or getattr(config, "num_heads", None)
            or getattr(config, "n_head", None)
        )
    except Exception as e:
        logger.warning(f"Could not load config for {model_name}: {e}")
        return None


def get_valid_tensor_parallel_sizes(num_attention_heads: int, max_tp: int = 40) -> list[int]:
    """Get valid tensor parallel sizes for a model."""
    valid_sizes = []
    for tp in range(1, min(max_tp, num_attention_heads) + 1):
        if num_attention_heads % tp == 0:
            valid_sizes.append(tp)
    return valid_sizes


def validate_tp_size(
    tp: int,
    valid_tp_sizes: list[int] | None,
    model_attention_heads: int | None,
    context: str = "",
) -> None:
    """Validate tensor parallel size against model architecture."""
    if valid_tp_sizes and tp not in valid_tp_sizes:
        msg = f"Error: tensor_parallel={tp} is invalid"
        if model_attention_heads:
            msg += f" for model with {model_attention_heads} attention heads"
        if context:
            msg += f" ({context})"
        typer.echo(msg)
        typer.echo(f"Valid TP sizes: {valid_tp_sizes}")
        typer.echo(f"Suggested GPU counts: {', '.join(map(str, valid_tp_sizes))}")
        raise typer.Exit(1)
