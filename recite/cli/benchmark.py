"""Benchmark CLI commands."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Union

import typer
import yaml
from loguru import logger

from recite.benchmark.db import get_connection, get_db_path, init_database
from recite.benchmark.pipeline import get_pipeline_stats, run_e2e_pipeline
from recite.utils.logging_config import configure_logging
from recite.utils.path_loader import get_project_root

app = typer.Typer()


@contextmanager
def _managed_connection(db_path: Optional[Path] = None) -> Generator:
    """Context manager for database connection."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def download_all_trial_versions(
    instance_ids: Optional[List[str]] = None,
    max_trials: Optional[int] = None,
    db_path: Optional[Path] = None,
):
    """Download all trial versions from ClinicalTrials.gov."""
    from recite.benchmark.downloaders import download_versions

    if instance_ids is None:
        instance_ids = _load_instance_ids(None, max_trials)
    with _managed_connection(db_path) as conn:
        download_versions(instance_ids, max_trials, conn)


def download_trial_eligibility_criteria(
    instance_ids: Optional[List[str]] = None,
    max_trials: Optional[int] = None,
    db_path: Optional[Path] = None,
):
    """Download trial eligibility criteria."""
    from recite.benchmark.downloaders import download_ecs

    if instance_ids is None:
        instance_ids = _load_instance_ids(None, max_trials)
    with _managed_connection(db_path) as conn:
        download_ecs(instance_ids, max_trials, conn)


def identify_trials_with_eligibility_criteria_amendments(
    max_trials: Optional[int] = None,
    db_path: Optional[Path] = None,
):
    """Identify trials with eligibility criteria amendments."""
    from recite.benchmark.processors import identify_amendments

    with _managed_connection(db_path) as conn:
        identify_amendments(max_trials, conn)


def download_full_trial(
    instance_ids: Optional[List[str]] = None,
    max_trials: Optional[int] = None,
    db_path: Optional[Path] = None,
):
    """Download full trial protocol PDFs."""
    from recite.benchmark.downloaders import download_protocols

    if instance_ids is None:
        instance_ids = _load_instance_ids(None, max_trials)
    with _managed_connection(db_path) as conn:
        download_protocols(instance_ids, max_trials, conn)


def extract_trial_data(
    max_trials: Optional[int] = None,
    db_path: Optional[Path] = None,
):
    """Extract trial data (evidence from protocols)."""
    from recite.benchmark.processors import extract_evidence

    with _managed_connection(db_path) as conn:
        extract_evidence(max_trials, conn)


def download_extract_all_relevant_trials(
    instance_ids: Optional[List[str]] = None,
    max_trials: Optional[int] = None,
    db_path: Optional[Path] = None,
):
    """Download and extract all relevant trial data."""
    from recite.benchmark.downloaders import download_protocols
    from recite.benchmark.processors import extract_evidence

    if instance_ids is None:
        instance_ids = _load_instance_ids(None, max_trials)
    with _managed_connection(db_path) as conn:
        download_protocols(instance_ids, max_trials, conn)
        extract_evidence(max_trials, conn)


def create_benchmark_db(
    max_trials: Optional[int] = None,
    db_path: Optional[Path] = None,
):
    """Create benchmark database (build RECITE instances)."""
    from recite.benchmark.builders import create_recite_instances

    with _managed_connection(db_path) as conn:
        create_recite_instances(max_trials, conn)


@app.command()
def init_benchmark(
    instance_ids: Optional[List[str]] = typer.Argument(
        None,
        help="One or more NCT IDs to process. If provided, uses manual NCT IDs instead of auto-discovery."
    ),
    max_trials: Optional[int] = typer.Option(None, "--max-trials", "-n", help="Maximum number of trials to process"),
    instance_ids_file: Optional[Path] = typer.Option(
        None, 
        "--nct-ids-file", 
        help="Path to file containing NCT IDs (one per line). Alternative to providing NCT IDs as arguments."
    ),
    discovery_method: str = typer.Option(
        "bulk_xml",
        "--discovery-method",
        help="Method to discover NCT IDs: 'bulk_xml' (default, more robust) or 'api_pagination'"
    ),
    db_path: Optional[Path] = typer.Option(
        str(get_db_path()),
        "--db-path",
        help="Path to database file (default: under LOCAL_DB_DIR/recite.db)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force fresh start: backup existing database to data/legacy/ and run from scratch (no skipping)"
    ),
    use_expedited: bool = typer.Option(
        True,
        "--use-expedited/--no-expedited",
        help="Use expedited strategy: filter by moduleLabels and download only EC-relevant versions (default: True)"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: TRACE, DEBUG, INFO, WARNING, ERROR (default: INFO). Set to INFO or higher to reduce stdout clutter."
    ),
    chunk_index: Optional[int] = typer.Option(
        None,
        "--chunk",
        help="Process only a specific chunk (0-indexed). Requires --total-chunks."
    ),
    total_chunks: Optional[int] = typer.Option(
        None,
        "--total-chunks",
        help="Divide NCT IDs into this many chunks. Use with --chunk to process incrementally."
    ),
    stop_after: Optional[str] = typer.Option(
        None,
        "--stop-after",
        help="Stop after a specific stage (e.g. 'metadata' to stop after populating trial_metadata table). Only works with auto-discovery and bulk_xml method."
    ),
):
    """
    Initialize benchmark - run full E2E pipeline with auto-discovery.
    
    By default, automatically discovers all NCT IDs from ClinicalTrials.gov,
    progressively filters through pipeline stages, and creates benchmark samples.
    
    Skip logic is granular: only individual trials/samples that are already complete are skipped.
    Use --force to backup existing database and run everything from scratch.
    
    Use --stop-after metadata to stop after populating the trial_metadata table
    (only works with auto-discovery and bulk_xml method).
    
    For manual override, provide NCT IDs as arguments:
      uv run benchmark.py init-benchmark NCT07252128 NCT07257666 ...
    
    Or use --nct-ids-file with a file containing one NCT ID per line.
    """
    import time
    from datetime import datetime

    configure_logging(level=log_level, app_name="benchmark", also_stderr=True)

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Starting RECITE benchmark pipeline")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Configuration:")
    logger.info(f"  - Database path: {db_path}")
    logger.info(f"  - Discovery method: {discovery_method}")
    logger.info(f"  - Max trials: {max_trials if max_trials else 'All'}")
    logger.info(f"  - Force mode: {force}")
    logger.info(f"  - Expedited mode: {use_expedited}")
    logger.info(f"  - Log level: {log_level.upper()}")
    if chunk_index is not None:
        if total_chunks is None:
            logger.error("--total-chunks must be provided when --chunk is specified")
            raise typer.Exit(code=1)
        logger.info(f"  - Chunking: Processing chunk {chunk_index + 1}/{total_chunks}")
    if force:
        logger.warning("  ⚠ Force mode enabled: existing database will be backed up to data/legacy/")
    if instance_ids_file:
        logger.info(f"  - Manual NCT IDs file: {instance_ids_file}")
    if instance_ids:
        logger.info(f"  - Manual NCT IDs: {len(instance_ids)} IDs provided")
    if stop_after:
        logger.info(f"  - Stop after: {stop_after}")
        if stop_after == "metadata" and discovery_method != "bulk_xml":
            logger.error("--stop-after metadata only works with --discovery-method bulk_xml")
            raise typer.Exit(code=1)
    logger.info("=" * 80)
    
    if isinstance(db_path, str):
        db_path = Path(db_path)
    
    manual_instance_ids = None
    if instance_ids:
        manual_instance_ids = instance_ids
    elif instance_ids_file and instance_ids_file.exists():
        manual_instance_ids = _load_instance_ids(instance_ids_file, max_trials)
    
    if manual_instance_ids:
        logger.info("=" * 80)
        logger.info("MODE: Manual NCT IDs")
        logger.info("=" * 80)
        logger.info(f"Using manual NCT IDs")
        if instance_ids_file:
            logger.info(f"  From file: {instance_ids_file}")
        from recite.benchmark.builders import create_recite_instances
        from recite.benchmark.downloaders import (
            download_ecs,
            download_protocols,
            download_versions,
        )
        from recite.benchmark.discovery import check_trial_versions_batch
        from recite.benchmark.processors import extract_evidence, identify_amendments
        
        conn = init_database(db_path, force=force)
        
        if max_trials:
            manual_instance_ids = manual_instance_ids[:max_trials]
        
        from recite.crawler.adapters import ClinicalTrialsGovAdapter
        adapter = ClinicalTrialsGovAdapter()
        
        logger.info("Checking version counts for manual NCT IDs...")
        check_trial_versions_batch(
            manual_instance_ids,
            adapter=adapter,
            conn=conn,
            discovery_method="manual",
            use_expedited=use_expedited,
        )
        
        logger.info("Downloading versions...")
        download_versions(manual_instance_ids, max_trials, conn, use_expedited=use_expedited)
        
        logger.info("Identifying EC changes...")
        identify_amendments(max_trials, conn)
        
        logger.info("Downloading protocols...")
        download_protocols(manual_instance_ids, max_trials, conn)
        
        logger.info("Extracting evidence...")
        extract_evidence(max_trials, conn)
        
        logger.info("Building RECITE instances...")
        create_recite_instances(max_trials, conn)
        
        conn.close()
    else:
        # Use auto-discovery pipeline
        logger.info("=" * 80)
        logger.info("MODE: Auto-discovery pipeline")
        logger.info("=" * 80)
        logger.info(f"Using auto-discovery method: {discovery_method}")
        
        try:
            stats = run_e2e_pipeline(
                discovery_method=discovery_method,
                max_trials=max_trials,
                db_path=db_path,
                force=force,
                use_expedited=use_expedited,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                stop_after=stop_after,
            )
            
            # Display statistics
            elapsed_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info("Pipeline Statistics:")
            logger.info("=" * 80)
            for stage, count in stats.items():
                logger.info(f"  {stage}: {count}")
            logger.info("=" * 80)
            logger.info(f"Total execution time: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
            logger.info("=" * 80)
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error("=" * 80)
            logger.error(f"Pipeline failed after {elapsed_time/60:.1f} minutes")
            logger.error(f"Error: {e}")
            logger.error("=" * 80)
            raise
    
    elapsed_total = time.time() - start_time
    logger.info("=" * 80)
    logger.info("Benchmark initialization complete")
    logger.info(f"Total time: {elapsed_total/60:.1f} minutes ({elapsed_total:.1f} seconds)")
    logger.info("=" * 80)

@app.command()
def export_splits(
    output_dir: Path = typer.Option(
        Path("data/benchmark_splits"),
        "--output-dir",
        help="Directory to write parquet file and statistics"
    ),
    db_path: Optional[Path] = typer.Option(
        str(get_db_path()),
        "--db-path",
        help="Path to database file (default: under LOCAL_DB_DIR/recite.db)",
    ),
    legacy_splits: bool = typer.Option(
        False,
        "--legacy-splits",
        help="Write train/val/test.parquet instead of single benchmark.parquet"
    ),
    train_ratio: float = typer.Option(
        0.8,
        "--train-ratio",
        help="(Legacy) Proportion for training set"
    ),
    val_ratio: float = typer.Option(
        0.1,
        "--val-ratio",
        help="(Legacy) Proportion for validation set"
    ),
    test_ratio: float = typer.Option(
        0.1,
        "--test-ratio",
        help="(Legacy) Proportion for test set"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="(Legacy) Random seed for shuffling"
    ),
    min_quality_score: Optional[float] = typer.Option(
        None,
        "--min-quality-score",
        help="Optional minimum quality_score to filter samples"
    ),
):
    """
    Export RECITE benchmark data to parquet.
    Default: one combined file (benchmark.parquet). Use --legacy-splits for train/val/test.
    """
    from recite.benchmark.db import get_connection
    from recite.benchmark.parquet_exporter import export_to_parquet_combined, export_to_parquet_splits

    project_root = get_project_root()
    if isinstance(db_path, str):
        db_path = Path(db_path)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(db_path)
    try:
        if legacy_splits:
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                logger.error(f"Ratios must sum to 1.0, got {total_ratio}")
                raise typer.Exit(code=1)
            logger.info("Exporting RECITE benchmark (legacy train/val/test splits)")
            stats = export_to_parquet_splits(
                conn=conn,
                output_dir=output_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                min_quality_score=min_quality_score,
            )
            logger.info(f"Train: {stats.get('train_samples', 0)}, Val: {stats.get('val_samples', 0)}, Test: {stats.get('test_samples', 0)}")
        else:
            logger.info("Exporting RECITE benchmark (combined benchmark.parquet)")
            stats = export_to_parquet_combined(
                conn=conn,
                output_dir=output_dir,
                min_quality_score=min_quality_score,
            )
            logger.info(f"Total samples: {stats.get('total_samples', 0)}")
        logger.info(f"Parquet and statistics written to {output_dir}")
    finally:
        conn.close()


def _resolve_path(path_val: Any, project_root: Path) -> Path:
    """Resolve path from config (may be str relative to project root) to Path."""
    if path_val is None:
        return project_root / "data" / "benchmark_predictions"
    p = Path(path_val) if not isinstance(path_val, Path) else path_val
    if not p.is_absolute():
        p = project_root / p
    return p


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize model id for use as directory name."""
    return re.sub(r"[^\w\-]", "_", model_id).strip("_") or "model"


def _run_timestamp() -> str:
    """Filesystem-safe run timestamp (e.g. 2025-01-29T20-45-00)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _load_prompts_snapshot(prompts_file: Path) -> Optional[Dict[str, Any]]:
    """Load prompts JSON from prompts_file for run_config snapshot. Returns None if file missing."""
    path = Path(prompts_file) if not isinstance(prompts_file, Path) else prompts_file
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _make_run_dir_and_save_config(
    base_dir: Path,
    run_config: Dict[str, Any],
    config_filename: str = "run_config.yaml",
    run_suffix: str = "",
) -> Path:
    """
    Create a timestamped run subdir under base_dir, write run_config to it, return the run dir.
    Paths in run_config are serialized as strings; API keys are redacted.
    run_suffix: Optional suffix for run dir name (e.g. _no_rag, _topk10).
    """
    run_name = f"run_{_run_timestamp()}{run_suffix}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Serialize for YAML: Path -> str, redact API keys
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(x) for x in obj]
        return obj

    to_save = _serialize(run_config)
    # Redact keys that may contain secrets (top-level and under rag_config)
    if isinstance(to_save, dict):
        for key in ("embed_api_key", "api_key"):
            if key in to_save and to_save[key]:
                to_save[key] = "(redacted)"
        if isinstance(to_save.get("rag_config"), dict):
            for key in ("embed_api_key", "api_key"):
                if key in to_save["rag_config"] and to_save["rag_config"][key]:
                    to_save["rag_config"][key] = "(redacted)"

    config_path = run_dir / config_filename
    with open(config_path, "w") as f:
        yaml.dump(to_save, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Run config saved to {config_path}")
    return run_dir


def _normalize_top_k_list(
    cli_top_k: Optional[int],
    config_top_k: Optional[Union[int, List[int]]],
) -> List[int]:
    """Normalize CLI and config top_k to a list of ints. CLI overrides config."""
    if cli_top_k is not None:
        return [cli_top_k]
    if config_top_k is None:
        return [2]
    if isinstance(config_top_k, list):
        return config_top_k
    return [config_top_k]


# Reserved tokens for system + user prompt when deriving no_rag evidence cap from context_window
_RESERVED_PROMPT_TOKENS = 4096


def _effective_no_rag_max_tokens(
    model_config: Dict[str, Any],
    default_arg: Optional[int],
    reserved: int = _RESERVED_PROMPT_TOKENS,
) -> Optional[int]:
    """
    Per-model no_rag_max_tokens for evidence truncation (no global in config).
    Uses model.no_rag_max_tokens if set; else if model.context_window set uses
    (context_window - reserved); else returns default_arg (None → RAG layer uses DEFAULT_NO_RAG_MAX_TOKENS).
    """
    if not model_config:
        return default_arg
    explicit = model_config.get("no_rag_max_tokens")
    if explicit is not None:
        return int(explicit)
    ctx = model_config.get("context_window")
    if ctx is not None:
        ctx = int(ctx)
        return max(0, ctx - reserved)
    return default_arg


def _normalize_parquet_paths_for_match(parquet_paths: Dict[str, Path], project_root: Path) -> Dict[str, str]:
    """Normalize parquet paths to absolute strings for config matching."""
    out = {}
    for k, p in parquet_paths.items():
        path = Path(p)
        if not path.is_absolute():
            path = project_root / path
        out[k] = str(path.resolve())
    return out


def _load_run_config_from_dir(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load run_config.yaml from a run directory. Return None on error."""
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _run_config_matches(
    run_config: Dict[str, Any],
    model_id: str,
    top_k: int,
    no_rag: bool,
    parquet_paths_normalized: Dict[str, str],
) -> bool:
    """Return True if run_config matches the given model_id, top_k, no_rag, and parquet_paths."""
    if run_config.get("model_id") != model_id:
        return False
    cfg_top_k = run_config.get("top_k")
    if cfg_top_k is not None and int(cfg_top_k) != top_k:
        return False
    cfg_no_rag = run_config.get("no_rag")
    if cfg_no_rag is not no_rag:
        return False
    cfg_paths = run_config.get("parquet_paths")
    if not isinstance(cfg_paths, dict):
        return False
    for split, path_str in parquet_paths_normalized.items():
        cfg_str = cfg_paths.get(split) if isinstance(cfg_paths.get(split), str) else None
        if cfg_str is None:
            return False
        # Normalize for comparison (resolve relative paths)
        cfg_resolved = str(Path(cfg_str).resolve()) if cfg_str else ""
        if cfg_resolved != path_str:
            return False
    return True


def _find_matching_run_dir(
    out_dir: Path,
    model_id: str,
    top_k: int,
    no_rag: bool,
    parquet_paths_base: Dict[str, Path],
) -> Optional[Path]:
    """
    Find an existing run dir under out_dir that matches (model_id, top_k, no_rag, parquet_paths).
    Returns the run_dir path if found, else None.
    """
    project_root = get_project_root()
    parquet_normalized = _normalize_parquet_paths_for_match(parquet_paths_base, project_root)
    # out_dir is already model-specific (output_dir / model_id); use it directly
    model_dir = out_dir
    if not model_dir.is_dir():
        return None
    suffix = "_no_rag" if no_rag else f"_topk{top_k}"
    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_") or not run_dir.name.endswith(suffix):
            continue
        cfg = _load_run_config_from_dir(run_dir)
        if cfg is None:
            continue
        if _run_config_matches(cfg, model_id, top_k, no_rag, parquet_normalized):
            return run_dir
    return None


def _get_parquet_row_counts(parquet_paths_base: Dict[str, Path], num_samples: Optional[int]) -> Dict[str, int]:
    """Return expected row count per split (from parquet, capped by num_samples if set)."""
    import pandas as pd
    counts: Dict[str, int] = {}
    for split_name, path in parquet_paths_base.items():
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            n = len(df)
            if num_samples is not None and num_samples >= 0:
                n = min(n, num_samples)
            counts[split_name] = n
        except Exception:
            continue
    return counts


def _load_done_sample_ids(run_dir: Path, split_names: List[str]) -> Dict[str, Set[int]]:
    """
    Load sample ids that are already evaluated per split from results_*.jsonl or predictions_*.jsonl.
    Returns {split_name: set of id}.
    """
    done: Dict[str, Set[int]] = {s: set() for s in split_names}
    for split_name in split_names:
        # Prefer results (they imply prediction + evaluation); fallback to predictions
        for name in (f"results_{split_name}.jsonl", f"predictions_{split_name}.jsonl"):
            path = run_dir / name
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            sid = row.get("id")
                            if sid is not None:
                                done[split_name].add(int(sid))
                        except Exception:
                            continue
            except Exception:
                pass
            break  # one file per split
    return done


def _run_one_benchmark_run(
    model: Dict[str, Any],
    k: int,
    out_dir: Path,
    run_config_base: Dict[str, Any],
    parquet_paths_base: Dict[str, Path],
    evaluator_type: str,
    evaluator_config: Optional[Dict[str, Any]],
    batch_size: int,
    num_samples: Optional[int],
    prompts_file: Path,
    two_step_val: bool,
    rag_config: Optional[Dict[str, Any]],
    wait_for_revive_seconds: int,
    e2e_smoke: bool,
    cfg_path: Path,
    no_rag: bool = False,
    no_rag_max_tokens: Optional[int] = None,
    sweep_no_rag: bool = False,
    max_concurrent_requests: int = 1,
) -> None:
    """Run a single benchmark (one model, one top_k, optionally no_rag). Uses existing run dir if config matches and continues from done samples."""
    from recite.benchmark.evaluator import run_benchmark as _run_benchmark

    model_id = run_config_base.get("model_id") or "unknown"
    existing_run_dir = _find_matching_run_dir(out_dir, model_id, k, no_rag, parquet_paths_base)
    done_sample_ids: Optional[Dict[str, Set[int]]] = None
    run_dir: Path

    if existing_run_dir is not None:
        split_names = list(parquet_paths_base.keys())
        done_sample_ids = _load_done_sample_ids(existing_run_dir, split_names)
        expected_counts = _get_parquet_row_counts(parquet_paths_base, num_samples)
        all_done = all(
            done_sample_ids.get(split, set()).__len__() >= expected_counts.get(split, 0)
            for split in split_names
            if expected_counts.get(split, 0) > 0
        )
        if all_done and expected_counts:
            logger.info(f"Run already complete (model={model_id}, top_k={k}, no_rag={no_rag}), skipping.")
            return
        run_dir = existing_run_dir
        todo = sum(
            max(0, expected_counts.get(s, 0) - len(done_sample_ids.get(s, set())))
            for s in split_names
        )
        logger.info(f"Continuing existing run {run_dir.name}: {todo} samples remaining.")
    else:
        run_dir = None  # will set below

    model_cpy = dict(model)
    model_cpy["top_k"] = k
    rag_cfg = dict(rag_config) if rag_config else None
    if rag_cfg is not None:
        if no_rag:
            rag_cfg["no_rag"] = True
            if no_rag_max_tokens is not None:
                rag_cfg["no_rag_max_tokens"] = no_rag_max_tokens
            rag_cfg["similarity_top_k"] = k  # for run_config record only
        else:
            rag_cfg["similarity_top_k"] = k
            rag_cfg["no_rag"] = False
    run_suffix = "_no_rag" if no_rag else (f"_topk{k}" if sweep_no_rag else "")
    run_config = {
        **run_config_base,
        "top_k": k,
        "no_rag": no_rag,
        "model": model_cpy,
        "rag_config": rag_cfg,
    }
    if run_dir is None:
        run_dir = _make_run_dir_and_save_config(out_dir, run_config, run_suffix=run_suffix)
    try:
        summary = _run_benchmark(
            model=model_cpy,
            parquet_paths=parquet_paths_base,
            output_dir=run_dir,
            evaluator_type=evaluator_type,
            evaluator_config=evaluator_config,
            batch_size=batch_size,
            num_samples=num_samples,
            prompts_path=prompts_file,
            two_step=two_step_val,
            rag_config=rag_cfg,
            wait_for_revive_seconds=wait_for_revive_seconds,
            done_sample_ids=done_sample_ids,
            max_concurrent_requests=max_concurrent_requests,
        )
        _log_summary(summary, run_dir)
    except Exception as e:
        logger.error(f"Benchmark evaluation failed: {e}")
        raise


def _log_summary(summary: Dict[str, Any], out_dir: Path) -> None:
    """Log evaluation summary to stdout."""
    logger.info("=" * 80)
    logger.info("Evaluation Summary:")
    logger.info("=" * 80)
    logger.info(f"Total predictions: {summary['total_predictions']}")
    for split_name, split_results in summary["splits"].items():
        logger.info(f"\n{split_name.upper()} Split:")
        logger.info(f"  Count: {split_results['count']}")
        logger.info("  Metrics:")
        for metric_name, metric_stats in split_results["metrics"].items():
            logger.info(f"    {metric_name}:")
            logger.info(f"      Mean: {metric_stats['mean']:.4f}")
            logger.info(f"      Std: {metric_stats['std']:.4f}")
            logger.info(f"      Min: {metric_stats['min']:.4f}")
            logger.info(f"      Max: {metric_stats['max']:.4f}")
    logger.info("=" * 80)
    logger.info(f"Results saved to {out_dir}")
    logger.info("=" * 80)


def _emit_run_summary(output_dir: Path) -> None:
    """Generate and write BENCHMARK_SUMMARY.md under output_dir after a run completes."""
    from recite.benchmark.summary_table import generate_benchmark_summary_md

    summary_path = output_dir / "BENCHMARK_SUMMARY.md"
    try:
        generate_benchmark_summary_md(
            root_dir=output_dir,
            output_path=summary_path,
            include_run_config=True,
        )
        logger.info(f"Results summary written to {summary_path}")
    except Exception as e:
        logger.warning(f"Could not generate results summary: {e}")


@app.command()
def run_benchmark(
    train_parquet: Path = typer.Option(
        Path("data/benchmark_splits/train.parquet"),
        "--train",
        help="Path to train.parquet file (default: data/benchmark_splits/train.parquet)"
    ),
    val_parquet: Path = typer.Option(
        Path("data/benchmark_splits/val.parquet"),
        "--val",
        help="Path to val.parquet file (default: data/benchmark_splits/val.parquet)"
    ),
    test_parquet: Path = typer.Option(
        Path("data/benchmark_splits/test.parquet"),
        "--test",
        help="Path to test.parquet file (default: data/benchmark_splits/test.parquet)"
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to benchmarks config YAML (default: config/benchmarks.yaml). When set and no --model-endpoint/--model-name, run each model from config."
    ),
    include_test: bool = typer.Option(
        False,
        "--include-test",
        help="Include test split in evaluation (default: train + val only; test preserved for final evaluation)"
    ),
    e2e_smoke: bool = typer.Option(
        False,
        "--e2e-smoke",
        help="E2E smoke: num_samples=1, train split only; optionally use llm_judge if configured"
    ),
    model_endpoint: Optional[str] = typer.Option(
        None,
        "--model-endpoint",
        help="LLM endpoint URL (e.g., http://localhost:8000/v1). Required with --model-name for single-model run; omit to use models from --config."
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model-name",
        help="Model name/ID for the predictor (e.g., 'llama-3.1-8b'). Required with --model-endpoint for single-model run."
    ),
    output_dir: Path = typer.Option(
        Path("data/benchmark_predictions"),
        "--output-dir",
        help="Directory to save predictions and results"
    ),
    evaluator_type: str = typer.Option(
        "default",
        "--evaluator",
        help="Enable LLM judge evaluator: 'llm_judge' (adds LLM judge to default metrics) or 'default' (only default metrics: BLEU/ROUGE/edit distance). Default evaluator always runs."
    ),
    judge_api_type: str = typer.Option(
        "ucsf_versa",
        "--judge-api-type",
        help="Judge API type: 'ucsf_versa' (default) or 'endpoint'"
    ),
    judge_model_type: str = typer.Option(
        "4o",
        "--judge-model-type",
        help="Judge model type for UCSF Versa API: '4o' (default, gpt-4o-2024-08-06), '4o-mini', '4.5-preview', or full model name"
    ),
    confirm_paid_judge: bool = typer.Option(
        False,
        "--confirm-paid-judge",
        help="REQUIRED when using paid judge APIs (ucsf_versa). Acknowledges that judge calls cost real money. Without this flag, paid judge runs will abort with a cost warning."
    ),
    judge_endpoint: Optional[str] = typer.Option(
        None,
        "--judge-endpoint",
        help="LLM endpoint for judge evaluator (required if --judge-api-type is 'endpoint')"
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Model name for judge evaluator. If not provided, uses --judge-model-type for UCSF or required for endpoint"
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        "--num-samples",
        "-n",
        help="Max samples per split for quick test runs (default: no limit)"
    ),
    prompts_file: Path = typer.Option(
        Path("config/benchmark_prompts.json"),
        "--prompts-file",
        help="Path to benchmark prompts config file"
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size",
        help="Number of samples to process before saving checkpoint"
    ),
    two_step: Optional[bool] = typer.Option(
        None,
        "--two-step/--no-two-step",
        help="Use two-step flow (schema then amended EC). Default from config or False."
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        help="RAG retrieval top_k (similarity_top_k). Default from config or server default (2)."
    ),
    max_parallel_runs: Optional[int] = typer.Option(
        None,
        "--max-parallel-runs",
        help="Run up to N benchmark runs concurrently (1 = sequential). Overrides config max_parallel_runs."
    ),
    max_concurrent_requests: Optional[int] = typer.Option(
        None,
        "--max-concurrent-requests",
        help="Within a run, process up to N samples per chunk in parallel (1 = sequential). Overrides config."
    ),
):
    """
    Run benchmark evaluation on parquet files.

    Evaluates a model on the RECITE benchmark dataset and saves predictions and metrics.
    RAG runs in-process: benchmark calls recite.rag.query_with_rag with embed config
    from config/benchmarks.yaml (rag section) or env. vLLM (or wrapper) at --model-endpoint
    is used for LLM only; ensure the server is already running.

    With --config (default config/benchmarks.yaml): run each model from config; by default
    train + val only (use --include-test to add test). With --model-endpoint and --model-name:
    single-model run (config still supplies prompts, evaluator, two_step, top_k unless overridden).

    Example (single model):
      uv run recite benchmark run-benchmark --model-endpoint http://localhost:8001/v1 --model-name llama-3.1-8b

    Example (config-driven, multiple models):
      uv run recite benchmark run-benchmark --config config/benchmarks.yaml --include-test
    """
    from recite.benchmark.evaluator import run_benchmark as _run_benchmark

    def check_server_health(endpoint: str, timeout: int = 5) -> bool:
        """Quick reachability check: GET the endpoint root."""
        import httpx
        try:
            r = httpx.get(f"{endpoint}/models", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    project_root = get_project_root()
    cfg_path = config_path if config_path is not None else project_root / "config" / "benchmarks.yaml"
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    bench_config: Optional[Dict[str, Any]] = None
    if cfg_path.exists():
        with open(cfg_path) as f:
            bench_config = yaml.safe_load(f)
        logger.info(f"Loaded benchmarks config from {cfg_path}")

    if e2e_smoke:
        num_samples = 1
        include_test = False

    if not train_parquet.is_absolute():
        train_parquet = project_root / train_parquet
    if not val_parquet.is_absolute():
        val_parquet = project_root / val_parquet
    if not test_parquet.is_absolute():
        test_parquet = project_root / test_parquet
    if bench_config and "parquet_paths" in bench_config:
        pp = bench_config["parquet_paths"]
        train_parquet = _resolve_path(pp.get("train", train_parquet), project_root)
        val_parquet = _resolve_path(pp.get("val", val_parquet), project_root)
        test_parquet = _resolve_path(pp.get("test", test_parquet), project_root)
    parquet_paths_base: Dict[str, Path] = {
        "train": train_parquet,
        "val": val_parquet,
    }
    if include_test:
        parquet_paths_base["test"] = test_parquet

    if e2e_smoke:
        parquet_paths_base = {"train": train_parquet}

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if bench_config:
        if "output_dir" in bench_config:
            output_dir = _resolve_path(bench_config["output_dir"], project_root)
        if "prompts_file" in bench_config and str(prompts_file) == str(Path("config/benchmark_prompts.json")):
            prompts_file = project_root / bench_config["prompts_file"]
        if "evaluator_type" in bench_config and evaluator_type == "default":
            evaluator_type = bench_config["evaluator_type"]
        if "batch_size" in bench_config and batch_size == 10:
            batch_size = int(bench_config["batch_size"])
    cfg_two_step = bench_config.get("two_step", False) if bench_config else False
    use_two_step = two_step if two_step is not None else cfg_two_step
    cfg_top_k = bench_config.get("top_k") if bench_config else None
    top_k_list = _normalize_top_k_list(top_k, cfg_top_k)

    rag_config: Optional[Dict[str, Any]] = None
    if bench_config and "rag" in bench_config:
        import os
        r = bench_config["rag"]
        rag_config = {
            "embed_base_url": r.get("embed_base_url") or os.environ.get("RAG_EMBED_URL", ""),
            "embed_model": r.get("embed_model") or os.environ.get("RAG_EMBED_MODEL", ""),
            "persist_dir": r.get("persist_dir", "data/llamaindex_cache"),
            "embed_api_key": r.get("embed_api_key") or os.environ.get("RAG_EMBED_API_KEY") or os.environ.get("UCSF_API_KEY"),
            "embed_api_version": r.get("embed_api_version") or os.environ.get("UCSF_API_VER") or os.environ.get("API_VERSION"),
        }
        r_top_k = r.get("top_k")
        if r_top_k is not None and not isinstance(r_top_k, list):
            rag_config["similarity_top_k"] = r_top_k
        if r.get("similarity_top_k") is not None:
            rag_config["similarity_top_k"] = r["similarity_top_k"]
        ucsf_endpoint = (os.environ.get("UCSF_RESOURCE_ENDPOINT") or "").strip().rstrip("/")
        if ucsf_endpoint and not rag_config["embed_base_url"] and rag_config["embed_model"]:
            rag_config["embed_base_url"] = f"{ucsf_endpoint}/openai/deployments/{rag_config['embed_model']}"
    sweep_no_rag = bool(bench_config.get("rag", {}).get("sweep_no_rag", False)) if bench_config else False
    sweep_rag = bool(bench_config.get("rag", {}).get("sweep_rag", True)) if bench_config else True
    no_rag_max_tokens: Optional[int] = None
    wait_for_revive_seconds = int(bench_config.get("wait_for_revive_seconds", 0)) if bench_config else 0
    _max_parallel_runs = max_parallel_runs if max_parallel_runs is not None else (int(bench_config.get("max_parallel_runs", 1)) if bench_config else 1)
    _max_parallel_runs = max(1, _max_parallel_runs)
    _max_concurrent_requests = max_concurrent_requests if max_concurrent_requests is not None else (int(bench_config.get("max_concurrent_requests", 1)) if bench_config else 1)
    _max_concurrent_requests = max(1, _max_concurrent_requests)

    single_model = model_endpoint and model_name
    if single_model and rag_config is None:
        import os
        ucsf = (os.environ.get("UCSF_RESOURCE_ENDPOINT") or "").strip().rstrip("/")
        embed_model = os.environ.get("RAG_EMBED_MODEL") or ("text-embedding-3-small-1-brim" if ucsf else "text-embedding-3-small")
        rag_config = {
            "embed_base_url": os.environ.get("RAG_EMBED_URL", ""),
            "embed_model": embed_model,
            "persist_dir": "data/llamaindex_cache",
            "embed_api_key": os.environ.get("RAG_EMBED_API_KEY") or (os.environ.get("UCSF_API_KEY") if ucsf else None),
            "embed_api_version": os.environ.get("UCSF_API_VER") or os.environ.get("API_VERSION"),
        }
        if ucsf and not rag_config["embed_base_url"]:
            rag_config["embed_base_url"] = f"{ucsf}/openai/deployments/{embed_model}"
    if single_model:
        if not check_server_health(model_endpoint, timeout=5):
            logger.error(
                f"Server at {model_endpoint} is not reachable. "
                "Start the server first."
            )
            raise typer.Exit(code=1)

    if not single_model and (not bench_config or "models" not in bench_config or not bench_config["models"]):
        logger.error(
            "Either provide --model-endpoint and --model-name, or use a config file with a 'models' list (e.g. config/benchmarks.yaml)."
        )
        raise typer.Exit(code=1)

    evaluator_config = None
    if evaluator_type == "llm_judge":
        from recite.llmapis import UCSFVersaAPI

        if judge_api_type == "ucsf_versa" and not confirm_paid_judge:
            logger.error(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  COST WARNING: LLM judge uses paid API (GPT-4o).            ║\n"
                "║  Estimated cost: ~$30-60 per 3K samples.                    ║\n"
                "║                                                             ║\n"
                "║  To proceed, add: --confirm-paid-judge                      ║\n"
                "║  For free local judge, use: --judge-api-type endpoint       ║\n"
                "║  To skip judging entirely, use: --evaluator default         ║\n"
                "╚══════════════════════════════════════════════════════════════╝"
            )
            raise typer.Exit(code=1)

        if judge_api_type == "ucsf_versa":
            model_type_map = {
                "4o": "gpt-4o-2024-08-06",
                "4o-mini": "gpt-4o-mini-2024-07-18",
                "4.5-preview": "gpt-4.5-preview",
                "4-turbo": "gpt-4-turbo-128k",
            }
            
            if judge_model is None:
                judge_model = model_type_map.get(judge_model_type, judge_model_type)
                logger.info(f"Using judge model type '{judge_model_type}' -> '{judge_model}'")
            else:
                logger.info(f"Using judge model: {judge_model}")
            
            if judge_model not in UCSFVersaAPI.available_models:
                logger.error(
                    f"Judge model '{judge_model}' is not in available models: {UCSFVersaAPI.available_models}"
                )
                raise typer.Exit(code=1)
            
            import os
            missing_env = []
            if not os.getenv("UCSF_API_KEY"):
                missing_env.append("UCSF_API_KEY")
            if not os.getenv("UCSF_API_VER"):
                missing_env.append("UCSF_API_VER")
            if not os.getenv("UCSF_RESOURCE_ENDPOINT"):
                missing_env.append("UCSF_RESOURCE_ENDPOINT")
            if missing_env:
                logger.warning(
                    f"API environment variables not set: {', '.join(missing_env)}. "
                    f"UCSFVersaAPI requires these for initialization."
                )
            
            evaluator_config = {
                "api_type": "ucsf_versa",
                "model": judge_model,
            }
        elif judge_api_type == "endpoint":
            if not judge_endpoint or not judge_model:
                logger.error("--judge-endpoint and --judge-model are required when --judge-api-type is 'endpoint'")
                raise typer.Exit(code=1)
            evaluator_config = {
                "api_type": "endpoint",
                "endpoint": judge_endpoint,
                "model": judge_model,
            }
        else:
            logger.error(f"Unknown --judge-api-type: {judge_api_type}. Use 'ucsf_versa' or 'endpoint'")
            raise typer.Exit(code=1)
    
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if single_model:
        single_model_id = _sanitize_model_id(model_name)
        single_out_dir = output_dir / single_model_id
        model = {"endpoint": model_endpoint, "model": model_name}
        run_config_base = {
            "run_started_at": datetime.now(timezone.utc).isoformat(),
            "model_id": single_model_id,
            "model": model,
            "parquet_paths": {k: str(v) for k, v in parquet_paths_base.items()},
            "rag_config": dict(rag_config) if rag_config else None,
            "evaluator_type": evaluator_type,
            "evaluator_config": evaluator_config,
            "two_step": use_two_step,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "prompts_file": str(prompts_file),
            "prompts_snapshot": _load_prompts_snapshot(prompts_file),
            "wait_for_revive_seconds": wait_for_revive_seconds,
            "e2e_smoke": e2e_smoke,
            "config_path": str(cfg_path) if cfg_path.exists() else None,
        }
        single_specs: List[Dict[str, Any]] = []
        for k in top_k_list:
            if sweep_no_rag:
                single_specs.append({
                    "model": model,
                    "k": k,
                    "out_dir": single_out_dir,
                    "run_config_base": run_config_base,
                    "parquet_paths_base": parquet_paths_base,
                    "evaluator_type": evaluator_type,
                    "evaluator_config": evaluator_config,
                    "batch_size": batch_size,
                    "num_samples": num_samples,
                    "prompts_file": prompts_file,
                    "two_step_val": use_two_step,
                    "rag_config": rag_config,
                    "wait_for_revive_seconds": wait_for_revive_seconds,
                    "e2e_smoke": e2e_smoke,
                    "cfg_path": cfg_path,
                    "no_rag": True,
                    "no_rag_max_tokens": no_rag_max_tokens,
                    "sweep_no_rag": True,
                    "max_concurrent_requests": _max_concurrent_requests,
                })
            single_specs.append({
                "model": model,
                "k": k,
                "out_dir": single_out_dir,
                "run_config_base": run_config_base,
                "parquet_paths_base": parquet_paths_base,
                "evaluator_type": evaluator_type,
                "evaluator_config": evaluator_config,
                "batch_size": batch_size,
                "num_samples": num_samples,
                "prompts_file": prompts_file,
                "two_step_val": use_two_step,
                "rag_config": rag_config,
                "wait_for_revive_seconds": wait_for_revive_seconds,
                "e2e_smoke": e2e_smoke,
                "cfg_path": cfg_path,
                "no_rag": False,
                "no_rag_max_tokens": no_rag_max_tokens,
                "sweep_no_rag": sweep_no_rag,
                "max_concurrent_requests": _max_concurrent_requests,
            })
        logger.info("=" * 80)
        logger.info(f"Running {len(single_specs)} benchmark run(s) (single model, max_parallel_runs={_max_parallel_runs})")
        logger.info("=" * 80)
        with ThreadPoolExecutor(max_workers=_max_parallel_runs) as executor:
            futures = [executor.submit(_run_one_benchmark_run, **spec) for spec in single_specs]
            for fut in as_completed(futures):
                fut.result()
        _emit_run_summary(output_dir)
        return

    models_list = bench_config["models"]
    multi_specs: List[Dict[str, Any]] = []
    for m in models_list:
        api_type = m.get("api_type", "endpoint")
        mod = m.get("model")
        if api_type == "ucsf_versa":
            if not mod:
                logger.warning(f"Skipping model entry missing model (ucsf_versa): {m}")
                continue
            from recite.llmapis import UCSFVersaAPI
            if mod not in UCSFVersaAPI.available_models:
                logger.warning(
                    f"Model '{mod}' not in UCSFVersaAPI.available_models; skipping. "
                    f"Available: {UCSFVersaAPI.available_models}"
                )
                continue
            model_id = _sanitize_model_id(m.get("id", mod))
            out_dir = output_dir / model_id
            model = {"api_type": "ucsf_versa", "model": mod}
            if m.get("context_window") is not None:
                model["context_window"] = int(m["context_window"])
            if m.get("no_rag_max_tokens") is not None:
                model["no_rag_max_tokens"] = int(m["no_rag_max_tokens"])
            two_step_m = m.get("two_step")
            if two_step_m is None and bench_config:
                two_step_m = bench_config.get("two_step", False)
        elif api_type == "python_gpu":
            if not mod:
                logger.warning(f"Skipping model entry missing model (python_gpu): {m}")
                continue
            model_id = _sanitize_model_id(m.get("id", mod))
            out_dir = output_dir / model_id
            model = {"api_type": "python_gpu", "model": mod}
            if m.get("context_window") is not None:
                model["context_window"] = int(m["context_window"])
            if m.get("no_rag_max_tokens") is not None:
                model["no_rag_max_tokens"] = int(m["no_rag_max_tokens"])
            model["device"] = m.get("device", "cuda")
            model["gpus"] = int(m.get("gpus", 1))
            two_step_m = m.get("two_step")
            if two_step_m is None and bench_config:
                two_step_m = bench_config.get("two_step", False)
        elif api_type == "vllm_endpoint":
            ep = m.get("endpoint")
            if not ep or not mod:
                logger.warning(f"Skipping model entry missing endpoint or model (vllm_endpoint): {m}")
                continue
            if not check_server_health(ep, timeout=5):
                logger.warning(f"Server at {ep} not reachable; skipping model {m.get('id', mod)}")
                continue
            model_id = _sanitize_model_id(m.get("id", mod))
            out_dir = output_dir / model_id
            model = {"api_type": "vllm_endpoint", "model": mod, "endpoint": ep}
            if m.get("context_window") is not None:
                model["context_window"] = int(m["context_window"])
            if m.get("no_rag_max_tokens") is not None:
                model["no_rag_max_tokens"] = int(m["no_rag_max_tokens"])
            if m.get("max_tokens") is not None:
                model["max_tokens"] = int(m["max_tokens"])
            if m.get("timeout") is not None:
                model["timeout"] = float(m["timeout"])
            if m.get("max_concurrent") is not None:
                model["max_concurrent"] = int(m["max_concurrent"])
            if m.get("save_every") is not None:
                model["save_every"] = int(m["save_every"])
            if m.get("prompt_suffix") is not None:
                model["prompt_suffix"] = str(m["prompt_suffix"])
            two_step_m = m.get("two_step")
            if two_step_m is None and bench_config:
                two_step_m = bench_config.get("two_step", False)
        else:
            ep = m.get("endpoint")
            if not ep or not mod:
                logger.warning(f"Skipping model entry missing endpoint or model: {m}")
                continue
            if not check_server_health(ep, timeout=5):
                logger.warning(f"Server at {ep} not reachable; skipping model {m.get('id', mod)}")
                continue
            model_id = _sanitize_model_id(m.get("id", mod))
            out_dir = output_dir / model_id
            model = {"endpoint": ep, "model": mod}
            two_step_m = m.get("two_step")
            if two_step_m is None and bench_config:
                two_step_m = bench_config.get("two_step", False)

        two_step_val = two_step_m if two_step_m is not None else use_two_step
        run_config_base = {
            "run_started_at": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "model": model,
            "parquet_paths": {k: str(v) for k, v in parquet_paths_base.items()},
            "rag_config": dict(rag_config) if rag_config else None,
            "evaluator_type": evaluator_type,
            "evaluator_config": evaluator_config,
            "two_step": two_step_val,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "prompts_file": str(prompts_file),
            "prompts_snapshot": _load_prompts_snapshot(prompts_file),
            "wait_for_revive_seconds": wait_for_revive_seconds,
            "e2e_smoke": e2e_smoke,
            "config_path": str(cfg_path) if cfg_path.exists() else None,
        }
        no_rag_max_m = _effective_no_rag_max_tokens(m, no_rag_max_tokens)
        for k in top_k_list:
            if sweep_no_rag:
                multi_specs.append({
                    "model": model,
                    "k": k,
                    "out_dir": out_dir,
                    "run_config_base": run_config_base,
                    "parquet_paths_base": parquet_paths_base,
                    "evaluator_type": evaluator_type,
                    "evaluator_config": evaluator_config,
                    "batch_size": batch_size,
                    "num_samples": num_samples,
                    "prompts_file": prompts_file,
                    "two_step_val": two_step_val,
                    "rag_config": rag_config,
                    "wait_for_revive_seconds": wait_for_revive_seconds,
                    "e2e_smoke": e2e_smoke,
                    "cfg_path": cfg_path,
                    "no_rag": True,
                    "no_rag_max_tokens": no_rag_max_m,
                    "sweep_no_rag": True,
                    "max_concurrent_requests": _max_concurrent_requests,
                })
            if sweep_rag:
                multi_specs.append({
                    "model": model,
                    "k": k,
                    "out_dir": out_dir,
                    "run_config_base": run_config_base,
                    "parquet_paths_base": parquet_paths_base,
                    "evaluator_type": evaluator_type,
                    "evaluator_config": evaluator_config,
                    "batch_size": batch_size,
                    "num_samples": num_samples,
                    "prompts_file": prompts_file,
                    "two_step_val": two_step_val,
                    "rag_config": rag_config,
                    "wait_for_revive_seconds": wait_for_revive_seconds,
                    "e2e_smoke": e2e_smoke,
                    "cfg_path": cfg_path,
                    "no_rag": False,
                    "no_rag_max_tokens": no_rag_max_m,
                    "sweep_no_rag": sweep_no_rag,
                    "max_concurrent_requests": _max_concurrent_requests,
                })

    logger.info("=" * 80)
    logger.info(f"Running {len(multi_specs)} benchmark run(s) (max_parallel_runs={_max_parallel_runs})")
    logger.info("=" * 80)
    with ThreadPoolExecutor(max_workers=_max_parallel_runs) as executor:
        futures = [executor.submit(_run_one_benchmark_run, **spec) for spec in multi_specs]
        for fut in as_completed(futures):
            fut.result()
    _emit_run_summary(output_dir)


@app.command()
def summarize(
    output_dir: Path = typer.Option(
        Path("data/benchmark_predictions"),
        "--output-dir",
        help="Root directory containing model_id/run_*/ subdirs (benchmark_predictions)."
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Path to write markdown summary file. Default: <output-dir>/BENCHMARK_SUMMARY.md"
    ),
    include_run_config: bool = typer.Option(
        True,
        "--include-run-config/--no-run-config",
        help="Include top_k and run_started_at from run_config.yaml in table."
    ),
):
    """Generate a markdown summary table from benchmark_predictions/ run directories."""
    from recite.benchmark.summary_table import generate_benchmark_summary_md

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = get_project_root() / output_dir
    output_path = out
    if output_path is not None and isinstance(output_path, str):
        output_path = Path(output_path)
    if output_path is not None and not output_path.is_absolute():
        output_path = get_project_root() / output_path
    if output_path is None:
        output_path = output_dir / "BENCHMARK_SUMMARY.md"
    md = generate_benchmark_summary_md(
        root_dir=output_dir,
        output_path=output_path,
        include_run_config=include_run_config,
    )
    logger.info(f"Summary written to {output_path}")
    typer.echo(md[:500] + "..." if len(md) > 500 else md)


def _load_instance_ids(instance_ids_file: Optional[Path], max_trials: Optional[int]) -> List[str]:
    """Load NCT IDs from file or use defaults."""
    if instance_ids_file and instance_ids_file.exists():
        with open(instance_ids_file) as f:
            instance_ids = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        logger.info(f"Loaded {len(instance_ids)} NCT IDs from {instance_ids_file}")
    else:
        instance_ids = [
            "NCT02110043",
            "NCT03281616",
            "NCT00942747",
            "NCT04424641",
            "NCT04372602",
        ]
        logger.warning(
            f"No NCT IDs file provided. Using default sample of {len(instance_ids)} trials only. "
            f"For large-scale processing, provide --nct-ids <file> with one NCT ID per line."
        )
    
    if max_trials:
        instance_ids = instance_ids[:max_trials]
        logger.info(f"Limited to {max_trials} trials")
    
    return instance_ids


if __name__ == "__main__":
    app()
