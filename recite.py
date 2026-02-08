"""
recite.py — Orchestrate RECITE benchmark runs.

Delegates to recite.benchmark: ensure_splits, dataloader, config_loader,
results_db, evaluator.run_single_sample. Keeps orchestration logic minimal.
"""

# Session log version: bump when changing RAG/embed/run behavior so logs confirm which code ran
RECITE_RUN_SESSION_VERSION = "embed-fix-v4"

import copy
import json
import multiprocessing
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Default sec/sample for rough ETA (local GPU ~60s, Versa API ~10s)
_DEFAULT_SEC_PER_SAMPLE_LOCAL_GPU = 60
_DEFAULT_SEC_PER_SAMPLE_VERSA = 10

from loguru import logger
import pyarrow.parquet as pq
import typer
from tqdm import tqdm

# Load .env so HF_TOKEN, OPENAI_API_KEY, etc. are available
try:
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
except ImportError:
    pass

from recite.benchmark.config_loader import get_experiment_specs, load_benchmark_config
from recite.benchmark.db import get_connection
from recite.benchmark.dataloader import (
    count_samples_in_db,
    stream_from_db,
    stream_parquet_splits,
    validate_train_split,
)
from recite.benchmark.evaluator import batched_scorer, clear_python_gpu_cache, load_benchmark_prompts, run_single_sample
from recite.benchmark.parquet_exporter import (
    export_final_test_to_parquet,
    export_to_parquet_combined,
    export_to_parquet_splits,
)
from recite.benchmark.results_db import (
    BENCHMARK_METRIC_COLUMNS,
    clear_results_for_config,
    count_samples,
    ensure_config,
    ensure_results_table,
    find_config,
    get_benchmark_summary_rows,
    get_connection as get_results_connection,
    get_predictions_without_judge,
    has_sample,
    insert_result,
    merge_duplicate_configs,
    migrate_from_benchmark_predictions,
    update_judge_scores,
)
from recite.rag.query import build_index_for_document
from recite.utils.logging_config import configure_logging
from recite.utils.path_loader import get_data_root, get_local_db_dir, get_project_root, resolve_path

app = typer.Typer()


@app.callback()
def _main(
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (DEBUG, INFO, WARNING, ERROR)"),
) -> None:
    """RECITE benchmark orchestrator. Logs go to logs/recite.log and stderr."""
    configure_logging(level=log_level, app_name="recite", also_stderr=True)

# Sentinel: worker returns (config_id, None) when skip
SKIP = None


def _versa_task(
    sample_row: Dict[str, Any],
    spec: Dict[str, Any],
    split_name: str,
) -> tuple:
    """Run one Versa sample in a worker; returns (config_id, result_dict or None)."""
    config_id = spec["config_id"]
    result = run_single_sample(
        sample_row=sample_row,
        model=spec["model"],
        rag_config=spec.get("rag_config"),
        evaluator_type=spec.get("evaluator_type", "default"),
        evaluator_config=spec.get("evaluator_config"),
        prompts_path=Path(spec["prompts_file"]),
        split_name=split_name,
        two_step=spec.get("two_step", False),
        wait_for_revive_seconds=spec.get("wait_for_revive_seconds", 0),
        max_retries=2,
        max_delay=5.0,
    )
    return (config_id, result)


def ensure_splits(
    db_path: Path,
    output_dir: Path,
    include_final_test: bool = False,
    final_test_num_samples: Optional[int] = None,
) -> Tuple[Path, int, Optional[int]]:
    """
    Export recite.db to parquet(s): benchmark.parquet (train/standard) and optionally final_test.parquet (eval).
    Returns (output_dir, n_benchmark, n_final_test or None).
    """
    output_dir = Path(output_dir)
    logger.info("Exporting splits from {} to {}", db_path, output_dir)
    conn = get_connection(db_path)
    try:
        logger.info("Writing train split (benchmark.parquet)")
        stats = export_to_parquet_combined(conn, output_dir)
        n_benchmark = stats.get("total_samples", 0)
        n_final_test: Optional[int] = None
        if include_final_test:
            logger.info("Writing test split (final_test.parquet)")
            final_stats = export_final_test_to_parquet(
                conn,
                output_dir,
                num_samples=final_test_num_samples,
            )
            n_final_test = final_stats.get("total_samples") or 0
    finally:
        conn.close()
    return (output_dir, n_benchmark, n_final_test)


def get_dataloader(
    parquet_dir: Optional[Path] = None,
    parquet_paths: Optional[Dict[str, Path]] = None,
    batch_size: Optional[int] = None,
    splits_order: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
    from_db: bool = False,
    limit: Optional[int] = None,
) -> Iterator[tuple]:
    """
    Return a lazy iterator over (split_name, row_dict).

    Two modes:
      1. from_db=True or db_path provided: Stream directly from recite.db (no parquet needed).
         Uses merge_source: benchmark/train=cluster1 (~3k), final_test=local (~8.8k).
      2. from_db=False (default): Stream from parquet files (legacy).

    Args:
        parquet_dir: Dir with benchmark.parquet or train/val/test.parquet (parquet mode).
        parquet_paths: Dict of split_name -> parquet path (parquet mode).
        batch_size: Rows per batch for memory control.
        splits_order: Order of splits to iterate (e.g. ["benchmark"] or ["benchmark", "final_test"]).
        db_path: Path to recite.db (DB mode).
        from_db: If True, use DB mode even if parquet_paths are provided.
    """
    # DB-based loading
    if from_db or db_path is not None:
        if db_path is None:
            db_path = get_local_db_dir() / "recite.db"
        splits = splits_order if splits_order else ["benchmark"]
        return stream_from_db(db_path, splits, batch_size=batch_size or 1000, limit=limit)

    # Parquet-based loading (legacy)
    if parquet_paths is not None:
        paths = {k: Path(v) for k, v in parquet_paths.items() if Path(v).exists()}
        order = splits_order if splits_order is not None else sorted(paths.keys())
    else:
        if parquet_dir is None:
            raise ValueError("Provide parquet_dir, parquet_paths, or db_path")
        paths = {}
        for name in ("benchmark", "train", "val", "test", "final_test"):
            p = Path(parquet_dir) / (f"{name}.parquet")
            if p.exists():
                paths[name] = p
        order = splits_order if splits_order is not None else sorted(paths.keys())
    if not paths:
        raise FileNotFoundError(
            f"No parquet files in {parquet_dir or list(parquet_paths.values()) if parquet_paths else 'N/A'}"
        )
    return stream_parquet_splits(paths, batch_size=batch_size, splits_order=order)


def _resolve_parquet_paths_from_config(
    bench_config: Dict[str, Any],
    root: Path,
    parquet_dir_override: Optional[Path] = None,
) -> Tuple[Dict[str, Path], Path]:
    """
    Resolve parquet paths from config. Respects splits_to_run.
    Returns (parquet_paths, parquet_dir). Only includes splits that exist.
    """
    pp = bench_config.get("parquet_paths") or {}
    splits_to_run = bench_config.get("splits_to_run")
    default_dir = get_data_root() / "benchmark_splits"

    if parquet_dir_override is not None:
        parquet_dir = Path(parquet_dir_override)
        candidate_splits = splits_to_run if splits_to_run is not None else ["benchmark", "train", "val", "test", "final_test"]
        paths = {}
        for s in candidate_splits:
            if s in pp:
                paths[s] = resolve_path(Path(pp[s]), root)
            else:
                fname = "benchmark.parquet" if s == "benchmark" else f"{s}.parquet"
                p = parquet_dir / fname
                if p.exists():
                    paths[s] = p
        return (paths, parquet_dir)

    # Resolve from config keys; default to benchmark then train/val/test
    keys = splits_to_run if splits_to_run is not None else list(pp.keys()) or ["benchmark", "train", "val", "test"]
    paths = {}
    for s in keys:
        if s in pp:
            p = resolve_path(Path(pp[s]), root)
            if p.exists():
                paths[s] = p
        else:
            fname = "benchmark.parquet" if s == "benchmark" else f"{s}.parquet"
            p = default_dir / fname
            if p.exists():
                paths[s] = resolve_path(p, root)
    if "final_test" in keys and "final_test" not in paths:
        logger.warning(
            "final_test split requested but file not found. Run first: uv run python scripts/create_final_test_split.py"
        )
    if not paths:
        parquet_dir = resolve_path(default_dir, root)
        for s in ("benchmark", "train", "val", "test"):
            fname = "benchmark.parquet" if s == "benchmark" else f"{s}.parquet"
            if (parquet_dir / fname).exists():
                paths[s] = parquet_dir / fname
        return (paths, parquet_dir)
    parquet_dir = next(iter(paths.values())).parent
    return (paths, parquet_dir)


def load_experiments(
    config_path: Path,
    parquet_paths: Optional[Dict[str, Path]] = None,
    project_root: Optional[Path] = None,
    include_test: bool = False,
) -> List[Dict[str, Any]]:
    """Load YAML config and return list of experiment specs with config_id."""
    return get_experiment_specs(
        config_path,
        parquet_paths=parquet_paths,
        project_root=project_root or get_project_root(),
        include_test=include_test,
    )


def run_ucsf_versa(
    config_path: Path,
    results_db_path: Path,
    db_path: Optional[Path] = None,
    parquet_dir: Optional[Path] = None,
    predictions_dir: Optional[Path] = None,
    migrate_first: bool = True,
    project_root: Optional[Path] = None,
    max_workers: int = 4,
    phase: str = "all",
    from_db: bool = False,
) -> None:
    """
    Run UCSF Versa benchmark: load experiments, stream samples, dispatch to workers
    (parallel only for Versa API calls; max_workers > 1 uses a thread pool).
    Skip if result exists or on failure; insert results into results DB.
    Optionally migrate existing benchmark_predictions into results DB first.
    """
    root = project_root or get_project_root()
    config_path = resolve_path(Path(config_path), root)
    results_db_path = resolve_path(Path(results_db_path), root)

    logger.info("run_ucsf_versa: start config={} results_db={} from_db={}", config_path.name, results_db_path, from_db)

    bench_config = load_benchmark_config(config_path)
    models_list = bench_config.get("models") or []
    ucsf_models = [m for m in models_list if m.get("api_type") == "ucsf_versa"]
    if not ucsf_models:
        logger.warning("No UCSF Versa models in config; nothing to run.")

    splits_to_run = bench_config.get("splits_to_run") or ["benchmark"]

    if from_db:
        # Load directly from recite.db
        if db_path is None:
            db_path = get_local_db_dir() / "recite.db"
        db_path = resolve_path(Path(db_path), root)
        specs = load_experiments(config_path, project_root=root, include_test=False)
        specs = [s for s in specs if s.get("model", {}).get("api_type") == "ucsf_versa"]
        if not specs:
            logger.warning("No UCSF Versa experiment specs; nothing to run.")
            return
        # Use num_samples from config to limit DB query (for smoke tests)
        num_samples_limit = bench_config.get("num_samples")
        total_samples = min(sum(count_samples_in_db(db_path, s) for s in splits_to_run), num_samples_limit or float("inf"))
        dataloader = get_dataloader(db_path=db_path, from_db=True, splits_order=splits_to_run, limit=num_samples_limit)
    else:
        # Resolve parquet paths from config (respects splits_to_run)
        parquet_paths, parquet_dir = _resolve_parquet_paths_from_config(
            bench_config, root, parquet_dir_override=Path(parquet_dir) if parquet_dir else None
        )
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet splits in {parquet_dir or 'config'}")

        specs = load_experiments(config_path, parquet_paths={k: str(v) for k, v in parquet_paths.items()}, project_root=root, include_test=False)
        specs = [s for s in specs if s.get("model", {}).get("api_type") == "ucsf_versa"]
        if not specs:
            logger.warning("No UCSF Versa experiment specs; nothing to run.")
            return
        total_samples = sum(pq.read_metadata(p).num_rows for p in parquet_paths.values())
        dataloader = get_dataloader(parquet_paths=parquet_paths, splits_order=list(parquet_paths.keys()))

    if phase == "predict":
        for s in specs:
            s["evaluator_type"] = "default"
        logger.info("run_ucsf_versa: phase=predict — skipping LLM judge")
    logger.info("run_ucsf_versa: loaded {} specs, total_samples={}", len(specs), total_samples)

    if migrate_first and predictions_dir:
        predictions_dir = resolve_path(Path(predictions_dir), root)
        if predictions_dir.is_dir():
            logger.info("run_ucsf_versa: migrating from {}", predictions_dir)
            migrate_from_benchmark_predictions(predictions_dir, results_db_path, root)

    use_parallel = max_workers > 1
    _start_versa = time.time()
    logger.info(
        "run_ucsf_versa: starting sample loop total_samples={} max_workers={} — ETA in progress bar below",
        total_samples,
        max_workers if use_parallel else 1,
    )
    progress = tqdm(dataloader, total=total_samples, desc="UCSF Versa", unit="sample", dynamic_ncols=True)

    conn = get_results_connection(results_db_path)
    n_skipped = 0
    n_written = 0
    max_pending = max_workers * 4 if use_parallel else 1
    try:
        for spec in specs:
            spec["config_id"] = ensure_config(conn, spec)
        # Log existing result counts so user sees Versa won't be re-run for those (avoids API cost)
        logger.info("Samples already in results DB are skipped (no API call).")
        # Build spec summary for progress tracking
        spec_summary = []
        for i, spec in enumerate(specs):
            config_id = spec["config_id"]
            model_id = spec.get("model_id", config_id[:12])
            no_rag = spec.get("no_rag", False)
            total_existing = sum(
                count_samples(conn, config_id, split_name)
                for split_name in splits_to_run
            )
            spec_summary.append({
                "idx": i + 1,
                "model_id": model_id,
                "config_id": config_id[:12],
                "no_rag": no_rag,
                "existing": total_existing,
                "target": total_samples,
            })
            # Log every spec so user sees why some still trigger API calls (e.g. no-RAG sweep has 0 rows)
            rag_mode = "no_rag" if no_rag else "rag"
            logger.info(
                "Spec {}/{}: {} ({}) — {}/{} done{}",
                i + 1, len(specs), model_id, rag_mode, total_existing, total_samples,
                " [COMPLETE]" if total_existing >= total_samples else "",
            )

        # Periodic progress logging helper
        _last_progress_log = [time.time()]
        _progress_interval = 60  # Log overall progress every 60 seconds

        def _log_overall_progress():
            now = time.time()
            if now - _last_progress_log[0] < _progress_interval:
                return
            _last_progress_log[0] = now
            # Count current progress per spec
            lines = []
            total_done = 0
            total_target = 0
            for s in spec_summary:
                done = sum(count_samples(conn, s["config_id"], split) for split in splits_to_run)
                pct = 100 * done / s["target"] if s["target"] > 0 else 0
                rag_mode = "no_rag" if s["no_rag"] else "rag"
                lines.append(f"  {s['idx']}/{len(specs)} {s['model_id']} ({rag_mode}): {done}/{s['target']} ({pct:.0f}%)")
                total_done += done
                total_target += s["target"]
            overall_pct = 100 * total_done / total_target if total_target > 0 else 0
            logger.info("=== OVERALL PROGRESS: {}/{} ({:.1f}%) ===", total_done, total_target, overall_pct)
            for line in lines:
                logger.info(line)
        if use_parallel:
            pending: List[tuple] = []  # (future, config_id)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for split_name, row in progress:
                    row_id = int(row.get("id", 0))
                    for spec in specs:
                        config_id = spec["config_id"]
                        num_samples_limit = spec.get("num_samples")
                        if num_samples_limit is not None:
                            if count_samples(conn, config_id, split_name) >= num_samples_limit:
                                continue
                        if has_sample(conn, config_id, row_id, split_name):
                            n_skipped += 1
                            continue
                        logger.info(f"Run id={row_id} split={split_name} config_id={config_id[:12]}")
                        while len(pending) >= max_pending:
                            done = [p for p in pending if p[0].done()]
                            for (f, cid) in done:
                                try:
                                    _, result = f.result()
                                    if result is not None:
                                        ensure_results_table(conn, cid, result)
                                        insert_result(conn, cid, result)
                                        n_written += 1
                                except Exception as e:
                                    logger.warning("Versa worker failed: {}", e)
                            pending = [p for p in pending if p not in done]
                            if not done:
                                for f in as_completed([p[0] for p in pending], timeout=300):
                                    for (pf, cid) in pending:
                                        if pf is f:
                                            try:
                                                _, result = pf.result()
                                                if result is not None:
                                                    ensure_results_table(conn, cid, result)
                                                    insert_result(conn, cid, result)
                                                    n_written += 1
                                            except Exception as e:
                                                logger.warning("Versa worker failed: {}", e)
                                            pending = [p for p in pending if p != (pf, cid)]
                                            break
                                    break
                        future = executor.submit(_versa_task, row, spec, split_name)
                        pending.append((future, config_id))
                    progress.set_postfix(skipped=n_skipped, written=n_written, pending=len(pending), refresh=False)
                    _log_overall_progress()
                for (f, cid) in pending:
                    try:
                        _, result = f.result()
                        if result is not None:
                            ensure_results_table(conn, cid, result)
                            insert_result(conn, cid, result)
                            n_written += 1
                    except Exception as e:
                        logger.warning("Versa worker failed: {}", e)
        else:
            for split_name, row in progress:
                row_id = int(row.get("id", 0))
                for spec in specs:
                    config_id = spec["config_id"]
                    num_samples_limit = spec.get("num_samples")
                    if num_samples_limit is not None:
                        if count_samples(conn, config_id, split_name) >= num_samples_limit:
                            continue
                    if has_sample(conn, config_id, row_id, split_name):
                        n_skipped += 1
                        logger.debug("Skip config_id={} id={} split={} (already in results)", config_id[:12], row_id, split_name)
                        continue
                    logger.info(f"Run id={row_id} split={split_name} config_id={config_id[:12]}")
                    result = run_single_sample(
                        sample_row=row,
                        model=spec["model"],
                        rag_config=spec.get("rag_config"),
                        evaluator_type=spec.get("evaluator_type", "default"),
                        evaluator_config=spec.get("evaluator_config"),
                        prompts_path=Path(spec["prompts_file"]),
                        split_name=split_name,
                        two_step=spec.get("two_step", False),
                        wait_for_revive_seconds=spec.get("wait_for_revive_seconds", 0),
                        max_retries=2,
                        max_delay=5.0,
                    )
                    if result is None:
                        continue
                    ensure_results_table(conn, config_id, result)
                    insert_result(conn, config_id, result)
                    n_written += 1
                progress.set_postfix(skipped=n_skipped, written=n_written, refresh=False)
                _log_overall_progress()
    finally:
        conn.close()

    # Final progress summary
    logger.info("=== FINAL SUMMARY ===")
    conn_final = get_results_connection(results_db_path)
    try:
        for s in spec_summary:
            done = sum(count_samples(conn_final, s["config_id"], split) for split in splits_to_run)
            pct = 100 * done / s["target"] if s["target"] > 0 else 0
            rag_mode = "no_rag" if s["no_rag"] else "rag"
            status = "COMPLETE" if done >= s["target"] else f"{s['target'] - done} remaining"
            logger.info("  {}/{} {} ({}): {}/{} ({:.0f}%) — {}", s["idx"], len(specs), s["model_id"], rag_mode, done, s["target"], pct, status)
    finally:
        conn_final.close()

    _elapsed_versa = time.time() - _start_versa
    logger.info(
        "run_ucsf_versa finished in {:.1f} min (skipped={}, written={}).",
        _elapsed_versa / 60.0,
        n_skipped,
        n_written,
    )


def _is_local_gpu_spec(spec: Dict[str, Any]) -> bool:
    """True if spec is for a local GPU model (endpoint-based or in-process python_gpu)."""
    model = spec.get("model") or {}
    return "endpoint" in model or model.get("api_type") == "python_gpu"


def _run_one_local_gpu_spec(
    spec: Dict[str, Any],
    parquet_dir: Path,
    results_db_path: Path,
    total_samples: int,
    parquet_paths: Optional[Dict[str, Path]] = None,
    db_path: Optional[Path] = None,
    from_db: bool = False,
    splits_order: Optional[List[str]] = None,
) -> tuple:
    """Run one local GPU spec over all samples; uses its own DB connection. Returns (n_skipped, n_written)."""
    conn = get_results_connection(results_db_path)
    try:
        config_id = ensure_config(conn, spec)
        model_id = spec.get("model_id", config_id[:12])
        num_samples_limit = spec.get("num_samples")
        if from_db and db_path is not None:
            dataloader = get_dataloader(db_path=db_path, from_db=True, splits_order=splits_order or ["benchmark"], limit=num_samples_limit)
        elif parquet_paths is not None:
            dataloader = get_dataloader(parquet_paths=parquet_paths, splits_order=list(parquet_paths.keys()))
        else:
            dataloader = get_dataloader(parquet_dir=parquet_dir)
        progress = tqdm(
            dataloader,
            total=total_samples,
            desc=f"Local GPU {model_id}",
            unit="sample",
            dynamic_ncols=True,
        )
        n_skipped = 0
        n_written = 0
        for split_name, row in progress:
            row_id = int(row.get("id", 0))
            num_samples_limit = spec.get("num_samples")
            if num_samples_limit is not None:
                if count_samples(conn, config_id, split_name) >= num_samples_limit:
                    continue
            if has_sample(conn, config_id, row_id, split_name):
                n_skipped += 1
                progress.set_postfix(skipped=n_skipped, written=n_written, refresh=False)
                continue
            logger.info(f"Run id={row_id} split={split_name} config_id={config_id[:12]}")
            result = run_single_sample(
                sample_row=row,
                model=spec["model"],
                rag_config=spec.get("rag_config"),
                evaluator_type=spec.get("evaluator_type", "default"),
                evaluator_config=spec.get("evaluator_config"),
                prompts_path=Path(spec["prompts_file"]),
                split_name=split_name,
                two_step=spec.get("two_step", False),
                wait_for_revive_seconds=spec.get("wait_for_revive_seconds", 0),
                max_retries=2,
                max_delay=5.0,
            )
            if result is None:
                continue
            ensure_results_table(conn, config_id, result)
            insert_result(conn, config_id, result)
            n_written += 1
            progress.set_postfix(skipped=n_skipped, written=n_written, refresh=False)
        logger.info(
            "run_local_gpu: finished spec model_id={} (skipped={}, written={})",
            model_id,
            n_skipped,
            n_written,
        )
        return (n_skipped, n_written)
    finally:
        conn.close()


def _gpu_worker_process(
    spec: Dict[str, Any],
    gpu_ids: List[int],
    samples: List[Tuple[str, Dict[str, Any]]],
    results_db_path: Path,
    worker_id: int,
) -> None:
    """
    Worker subprocess: load model, process all assigned samples, write directly to DB.

    Each subprocess is fully independent - no shared state with other workers.
    Sets CUDA_VISIBLE_DEVICES to pin this process to specific GPU(s).

    Args:
        spec: Experiment spec dict with model config, prompts_file, etc.
        gpu_ids: List of GPU indices this worker should use (e.g. [0] or [0,1]).
        samples: List of (split_name, row_dict) tuples to process.
        results_db_path: Path to the SQLite results database.
        worker_id: Worker index for logging.
    """
    # Pin this process to specific GPU(s) BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    # After CUDA_VISIBLE_DEVICES, this process only sees GPUs 0, 1, ... (n-1).
    # Override spec so the evaluator uses cuda:0 (and cuda:1, ... for multi-GPU) not global indices.
    if spec.get("model") and spec["model"].get("api_type") == "python_gpu":
        spec["model"] = dict(spec["model"])
        spec["model"]["gpu_ids"] = list(range(len(gpu_ids)))  # [0] or [0, 1] etc.

    # Re-import inside subprocess (fresh Python interpreter state)
    from loguru import logger
    from recite.benchmark.evaluator import run_single_sample
    from recite.benchmark.results_db import (
        get_connection as get_results_connection,
        ensure_results_table,
        insert_result,
    )
    from recite.utils.logging_config import configure_logging

    # Configure logging for this subprocess
    configure_logging("INFO")

    config_id = spec["config_id"]
    model_id = spec.get("model_id", config_id[:12])

    logger.info(
        "Worker {}: starting with {} samples on GPU {} (model={})",
        worker_id,
        len(samples),
        gpu_ids,
        model_id,
    )

    # Open DB connection for this worker (SQLite handles concurrent writes)
    conn = get_results_connection(results_db_path)
    # Enable WAL mode for better concurrent write performance
    conn.execute("PRAGMA journal_mode=WAL")

    n_written = 0
    n_failed = 0

    for idx, (split_name, row) in enumerate(samples):
        row_id = int(row.get("id", 0))
        try:
            result = run_single_sample(
                sample_row=row,
                model=spec["model"],
                rag_config=spec.get("rag_config"),
                evaluator_type=spec.get("evaluator_type", "default"),
                evaluator_config=spec.get("evaluator_config"),
                prompts_path=Path(spec["prompts_file"]),
                split_name=split_name,
                two_step=spec.get("two_step", False),
                wait_for_revive_seconds=spec.get("wait_for_revive_seconds", 0),
                max_retries=2,
                max_delay=5.0,
            )
            if result is not None:
                ensure_results_table(conn, config_id, result)
                insert_result(conn, config_id, result)
                n_written += 1
        except Exception as e:
            logger.warning("Worker {}: sample id={} failed: {}", worker_id, row_id, e)
            n_failed += 1

        # Log progress periodically
        processed = idx + 1
        if processed == 1:
            logger.info("Worker {}: first sample processed (id={})", worker_id, row_id)
        elif processed % 50 == 0 or processed == len(samples):
            logger.info(
                "Worker {}: processed {}/{}, written={}, failed={}",
                worker_id,
                processed,
                len(samples),
                n_written,
                n_failed,
            )

    conn.close()
    logger.info(
        "Worker {}: finished (written={}, failed={}, total={})",
        worker_id,
        n_written,
        n_failed,
        len(samples),
    )


def _run_spec_sample_parallel(
    spec: Dict[str, Any],
    parquet_dir: Path,
    results_db_path: Path,
    total_samples: int,
    n_gpus: int,
    max_workers: int,
    parquet_paths: Optional[Dict[str, Path]] = None,
    db_path: Optional[Path] = None,
    from_db: bool = False,
    splits_order: Optional[List[str]] = None,
) -> tuple:
    """
    Run one spec with sample-level parallelism using subprocesses.

    Each subprocess:
    1. Gets a pre-partitioned slice of samples
    2. Loads model onto its assigned GPU
    3. Processes all samples and writes directly to DB
    4. Exits when done

    No IPC queues - fully independent workers with direct DB writes.

    GPU allocation uses spec["model"]["gpus"] (per-model GPU count, default 1).
    n_workers = min(n_gpus // gpus_per_model, max_workers).

    Returns (n_skipped, n_written).
    """
    # Resolve config_id first (identity fallback so path-case differences reuse same config)
    conn = get_results_connection(results_db_path)
    try:
        config_id = ensure_config(conn, spec)
        model_id = spec.get("model_id", config_id[:12])
    finally:
        conn.close()

    # Get per-model GPU requirement
    gpus_per_model = 1
    if spec.get("model") and spec["model"].get("api_type") == "python_gpu":
        gpus_per_model = spec["model"].get("gpus", 1)

    # Calculate effective workers for this spec
    n_workers = n_gpus // gpus_per_model if gpus_per_model > 0 else 1
    n_workers = max(1, min(n_workers, max_workers))

    logger.info(
        "run_local_gpu: spec {} needs {} GPU(s)/model → {} subprocess workers",
        model_id,
        gpus_per_model,
        n_workers,
    )

    num_samples_limit = spec.get("num_samples")
    if from_db and db_path is not None:
        dataloader = get_dataloader(db_path=db_path, from_db=True, splits_order=splits_order or ["benchmark"], limit=num_samples_limit)
    elif parquet_paths is not None:
        dataloader = get_dataloader(parquet_paths=parquet_paths, splits_order=list(parquet_paths.keys()))
    else:
        dataloader = get_dataloader(parquet_dir=parquet_dir)
    samples_to_process: List[Tuple[str, Dict[str, Any]]] = []
    n_skipped = 0
    conn = get_results_connection(results_db_path)
    try:
        for split_name, row in dataloader:
            row_id = int(row.get("id", 0))
            if num_samples_limit is not None:
                current_count = count_samples(conn, config_id, split_name)
                if current_count + len(samples_to_process) >= num_samples_limit:
                    continue
            if has_sample(conn, config_id, row_id, split_name):
                n_skipped += 1
                continue
            samples_to_process.append((split_name, row))
    finally:
        conn.close()

    if not samples_to_process:
        logger.info(
            "run_local_gpu: spec {} has no samples to process (all skipped={})",
            model_id,
            n_skipped,
        )
        return (n_skipped, 0)

    # Partition samples among workers (round-robin: worker 0 gets [0, n, 2n, ...])
    slices: List[List[Tuple[str, Dict[str, Any]]]] = [[] for _ in range(n_workers)]
    for idx, sample in enumerate(samples_to_process):
        slices[idx % n_workers].append(sample)

    logger.info(
        "run_local_gpu: spec {} spawning {} subprocesses for {} samples (~{} each)",
        model_id,
        n_workers,
        len(samples_to_process),
        len(slices[0]) if slices else 0,
    )

    # Spawn worker subprocesses
    # Use 'spawn' method to ensure clean CUDA state in each subprocess
    ctx = multiprocessing.get_context("spawn")
    processes: List[multiprocessing.Process] = []

    for worker_idx in range(n_workers):
        # Compute GPU IDs for this worker
        gpu_start = worker_idx * gpus_per_model
        gpu_ids = list(range(gpu_start, gpu_start + gpus_per_model))

        # Create spec copy with gpu_ids and resolved config_id (so workers write to same table)
        spec_copy = copy.deepcopy(spec)
        spec_copy["config_id"] = config_id
        if spec_copy.get("model") and spec_copy["model"].get("api_type") == "python_gpu":
            spec_copy["model"] = dict(spec_copy["model"])
            spec_copy["model"]["gpu_ids"] = gpu_ids

        p = ctx.Process(
            target=_gpu_worker_process,
            args=(spec_copy, gpu_ids, slices[worker_idx], results_db_path, worker_idx),
        )
        p.start()
        processes.append(p)
        logger.debug("run_local_gpu: spawned worker {} (pid={}, gpus={})", worker_idx, p.pid, gpu_ids)

    # Wait for all workers to finish
    logger.info("run_local_gpu: waiting for {} workers to finish...", n_workers)
    for p in processes:
        p.join()

    # Check exit codes
    failed_workers = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed_workers:
        logger.warning("run_local_gpu: workers {} exited with non-zero code", failed_workers)

    # Count final results from DB
    conn = get_results_connection(results_db_path)
    try:
        n_written = count_samples(conn, config_id, None)
    finally:
        conn.close()

    logger.info(
        "run_local_gpu: finished spec {} (skipped={}, written={})",
        model_id,
        n_skipped,
        n_written,
    )
    return (n_skipped, n_written)


def run_local_gpu(
    config_path: Path,
    results_db_path: Path,
    db_path: Optional[Path] = None,
    parquet_dir: Optional[Path] = None,
    predictions_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    local_gpu_workers: int = 1,
    phase: str = "all",
    from_db: bool = False,
) -> None:
    """
    Run benchmark using local GPU (endpoint or python_gpu models). Same flow as run_ucsf_versa
    but only processes specs with endpoint or api_type=python_gpu (no Versa). No migration
    (migration runs in Versa phase when using backend=all; local GPU runs first).

    Parallelism modes:
    - local_gpu_workers == 1: Sequential, one spec at a time, one sample at a time.
    - local_gpu_workers > 1: Sample-parallel within each spec.
      - n_workers = n_gpus // model["gpus"] (per-model GPU count from config).
      - Each worker has its own copy of the model on its assigned GPU(s).
      - Samples are dispatched to workers in parallel; each worker → model → LLM judge.
    """
    root = project_root or get_project_root()
    config_path = resolve_path(Path(config_path), root)
    results_db_path = resolve_path(Path(results_db_path), root)

    logger.info("run_local_gpu: start config={} results_db={} from_db={}", config_path.name, results_db_path, from_db)

    bench_config = load_benchmark_config(config_path)
    models_list = bench_config.get("models") or []
    local_models = [
        m for m in models_list
        if m.get("api_type") in ("endpoint", "python_gpu")
    ]
    if not local_models:
        logger.warning("No local GPU (endpoint or python_gpu) models in config; nothing to run.")
        return

    splits_to_run = bench_config.get("splits_to_run") or ["benchmark"]

    if from_db:
        # Load directly from recite.db
        if db_path is None:
            db_path = get_local_db_dir() / "recite.db"
        db_path = resolve_path(Path(db_path), root)
        specs = load_experiments(config_path, project_root=root, include_test=False)
        specs = [s for s in specs if _is_local_gpu_spec(s)]
        if not specs:
            logger.warning("No local GPU experiment specs; nothing to run.")
            return
        total_samples = sum(count_samples_in_db(db_path, s) for s in splits_to_run)
        parquet_paths = None  # Not used in DB mode
    else:
        parquet_paths, parquet_dir = _resolve_parquet_paths_from_config(
            bench_config, root, parquet_dir_override=Path(parquet_dir) if parquet_dir else None
        )
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet splits in {parquet_dir or 'config'}")

        specs = load_experiments(
            config_path,
            parquet_paths={k: str(v) for k, v in parquet_paths.items()},
            project_root=root,
            include_test=False,
        )
        specs = [s for s in specs if _is_local_gpu_spec(s)]
        if not specs:
            logger.warning("No local GPU experiment specs; nothing to run.")
            return
        total_samples = sum(pq.read_metadata(p).num_rows for p in parquet_paths.values())

    if phase == "predict":
        for s in specs:
            s["evaluator_type"] = "default"
        logger.info("run_local_gpu: phase=predict — skipping LLM judge")

    logger.info("run_local_gpu: loaded {} specs, total_samples={}", len(specs), total_samples)
    _start_local = time.time()
    logger.info(
        "run_local_gpu: {} specs, {} samples — ETA in progress bar(s) below",
        len(specs),
        total_samples,
    )

    # Determine GPU count
    n_gpus = 1
    try:
        import torch
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except ImportError:
        pass

    use_sample_parallel = local_gpu_workers > 1 and n_gpus > 1
    if use_sample_parallel:
        logger.info(
            "run_local_gpu: sample-parallel mode enabled ({} GPUs available, max {} workers)",
            n_gpus,
            local_gpu_workers,
        )

    conn = get_results_connection(results_db_path)
    n_skipped_total = 0
    n_written_total = 0
    try:
        for spec in specs:
            ensure_config(conn, spec)
    finally:
        conn.close()

    if not use_sample_parallel:
        # Sequential: one spec (one model) at a time, one sample at a time.
        for spec_idx, spec in enumerate(specs):
            logger.info(
                "run_local_gpu: spec {}/{} (sequential)",
                spec_idx + 1,
                len(specs),
            )
            n_skipped, n_written = _run_one_local_gpu_spec(
                spec, parquet_dir, results_db_path, total_samples,
                parquet_paths=parquet_paths,
                db_path=db_path if from_db else None,
                from_db=from_db,
                splits_order=splits_to_run,
            )
            n_skipped_total += n_skipped
            n_written_total += n_written
            clear_python_gpu_cache()
    else:
        # Sample-parallel: for each spec, dispatch samples to N workers (each with model on its GPUs).
        # Workers count is determined per-spec based on model["gpus"].
        for spec_idx, spec in enumerate(specs):
            logger.info(
                "run_local_gpu: spec {}/{} (sample-parallel)",
                spec_idx + 1,
                len(specs),
            )
            n_skipped, n_written = _run_spec_sample_parallel(
                spec,
                parquet_dir,
                results_db_path,
                total_samples,
                n_gpus=n_gpus,
                max_workers=local_gpu_workers,
                parquet_paths=parquet_paths,
                db_path=db_path if from_db else None,
                from_db=from_db,
                splits_order=splits_to_run,
            )
            n_skipped_total += n_skipped
            n_written_total += n_written
            clear_python_gpu_cache()

    _elapsed_local = time.time() - _start_local
    logger.info(
        "run_local_gpu finished in {:.1f} min (skipped={}, written={}).",
        _elapsed_local / 60.0,
        n_skipped_total,
        n_written_total,
    )


def _judge_batch_task(
    config_id: str,
    batch: List[Dict[str, Any]],
    judge_model: str,
    prompts: Any,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, float]]]:
    """Run batched judge on one batch; return (config_id, batch, metrics_list) for DB updates."""
    metrics_list = batched_scorer(
        batch,
        model=judge_model,
        prompts=prompts,
        batch_size=len(batch),
    )
    return (config_id, batch, metrics_list)


def run_judge_only(
    results_db_path: Path,
    judge_batch_size: int = 10,
    judge_workers: int = 4,
    prompts_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> None:
    """
    Run batched LLM judge on all stored predictions that do not yet have judge scores.
    Reads configs from results DB; for each config with evaluator_config (ucsf_versa),
    fetches rows with prediction but no llm_judge_score, runs batched judge, updates rows.
    When judge_workers > 1, batch API calls run in parallel (ThreadPoolExecutor).
    """
    root = project_root or get_project_root()
    results_db_path = resolve_path(Path(results_db_path), root)
    prompts = load_benchmark_prompts(prompts_path)
    conn = get_results_connection(results_db_path)
    try:
        rows = conn.execute("SELECT id, evaluator_config FROM configs").fetchall()
        total_updated = 0
        tasks: List[Tuple[str, List[Dict[str, Any]], str]] = []
        total_pending = 0
        for row in rows:
            config_id = row["id"]
            eval_cfg = row["evaluator_config"]
            if not eval_cfg:
                continue
            try:
                cfg = json.loads(eval_cfg) if isinstance(eval_cfg, str) else eval_cfg
            except Exception:
                continue
            if cfg.get("api_type") != "ucsf_versa" or not cfg.get("model"):
                continue
            judge_model = cfg["model"]
            pending = get_predictions_without_judge(conn, config_id)
            if not pending:
                continue
            logger.info(
                "run_judge_only: config_id={} model={} pending={}",
                config_id[:12],
                judge_model,
                len(pending),
            )
            total_pending += len(pending)
            for start in range(0, len(pending), judge_batch_size):
                batch = pending[start : start + judge_batch_size]
                tasks.append((config_id, batch, judge_model))
        if not tasks:
            logger.info("run_judge_only: no pending predictions to judge.")
            return
        use_parallel = judge_workers > 1
        with tqdm(total=total_pending, desc="Judge", unit="sample", dynamic_ncols=True) as pbar:
            if use_parallel:
                with ThreadPoolExecutor(max_workers=judge_workers) as executor:
                    futures = [
                        executor.submit(_judge_batch_task, cid, batch, jmodel, prompts)
                        for (cid, batch, jmodel) in tasks
                    ]
                    for fut in as_completed(futures):
                        config_id, batch, metrics_list = fut.result()
                        for sample, metrics in zip(batch, metrics_list):
                            update_judge_scores(
                                conn,
                                config_id,
                                sample["id"],
                                sample["split_name"],
                                metrics["llm_judge_binary"],
                                metrics["llm_judge_score"],
                                metrics["llm_judge_normalized"],
                                metrics.get("llm_judge_raw_response"),
                            )
                            total_updated += 1
                        pbar.update(len(batch))
            else:
                for (config_id, batch, judge_model) in tasks:
                    _, _, metrics_list = _judge_batch_task(config_id, batch, judge_model, prompts)
                    for sample, metrics in zip(batch, metrics_list):
                        update_judge_scores(
                            conn,
                            config_id,
                            sample["id"],
                            sample["split_name"],
                            metrics["llm_judge_binary"],
                            metrics["llm_judge_score"],
                            metrics["llm_judge_normalized"],
                            metrics.get("llm_judge_raw_response"),
                        )
                        total_updated += 1
                    pbar.update(len(batch))
        logger.info("run_judge_only finished: {} rows updated.", total_updated)
    finally:
        conn.close()


@app.command()
def run(
    config: Path = typer.Option(Path("config/benchmarks.yaml"), "--config", "-c", help="Path to benchmarks YAML"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path to recite.db (default: LOCAL_DB_DIR/recite.db)"),
    results_db: Optional[Path] = typer.Option(None, "--results-db", help="Path to benchmark results DB (default: LOCAL_DB_DIR/results.db)"),
    parquet_dir: Optional[Path] = typer.Option(None, "--parquet-dir", help="Dir with train/val/test.parquet (only when --no-from-db)"),
    predictions_dir: Optional[Path] = typer.Option(Path("data/benchmark_predictions"), "--predictions-dir", help="Dir to migrate from (benchmark_predictions)"),
    migrate: bool = typer.Option(True, "--migrate/--no-migrate", help="Migrate existing predictions into results DB first"),
    from_db: bool = typer.Option(True, "--from-db/--no-from-db", help="Load data directly from recite.db (default). Use --no-from-db to load from parquet files."),
    backend: str = typer.Option("ucsf_versa", "--backend", "-b", help="Backend: ucsf_versa, local_gpu, or all (local GPU first, then Versa; no intermingling)"),
    phase: str = typer.Option("all", "--phase", "-p", help="Phase: all (predict+judge per sample), predict (store predictions only), judge (run batched judge on stored predictions only)"),
    judge_batch_size: int = typer.Option(10, "--judge-batch-size", help="When phase=judge, number of samples per batched API call"),
    judge_workers: int = typer.Option(1, "--judge-workers", help="When phase=judge, parallel batch API calls (1=one batch at a time; batching is the main win). Config judge_workers overrides if set."),
    versa_workers: int = typer.Option(16, "--versa-workers", "-w", help="Parallel workers for UCSF Versa API (1=sequential). 16 is good for 128-core machines."),
    local_gpu_workers: int = typer.Option(8, "--local-gpu-workers", help="Max parallel sample workers for local GPU. Actual workers = min(n_gpus // model.gpus, this)."),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (DEBUG, INFO, WARNING, ERROR). Use DEBUG to see full tracebacks on sample failures."),
) -> None:
    """Run the RECITE benchmark. Default: load data directly from recite.db (--from-db). Use backend=all to run local GPU first, then Versa. Use --phase predict then --phase judge for two-phase."""
    configure_logging(level=log_level, app_name="recite", also_stderr=True)
    logger.info("recite session_version={}", RECITE_RUN_SESSION_VERSION)
    root = get_project_root()
    config = resolve_path(config, root)
    # Use LOCAL_DB_DIR from .env when paths not explicitly provided
    local_db_dir = get_local_db_dir()
    if db_path is None:
        db_path = local_db_dir / "recite.db"
    if results_db is None:
        results_db = local_db_dir / "results.db"
    db_path = resolve_path(db_path, root)
    results_db = resolve_path(results_db, root)
    parquet_dir = resolve_path(parquet_dir, root) if parquet_dir is not None else None
    predictions_dir = resolve_path(predictions_dir, root) if predictions_dir is not None else None

    # Config can override from_db (e.g., smoke tests set from_db: false to use tiny parquet)
    _bench_early = load_benchmark_config(config)
    if "from_db" in _bench_early:
        from_db = bool(_bench_early["from_db"])
        logger.info("Using from_db={} from config (overrides CLI default)", from_db)

    logger.info(
        "recite run: session={} backend={} config={} results_db={} phase={} from_db={}",
        RECITE_RUN_SESSION_VERSION,
        backend,
        config.name,
        results_db,
        phase,
        from_db,
    )
    _run_start = time.time()

    # Validate DB when loading from DB
    if from_db and phase != "judge":
        ok, n_train, msg = validate_train_split(db_path)
        if ok:
            logger.info("DB validation: {}", msg)
        else:
            logger.warning("DB validation: {}", msg)
        # Also count final_test if it will be used
        bench_config = load_benchmark_config(config)
        splits_to_run = bench_config.get("splits_to_run") or ["benchmark"]
        if "final_test" in splits_to_run:
            n_test = count_samples_in_db(db_path, "final_test")
            logger.info("DB validation: final_test {} samples (merge_source=local)", n_test)

    if phase == "judge":
        _bench = load_benchmark_config(config)
        if _bench.get("judge_workers") is not None:
            judge_workers = int(_bench["judge_workers"])
            logger.info("Using judge_workers={} from config", judge_workers)
        run_judge_only(
            results_db_path=results_db,
            judge_batch_size=judge_batch_size,
            judge_workers=judge_workers,
            prompts_path=root / "config" / "benchmark_prompts.json",
            project_root=root,
        )
        logger.info("recite run finished in {:.1f} min (phase=judge only).", (time.time() - _run_start) / 60.0)
        return

    # Config can set versa_workers / local_gpu_workers; CLI overrides when passed
    _bench = load_benchmark_config(config)
    if backend in ("ucsf_versa", "all") and _bench.get("versa_workers") is not None:
        versa_workers = int(_bench["versa_workers"])
        logger.info("Using versa_workers={} from config", versa_workers)
    if backend in ("local_gpu", "all") and _bench.get("local_gpu_workers") is not None:
        local_gpu_workers = int(_bench["local_gpu_workers"])
        logger.info("Using local_gpu_workers={} from config", local_gpu_workers)

    if backend == "ucsf_versa":
        run_ucsf_versa(
            config_path=config,
            results_db_path=results_db,
            db_path=db_path,
            parquet_dir=parquet_dir,
            predictions_dir=predictions_dir,
            migrate_first=migrate,
            project_root=root,
            max_workers=versa_workers,
            phase=phase,
            from_db=from_db,
        )
        logger.info("recite run finished in {:.1f} min (backend=ucsf_versa).", (time.time() - _run_start) / 60.0)
    elif backend == "local_gpu":
        run_local_gpu(
            config_path=config,
            results_db_path=results_db,
            db_path=db_path,
            parquet_dir=parquet_dir,
            predictions_dir=predictions_dir,
            project_root=root,
            local_gpu_workers=local_gpu_workers,
            from_db=from_db,
        )
        logger.info("recite run finished in {:.1f} min (backend=local_gpu).", (time.time() - _run_start) / 60.0)
    elif backend == "all":
        # Run local GPU first, then Versa (no intermingling)
        # Pre-compute rough ETA for full run
        bench_config = load_benchmark_config(config)
        splits_to_run = bench_config.get("splits_to_run") or ["benchmark"]
        if from_db:
            _total_samples_all = sum(count_samples_in_db(db_path, s) for s in splits_to_run)
            _specs_all = load_experiments(config, project_root=root, include_test=False)
        else:
            pp = bench_config.get("parquet_paths") or {}
            default = get_data_root() / "benchmark_splits"
            first_val = pp.get("benchmark") or pp.get("train") or str(default / "benchmark.parquet")
            first_path = resolve_path(Path(first_val), root)
            _parquet_dir_all = first_path.parent if first_path.suffix == ".parquet" else first_path
            _parquet_paths_all = {}
            for s in splits_to_run:
                p = _parquet_dir_all / (f"{s}.parquet")
                if p.exists():
                    _parquet_paths_all[s] = p
            _total_samples_all = sum(pq.read_metadata(p).num_rows for p in _parquet_paths_all.values()) if _parquet_paths_all else 0
            _specs_all = load_experiments(config, parquet_paths=_parquet_paths_all, project_root=root, include_test=False) if _parquet_paths_all else []
        if _total_samples_all > 0 and _specs_all:
            _n_local = len([s for s in _specs_all if _is_local_gpu_spec(s)])
            _n_versa = len([s for s in _specs_all if (s.get("model") or {}).get("api_type") == "ucsf_versa"])
            _rough_eta_sec = (
                _total_samples_all * _n_local * _DEFAULT_SEC_PER_SAMPLE_LOCAL_GPU
                + _total_samples_all * _n_versa * _DEFAULT_SEC_PER_SAMPLE_VERSA
            )
            logger.info(
                "Rough ETA for full run: ~{:.0f} min ({} samples × {} local + {} versa specs)",
                _rough_eta_sec / 60.0,
                _total_samples_all,
                _n_local,
                _n_versa,
            )
        _t0 = time.time()
        logger.info("recite run (all): phase 1 — local GPU")
        run_local_gpu(
            config_path=config,
            results_db_path=results_db,
            db_path=db_path,
            parquet_dir=parquet_dir,
            predictions_dir=predictions_dir,
            project_root=root,
            local_gpu_workers=local_gpu_workers,
            phase=phase,
            from_db=from_db,
        )
        _t1 = time.time()
        logger.info("Phase 1 (local GPU) finished in {:.1f} min", (_t1 - _t0) / 60.0)
        logger.info("recite run (all): phase 2 — UCSF Versa")
        run_ucsf_versa(
            config_path=config,
            results_db_path=results_db,
            db_path=db_path,
            parquet_dir=parquet_dir,
            predictions_dir=predictions_dir,
            migrate_first=migrate,
            project_root=root,
            max_workers=versa_workers,
            phase=phase,
            from_db=from_db,
        )
        _t2 = time.time()
        logger.info("Phase 2 (Versa) finished in {:.1f} min", (_t2 - _t1) / 60.0)
        logger.info("Total run time: {:.1f} min", (_t2 - _t0) / 60.0)
    else:
        raise typer.BadParameter(f"Unknown backend: {backend}. Use ucsf_versa, local_gpu, or all.")


def _run_ready_checks(
    config_path: Path,
    results_db_path: Path,
    parquet_dir: Optional[Path],
    project_root: Path,
) -> List[Tuple[str, bool, str]]:
    """
    Run pre-flight checks: config, parquet splits, results DB, RAG persist_dir, local GPU.
    Returns list of (check_name, passed, message).
    """
    out: List[Tuple[str, bool, str]] = []
    specs: List[Dict[str, Any]] = []

    # 1. Config exists and loads
    try:
        bench_config = load_benchmark_config(config_path)
        out.append(("config", True, f"Loaded {config_path.name}"))
    except FileNotFoundError as e:
        out.append(("config", False, str(e)))
        return out  # Can't continue without config
    except Exception as e:
        out.append(("config", False, str(e)))
        return out

    # 2. Parquet paths
    pp = bench_config.get("parquet_paths") or {}
    default = get_data_root() / "benchmark_splits"
    first_key = pp.get("benchmark") or pp.get("train") or str(default / "benchmark.parquet")
    first_path = Path(project_root) / first_key
    if parquet_dir is None:
        parquet_dir = first_path.parent if first_path.suffix == ".parquet" else first_path
    parquet_dir = Path(parquet_dir)
    parquet_paths = {}
    for s in ("benchmark", "train", "val", "test"):
        p = parquet_dir / (f"{s}.parquet")
        if p.exists():
            parquet_paths[s] = p
    if not parquet_paths:
        out.append(("parquet", False, f"No parquet in {parquet_dir} (need benchmark.parquet or train/val/test.parquet)"))
    else:
        try:
            total_rows = sum(pq.read_metadata(p).num_rows for p in parquet_paths.values())
            out.append(("parquet", True, f"{parquet_dir}: {len(parquet_paths)} splits, {total_rows} rows"))
        except Exception as e:
            out.append(("parquet", False, str(e)))

    # 3. Results DB: parent writable and can init
    try:
        results_db_path = Path(results_db_path)
        if not results_db_path.is_absolute():
            results_db_path = project_root / results_db_path
        parent = results_db_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not os.access(parent, os.W_OK):
            out.append(("results_db", False, f"Results DB parent not writable: {parent}"))
        else:
            conn = get_results_connection(results_db_path)
            conn.execute("SELECT 1 FROM configs LIMIT 1")
            conn.close()
            out.append(("results_db", True, f"Writable and initialized: {results_db_path}"))
    except sqlite3.OperationalError:
        # configs table might not exist yet; init_results_db creates it
        try:
            conn = get_results_connection(results_db_path)
            conn.close()
            out.append(("results_db", True, f"Writable and initialized: {results_db_path}"))
        except Exception as e2:
            out.append(("results_db", False, str(e2)))
    except Exception as e:
        out.append(("results_db", False, str(e)))

    # 4. Specs load (depends on parquet)
    if not parquet_paths:
        out.append(("specs", False, "Skipped (no parquet)"))
    else:
        try:
            specs.extend(load_experiments(config_path, parquet_paths=parquet_paths, project_root=project_root, include_test=False))
            n_local = len([s for s in specs if _is_local_gpu_spec(s)])
            n_versa = len([s for s in specs if (s.get("model") or {}).get("api_type") == "ucsf_versa"])
            out.append(("specs", True, f"{len(specs)} specs ({n_local} local GPU, {n_versa} Versa)"))
        except Exception as e:
            out.append(("specs", False, str(e)))

    # 5. RAG persist_dir (if any spec uses RAG)
    specs_ok = any(r[0] == "specs" and r[1] for r in out)
    if not specs_ok:
        pass
    else:
        needs_rag = any(
            (s.get("model") or {}).get("api_type") in ("ucsf_versa", "endpoint")
            for s in specs
        )
        if needs_rag:
            rag = bench_config.get("rag") or {}
            persist_raw = rag.get("persist_dir") or "data/llamaindex_cache"
            persist_dir_path = resolve_path(Path(persist_raw), project_root)
            if not persist_dir_path.exists():
                out.append(("rag_persist_dir", False, f"RAG persist_dir missing (run build-rag-index first): {persist_dir_path}"))
            elif not os.access(persist_dir_path, os.W_OK):
                out.append(("rag_persist_dir", False, f"RAG persist_dir not writable: {persist_dir_path}"))
            else:
                # Check for at least one index (docstore.json in a subdir)
                subdirs = [d for d in persist_dir_path.iterdir() if d.is_dir()]
                has_index = any((d / "docstore.json").exists() for d in subdirs)
                if not has_index:
                    out.append(("rag_index", False, f"RAG persist_dir has no indices (run: recite build-rag-index --config {config_path.name})"))
                else:
                    out.append(("rag_index", True, f"{persist_dir_path}: {len(subdirs)} index dirs"))

    # 6. Local GPU: CUDA available (if any local GPU spec)
    if specs_ok:
        if any(_is_local_gpu_spec(s) for s in specs):
            try:
                import torch
                cuda_ok = torch.cuda.is_available()
                n_gpus = torch.cuda.device_count() if cuda_ok else 0
                if not cuda_ok:
                    out.append(("local_gpu", False, "CUDA not available (local GPU specs need GPU)"))
                elif n_gpus < 1:
                    out.append(("local_gpu", False, "No GPUs visible"))
                else:
                    out.append(("local_gpu", True, f"CUDA OK, {n_gpus} GPU(s) available"))
            except ImportError:
                out.append(("local_gpu", False, "torch not installed (required for local GPU)"))

    return out


@app.command("ready")
def ready(
    config: Path = typer.Option(Path("config/benchmarks.yaml"), "--config", "-c", help="Path to benchmarks YAML"),
    results_db: Path = typer.Option(Path("data/dev/benchmark_results.db"), "--results-db", help="Path to benchmark results DB"),
    parquet_dir: Optional[Path] = typer.Option(None, "--parquet-dir", help="Dir with train/val/test.parquet (default from config)"),
) -> None:
    """Check that config, parquet splits, results DB, RAG index, and GPUs are ready for a benchmark run.
    Exits 0 if all checks pass, 1 otherwise. Use before 'recite run' to avoid mid-run failures."""
    root = get_project_root()
    config = resolve_path(config, root)
    results_db = resolve_path(results_db, root)
    parquet_dir = resolve_path(parquet_dir, root) if parquet_dir is not None else None

    checks = _run_ready_checks(config, results_db, parquet_dir, root)

    all_pass = all(passed for (_name, passed, _msg) in checks)
    for name, passed, msg in checks:
        status = "OK" if passed else "FAIL"
        logger.info("{} {}: {}", status, name, msg)
    if all_pass:
        logger.info("All checks passed. Ready to run.")
        raise SystemExit(0)
    logger.error("Some checks failed. Fix the issues above before running.")
    raise SystemExit(1)


@app.command()
def verify_local_gpu(
    config: Path = typer.Option(Path("config/benchmarks_large.yaml"), "--config", "-c", help="Path to benchmarks YAML"),
    results_db: Path = typer.Option(Path("data/benchmark_results_verify.db"), "--results-db", help="Path to verification results DB"),
    parquet_dir: Optional[Path] = typer.Option(None, "--parquet-dir", help="Dir with train/val/test.parquet (default from config)"),
    samples_per_model: int = typer.Option(2, "--samples-per-model", "-n", help="Number of samples to run per model for verification"),
) -> None:
    """Load each local GPU model from config, run a few samples through the full pipeline (model + judge),
    write to results DB, and verify DB rows and non-empty predictions. Use to confirm models load and
    responses are written correctly before a full benchmark run."""
    root = get_project_root()
    if not config.is_absolute():
        config = root / config
    if not results_db.is_absolute():
        results_db = root / results_db
    if parquet_dir is not None and not parquet_dir.is_absolute():
        parquet_dir = root / parquet_dir

    bench_config = load_benchmark_config(config)
    parquet_paths, parquet_dir = _resolve_parquet_paths_from_config(
        bench_config, root, parquet_dir_override=Path(parquet_dir) if parquet_dir else None
    )
    if not parquet_paths:
        raise FileNotFoundError(
            "No parquet splits in config or dir. Create splits first (e.g. scripts/create_tiny_benchmark_splits.py) "
            "or set parquet_paths in config."
        )

    specs = load_experiments(
        config,
        parquet_paths={k: str(v) for k, v in parquet_paths.items()},
        project_root=root,
        include_test=False,
    )
    specs = [s for s in specs if _is_local_gpu_spec(s)]
    if not specs:
        logger.warning("No local GPU (endpoint or python_gpu) models in config; nothing to verify.")
        raise SystemExit(0)

    # Collect first N samples once (shared across all specs)
    dataloader = get_dataloader(parquet_paths=parquet_paths, splits_order=list(parquet_paths.keys()))
    samples: List[tuple] = []
    for split_name, row in dataloader:
        if len(samples) >= samples_per_model:
            break
        samples.append((split_name, row))
    if not samples:
        raise FileNotFoundError(f"No rows in parquet splits under {parquet_dir}")

    logger.info(
        "verify-local-gpu: config={} results_db={} parquet_dir={} specs={} samples={} — ETA in progress per model",
        config.name,
        results_db,
        parquet_dir,
        len(specs),
        len(samples),
    )
    _start_verify = time.time()

    conn = get_results_connection(results_db)
    try:
        for spec in specs:
            spec["config_id"] = ensure_config(conn, spec)
        for spec_idx, spec in enumerate(specs):
            config_id = spec["config_id"]
            model_id = spec.get("model_id", config_id[:12])
            logger.info("verify-local-gpu: loading model {} ({}/{})", model_id, spec_idx + 1, len(specs))
            for split_name, row in samples:
                row_id = int(row.get("id", 0))
                result = run_single_sample(
                    sample_row=row,
                    model=spec["model"],
                    rag_config=spec.get("rag_config"),
                    evaluator_type=spec.get("evaluator_type", "default"),
                    evaluator_config=spec.get("evaluator_config"),
                    prompts_path=Path(spec["prompts_file"]),
                    split_name=split_name,
                    two_step=spec.get("two_step", False),
                    wait_for_revive_seconds=spec.get("wait_for_revive_seconds", 0),
                    max_retries=2,
                    max_delay=5.0,
                )
                if result is None:
                    raise RuntimeError(
                        f"verify-local-gpu: run_single_sample returned None for model_id={model_id} id={row_id} split={split_name}"
                    )
                ensure_results_table(conn, config_id, result)
                insert_result(conn, config_id, result)
                if not has_sample(conn, config_id, row_id, split_name):
                    raise RuntimeError(
                        f"verify-local-gpu: DB verify failed (has_sample false after insert) model_id={model_id} id={row_id} split={split_name}"
                    )
                pred = result.get("prediction")
                if pred is None or not (isinstance(pred, str) and pred.strip()):
                    raise RuntimeError(
                        f"verify-local-gpu: empty prediction for model_id={model_id} id={row_id} split={split_name}"
                    )
            clear_python_gpu_cache()
        _elapsed_verify = time.time() - _start_verify
        logger.info(
            "verify-local-gpu finished in {:.1f} min. All {} model(s) passed (DB writes and non-empty predictions).",
            _elapsed_verify / 60.0,
            len(specs),
        )
    finally:
        conn.close()


def _build_rag_index_impl(
    config_path: Path,
    project_root: Path,
    db_path: Optional[Path] = None,
) -> int:
    """Load config, collect unique evidence from DB or parquet splits, build indices with local embedder on GPU. Returns count of indices built."""
    bench_config = load_benchmark_config(config_path)
    rag = bench_config.get("rag") or {}
    embed_local_model = rag.get("embed_local_model")
    if not embed_local_model:
        raise ValueError("rag.embed_local_model required for build-rag-index (e.g. BAAI/bge-small-en-v1.5)")
    persist_dir = resolve_path(Path(rag.get("persist_dir") or "data/llamaindex_cache"), project_root)
    persist_dir.mkdir(parents=True, exist_ok=True)
    embed_device_index = rag.get("embed_device_index", "cuda:0")

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError:
        raise ImportError("build-rag-index requires llama-index-embeddings-huggingface. Install with: pip install llama-index-embeddings-huggingface") from None

    embed = HuggingFaceEmbedding(
        model_name=embed_local_model.strip(),
        device=embed_device_index.strip(),
        model_kwargs={"low_cpu_mem_usage": False, "device_map": None},
    )
    logger.info("Loaded local embedder {} on {}", embed_local_model, embed_device_index)

    # Check if we should use DB mode (default) or parquet mode
    from_db = bench_config.get("from_db", True)  # Default to DB mode

    if from_db:
        # Load from recite.db
        if db_path is None:
            db_path = get_local_db_dir() / "recite.db"
        db_path = resolve_path(Path(db_path), project_root)
        if not db_path.exists():
            raise FileNotFoundError(f"DB not found: {db_path}. Use --db-path or set from_db: false in config.")

        # Index both benchmark (cluster1) and final_test (local) splits
        splits_to_index = ["benchmark", "final_test"]
        total_rows = sum(count_samples_in_db(db_path, s) for s in splits_to_index)
        logger.info("Building RAG index from DB: {} ({} total rows from {})", db_path, total_rows, splits_to_index)

        def _iter_rows() -> Iterator[tuple]:
            for split_name, row in stream_from_db(db_path, splits_to_index, batch_size=1000):
                yield split_name, row
    else:
        # Use parquet files (legacy mode or smoke tests)
        pp = bench_config.get("parquet_paths") or {}
        default_dir = get_data_root() / "benchmark_splits"
        paths = {}
        for s, p in pp.items():
            path = resolve_path(Path(p), project_root)
            if path.exists():
                paths[s] = path
        if not paths:
            for s in ("benchmark", "train", "val", "test", "final_test"):
                fname = "benchmark.parquet" if s == "benchmark" else f"{s}.parquet"
                candidate = default_dir / fname
                if candidate.exists():
                    paths[s] = resolve_path(candidate, project_root)
        if not paths:
            raise FileNotFoundError(
                "No parquet paths found in config or files missing; set parquet_paths or use from_db: true"
            )

        total_rows = sum(pq.read_metadata(p).num_rows for p in paths.values())
        logger.info("Building RAG index from parquet: {} splits, {} total rows", len(paths), total_rows)

        def _iter_rows() -> Iterator[tuple]:
            for split_name, path in paths.items():
                for _split, row in stream_parquet_splits({split_name: path}):
                    yield split_name, row

    seen: set = set()
    built = 0
    pbar = tqdm(
        _iter_rows(),
        total=total_rows,
        desc="RAG index",
        unit="row",
        dynamic_ncols=True,
    )
    for _split_name, row in pbar:
        ev = row.get("evidence")
        if ev is None or (hasattr(ev, "__float__") and getattr(ev, "__float__", None) and str(ev) == "nan"):
            continue
        doc = str(ev).strip()
        if not doc or doc in seen:
            continue
        seen.add(doc)
        if build_index_for_document(doc, persist_dir, embed):
            built += 1
        pbar.set_postfix(built=built, unique=len(seen), refresh=False)
    logger.info("Built {} new RAG indices under {} ({} unique documents)", built, persist_dir, len(seen))
    return built


@app.command("build-rag-index")
def build_rag_index(
    config: Path = typer.Option(Path("config/benchmarks.yaml"), "--config", "-c", help="Path to benchmarks YAML"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path to recite.db (default: data/dev/recite.db)"),
) -> None:
    """Build RAG indices for all unique documents using local embedder on GPU. Loads from DB by default (both benchmark and final_test splits). Run once before benchmark when using rag.embed_local_model."""
    root = get_project_root()
    if not config.is_absolute():
        config = root / config
    if db_path is not None:
        db_path = resolve_path(db_path, root)
    _start_rag = time.time()
    n = _build_rag_index_impl(config, root, db_path=db_path)
    _elapsed_rag = time.time() - _start_rag
    logger.info("build-rag-index finished: {} indices built in {:.1f} min.", n, _elapsed_rag / 60.0)


@app.command()
def export_splits(
    db_path: Path = typer.Option(Path("data/dev/recite.db"), "--db-path", help="Path to recite.db"),
    output_dir: Path = typer.Option(Path("data/benchmark_splits"), "--output-dir", "-o", help="Output dir for parquets"),
    include_final_test: bool = typer.Option(True, "--include-final-test/--no-final-test", help="Also export final_test.parquet from merge_source=local"),
    num_final_test: Optional[int] = typer.Option(None, "--num-final-test", help="Limit final_test to N samples (default: all)"),
) -> None:
    """Export train (benchmark.parquet) and optionally test (final_test.parquet) from recite.db."""
    root = get_project_root()
    db_path = resolve_path(db_path, root)
    output_dir = resolve_path(output_dir, root)
    logger.info(
        "export_splits: db_path={} output_dir={} include_final_test={}",
        db_path,
        output_dir,
        include_final_test,
    )
    out_dir, n_benchmark, n_final_test = ensure_splits(
        db_path,
        output_dir,
        include_final_test=include_final_test,
        final_test_num_samples=num_final_test,
    )
    logger.info("Exported to {}: benchmark {} samples", out_dir, n_benchmark)
    if n_final_test is not None:
        logger.info("Exported to {}: final_test {} samples", out_dir, n_final_test)


@app.command()
def migrate(
    predictions_dir: Path = typer.Option(Path("data/benchmark_predictions"), "--predictions-dir", help="benchmark_predictions root"),
    results_db: Path = typer.Option(Path("data/dev/benchmark_results.db"), "--results-db", help="Benchmark results DB path"),
) -> None:
    """Migrate existing benchmark_predictions run dirs into benchmark_results.db."""
    root = get_project_root()
    predictions_dir = resolve_path(predictions_dir, root)
    results_db = resolve_path(results_db, root)
    n = migrate_from_benchmark_predictions(predictions_dir, results_db, root)
    logger.info(f"Migrated {n} result rows.")


@app.command("merge-duplicate-configs")
def merge_duplicate_configs_cmd(
    results_db: Path = typer.Option(Path("data/dev/benchmark_results.db"), "--results-db", help="Benchmark results DB path"),
) -> None:
    """Merge duplicate configs in the results DB (same identity: model, RAG, prompts, evaluator). Keeps earliest config, merges result rows, drops duplicate tables."""
    root = get_project_root()
    results_db = resolve_path(Path(results_db), root)
    conn = get_results_connection(results_db)
    try:
        n = merge_duplicate_configs(conn)
        logger.info("merge-duplicate-configs: removed {} duplicate config(s).", n)
    finally:
        conn.close()


@app.command("benchmark-summary")
def benchmark_summary(
    results_db: Path = typer.Option(Path("data/dev/benchmark_results.db"), "--results-db", help="Benchmark results DB path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write markdown to file"),
) -> None:
    """Print (or write) benchmark summary table from benchmark_results.db: model_id, no_rag, top_k, split, n, and mean metrics for comparing models and RAG."""
    root = get_project_root()
    results_db = Path(results_db)
    if not results_db.is_absolute():
        results_db = root / results_db
    conn = get_results_connection(results_db)
    try:
        rows = get_benchmark_summary_rows(conn)
    finally:
        conn.close()
    if not rows:
        logger.info("No benchmark results in DB.")
        return
    header = ["model_id", "no_rag", "top_k", "split_name", "n"] + [f"{c}_mean" for c in BENCHMARK_METRIC_COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [
            str(r.get("model_id", "")),
            "yes" if r.get("no_rag") else "no",
            str(r.get("top_k", "")),
            str(r.get("split_name", "")),
            str(r.get("n", 0)),
        ]
        for c in BENCHMARK_METRIC_COLUMNS:
            v = r.get(f"{c}_mean")
            if v is None:
                cells.append("—")
            else:
                try:
                    cells.append(f"{float(v):.4f}")
                except (TypeError, ValueError):
                    cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md = "# Benchmark summary (from DB)\n\n" + "\n".join(lines)
    if output:
        output = Path(output)
        output = resolve_path(output, root)
        output.write_text(md)
        logger.info("Wrote {}", output)
    else:
        print(md)


@app.command("clear-local-gpu-results")
def clear_local_gpu_results(
    config: Path = typer.Option(Path("config/benchmarks_large.yaml"), "--config", "-c", help="Path to benchmarks YAML"),
    results_db: Path = typer.Option(Path("data/dev/benchmark_results.db"), "--results-db", help="Benchmark results DB path"),
) -> None:
    """Delete all result rows for local GPU (python_gpu/endpoint) configs from the given config.
    Use after a botched run so the next run re-processes those samples instead of skipping them."""
    root = get_project_root()
    config = resolve_path(config, root)
    results_db = resolve_path(results_db, root)

    bench_config = load_benchmark_config(config)
    parquet_paths, parquet_dir = _resolve_parquet_paths_from_config(bench_config, root, None)
    if not parquet_paths:
        raise FileNotFoundError("No parquet splits in config; cannot resolve config_ids.")

    specs = load_experiments(
        config,
        parquet_paths={k: str(v) for k, v in parquet_paths.items()},
        project_root=root,
        include_test=False,
    )
    specs = [s for s in specs if _is_local_gpu_spec(s)]
    if not specs:
        logger.warning("No local GPU specs in config; nothing to clear.")
        return

    conn = get_results_connection(results_db)
    total_deleted = 0
    try:
        for spec in specs:
            existing = find_config(conn, spec)
            config_id = existing["id"] if existing else spec["config_id"]
            model_id = spec.get("model_id", config_id[:12])
            n = clear_results_for_config(conn, config_id)
            total_deleted += n
            if n > 0:
                logger.info("Cleared {} rows for config {} (model_id={}).", n, config_id[:12], model_id)
    finally:
        conn.close()
    logger.info("clear-local-gpu-results: deleted {} rows across {} config(s).", total_deleted, len(specs))


if __name__ == "__main__":
    app()
