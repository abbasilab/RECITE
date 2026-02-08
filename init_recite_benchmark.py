"""
init_recite_benchmark.py

Multi-worker init-benchmark: scale RECITE pipeline by distributing samples from
chunks to workers in batches. Uses same chunking seed as init-benchmark (42).

Usage:
  uv run init_recite_benchmark.py --workers 4 --num-chunks 10 --chunks 0,1,2,3

DB: Reads/writes RECITE DB under data/dev/ by default (LOCAL_DB_DIR/recite.db).
Set .env LOCAL_DB_DIR or --db-path to override.

Rate limits: ClinicalTrials.gov does not publish a strict N req/s; this script
uses a shared rate limiter (default 5 req/s). Recommend workers 3-5. See:
  https://www.clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Set

import typer
from loguru import logger
from tqdm import tqdm

from recite.benchmark.db import get_db_path, init_database
from recite.benchmark.discovery import check_trial_versions_batch, get_chunked_instance_ids
from recite.benchmark.downloaders import download_protocols, download_versions
from recite.benchmark.builders import create_recite_instances
from recite.benchmark.processors import extract_evidence, identify_amendments
from recite.benchmark.rate_limiter import SharedRateLimiter
from recite.benchmark.utils import get_trials_with_protocols, get_trials_with_versions
from recite.crawler.adapters import ClinicalTrialsGovAdapter

app = typer.Typer(help="Multi-worker init-benchmark: process chunks in parallel with batched workers. DB: data/dev/recite.db by default (LOCAL_DB_DIR).")


def _completed_instance_ids(conn) -> Set[str]:
    """NCT IDs that have reached the end (recite or ec_changes with evidence)."""
    cursor = conn.cursor()
    try:
        rows = cursor.execute(
            "SELECT DISTINCT instance_id FROM recite"
        ).fetchall()
        from_recite = {row["instance_id"] for row in rows}
    except Exception:
        from_recite = set()
    try:
        rows = cursor.execute(
            """SELECT DISTINCT instance_id FROM ec_changes
               WHERE evidence_source_path IS NOT NULL AND evidence_source_path != ''"""
        ).fetchall()
        from_ec = {row["instance_id"] for row in rows}
    except Exception:
        from_ec = set()
    return from_recite | from_ec


def _stage2_completed(conn) -> Set[str]:
    """NCT IDs already in trials_with_versions (version check done)."""
    cursor = conn.cursor()
    rows = cursor.execute("SELECT instance_id FROM trials_with_versions").fetchall()
    return {row["instance_id"] for row in rows}


def _stage3_completed(conn) -> Set[str]:
    """NCT IDs that have trial_versions (versions downloaded)."""
    cursor = conn.cursor()
    rows = cursor.execute("SELECT DISTINCT instance_id FROM trial_versions").fetchall()
    return {row["instance_id"] for row in rows}


def _stage5_completed(conn) -> Set[str]:
    """NCT IDs that have been checked for protocols (either downloaded or marked unavailable)."""
    cursor = conn.cursor()
    rows = cursor.execute(
        """SELECT DISTINCT instance_id FROM ec_changes
           WHERE evidence_source IS NOT NULL"""
    ).fetchall()
    return {row["instance_id"] for row in rows}


def _stage6_completed_instance_ids(conn) -> Set[str]:
    """NCT IDs that have protocol text extracted (in protocol_texts table)."""
    cursor = conn.cursor()
    try:
        rows = cursor.execute(
            "SELECT instance_id FROM protocol_texts"
        ).fetchall()
        return {row["instance_id"] for row in rows}
    except sqlite3.OperationalError:
        # protocol_texts table may not exist in old DBs
        return set()


def _get_stage_status(conn) -> Dict[str, Dict[str, Any]]:
    """Return completion status for each pipeline stage."""
    cursor = conn.cursor()

    def _safe_count(sql: str, default: int = 0) -> int:
        try:
            return cursor.execute(sql).fetchone()[0]
        except (sqlite3.OperationalError, TypeError):
            return default

    stage2_total = _safe_count("SELECT COUNT(*) FROM trials_with_versions")
    stage3_remaining = _safe_count("SELECT COUNT(*) FROM trials_with_versions WHERE versions_downloaded = 0")
    stage4_total = _safe_count("SELECT COUNT(*) FROM ec_changes")
    stage4_unique_trials = _safe_count("SELECT COUNT(DISTINCT instance_id) FROM ec_changes")
    # Stage 5: protocol check status
    stage5_total = stage4_unique_trials  # total trials to check
    stage5_checked = _safe_count("SELECT COUNT(DISTINCT instance_id) FROM ec_changes WHERE evidence_source IS NOT NULL")
    stage5_with_protocol = _safe_count(
        "SELECT COUNT(DISTINCT instance_id) FROM ec_changes WHERE evidence_source = 'protocol_pdf'"
    )
    stage5_remaining = stage5_total - stage5_checked
    # Stage 6: extract text from protocols
    stage6_total = stage5_with_protocol
    stage6_remaining = stage6_total  # default
    try:
        stage6_remaining = _safe_count(
            """SELECT COUNT(DISTINCT ec.instance_id) FROM ec_changes ec
               WHERE ec.evidence_source = 'protocol_pdf'
               AND ec.instance_id NOT IN (SELECT instance_id FROM protocol_texts)"""
        )
    except Exception:
        pass
    recite_count = _safe_count("SELECT COUNT(*) FROM recite")

    return {
        "stage1": {
            "name": "Discovery",
            "total": _safe_count("SELECT COUNT(*) FROM discovered_trials"),
            "complete": _safe_count("SELECT COUNT(*) FROM discovered_trials") > 0,
        },
        "stage2": {
            "name": "Version Check",
            "total": stage2_total,
            "remaining": 0,
            "complete": stage2_total > 0,
        },
        "stage3": {
            "name": "Download Versions",
            "total": stage2_total,
            "remaining": stage3_remaining,
            "complete": stage3_remaining == 0 and stage2_total > 0,
        },
        "stage4": {
            "name": "Identify EC Changes",
            "total": stage4_total,
            "complete": stage4_total > 0,
        },
        "stage5": {
            "name": "Check/Download Protocols",
            "total": stage5_total,
            "checked": stage5_checked,
            "with_protocol": stage5_with_protocol,
            "remaining": stage5_remaining,
            "complete": stage5_remaining == 0,
        },
        "stage6": {
            "name": "Extract Protocol Text",
            "total": stage6_total,
            "remaining": stage6_remaining,
            "complete": stage6_remaining == 0 and stage6_total > 0,
        },
        "stage7": {
            "name": "Build RECITE",
            "total": stage5_with_protocol,
            "existing": recite_count,
        },
    }


def _print_stage_status(status: Dict[str, Dict[str, Any]]) -> None:
    """Log stage completion status for user."""
    s1 = status["stage1"]
    logger.info(f"Stage 1 ({s1['name']}): {s1['total']:,} NCT IDs" + (" ✓ Complete" if s1["complete"] else ""))
    s2 = status["stage2"]
    logger.info(f"Stage 2 ({s2['name']}): {s2['total']:,} trials" + (" ✓ Complete" if s2["complete"] else ""))
    s3 = status["stage3"]
    logger.info(f"Stage 3 ({s3['name']}): {s3['remaining']:,}/{s3['total']:,} remaining")
    s4 = status["stage4"]
    logger.info(f"Stage 4 ({s4['name']}): {s4['total']:,} EC changes" + (" ✓ Complete" if s4.get("complete") else ""))
    s5 = status["stage5"]
    logger.info(f"Stage 5 ({s5['name']}): {s5['checked']:,}/{s5['total']:,} checked, {s5['with_protocol']:,} have protocols, {s5['remaining']:,} unchecked")
    s6 = status["stage6"]
    logger.info(f"Stage 6 ({s6['name']}): {s6['remaining']:,}/{s6['total']:,} trials needing extraction")
    s7 = status["stage7"]
    logger.info(f"Stage 7 ({s7['name']}): {s7['existing']:,} RECITE instances (from {s7['total']:,} trials with protocols)")


def _ids_for_chunks(
    all_instance_ids: List[str],
    chunk_indices: List[int],
    num_chunks: int,
    chunk_seed: int,
) -> List[tuple[int, List[str]]]:
    """Return (chunk_index, instance_ids) for each requested chunk."""
    out = []
    for ci in chunk_indices:
        ids = get_chunked_instance_ids(all_instance_ids, ci, num_chunks, seed=chunk_seed)
        out.append((ci, ids))
    return out


def _build_batches(
    chunk_tuples: List[tuple[int, List[str]]],
    completed: Set[str],
    worker_batch_size: int,
) -> List[List[str]]:
    """Stream over chunk NCT IDs; skip completed; fill batches of worker_batch_size."""
    batches = []
    current = []
    for _ci, instance_ids in chunk_tuples:
        for instance_id in instance_ids:
            if instance_id in completed:
                continue
            current.append(instance_id)
            if len(current) >= worker_batch_size:
                batches.append(current)
                current = []
    if current:
        batches.append(current)
    return batches


def _run_stage_with_workers(
    stage_name: str,
    batches: List[List[str]],
    db_path: Path,
    lock: threading.Lock,
    adapter: Optional[ClinicalTrialsGovAdapter],
    rate_limiter: Optional[SharedRateLimiter],
    max_retries: int,
    retry_min_secs: float,
    use_expedited: bool,
    discovery_method: str,
    num_workers: int,
) -> None:
    """Run a parallel stage: workers process batches under lock with retries.
    Each worker opens its own DB connection (SQLite connections are not thread-safe).
    """
    if not batches:
        logger.info(f"{stage_name}: no batches to process")
        return
    total_ids = sum(len(b) for b in batches)
    stage_start = time.time()
    logger.info(f"{stage_name}: {len(batches)} batches, {total_ids} IDs")
    work_queue: Queue[Optional[List[str]]] = Queue()
    for b in batches:
        work_queue.put(b)

    done_count = 0
    done_lock = threading.Lock()

    with tqdm(total=total_ids, desc=stage_name, unit="id", dynamic_ncols=True) as pbar:

        def worker():
            nonlocal done_count
            worker_conn = init_database(db_path, force=False)
            try:
                while True:
                    batch = work_queue.get()
                    if batch is None:
                        break
                    for attempt in range(max_retries + 1):
                        try:
                            # Acquire rate limiter tokens BEFORE work (under lock)
                            if rate_limiter:
                                with lock:
                                    for _ in batch:
                                        rate_limiter.acquire()
                            
                            # Do actual work OUTSIDE lock (allows parallelism)
                            if stage_name == "Stage 2":
                                check_trial_versions_batch(
                                    batch,
                                    adapter=adapter or ClinicalTrialsGovAdapter(),
                                    conn=worker_conn,
                                    discovery_method=discovery_method,
                                    use_expedited=use_expedited,
                                    chunk_index=None,
                                    total_chunks=None,
                                )
                            elif stage_name == "Stage 3":
                                download_versions(batch, None, worker_conn, use_expedited=use_expedited)
                            elif stage_name == "Stage 5":
                                download_protocols(batch, None, worker_conn)
                            elif stage_name == "Stage 6":
                                extract_evidence(None, worker_conn, instance_ids=batch)
                            else:
                                raise ValueError(f"Unknown stage {stage_name}")
                            with done_lock:
                                done_count += len(batch)
                                pbar.update(len(batch))
                                elapsed = time.time() - stage_start
                                rate = done_count / elapsed if elapsed > 0 else 0
                                eta_sec = (total_ids - done_count) / rate if rate > 0 else 0
                                pbar.set_postfix(rate=f"{rate:.1f}/s", eta_min=f"{eta_sec/60:.1f}m", refresh=False)
                            break
                        except Exception as e:
                            if attempt < max_retries:
                                delay = retry_min_secs * (2**attempt)
                                logger.warning(f"{stage_name} batch failed (attempt {attempt+1}), retrying in {delay:.1f}s: {e}")
                                time.sleep(delay)
                            else:
                                logger.exception(f"{stage_name} batch failed after {max_retries+1} attempts: {e}")
            finally:
                worker_conn.close()

        import concurrent.futures
        n_workers = min(num_workers, len(batches), 32)
        for _ in range(n_workers):
            work_queue.put(None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(worker) for _ in range(n_workers)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

    elapsed_min = (time.time() - stage_start) / 60
    logger.info(f"{stage_name}: complete ({done_count}/{total_ids} IDs) in {elapsed_min:.1f} min")


def _get_instance_ids_from_source(
    conn,
    discovery_source: str,
    discovery_method: str,
    max_trials: Optional[int],
) -> List[str]:
    """
    Get NCT ID list from the chosen source.
    - discovered: use discovered_trials (run discovery if empty)
    - trials_with_versions: use DB table (skip Stage 2 for these; much faster)
    - trials_with_protocols: use ec_changes with protocol path (~8.8k; skip most of 2–5)
    """
    if discovery_source == "trials_with_versions":
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT instance_id FROM trials_with_versions ORDER BY checked_at DESC"
        ).fetchall()
        ids = [row["instance_id"] for row in rows]
        if max_trials:
            ids = ids[:max_trials]
        logger.info(f"Using {len(ids)} NCT IDs from trials_with_versions (Stage 2 will be skipped for these)")
        return ids
    if discovery_source == "trials_with_protocols":
        ids = get_trials_with_protocols(conn, max_trials=max_trials)
        logger.info(f"Using {len(ids)} NCT IDs from trials with protocol PDFs (Stage 2–5 largely skipped)")
        return ids
    # default: discovered
    return _run_discovery_if_needed(conn, discovery_method, max_trials)


def _run_discovery_if_needed(conn, discovery_method: str, max_trials: Optional[int]) -> List[str]:
    """Ensure discovered_trials has IDs; run stage 1 (discovery) if empty."""
    cursor = conn.cursor()
    try:
        rows = cursor.execute(
            "SELECT instance_id FROM discovered_trials WHERE discovery_method = ? ORDER BY discovered_at",
            (discovery_method,),
        ).fetchall()
        existing = [row["instance_id"] for row in rows]
    except Exception:
        existing = []
    if existing:
        if max_trials:
            existing = existing[:max_trials]
        logger.info(f"Using {len(existing)} existing discovered NCT IDs")
        return existing
    logger.info("No discovered NCT IDs; running stage 1 (discovery)...")
    from recite.benchmark.discovery import discover_all_instance_ids
    count = 0
    for instance_id in discover_all_instance_ids(method=discovery_method, max_results=max_trials):
        cursor.execute(
            "INSERT OR IGNORE INTO discovered_trials (instance_id, discovery_method, discovered_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (instance_id, discovery_method),
        )
        count += 1
        if count % 1000 == 0:
            conn.commit()
            logger.info(f"  Discovered {count} NCT IDs...")
        if max_trials and count >= max_trials:
            break
    conn.commit()
    logger.info(f"  Discovered {count} NCT IDs total")
    rows = cursor.execute(
        "SELECT instance_id FROM discovered_trials WHERE discovery_method = ? ORDER BY discovered_at",
        (discovery_method,),
    ).fetchall()
    return [row["instance_id"] for row in rows]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    workers: int = typer.Option(None, "--workers", "-w", help="Number of workers (required unless --status)"),
    num_chunks: int = typer.Option(10, "--num-chunks", help="Total number of chunks"),
    chunks: str = typer.Option(
        "0,1,2,3,4,5,6,7,8,9",
        "--chunks",
        help="Comma-separated chunk indices to process (e.g. 0,1,2,3,4)",
    ),
    chunk_seed: int = typer.Option(42, "--chunk-seed", help="Random seed for shuffling (must match init-benchmark)"),
    worker_batch_size: int = typer.Option(10, "--worker-batch-size", help="Samples per batch per worker"),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db-path",
        path_type=Path,
        help="Database path (default: LOCAL_DB_DIR/recite.db, e.g. data/dev/recite.db)",
    ),
    discovery_method: str = typer.Option("bulk_xml", "--discovery-method", help="Discovery method"),
    discovery_source: str = typer.Option(
        "discovered",
        "--discovery-source",
        help="Where to get NCT IDs: 'discovered' (default, or run discovery), 'trials_with_versions' (DB; skip Stage 2), 'trials_with_protocols' (~8.8k with PDFs; skip most of 2–5)",
    ),
    use_expedited: bool = typer.Option(True, "--use-expedited/--no-expedited", help="Expedited mode"),
    max_retries: int = typer.Option(2, "--max-retries", help="Retries per batch (1 initial + this)"),
    retry_min_secs: float = typer.Option(1.0, "--retry-min-secs", help="Min delay between retries (exponential)"),
    requests_per_second: float = typer.Option(
        5.0,
        "--requests-per-second",
        help="Shared rate limit (ClinicalTrials.gov; adjust if you have different guidance)",
    ),
    max_trials: Optional[int] = typer.Option(None, "--max-trials", help="Cap total trials (for discovery)"),
    status: bool = typer.Option(False, "--status", help="Print stage completion status and exit"),
) -> None:
    """Run multi-worker init-benchmark. DB defaults to data/dev/recite.db (set LOCAL_DB_DIR or --db-path)."""
    if ctx.invoked_subcommand is not None:
        return
    if db_path is None:
        db_path = get_db_path()
    db_path = Path(db_path)
    logger.info(f"DB path: {db_path.resolve()}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_database(db_path, force=False)
    try:
        stage_status = _get_stage_status(conn)
        _print_stage_status(stage_status)
        if status:
            return

        if workers is None:
            raise typer.BadParameter("--workers is required when not using --status")
        chunk_indices = [int(x.strip()) for x in chunks.split(",")]
        for ci in chunk_indices:
            if ci < 0 or ci >= num_chunks:
                raise typer.BadParameter(f"Chunk index {ci} must be in [0, {num_chunks - 1}]")

        # Auto-select discovery source: if we already have trials_with_versions, use them
        if stage_status["stage2"]["total"] > 0 and discovery_source == "discovered":
            logger.info(
                f"Auto-switching to trials_with_versions ({stage_status['stage2']['total']:,} trials)"
            )
            discovery_source = "trials_with_versions"

        lock = threading.Lock()
        rate_limiter = SharedRateLimiter(requests_per_second=requests_per_second)
        adapter = ClinicalTrialsGovAdapter(requests_per_second=requests_per_second)

        if discovery_source not in ("discovered", "trials_with_versions", "trials_with_protocols"):
            raise typer.BadParameter(
                f"discovery_source must be one of: discovered, trials_with_versions, trials_with_protocols (got {discovery_source!r})"
            )
        all_instance_ids = _get_instance_ids_from_source(conn, discovery_source, discovery_method, max_trials)
        if not all_instance_ids:
            logger.warning("No NCT IDs available. Exiting.")
            raise typer.Exit(1)

        completed = _completed_instance_ids(conn)
        chunk_tuples = _ids_for_chunks(all_instance_ids, chunk_indices, num_chunks, chunk_seed)
        full_chunks_skipped = 0
        for ci, ids in chunk_tuples:
            if ids and all(n in completed for n in ids):
                full_chunks_skipped += 1
                logger.info(f"Chunk {ci} fully completed (skipped)")
        todo_ids = []
        for _ci, ids in chunk_tuples:
            for n in ids:
                if n not in completed:
                    todo_ids.append(n)
        logger.info(f"Chunks fully skipped: {full_chunks_skipped}; IDs to process: {len(todo_ids)}")
        # Rough per-stage ETA: Stage 2 ~4.8 trials/s (API), others vary; show order of magnitude
        if todo_ids:
            est_s2_min = len(todo_ids) / (4.8 * 60)
            logger.info(f"Rough ETA for Stage 2 (version check): ~{est_s2_min:.0f} min at ~5 req/s (later stages may be faster)")

        if not todo_ids:
            logger.info("Nothing to do. Running stage 4 and 7 only.")
            identify_amendments(max_trials, conn)
            create_recite_instances(max_trials, conn)
            return

        # Stage 2: skip if using trials_with_versions or trials_with_protocols
        if discovery_source != "discovered":
            logger.info("Stage 2: Skipped (using existing trials_with_versions or trials_with_protocols)")
        else:
            completed_s2 = _stage2_completed(conn)
            chunk_tuples_s2 = _ids_for_chunks(all_instance_ids, chunk_indices, num_chunks, chunk_seed)
            batches_s2 = _build_batches(chunk_tuples_s2, completed_s2, worker_batch_size)
            if batches_s2:
                _run_stage_with_workers(
                    "Stage 2", batches_s2, db_path, lock, adapter, rate_limiter,
                    max_retries, retry_min_secs, use_expedited, discovery_method, workers,
                )
            else:
                logger.info("Stage 2: Complete (no batches to process)")

        # Stage 3: skip if none remaining; if < 100 remaining run single-threaded
        completed_s3 = _stage3_completed(conn)
        chunk_tuples_s3 = _ids_for_chunks(all_instance_ids, chunk_indices, num_chunks, chunk_seed)
        batches_s3 = _build_batches(chunk_tuples_s3, completed_s3, worker_batch_size)
        if not batches_s3:
            logger.info("Stage 3: Complete (0 remaining), skipping")
        else:
            total_s3 = sum(len(b) for b in batches_s3)
            if total_s3 < 100:
                logger.info(f"Stage 3: {total_s3} remaining, running single-threaded")
                ids_s3 = [instance_id for batch in batches_s3 for instance_id in batch]
                download_versions(ids_s3, max_trials, conn, use_expedited=use_expedited)
            else:
                _run_stage_with_workers(
                    "Stage 3", batches_s3, db_path, lock, adapter, rate_limiter,
                    max_retries, retry_min_secs, use_expedited, discovery_method, workers,
                )

        # Stage 4: skip if EC changes already exist
        if stage_status["stage4"]["total"] > 0:
            logger.info(f"Stage 4: Skipped ({stage_status['stage4']['total']:,} EC changes already exist)")
        else:
            logger.info("Stage 4: Identifying amendments...")
            identify_amendments(max_trials, conn)

        # Stage 5 & 6: Only process trials that have EC changes (not all discovered trials)
        ec_change_instance_ids = [
            row["instance_id"] for row in conn.execute(
                "SELECT DISTINCT instance_id FROM ec_changes"
            ).fetchall()
        ]
        logger.info(f"Stage 5-6: Processing {len(ec_change_instance_ids):,} trials with EC changes (not all {len(all_instance_ids):,} discovered)")
        
        completed_s5 = _stage5_completed(conn)
        chunk_tuples_s5 = _ids_for_chunks(ec_change_instance_ids, chunk_indices, num_chunks, chunk_seed)
        batches_s5 = _build_batches(chunk_tuples_s5, completed_s5, worker_batch_size)
        _run_stage_with_workers(
            "Stage 5", batches_s5, db_path, lock, adapter, rate_limiter,
            max_retries, retry_min_secs, use_expedited, discovery_method, workers,
        )

        completed_s6 = _stage6_completed_instance_ids(conn)
        chunk_tuples_s6 = _ids_for_chunks(ec_change_instance_ids, chunk_indices, num_chunks, chunk_seed)
        batches_s6 = _build_batches(chunk_tuples_s6, completed_s6, worker_batch_size)
        _run_stage_with_workers(
            "Stage 6", batches_s6, db_path, lock, adapter, rate_limiter,
            max_retries, retry_min_secs, use_expedited, discovery_method, workers,
        )

        logger.info("Stage 7: Building RECITE instances...")
        create_recite_instances(max_trials, conn)

        logger.info("Multi-worker init-benchmark complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    app()
