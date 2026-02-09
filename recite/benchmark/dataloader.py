"""
Lazy streaming over benchmark data.

Supports two backends:
1. Parquet files (stream_parquet_splits) - legacy, requires export step
2. Direct DB (stream_from_db) - no export needed, reads from recite.db

Yields (split_name, row_dict) in batches to control memory.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from loguru import logger

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore


# Split → merge_source mapping for DB-based loading
# "benchmark"/"train" = cluster1 (the ~3k standard set)
# "final_test"/"test" = local with evidence (the ~8.8k eval set)
SPLIT_TO_MERGE_SOURCE = {
    "benchmark": "cluster1",
    "train": "cluster1",
    "val": "cluster1",  # legacy; in merged DB val may not exist separately
    "test": "cluster1",  # legacy
    "final_test": "local",
}


def count_samples_in_db(
    db_path: Path,
    split_name: str = "benchmark",
) -> int:
    """
    Count samples for a split in the DB (without loading all data).
    Uses merge_source mapping: benchmark/train = cluster1, final_test = local with evidence.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Check if merge_source column exists
        cols = [row[1] for row in conn.execute("PRAGMA table_info(recite)").fetchall()]
        has_merge_source = "merge_source" in cols
        if not has_merge_source:
            # No merge_source - count all rows
            row = conn.execute("SELECT COUNT(*) FROM recite").fetchone()
            return row[0] if row else 0

        merge_source = SPLIT_TO_MERGE_SOURCE.get(split_name, "cluster1")
        if split_name in ("final_test",):
            # final_test = local with evidence
            row = conn.execute(
                """SELECT COUNT(*) FROM recite
                   WHERE merge_source = ?
                     AND evidence IS NOT NULL
                     AND TRIM(COALESCE(evidence, '')) != ''""",
                (merge_source,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM recite WHERE merge_source = ?",
                (merge_source,),
            ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def validate_train_split(db_path: Path, expected_min: int = 2500, expected_max: int = 3500) -> Tuple[bool, int, str]:
    """
    Quick validation that train (benchmark) split has the expected ~3k cluster1 samples.
    Returns (ok, count, message).
    """
    n = count_samples_in_db(db_path, "benchmark")
    if expected_min <= n <= expected_max:
        return (True, n, f"Train split OK: {n} samples (cluster1)")
    elif n == 0:
        return (False, n, f"Train split EMPTY: no cluster1 samples found in {db_path}")
    elif n < expected_min:
        return (False, n, f"Train split too small: {n} samples (expected {expected_min}-{expected_max})")
    else:
        return (False, n, f"Train split too large: {n} samples (expected {expected_min}-{expected_max})")


def stream_from_db(
    db_path: Path,
    splits: List[str],
    batch_size: int = 1000,
    limit: Optional[int] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Stream samples directly from recite.db, yielding (split_name, row_dict).

    Split mapping:
      - "benchmark" / "train" / "val" / "test" → merge_source = 'cluster1'
      - "final_test" → merge_source = 'local' with evidence

    Args:
        db_path: Path to recite.db
        splits: List of split names to stream (e.g. ["benchmark"] or ["benchmark", "final_test"])
        batch_size: Number of rows per batch (for memory control)
        limit: If set, only return this many samples total (for smoke tests)

    Yields:
        (split_name, row_dict) for each sample
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Check if merge_source column exists
        cols = [row[1] for row in conn.execute("PRAGMA table_info(recite)").fetchall()]
        has_merge_source = "merge_source" in cols

        for split_name in splits:
            merge_source = SPLIT_TO_MERGE_SOURCE.get(split_name, "cluster1")
            is_final_test = split_name in ("final_test",)

            # Build query (join on both nct_id and merge_source to avoid duplicates from merged DBs)
            base_query = """
                SELECT
                    r.id,
                    r.nct_id,
                    r.version_from,
                    r.version_to,
                    r.preamended_text,
                    r.evidence,
                    r.amended_text,
                    r.quality_score,
                    r.evidence_extraction_level,
                    r.evidence_extraction_score,
                    tm.year,
                    tm.conditions,
                    tm.keywords,
                    tm.phases,
                    tm.locations,
                    tm.study_type,
                    tm.enrollment,
                    tm.start_date,
                    tm.overall_status
                FROM recite r
                LEFT JOIN trial_metadata tm ON r.nct_id = tm.nct_id AND r.merge_source = tm.merge_source
            """
            if has_merge_source:
                if is_final_test:
                    query = base_query + """
                        WHERE r.merge_source = ?
                          AND r.evidence IS NOT NULL
                          AND TRIM(COALESCE(r.evidence, '')) != ''
                        ORDER BY r.id
                    """
                    params: tuple = (merge_source,)
                else:
                    query = base_query + " WHERE r.merge_source = ? ORDER BY r.id"
                    params = (merge_source,)
            else:
                # No merge_source column - return all (legacy single-source DB)
                query = base_query + " ORDER BY r.id"
                params = ()

            # Add LIMIT for smoke tests / sampling
            if limit is not None:
                query += f" LIMIT {int(limit)}"

            logger.info("Streaming {} from DB (merge_source={}, limit={})", split_name, merge_source if has_merge_source else "N/A", limit)
            cursor = conn.execute(query, params)

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    row_dict = dict(row)
                    # Parse JSON columns
                    for col in ("conditions", "keywords", "phases", "locations"):
                        val = row_dict.get(col)
                        if val and isinstance(val, str):
                            try:
                                row_dict[col] = json.loads(val)
                            except json.JSONDecodeError:
                                row_dict[col] = []
                    yield (split_name, row_dict)
    finally:
        conn.close()


def stream_parquet_splits(
    parquet_paths: Dict[str, Path],
    batch_size: Optional[int] = None,
    splits_order: Optional[List[str]] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Yield samples lazily from parquet files as (split_name, row_dict).

    Uses PyArrow to read in batches (by row group or fixed row count) to control memory.
    If batch_size is None, yields row-by-row (still using PyArrow, no full load).

    Args:
        parquet_paths: Dict mapping split name (e.g. 'train', 'val', 'test') to parquet file path.
        batch_size: If set, yield up to this many rows per batch (sliced from table).
                    If None, yield one row at a time.
        splits_order: Order of splits to iterate (default: sorted(parquet_paths.keys())).

    Yields:
        (split_name, row_dict) where row_dict has column names as keys.
    """
    if pq is None:
        raise ImportError("pyarrow is required for stream_parquet_splits. Install with: pip install pyarrow")

    order = splits_order if splits_order is not None else sorted(parquet_paths.keys())
    for split_name in order:
        path = parquet_paths.get(split_name)
        if path is None or not Path(path).exists():
            logger.warning(f"Parquet path missing or not found for split {split_name}: {path}")
            continue
        path = Path(path)
        try:
            table = pq.read_table(path)
        except Exception as e:
            logger.warning(f"Failed to read parquet {path}: {e}")
            continue
        n_rows = table.num_rows
        if n_rows == 0:
            continue
        col_names = table.column_names
        if batch_size is None:
            for i in range(n_rows):
                row_dict = {name: _pyarrow_scalar_to_python(table.column(name)[i]) for name in col_names}
                yield (split_name, row_dict)
        else:
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                batch = table.slice(start, end - start)
                for i in range(batch.num_rows):
                    row_dict = {name: _pyarrow_scalar_to_python(batch.column(name)[i]) for name in col_names}
                    yield (split_name, row_dict)


def _pyarrow_scalar_to_python(val: Any) -> Any:
    """Convert PyArrow scalar to Python native type for JSON/serialization."""
    if val is None:
        return None
    if hasattr(val, "as_py"):
        return val.as_py()
    return val
