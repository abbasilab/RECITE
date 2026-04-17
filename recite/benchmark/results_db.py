"""Benchmark results SQLite storage with content-addressed configs."""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Identity fields used for config_fingerprint (canonical order).
# prompts_snapshot is intentionally excluded so config_id is stable across prompt file
# content changes and we reuse existing result tables (avoid re-running same experiment).
IDENTITY_KEYS = [
    "model_id",
    "top_k",
    "no_rag",
    "parquet_paths",
    "prompts_file",
    "evaluator_type",
    "evaluator_config",
]

# Columns for per-config results table (exact inputs/outputs + metrics)
RESULT_COLUMNS = [
    "id",
    "split_name",
    "nct_id",
    "version_from",
    "version_to",
    "preamended_text",
    "evidence",
    "amended_text",
    "prediction",
    "quality_score",
    "year",
    "study_type",
    "predicted_at",
    "binary_correct",
    "edit_distance",
    "normalized_edit_distance",
    "edit_similarity",
    "bleu",
    "rouge_l",
    "llm_judge_binary",
    "llm_judge_score",
    "llm_judge_normalized",
    "llm_judge_raw_response",
]

FINGERPRINT_LEN = 24  # truncate hash for id/table name


def _canonical_json(obj: Any) -> str:
    """Serialize to JSON with sorted keys for stable fingerprint."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _normalize_path_string(s: str) -> str:
    """Resolve to absolute path and lowercase so path case does not change fingerprint.

    Same logical path with different case (e.g. .../Projects/... vs .../projects/...)
    yields the same config_id so existing results are reused and the API is not re-called.
    """
    if not s or not isinstance(s, str):
        return s
    try:
        resolved = str(Path(s).resolve())
        return resolved.lower()
    except Exception:
        return s


def compute_config_fingerprint(metadata: Dict[str, Any]) -> str:
    """
    Compute content-addressed fingerprint from config identity fields.
    Used for matching and as basis for config id (table name prefix).
    Paths (parquet_paths, prompts_file) are normalized to absolute so that
    the same logical experiment yields the same config_id across config files
    (e.g. benchmarks.yaml vs benchmarks_large.yaml) and relative vs absolute paths.
    """
    canonical: Dict[str, Any] = {}
    for key in IDENTITY_KEYS:
        val = metadata.get(key)
        if val is None:
            canonical[key] = None
        elif key == "parquet_paths" and isinstance(val, dict):
            # Normalize path strings so relative/absolute and resolution don't change id
            normalized = {k: _normalize_path_string(v) for k, v in val.items()}
            canonical[key] = json.loads(_canonical_json(normalized))
        elif key == "prompts_file" and isinstance(val, str):
            canonical[key] = _normalize_path_string(val)
        elif isinstance(val, dict):
            canonical[key] = json.loads(_canonical_json(val))
        else:
            canonical[key] = val
    payload = _canonical_json(canonical)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return h[:FINGERPRINT_LEN]


def sanitize_table_name(config_id: str) -> str:
    """Sanitize config_id for use as SQL table name (alphanumeric + underscore)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", config_id)


def init_results_db(conn: sqlite3.Connection) -> None:
    """Create configs table if not exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS configs (
            id TEXT PRIMARY KEY,
            config_fingerprint TEXT UNIQUE NOT NULL,
            model_id TEXT,
            top_k INTEGER,
            no_rag INTEGER,
            parquet_paths TEXT,
            prompts_file TEXT,
            prompts_snapshot TEXT,
            evaluator_type TEXT,
            evaluator_config TEXT,
            two_step INTEGER,
            batch_size INTEGER,
            num_samples INTEGER,
            wait_for_revive_seconds INTEGER,
            rag_config TEXT,
            model TEXT,
            config_path TEXT,
            run_started_at TEXT,
            config_json TEXT,
            prompt_version TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_configs_fingerprint ON configs(config_fingerprint)"
    )
    conn.commit()
    # Migration: add prompt_version to existing DBs
    try:
        conn.execute("ALTER TABLE configs ADD COLUMN prompt_version TEXT")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def _configs_has_merge_source(conn: sqlite3.Connection) -> bool:
    """True if configs table has merge_source column (e.g. merged benchmark_results DB)."""
    try:
        cur = conn.execute("PRAGMA table_info(configs)")
        return any(row[1] == "merge_source" for row in cur.fetchall())
    except sqlite3.OperationalError:
        return False


def find_config(conn: sqlite3.Connection, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find existing config by config_fingerprint (computed from spec).
    Returns config row as dict (including id) if found, else None.
    """
    fingerprint = compute_config_fingerprint(spec)
    row = conn.execute(
        "SELECT * FROM configs WHERE config_fingerprint = ?", (fingerprint,)
    ).fetchone()
    if row is not None:
        return dict(row)
    # Fallback: match by identity (same model/parquet/prompts/evaluator) so we reuse
    # result tables when fingerprint changed (e.g. prompts_snapshot was added/removed).
    row = _find_config_by_identity(conn, spec)
    return dict(row) if row else None


def _identity_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build dict of identity fields only (for matching); excludes prompts_snapshot."""
    return {k: spec.get(k) for k in IDENTITY_KEYS}


def _parquet_paths_for_identity_compare(parquet_paths: Any) -> Optional[str]:
    """Normalize parquet_paths (dict of path strings) for case-insensitive identity comparison."""
    if parquet_paths is None:
        return None
    if isinstance(parquet_paths, dict):
        normalized = {k: _normalize_path_string(v) for k, v in parquet_paths.items()}
        return _canonical_json(normalized)
    return _canonical_json(parquet_paths)


def _find_config_by_identity(conn: sqlite3.Connection, spec: Dict[str, Any]) -> Optional[sqlite3.Row]:
    """Find config row with same identity (model_id, top_k, no_rag, parquet_paths, prompts_file, evaluator_*).
    Paths are compared case-insensitively so existing results from .../Projects/... match .../projects/....
    """
    want = _identity_from_spec(spec)
    want_parquet = _parquet_paths_for_identity_compare(want.get("parquet_paths"))
    want_prompts = _normalize_path_string(want.get("prompts_file") or "") or want.get("prompts_file")
    want_eval = _canonical_json(want.get("evaluator_config")) if want.get("evaluator_config") is not None else None
    for row in conn.execute(
        "SELECT * FROM configs WHERE model_id = ? AND top_k = ? AND no_rag = ? AND evaluator_type = ?",
        (want.get("model_id"), want.get("top_k"), 1 if want.get("no_rag") else 0, want.get("evaluator_type")),
    ).fetchall():
        row_parquet = row["parquet_paths"]
        if row_parquet is not None and isinstance(row_parquet, str):
            try:
                row_parquet = _parquet_paths_for_identity_compare(json.loads(row_parquet))
            except Exception:
                row_parquet = None
        else:
            row_parquet = None
        row_prompts = _normalize_path_string(row["prompts_file"] or "") or row["prompts_file"]
        row_eval = row["evaluator_config"]
        if row_eval is not None and isinstance(row_eval, str):
            try:
                row_eval = _canonical_json(json.loads(row_eval))
            except Exception:
                row_eval = row_eval
        else:
            row_eval = None
        if row_prompts != want_prompts:
            continue
        if row_parquet != want_parquet:
            continue
        if row_eval != want_eval:
            continue
        return row
    return None


def ensure_config(conn: sqlite3.Connection, config_metadata: Dict[str, Any]) -> str:
    """
    Ensure config row exists. Compute fingerprint; if no row with that fingerprint,
    try finding by identity (so we reuse tables when only prompts_snapshot differed).
    If found, return existing id; else insert and set id = fingerprint; return id.
    """
    fingerprint = compute_config_fingerprint(config_metadata)
    row = conn.execute(
        "SELECT id FROM configs WHERE config_fingerprint = ?", (fingerprint,)
    ).fetchone()
    if row is not None:
        return row[0]
    # Reuse existing config with same identity (avoids rework when fingerprint formula changed)
    existing = _find_config_by_identity(conn, config_metadata)
    if existing is not None:
        return existing["id"]

    config_id = fingerprint  # use full truncated hash as id
    now = datetime.now(timezone.utc).isoformat()
    include_merge_source = _configs_has_merge_source(conn)
    merge_source_value = config_metadata.get("merge_source") or "local"

    columns = [
        "id", "config_fingerprint", "model_id", "top_k", "no_rag", "parquet_paths",
        "prompts_file", "prompts_snapshot", "evaluator_type", "evaluator_config",
        "two_step", "batch_size", "num_samples", "wait_for_revive_seconds",
        "rag_config", "model", "config_path", "run_started_at", "config_json", "prompt_version", "created_at",
    ]
    values: List[Any] = [
        config_id,
        fingerprint,
        config_metadata.get("model_id"),
        config_metadata.get("top_k"),
        1 if config_metadata.get("no_rag") else 0,
        json.dumps(config_metadata.get("parquet_paths")) if config_metadata.get("parquet_paths") is not None else None,
        config_metadata.get("prompts_file"),
        json.dumps(config_metadata.get("prompts_snapshot")) if isinstance(config_metadata.get("prompts_snapshot"), dict) else config_metadata.get("prompts_snapshot"),
        config_metadata.get("evaluator_type"),
        json.dumps(config_metadata.get("evaluator_config")) if config_metadata.get("evaluator_config") is not None else None,
        1 if config_metadata.get("two_step") else 0,
        config_metadata.get("batch_size"),
        config_metadata.get("num_samples"),
        config_metadata.get("wait_for_revive_seconds"),
        json.dumps(config_metadata.get("rag_config")) if config_metadata.get("rag_config") is not None else None,
        json.dumps(config_metadata.get("model")) if config_metadata.get("model") is not None else None,
        config_metadata.get("config_path"),
        config_metadata.get("run_started_at"),
        json.dumps(config_metadata.get("config_json")) if config_metadata.get("config_json") is not None else None,
        config_metadata.get("prompt_version"),
        now,
    ]
    if include_merge_source:
        columns.append("merge_source")
        values.append(merge_source_value)

    placeholders = ", ".join(["?"] * len(values))
    conn.execute(
        f"INSERT INTO configs ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    logger.debug(f"Inserted config id={config_id}")
    return config_id


def _results_table_name(config_id: str) -> str:
    """Per-config results table name (sanitized)."""
    return "results_" + sanitize_table_name(config_id)


def ensure_results_table(conn: sqlite3.Connection, config_id: str, result_row: Optional[Dict[str, Any]] = None) -> None:
    """
    Create per-config results table if not exists.
    Schema matches RESULT_COLUMNS; result_row can be used to infer types (all TEXT/REAL/INTEGER).
    """
    name = _results_table_name(config_id)
    # Use a fixed schema from RESULT_COLUMNS
    columns_sql = [
        "id INTEGER NOT NULL",
        "split_name TEXT NOT NULL",
        "nct_id TEXT",
        "version_from INTEGER",
        "version_to INTEGER",
        "preamended_text TEXT",
        "evidence TEXT",
        "amended_text TEXT",
        "prediction TEXT",
        "quality_score REAL",
        "year INTEGER",
        "study_type TEXT",
        "predicted_at TEXT",
        "binary_correct REAL",
        "edit_distance REAL",
        "normalized_edit_distance REAL",
        "edit_similarity REAL",
        "bleu REAL",
        "rouge_l REAL",
        "llm_judge_binary REAL",
        "llm_judge_score REAL",
        "llm_judge_normalized REAL",
        "llm_judge_raw_response TEXT",
        "PRIMARY KEY (id, split_name)",
    ]
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(columns_sql)})"
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{sanitize_table_name(config_id)}_id ON {name}(id)")
    conn.commit()


def has_sample(
    conn: sqlite3.Connection,
    config_id: str,
    sample_id: int,
    split_name: Optional[str] = None,
) -> bool:
    """Return True if a result row exists for (config_id, sample_id), optionally for split_name.
    When split_name is provided, matches (id, split_name) so skips are per-split."""
    name = _results_table_name(config_id)
    try:
        if split_name is not None:
            row = conn.execute(
                f"SELECT 1 FROM {name} WHERE id = ? AND split_name = ? LIMIT 1",
                (sample_id, split_name),
            ).fetchone()
        else:
            row = conn.execute(f"SELECT 1 FROM {name} WHERE id = ? LIMIT 1", (sample_id,)).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False


def count_samples(conn: sqlite3.Connection, config_id: str, split_name: str) -> int:
    """Return number of result rows for (config_id, split_name). Used for num_samples limit."""
    name = _results_table_name(config_id)
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {name} WHERE split_name = ?", (split_name,)
        ).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def insert_result(conn: sqlite3.Connection, config_id: str, result_row: Dict[str, Any]) -> None:
    """
    Insert one result row into the config's results table.
    result_row must include id, split_name, and other RESULT_COLUMNS as available.
    """
    ensure_results_table(conn, config_id, result_row)
    name = _results_table_name(config_id)
    # Map result_row keys to columns; use None for missing
    placeholders = []
    values = []
    for col in RESULT_COLUMNS:
        placeholders.append("?")
        val = result_row.get(col)
        if val is None and col in ("quality_score", "year", "study_type", "predicted_at"):
            values.append(None)
        elif col == "split_name" and val is None:
            values.append("")  # required; default to empty if not provided
        else:
            values.append(val)
    conn.execute(
        f"INSERT OR REPLACE INTO {name} ({', '.join(RESULT_COLUMNS)}) VALUES ({', '.join(placeholders)})",
        values,
    )
    conn.commit()


def get_existing_result(
    conn: sqlite3.Connection, config_id: str, sample_id: int, split_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Return existing result row for (config_id, sample_id), optionally filtered by split_name.
    Returns None if not found.
    """
    name = _results_table_name(config_id)
    try:
        if split_name is not None:
            row = conn.execute(
                f"SELECT * FROM {name} WHERE id = ? AND split_name = ?",
                (sample_id, split_name),
            ).fetchone()
        else:
            row = conn.execute(f"SELECT * FROM {name} WHERE id = ? LIMIT 1", (sample_id,)).fetchone()
        if row is None:
            return None
        return dict(row)
    except sqlite3.OperationalError:
        return None


def get_predictions_without_judge(
    conn: sqlite3.Connection,
    config_id: str,
) -> List[Dict[str, Any]]:
    """
    Return result rows that have a prediction but no LLM judge scores yet.
    Used for phase=judge to run batched judge on stored predictions.
    """
    name = _results_table_name(config_id)
    try:
        cursor = conn.execute(
            f"""SELECT * FROM {name}
               WHERE prediction IS NOT NULL AND llm_judge_score IS NULL
               ORDER BY id, split_name"""
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []


def update_judge_scores(
    conn: sqlite3.Connection,
    config_id: str,
    row_id: int,
    split_name: str,
    llm_judge_binary: float,
    llm_judge_score: float,
    llm_judge_normalized: float,
    llm_judge_raw_response: Optional[str],
) -> None:
    """Update judge score columns for an existing result row."""
    name = _results_table_name(config_id)
    conn.execute(
        f"""UPDATE {name}
            SET llm_judge_binary = ?, llm_judge_score = ?, llm_judge_normalized = ?, llm_judge_raw_response = ?
            WHERE id = ? AND split_name = ?""",
        (llm_judge_binary, llm_judge_score, llm_judge_normalized, llm_judge_raw_response, row_id, split_name),
    )
    conn.commit()


def clear_results_for_config(conn: sqlite3.Connection, config_id: str) -> int:
    """
    Delete all rows from the results table for this config_id.
    Used to reset botched runs so the next run re-processes those samples.
    Returns number of rows deleted. No-op if table does not exist.
    """
    name = _results_table_name(config_id)
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        conn.execute(f"DELETE FROM {name}")
        conn.commit()
        return n
    except sqlite3.OperationalError:
        conn.rollback()
        return 0


def get_connection(results_db_path: Path) -> sqlite3.Connection:
    """Open benchmark results DB and ensure configs table exists."""
    results_db_path = Path(results_db_path)
    results_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(results_db_path))
    conn.row_factory = sqlite3.Row
    init_results_db(conn)
    return conn


# Metrics to aggregate for benchmark summary (mean per config × split)
BENCHMARK_METRIC_COLUMNS = [
    "binary_correct",
    "edit_similarity",
    "bleu",
    "rouge_l",
    "llm_judge_binary",
    "llm_judge_score",
    "llm_judge_normalized",
]


def get_benchmark_summary_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Aggregate metrics from all configs and their results tables.
    Returns one row per (model_id, no_rag, top_k, split_name) with n and mean of each metric.
    Use this to build the final "table of benchmark vals" (models × RAG × metrics).
    """
    configs = conn.execute(
        "SELECT id, model_id, top_k, no_rag, evaluator_type FROM configs"
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in configs:
        config_id = row["id"]
        model_id = row["model_id"] or ""
        top_k = row["top_k"]
        no_rag = bool(row["no_rag"]) if row["no_rag"] is not None else False
        evaluator_type = row["evaluator_type"] or ""
        name = _results_table_name(config_id)
        try:
            agg_cols = ", ".join(
                f"AVG({c}) AS {c}_mean" for c in BENCHMARK_METRIC_COLUMNS
            )
            rows_agg = conn.execute(
                f"SELECT split_name, COUNT(*) AS n, {agg_cols} FROM {name} GROUP BY split_name"
            ).fetchall()
        except sqlite3.OperationalError:
            continue
        for r in rows_agg:
            out.append({
                "model_id": model_id,
                "no_rag": no_rag,
                "top_k": top_k,
                "evaluator_type": evaluator_type,
                "split_name": r["split_name"],
                "n": r["n"],
                **{f"{c}_mean": r[f"{c}_mean"] for c in BENCHMARK_METRIC_COLUMNS},
            })
    return out


def _identity_key_from_row(row: Dict[str, Any]) -> tuple:
    """Build a hashable identity key from a config row (for grouping duplicates)."""
    parquet = row.get("parquet_paths")
    if isinstance(parquet, str):
        try:
            parquet = json.loads(parquet)
        except Exception:
            parquet = None
    eval_cfg = row.get("evaluator_config")
    if isinstance(eval_cfg, str):
        try:
            eval_cfg = json.loads(eval_cfg)
        except Exception:
            eval_cfg = None
    return (
        row.get("model_id"),
        row.get("top_k"),
        1 if row.get("no_rag") else 0,
        row.get("evaluator_type"),
        row.get("prompts_file"),
        _canonical_json(parquet) if parquet is not None else None,
        _canonical_json(eval_cfg) if eval_cfg is not None else None,
    )


def merge_duplicate_configs(conn: sqlite3.Connection) -> int:
    """
    Merge configs that share the same identity (model_id, top_k, no_rag, parquet_paths,
    prompts_file, evaluator_type, evaluator_config). For each group of duplicates, keep
    one config (earliest by created_at), copy all result rows from duplicate tables into
    the keeper's table (INSERT OR REPLACE), drop duplicate results tables, and delete
    duplicate config rows.

    Returns:
        Number of duplicate config rows removed.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM configs ORDER BY created_at").fetchall()
    configs = [dict(r) for r in rows]
    if not configs:
        return 0

    # Group by identity key
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for c in configs:
        key = _identity_key_from_row(c)
        groups.setdefault(key, []).append(c)

    removed = 0
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        # Keeper = first (earliest created_at; we ordered by created_at)
        keeper = group[0]
        keeper_id = keeper["id"]
        keeper_table = _results_table_name(keeper_id)
        ensure_results_table(conn, keeper_id)
        duplicates = group[1:]

        for dup in duplicates:
            dup_id = dup["id"]
            dup_table = _results_table_name(dup_id)
            try:
                # Check if duplicate's results table exists
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (dup_table,),
                )
                if cur.fetchone() is None:
                    conn.execute("DELETE FROM configs WHERE id = ?", (dup_id,))
                    removed += 1
                    continue
                # Copy rows: INSERT OR REPLACE into keeper so keeper wins on (id, split_name) conflict
                conn.execute(
                    f"INSERT OR REPLACE INTO {keeper_table} SELECT * FROM {dup_table}"
                )
                conn.execute(f"DROP TABLE IF EXISTS {dup_table}")
                conn.execute("DELETE FROM configs WHERE id = ?", (dup_id,))
                removed += 1
            except sqlite3.OperationalError as e:
                logger.warning("merge_duplicate_configs: skip %s -> %s: %s", dup_id, keeper_id, e)
    conn.commit()
    if removed:
        logger.info("merge_duplicate_configs: removed {} duplicate config(s)", removed)
    return removed


def migrate_from_benchmark_predictions(
    predictions_dir: Path,
    results_db_path: Path,
    project_root: Optional[Path] = None,
) -> int:
    """
    Import existing run dirs from benchmark_predictions into the results DB.
    For each model_id/run_*/ dir: load run_config.yaml, derive config_id (fingerprint),
    ensure config row, load results_{split}.jsonl, insert each result row.
    Skips runs that lack run_config.yaml or results_*.jsonl.

    Returns:
        Total number of result rows inserted.
    """
    import yaml

    predictions_dir = Path(predictions_dir)
    if not predictions_dir.is_dir():
        logger.warning(f"Predictions dir not found: {predictions_dir}")
        return 0

    conn = get_connection(results_db_path)
    total_inserted = 0

    try:
        for model_dir in sorted(predictions_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_id = model_dir.name
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                    continue
                config_path = run_dir / "run_config.yaml"
                if not config_path.exists():
                    logger.debug(f"Skipping {run_dir}: no run_config.yaml")
                    continue
                try:
                    with open(config_path) as f:
                        run_config = yaml.safe_load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {config_path}: {e}")
                    continue
                if not run_config:
                    continue

                # Build config_metadata for fingerprint / ensure_config
                top_k = run_config.get("top_k")
                no_rag = run_config.get("no_rag", False)
                parquet_paths = run_config.get("parquet_paths")
                if isinstance(parquet_paths, dict):
                    parquet_paths = {k: str(v) for k, v in parquet_paths.items()}
                config_metadata = {
                    "model_id": model_id,
                    "top_k": int(top_k) if top_k is not None else 0,
                    "no_rag": bool(no_rag),
                    "parquet_paths": parquet_paths,
                    "prompts_file": run_config.get("prompts_file"),
                    "prompts_snapshot": run_config.get("prompts_snapshot"),
                    "evaluator_type": run_config.get("evaluator_type", "default"),
                    "evaluator_config": run_config.get("evaluator_config"),
                    "two_step": run_config.get("two_step", False),
                    "batch_size": run_config.get("batch_size"),
                    "num_samples": run_config.get("num_samples"),
                    "wait_for_revive_seconds": run_config.get("wait_for_revive_seconds"),
                    "rag_config": run_config.get("rag_config"),
                    "model": run_config.get("model"),
                    "config_path": run_config.get("config_path"),
                    "run_started_at": run_config.get("run_started_at"),
                }
                config_id = ensure_config(conn, config_metadata)

                # Load results_*.jsonl
                for results_file in sorted(run_dir.glob("results_*.jsonl")):
                    # results_train.jsonl -> train
                    stem = results_file.stem
                    if stem.startswith("results_"):
                        split_name = stem[8:]
                    else:
                        split_name = stem
                    with open(results_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            row["split_name"] = row.get("split_name", split_name)
                            row["predicted_at"] = row.get("predicted_at") or row.get("timestamp")
                            if has_sample(conn, config_id, int(row.get("id", 0))):
                                continue
                            insert_result(conn, config_id, row)
                            total_inserted += 1
                if total_inserted > 0 and total_inserted % 500 == 0:
                    logger.info(f"Migrated {total_inserted} result rows so far...")
    finally:
        conn.close()

    logger.info(f"Migration complete: {total_inserted} result rows inserted from {predictions_dir}")
    return total_inserted
