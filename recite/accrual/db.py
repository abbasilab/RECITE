"""
Accrual database: paper_answers, paper_trial_gains, summary_runs.
Thin wrapper for trial enrollment from recite.db (read-only).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def _coerce_for_sqlite(val: Any, kind: str = "text") -> Any:
    """Coerce value for SQLite binding; list/dict from LLM JSON must become scalar or None."""
    if val is None:
        return None
    if isinstance(val, list):
        if not val:
            return None
        val = val[0]
    if isinstance(val, dict):
        if kind == "text":
            return json.dumps(val)[:10000]
        return None
    if kind == "text" and val is not None and not isinstance(val, str):
        return str(val)
    if kind == "int" and val is not None:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None
    if kind == "float" and val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    return val

from recite.utils.path_loader import get_local_db_dir, get_project_root, resolve_path

# Default accrual DB path (can be overridden via ACCRUAL_DB_PATH env or config)
DEFAULT_ACCRUAL_DB_PATH = Path("accrual.db")


def get_accrual_db_path(config_path: Optional[Path] = None) -> Path:
    """Resolve accrual.db path: env, config, or default under this machine's DB dir (see LOCAL_DB_DIR in .env)."""
    env_path = os.getenv("ACCRUAL_DB_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else get_project_root() / p
    return get_local_db_dir() / DEFAULT_ACCRUAL_DB_PATH


def _get_conn(accrual_db_path: Path) -> sqlite3.Connection:
    accrual_db_path = Path(accrual_db_path)
    accrual_db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(accrual_db_path))


# --- Schema ---

PAPER_ANSWERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_source TEXT NOT NULL,
    paper_source_id TEXT NOT NULL,
    raw_response_directives TEXT,
    raw_response_impact TEXT,
    raw_response_population TEXT,
    raw_response_eligibility TEXT,
    directives_answer_location TEXT NOT NULL,
    directives_has_answer INTEGER NOT NULL,
    directives_exact_text TEXT,
    directives_count INTEGER,
    impact_answer_location TEXT NOT NULL,
    impact_has_answer INTEGER NOT NULL,
    impact_percent REAL,
    impact_absolute REAL,
    impact_unit TEXT,
    impact_qualitative TEXT,
    impact_evidence TEXT,
    population_size REAL,
    population_unit TEXT,
    eligibility_size REAL,
    eligible_count INTEGER,
    eligibility_unit TEXT,
    other_numeric_value REAL,
    other_numeric_descriptor TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(paper_source, paper_source_id)
);
"""

PAPER_TRIAL_GAINS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trial_gains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_source TEXT NOT NULL,
    paper_source_id TEXT NOT NULL,
    trial_instance_id TEXT NOT NULL,
    match_score INTEGER,
    applicability_score INTEGER,
    enrollment INTEGER,
    paper_pct_gain REAL,
    scalar_gain REAL,
    amended_ec_text TEXT,
    change_directive_quote TEXT,
    change_rationale_quote TEXT,
    evidenced_summary TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(paper_source, paper_source_id, trial_instance_id)
);
"""

SUMMARY_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS summary_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_path TEXT,
    llm_endpoint TEXT,
    papers_processed INTEGER,
    top_papers_count INTEGER,
    created_at TEXT NOT NULL
);
"""


def _ensure_accrual_columns(conn: sqlite3.Connection) -> None:
    """Add missing columns to existing tables (migration for existing DBs)."""
    for table, column in [
        ("paper_answers", "impact_evidence"),
        ("paper_trial_gains", "change_directive_quote"),
        ("paper_trial_gains", "change_rationale_quote"),
        ("paper_trial_gains", "evidenced_summary"),
    ]:
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            names = [r[1] for r in rows]
            if column not in names:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
                logger.info(f"Added column {table}.{column}")
        except sqlite3.OperationalError as e:
            logger.debug(f"ensure column {table}.{column}: {e}")


def init_accrual_db(accrual_db_path: Optional[Path] = None) -> None:
    """Create accrual.db and tables if they do not exist. Migrates existing DBs with new columns."""
    path = accrual_db_path or get_accrual_db_path()
    path = path if path.is_absolute() else get_project_root() / path
    conn = _get_conn(path)
    try:
        conn.executescript(PAPER_ANSWERS_SCHEMA)
        conn.executescript(PAPER_TRIAL_GAINS_SCHEMA)
        conn.executescript(SUMMARY_RUNS_SCHEMA)
        _ensure_accrual_columns(conn)
        conn.commit()
        logger.info(f"Initialized accrual DB at {path}")
    finally:
        conn.close()


def insert_paper_answer(
    accrual_db_path: Path,
    paper_source: str,
    paper_source_id: str,
    raw_response_directives: Optional[str],
    raw_response_impact: Optional[str],
    raw_response_population: Optional[str],
    raw_response_eligibility: Optional[str],
    directives_answer_location: str,
    directives_has_answer: int,
    directives_exact_text: Optional[str],
    directives_count: Optional[int],
    impact_answer_location: str,
    impact_has_answer: int,
    impact_percent: Optional[float],
    impact_absolute: Optional[float],
    impact_unit: Optional[str],
    impact_qualitative: Optional[str],
    impact_evidence: Optional[str] = None,
    population_size: Optional[float] = None,
    population_unit: Optional[str] = None,
    eligibility_size: Optional[float] = None,
    eligible_count: Optional[int] = None,
    eligibility_unit: Optional[str] = None,
    other_numeric_value: Optional[float] = None,
    other_numeric_descriptor: Optional[str] = None,
) -> None:
    """Insert or replace one row in paper_answers."""
    now = datetime.now(timezone.utc).isoformat()
    # Coerce any list/dict from LLM so SQLite binding does not fail (e.g. parameter 9 = directives_exact_text)
    directives_exact_text_s = _coerce_for_sqlite(directives_exact_text, "text")
    directives_count_i = _coerce_for_sqlite(directives_count, "int")
    impact_percent_f = _coerce_for_sqlite(impact_percent, "float")
    impact_absolute_f = _coerce_for_sqlite(impact_absolute, "float")
    impact_unit_s = _coerce_for_sqlite(impact_unit, "text")
    impact_qualitative_s = _coerce_for_sqlite(impact_qualitative, "text")
    impact_evidence_s = _coerce_for_sqlite(impact_evidence, "text")
    population_size_f = _coerce_for_sqlite(population_size, "float")
    population_unit_s = _coerce_for_sqlite(population_unit, "text")
    eligibility_size_f = _coerce_for_sqlite(eligibility_size, "float")
    eligible_count_i = _coerce_for_sqlite(eligible_count, "int")
    eligibility_unit_s = _coerce_for_sqlite(eligibility_unit, "text")
    other_numeric_value_f = _coerce_for_sqlite(other_numeric_value, "float")
    other_numeric_descriptor_s = _coerce_for_sqlite(other_numeric_descriptor, "text")
    conn = _get_conn(accrual_db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_answers (
                paper_source, paper_source_id,
                raw_response_directives, raw_response_impact,
                raw_response_population, raw_response_eligibility,
                directives_answer_location, directives_has_answer,
                directives_exact_text, directives_count,
                impact_answer_location, impact_has_answer,
                impact_percent, impact_absolute, impact_unit, impact_qualitative, impact_evidence,
                population_size, population_unit,
                eligibility_size, eligible_count, eligibility_unit,
                other_numeric_value, other_numeric_descriptor,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_source,
                paper_source_id,
                raw_response_directives,
                raw_response_impact,
                raw_response_population,
                raw_response_eligibility,
                directives_answer_location,
                directives_has_answer,
                directives_exact_text_s,
                directives_count_i,
                impact_answer_location,
                impact_has_answer,
                impact_percent_f,
                impact_absolute_f,
                impact_unit_s,
                impact_qualitative_s,
                impact_evidence_s,
                population_size_f,
                population_unit_s,
                eligibility_size_f,
                eligible_count_i,
                eligibility_unit_s,
                other_numeric_value_f,
                other_numeric_descriptor_s,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_paper_answer_impact(
    accrual_db_path: Path,
    paper_source: str,
    paper_source_id: str,
    raw_response_impact: Optional[str],
    impact_answer_location: str,
    impact_has_answer: int,
    impact_percent: Optional[float],
    impact_absolute: Optional[float],
    impact_unit: Optional[str],
    impact_qualitative: Optional[str],
    impact_evidence: Optional[str],
) -> None:
    """Update impact-related columns for one paper_answers row."""
    impact_percent_f = _coerce_for_sqlite(impact_percent, "float")
    impact_absolute_f = _coerce_for_sqlite(impact_absolute, "float")
    impact_unit_s = _coerce_for_sqlite(impact_unit, "text")
    impact_qualitative_s = _coerce_for_sqlite(impact_qualitative, "text")
    impact_evidence_s = _coerce_for_sqlite(impact_evidence, "text")
    conn = _get_conn(accrual_db_path)
    try:
        conn.execute(
            """
            UPDATE paper_answers SET
                raw_response_impact = ?,
                impact_answer_location = ?,
                impact_has_answer = ?,
                impact_percent = ?,
                impact_absolute = ?,
                impact_unit = ?,
                impact_qualitative = ?,
                impact_evidence = ?
            WHERE paper_source = ? AND paper_source_id = ?
            """,
            (
                raw_response_impact,
                impact_answer_location,
                impact_has_answer,
                impact_percent_f,
                impact_absolute_f,
                impact_unit_s,
                impact_qualitative_s,
                impact_evidence_s,
                paper_source,
                paper_source_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def backfill_impact_evidence(accrual_db_path: Path) -> int:
    """Re-parse raw_response_impact for existing paper_answers and UPDATE impact_evidence when the parsed JSON has it. Returns number of rows updated."""
    from recite.accrual.parsing import parse_impact_response

    conn = _get_conn(accrual_db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, raw_response_impact FROM paper_answers
            WHERE raw_response_impact IS NOT NULL
            AND (impact_evidence IS NULL OR impact_evidence = '')
            """
        ).fetchall()
        updated = 0
        for (row_id, raw_impact) in rows:
            parsed = parse_impact_response(raw_impact)
            ev = parsed.get("impact_evidence")
            ev_s = _coerce_for_sqlite(ev, "text") if ev else None
            if ev_s and str(ev_s).strip():
                conn.execute(
                    "UPDATE paper_answers SET impact_evidence = ? WHERE id = ?",
                    (str(ev_s).strip()[:10000], row_id),
                )
                updated += 1
        conn.commit()
        logger.info(f"Backfill impact_evidence: updated {updated} rows")
        return updated
    finally:
        conn.close()


def get_paper_answers(
    accrual_db_path: Path,
    paper_source: Optional[str] = None,
    paper_source_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Get paper_answers rows, optionally filtered by paper_source and paper_source_id."""
    conn = _get_conn(accrual_db_path)
    conn.row_factory = sqlite3.Row
    try:
        if paper_source is not None and paper_source_id is not None:
            rows = conn.execute(
                "SELECT * FROM paper_answers WHERE paper_source=? AND paper_source_id=?",
                (paper_source, paper_source_id),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM paper_answers").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_top_papers(
    accrual_db_path: Path,
    require_directives_exact_text: bool = True,
    require_impact_numeric: bool = True,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Get papers that have directives_has_answer=1, non-null directives_exact_text, and impact (percent or absolute)."""
    conn = _get_conn(accrual_db_path)
    conn.row_factory = sqlite3.Row
    try:
        conditions = ["directives_has_answer = 1", "impact_has_answer = 1"]
        if require_directives_exact_text:
            conditions.append("directives_exact_text IS NOT NULL AND directives_exact_text != ''")
        if require_impact_numeric:
            conditions.append("(impact_percent IS NOT NULL OR impact_absolute IS NOT NULL)")
        where = " AND ".join(conditions)
        sql = f"SELECT * FROM paper_answers WHERE {where} ORDER BY id"
        if limit is not None:
            sql += f" LIMIT {limit}"
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def insert_paper_trial_gain(
    accrual_db_path: Path,
    paper_source: str,
    paper_source_id: str,
    trial_instance_id: str,
    match_score: Optional[int],
    applicability_score: Optional[int],
    enrollment: Optional[int],
    paper_pct_gain: Optional[float],
    scalar_gain: Optional[float],
    amended_ec_text: Optional[str] = None,
    change_directive_quote: Optional[str] = None,
    change_rationale_quote: Optional[str] = None,
    evidenced_summary: Optional[str] = None,
) -> None:
    """Insert or replace one row in paper_trial_gains."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn(accrual_db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_trial_gains (
                paper_source, paper_source_id, trial_instance_id,
                match_score, applicability_score, enrollment,
                paper_pct_gain, scalar_gain, amended_ec_text,
                change_directive_quote, change_rationale_quote, evidenced_summary,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_source,
                paper_source_id,
                trial_instance_id,
                match_score,
                applicability_score,
                enrollment,
                paper_pct_gain,
                scalar_gain,
                amended_ec_text,
                change_directive_quote,
                change_rationale_quote,
                evidenced_summary,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_paper_trial_gains(
    accrual_db_path: Path,
    only_null_evidenced_summary: bool = False,
) -> list[dict[str, Any]]:
    """Get paper_trial_gains rows, optionally only those with evidenced_summary IS NULL."""
    conn = _get_conn(accrual_db_path)
    conn.row_factory = sqlite3.Row
    try:
        if only_null_evidenced_summary:
            rows = conn.execute(
                "SELECT * FROM paper_trial_gains WHERE evidenced_summary IS NULL"
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM paper_trial_gains").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_paper_trial_evidenced_summary(
    accrual_db_path: Path,
    paper_source: str,
    paper_source_id: str,
    trial_instance_id: str,
    evidenced_summary: str,
) -> None:
    """Update evidenced_summary for one paper_trial_gains row."""
    conn = _get_conn(accrual_db_path)
    try:
        conn.execute(
            """
            UPDATE paper_trial_gains
            SET evidenced_summary = ?
            WHERE paper_source = ? AND paper_source_id = ? AND trial_instance_id = ?
            """,
            (evidenced_summary, paper_source, paper_source_id, trial_instance_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_trial_metadata_enrollment(recite_db_path: Path, instance_id: str) -> Optional[int]:
    """Return enrollment for instance_id from recite.db trial_metadata, or None if missing."""
    path = Path(recite_db_path)
    if not path.is_absolute():
        path = resolve_path(path)
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT enrollment FROM trial_metadata WHERE instance_id = ?", (instance_id,)
        ).fetchone()
        if row is None:
            return None
        return int(row[0]) if row[0] is not None else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
