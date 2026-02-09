"""
db.py

SQLite database schema and helper functions.
"""
import hashlib
import os
import re
import sqlite3
import statistics
from datetime import datetime, timedelta

from pathlib import Path
from loguru import logger
from recite.utils.path_loader import get_local_db_dir

# Default database path (can be overridden via DATABASE_PATH env var).
# Uses LOCAL_DB_DIR from .env when set (default data/dev).
DEFAULT_DB_PATH = get_local_db_dir() / "clintrialm.db"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB_PATH)))

# Archive database path (can be overridden via ARCHIVE_DATABASE_PATH env var)
# Default: same directory as main DB, named archive.db
ARCHIVE_DB_PATH = Path(os.getenv("ARCHIVE_DATABASE_PATH", str(DATABASE_PATH.parent / "archive.db")))

# Base schema for prompts table (shared across all models)
PROMPTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    query TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    model_preset TEXT NOT NULL,
    results_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, query_hash)
);
"""


def _get_table_name(model_preset: str) -> str:
    """Get sanitized table name for model preset (replace hyphens with underscores)."""
    return f"documents_{model_preset.replace('-', '_')}"


def _get_paper_trial_matches_table_name(model_preset: str) -> str:
    """Get sanitized table name for paper-trial matches (replace hyphens with underscores)."""
    return f"paper_trial_matches_{model_preset.replace('-', '_')}"


def _get_documents_schema(model_preset: str) -> str:
    """Get schema for model-specific documents table."""
    table_name = _get_table_name(model_preset)
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    doi TEXT,
    pmid INTEGER,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    journal TEXT,
    pub_date TEXT,
    relevance INTEGER CHECK (relevance BETWEEN 1 AND 5),
    extraction_confidence INTEGER CHECK (extraction_confidence BETWEEN 1 AND 5),
    accrual_ease INTEGER CHECK (accrual_ease BETWEEN 1 AND 5),
    total_score INTEGER,
    reasoning TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, source_id)
);
"""


def _get_paper_trial_matches_schema(model_preset: str) -> str:
    """Get schema for model-specific paper-trial matches table.
    
    This table stores matches between research papers and clinical trials,
    including LLM-generated scores and reasoning.
    """
    table_name = _get_paper_trial_matches_table_name(model_preset)
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_source TEXT NOT NULL,
    paper_source_id TEXT NOT NULL,
    paper_title TEXT NOT NULL,
    trial_nct_id TEXT NOT NULL,
    trial_title TEXT NOT NULL,
    match_score INTEGER CHECK (match_score BETWEEN 1 AND 5),
    match_reasoning TEXT,
    applicability_score INTEGER CHECK (applicability_score BETWEEN 1 AND 5),
    change_directive_quote TEXT,
    change_rationale_quote TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_source, paper_source_id, trial_nct_id)
);
"""


def _get_conn() -> sqlite3.Connection:
    """Get a database connection, creating the directory if needed."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DATABASE_PATH))


def _get_archive_conn() -> sqlite3.Connection:
    """Get a connection to the archive database, creating it if needed."""
    ARCHIVE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(ARCHIVE_DB_PATH))


def _archive_table_to_archive_db(source_table: str, model_preset: str) -> str:
    """Copy a table from clintrialm.db to archive.db with timestamp.
    
    This is a generic function that handles archiving for both documents and matches tables.
    After copying, the original table is dropped from clintrialm.db.
    
    Args:
        source_table: Name of the table in clintrialm.db to archive
        model_preset: Model preset (used for generating archive table name)
    
    Returns:
        The archived table name in archive.db (e.g., documents_local_archive_20260122_143022)
    
    Raises:
        ValueError: If source table doesn't exist
    """
    # Generate timestamped archive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_table_name = f"{source_table}_archive_{timestamp}"
    
    with _get_conn() as source_conn:
        # Check if source table exists
        cursor = source_conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (source_table,))
        if not cursor.fetchone():
            raise ValueError(f"Table {source_table} does not exist")
        
        # Get table schema
        cursor = source_conn.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (source_table,))
        schema_row = cursor.fetchone()
        if not schema_row or not schema_row[0]:
            raise ValueError(f"Could not get schema for table {source_table}")
        
        # Get all data from source table
        cursor = source_conn.execute(f"SELECT * FROM {source_table}")
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        row_count = len(rows)
        
        # Modify schema to use archive table name
        schema_sql = schema_row[0]
        # Replace CREATE TABLE statements - handle various formats
        # Need to be careful with replacement to avoid partial matches
        if f"CREATE TABLE IF NOT EXISTS {source_table}" in schema_sql:
            schema_sql = schema_sql.replace(f"CREATE TABLE IF NOT EXISTS {source_table}", f"CREATE TABLE IF NOT EXISTS {archive_table_name}", 1)
        elif f"CREATE TABLE {source_table}" in schema_sql:
            schema_sql = schema_sql.replace(f"CREATE TABLE {source_table}", f"CREATE TABLE {archive_table_name}", 1)
        else:
            # Fallback: replace table name (but be careful with partial matches)
            # Replace the table name when it appears as a standalone identifier
            import re
            # Match table name as a word boundary (not part of another name)
            pattern = r'\b' + re.escape(source_table) + r'\b'
            schema_sql = re.sub(pattern, archive_table_name, schema_sql, count=1)
        
        # Copy to archive.db
        with _get_archive_conn() as archive_conn:
            # Create table in archive.db with schema
            # Use CREATE TABLE (not IF NOT EXISTS) since archive names are timestamped and unique
            schema_sql_clean = schema_sql.replace("CREATE TABLE IF NOT EXISTS", "CREATE TABLE", 1)
            try:
                archive_conn.executescript(schema_sql_clean)
            except sqlite3.OperationalError as e:
                if "already exists" in str(e).lower():
                    # If table somehow already exists (very unlikely with timestamp), append a random suffix
                    import random
                    archive_table_name = f"{archive_table_name}_{random.randint(1000, 9999)}"
                    schema_sql_retry = schema_row[0]
                    if f"CREATE TABLE IF NOT EXISTS {source_table}" in schema_sql_retry:
                        schema_sql_retry = schema_sql_retry.replace(f"CREATE TABLE IF NOT EXISTS {source_table}", f"CREATE TABLE {archive_table_name}", 1)
                    elif f"CREATE TABLE {source_table}" in schema_sql_retry:
                        schema_sql_retry = schema_sql_retry.replace(f"CREATE TABLE {source_table}", f"CREATE TABLE {archive_table_name}", 1)
                    else:
                        pattern = r'\b' + re.escape(source_table) + r'\b'
                        schema_sql_retry = re.sub(pattern, archive_table_name, schema_sql_retry, count=1)
                    schema_sql_retry = schema_sql_retry.replace("CREATE TABLE IF NOT EXISTS", "CREATE TABLE", 1)
                    archive_conn.executescript(schema_sql_retry)
                else:
                    raise
            
            # Insert data
            if rows:
                placeholders = ",".join(["?"] * len(column_names))
                columns = ",".join([f'"{col}"' for col in column_names])  # Quote column names for safety
                archive_conn.executemany(
                    f'INSERT INTO "{archive_table_name}" ({columns}) VALUES ({placeholders})',
                    rows
                )
                archive_conn.commit()
            
            # Verify copy was successful (use the potentially updated archive_table_name)
            cursor = archive_conn.execute(f'SELECT COUNT(*) FROM "{archive_table_name}"')
            archive_count = cursor.fetchone()[0]
            
            if archive_count != row_count:
                raise RuntimeError(f"Archive copy failed: expected {row_count} rows, got {archive_count}")
        
        # Drop original table from source DB
        source_conn.execute(f"DROP TABLE {source_table}")
        source_conn.commit()
        
        logger.info(f"Archived {source_table} to {archive_table_name} in archive.db ({archive_count} rows)")
    
    return archive_table_name


def _ensure_table(model_preset: str):
    """Ensure the model-specific documents table exists and has updated_at column."""
    table_name = _get_table_name(model_preset)
    schema = _get_documents_schema(model_preset)
    with _get_conn() as conn:
        conn.executescript(schema)
        
        # Check if updated_at column exists, add it if missing (migration)
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        if "updated_at" not in columns:
            logger.info(f"Migrating {table_name}: adding updated_at column")
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN updated_at TEXT")
            # Backfill with created_at for existing rows
            conn.execute(f"UPDATE {table_name} SET updated_at = created_at WHERE updated_at IS NULL")
            conn.commit()
            logger.info(f"Migration complete for {table_name}")
        
        logger.debug(f"Ensured table {table_name} exists")


def _ensure_paper_trial_matches_table(model_preset: str):
    """Ensure the model-specific paper-trial matches table exists."""
    table_name = _get_paper_trial_matches_table_name(model_preset)
    schema = _get_paper_trial_matches_schema(model_preset)
    with _get_conn() as conn:
        conn.executescript(schema)
        logger.debug(f"Ensured table {table_name} exists")


def init_db(model_preset: str | None = None):
    """Initialize database with schema.
    
    Creates prompts table (shared) and optionally creates model-specific documents table.
    """
    with _get_conn() as conn:
        # Always create prompts table
        conn.executescript(PROMPTS_SCHEMA)
        
        # Create model-specific documents table if preset provided
        if model_preset:
            _ensure_table(model_preset)


def save_document(doc, scores, model_preset: str):
    """Save a document with its evaluation scores to model-specific table."""
    _ensure_table(model_preset)
    table_name = _get_table_name(model_preset)
    total_score = scores.relevance + scores.extraction_confidence + scores.accrual_ease
    with _get_conn() as conn:
        conn.execute(f"""
            INSERT OR REPLACE INTO {table_name} (source, source_id, doi, pmid, title, abstract, 
                                   relevance, extraction_confidence, accrual_ease, total_score, reasoning, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (doc.source, doc.source_id, doc.doi, doc.pmid, doc.title, doc.abstract,
              scores.relevance, scores.extraction_confidence, scores.accrual_ease, total_score, scores.reasoning))


def _query_hash(query: str) -> str:
    """Generate SHA-256 hash of query string."""
    return hashlib.sha256(query.encode()).hexdigest()


def has_query(source: str, query: str, model_preset: str) -> bool:
    """Check if a query has already been used for a source by this model."""
    h = _query_hash(query)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM prompts WHERE source=? AND query_hash=? AND model_preset=?",
            (source, h, model_preset)
        ).fetchone()
    return row is not None


def save_query(source: str, query: str, model_preset: str):
    """Save a query to prevent reuse."""
    h = _query_hash(query)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO prompts (source, query, query_hash, model_preset) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (source, query, h, model_preset)
        )


def get_used_queries(source: str, model_preset: str) -> list[str]:
    """Get all queries used for a source by this model."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT query FROM prompts WHERE source=? AND model_preset=?",
            (source, model_preset)
        ).fetchall()
    return [r[0] for r in rows]


def has_document(source: str, source_id: str, model_preset: str) -> bool:
    """Check if a document already exists in the model-specific table."""
    _ensure_table(model_preset)
    table_name = _get_table_name(model_preset)
    with _get_conn() as conn:
        row = conn.execute(
            f"SELECT 1 FROM {table_name} WHERE source=? AND source_id=?",
            (source, source_id)
        ).fetchone()
    return row is not None


def get_document_by_source_id(
    paper_source: str,
    paper_source_id: str,
    model_preset: str,
) -> dict | None:
    """Get a single document by (paper_source, paper_source_id) from the model-specific table.
    
    Returns:
        Dict with keys source, source_id, title, abstract (and optionally doi, pmid), or None if not found.
    """
    _ensure_table(model_preset)
    table_name = _get_table_name(model_preset)
    with _get_conn() as conn:
        row = conn.execute(
            f"SELECT source, source_id, title, abstract FROM {table_name} WHERE source=? AND source_id=?",
            (paper_source, paper_source_id),
        ).fetchone()
    if not row:
        return None
    return {"source": row[0], "source_id": row[1], "title": row[2] or "", "abstract": row[3] or ""}


def get_relevant_documents(
    min_relevance: int = 4,
    limit: int = 10,
    model_preset: str | list[str] | None = None,
    min_extraction_confidence: int | None = None,
    min_accrual_ease: int | None = None,
    min_total_score: int | None = None,
    sort_by: str = "total_score",
    sort_desc: bool = True,
) -> list[dict]:
    """Get highly relevant documents with filtering and sorting.
    
    Args:
        min_relevance: Minimum relevance score (1-5)
        limit: Maximum number of results
        model_preset: Single preset (str), list of presets (list[str]), or None for all active presets
        min_extraction_confidence: Minimum extraction_confidence score (1-5)
        min_accrual_ease: Minimum accrual_ease score (1-5)
        min_total_score: Minimum total_score (3-15)
        sort_by: Field to sort by: "relevance", "extraction_confidence", "accrual_ease", "total_score", "created_at"
        sort_desc: Sort descending (True) or ascending (False)
    
    Returns:
        List of document dicts with all score fields
    """
    # Validate sort_by field
    valid_sort_fields = ["relevance", "extraction_confidence", "accrual_ease", "total_score", "created_at"]
    if sort_by not in valid_sort_fields:
        sort_by = "total_score"
    
    # Determine which presets to query
    if model_preset is None:
        # Query all active presets (excludes backup tables)
        presets_to_query = get_all_model_presets()
    elif isinstance(model_preset, str):
        # Single preset
        presets_to_query = [model_preset]
    else:
        # List of presets
        presets_to_query = model_preset
    
    # Query each preset and collect results
    all_docs = []
    for preset in presets_to_query:
        _ensure_table(preset)
        table_name = _get_table_name(preset)
        
        # Build WHERE clause with filters
        conditions = ["relevance >= ?"]
        params = [min_relevance]
        
        if min_extraction_confidence is not None:
            conditions.append("extraction_confidence >= ?")
            params.append(min_extraction_confidence)
        
        if min_accrual_ease is not None:
            conditions.append("accrual_ease >= ?")
            params.append(min_accrual_ease)
        
        if min_total_score is not None:
            conditions.append("total_score >= ?")
            params.append(min_total_score)
        
        where_clause = " AND ".join(conditions)
        sort_direction = "DESC" if sort_desc else "ASC"
        
        # Always add created_at as secondary sort for consistency
        if sort_by != "created_at":
            order_by = f"{sort_by} {sort_direction}, created_at DESC"
        else:
            order_by = f"{sort_by} {sort_direction}"
        
        # Query with all score fields
        with _get_conn() as conn:
            rows = conn.execute(f"""
                SELECT source, source_id, title, abstract, doi, pmid,
                       relevance, extraction_confidence, accrual_ease, total_score
                FROM {table_name}
                WHERE {where_clause}
                ORDER BY {order_by}
            """, params).fetchall()
        
        # Convert to dicts with all fields
        for r in rows:
            all_docs.append({
                "source": r[0],
                "source_id": r[1],
                "title": r[2],
                "abstract": r[3],
                "doi": r[4],
                "pmid": r[5],
                "relevance": r[6],
                "extraction_confidence": r[7],
                "accrual_ease": r[8],
                "total_score": r[9],
            })
    
    # Deduplicate by (source, source_id) - prefer higher scores
    # Sort by total_score first to keep best duplicates
    all_docs.sort(key=lambda x: (x.get("total_score", 0), x.get("relevance", 0)), reverse=True)
    
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = (doc["source"], doc["source_id"])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    
    # Re-sort by requested field
    unique_docs.sort(key=lambda x: x.get(sort_by, 0), reverse=sort_desc)
    
    # Apply limit
    return unique_docs[:limit]


def get_all_model_presets() -> list[str]:
    """Get list of all model presets that have document tables.
    
    Only returns active presets from clintrialm.db (archived tables are in archive.db).
    """
    with _get_conn() as conn:
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'documents_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]
    # Extract preset names (remove 'documents_' prefix)
    presets = [t.replace("documents_", "") for t in tables]
    # Convert underscores back to hyphens for display
    return [p.replace("_", "-") for p in presets]


def archive_model_table(model_preset: str) -> str:
    """Archive a model's documents table to archive.db.
    
    Copies the table to archive.db with a timestamp, then drops the original.
    
    Args:
        model_preset: Model preset to archive
        
    Returns:
        The archived table name in archive.db (e.g., documents_local_archive_20260122_143022)
        
    Raises:
        ValueError: If table doesn't exist
    """
    table_name = _get_table_name(model_preset)
    archive_name = _archive_table_to_archive_db(table_name, model_preset)
    return archive_name


def get_all_documents_from_other_models(model_preset: str, min_relevance: int = 3) -> list[dict]:
    """Get all documents from other model tables, deduplicated.
    
    Queries all documents_% tables from clintrialm.db (except current model's active table)
    and all archived documents_% tables from archive.db.
    
    Deduplication priority: DOI > PMID > title match.
    Returns list of unique documents to review.
    """
    current_table_name = _get_table_name(model_preset)
    
    all_docs = []
    
    # Query active tables from clintrialm.db (excluding current model's table)
    with _get_conn() as conn:
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'documents_%'
        """)
        active_tables = [row[0] for row in cursor.fetchall()]
    
    # Filter out the current model's active table
    other_active_tables = [t for t in active_tables if t != current_table_name]
    
    for table_name in other_active_tables:
        with _get_conn() as conn:
            try:
                rows = conn.execute(f"""
                    SELECT source, source_id, doi, pmid, title, abstract
                    FROM {table_name}
                    WHERE relevance >= ?
                """, (min_relevance,)).fetchall()
                for row in rows:
                    all_docs.append({
                        "source": row[0],
                        "source_id": row[1],
                        "doi": row[2],
                        "pmid": row[3],
                        "title": row[4],
                        "abstract": row[5],
                    })
            except sqlite3.OperationalError as e:
                logger.warning(f"Skipping table {table_name} due to schema mismatch: {e}")
                continue
    
    # Query archived tables from archive.db (exclude current model's archived tables)
    try:
        with _get_archive_conn() as archive_conn:
            cursor = archive_conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'documents_%_archive_%'
            """)
            archived_tables = [row[0] for row in cursor.fetchall()]
        
        # Filter out archived tables from current model
        # Archived table name format: documents_{preset}_archive_YYYYMMDD_HHMMSS
        current_model_prefix = f"{current_table_name}_archive_"
        other_archived_tables = [t for t in archived_tables if not t.startswith(current_model_prefix)]
        
        for table_name in other_archived_tables:
            with _get_archive_conn() as archive_conn:
                try:
                    rows = archive_conn.execute(f"""
                        SELECT source, source_id, doi, pmid, title, abstract
                        FROM {table_name}
                        WHERE relevance >= ?
                    """, (min_relevance,)).fetchall()
                    for row in rows:
                        all_docs.append({
                            "source": row[0],
                            "source_id": row[1],
                            "doi": row[2],
                            "pmid": row[3],
                            "title": row[4],
                            "abstract": row[5],
                        })
                except sqlite3.OperationalError as e:
                    logger.warning(f"Skipping archived table {table_name} due to schema mismatch: {e}")
                    continue
    except sqlite3.OperationalError:
        # Archive.db might not exist yet, that's okay
        logger.debug("Archive.db does not exist or is not accessible")
    
    if not all_docs:
        return []
    
    # Deduplicate: prefer DOI, then PMID, then title
    # Use dicts to track seen items and keep first occurrence
    seen_dois = {}
    seen_pmids = {}
    seen_titles = {}
    unique_docs = []
    
    for doc in all_docs:
        # Check DOI first (highest priority)
        if doc.get("doi"):
            doi = doc["doi"]
            if doi not in seen_dois:
                seen_dois[doi] = doc
                unique_docs.append(doc)
            continue
        
        # Check PMID second
        if doc.get("pmid"):
            pmid = doc["pmid"]
            if pmid not in seen_pmids:
                seen_pmids[pmid] = doc
                unique_docs.append(doc)
            continue
        
        # Check title (normalized) - lowest priority
        title_key = doc.get("title", "").lower().strip()
        if title_key:
            if title_key not in seen_titles:
                seen_titles[title_key] = doc
                unique_docs.append(doc)
            continue
    
    return unique_docs


def get_documents_count(
    model_preset: str,
    min_relevance: int = 1,
    source_filter: list[str] | None = None,
    search_term: str | None = None,
) -> int:
    """Get count of documents matching filters.
    
    Args:
        model_preset: Model preset to query
        min_relevance: Minimum relevance score (1-5)
        source_filter: List of sources to include (e.g., ['pubmed', 's2', 'ctg'])
        search_term: Text to search in title/abstract
    
    Returns:
        Total count of matching documents
    """
    _ensure_table(model_preset)
    table_name = _get_table_name(model_preset)
    
    # Build query
    conditions = ["relevance >= ?"]
    params = [min_relevance]
    
    if source_filter and len(source_filter) > 0:
        placeholders = ",".join(["?"] * len(source_filter))
        conditions.append(f"source IN ({placeholders})")
        params.extend(source_filter)
    
    if search_term:
        conditions.append("(title LIKE ? OR abstract LIKE ?)")
        search_pattern = f"%{search_term}%"
        params.extend([search_pattern, search_pattern])
    
    where_clause = " AND ".join(conditions)
    
    query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"
    
    with _get_conn() as conn:
        count = conn.execute(query, params).fetchone()[0]
    
    return count


def get_stats(model_preset: str | None = None) -> dict:
    """Get database statistics.
    
    If model_preset is provided, returns stats for that model only.
    Otherwise, aggregates across all model tables.
    """
    with _get_conn() as conn:
        prompts = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
    
    if model_preset:
        _ensure_table(model_preset)
        table_name = _get_table_name(model_preset)
        with _get_conn() as conn:
            docs = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            avg = conn.execute(f"SELECT AVG(relevance) FROM {table_name} WHERE relevance IS NOT NULL").fetchone()[0] or 0
            relevant = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE relevance >= 4").fetchone()[0]
        return {"documents": docs, "prompts": prompts, "avg_relevance": avg, "relevant": relevant}
    else:
        # Aggregate across all models
        presets = get_all_model_presets()
        total_docs = 0
        total_relevant = 0
        all_relevances = []
        
        for preset in presets:
            _ensure_table(preset)
            table_name = _get_table_name(preset)
            with _get_conn() as conn:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                relevant_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE relevance >= 4").fetchone()[0]
                relevances = conn.execute(f"SELECT relevance FROM {table_name} WHERE relevance IS NOT NULL").fetchall()
                total_docs += count
                total_relevant += relevant_count
                all_relevances.extend([r[0] for r in relevances])
        
        avg = sum(all_relevances) / len(all_relevances) if all_relevances else 0
        return {"documents": total_docs, "prompts": prompts, "avg_relevance": avg, "relevant": total_relevant}


def get_documents_for_ui(
    model_preset: str,
    min_relevance: int = 1,
    source_filter: list[str] | None = None,
    search_term: str | None = None,
    limit: int = 1000,
    offset: int = 0,
    sort_by: str = "relevance",
    sort_desc: bool = True,
) -> list[dict]:
    """Get documents for UI display with filtering and pagination.
    
    Args:
        model_preset: Model preset to query
        min_relevance: Minimum relevance score (1-5)
        source_filter: List of sources to include (e.g., ['pubmed', 's2', 'ctg'])
        search_term: Text to search in title/abstract
        limit: Maximum number of results
        offset: Offset for pagination
        sort_by: Field to sort by (relevance, extraction_confidence, accrual_ease, total_score, created_at)
        sort_desc: Whether to sort descending (True) or ascending (False)
    
    Returns:
        List of document dicts with all fields
    """
    _ensure_table(model_preset)
    table_name = _get_table_name(model_preset)
    
    # Validate sort_by field
    valid_sort_fields = ["relevance", "extraction_confidence", "accrual_ease", "total_score", "created_at"]
    if sort_by not in valid_sort_fields:
        sort_by = "relevance"
    
    # Build query
    conditions = ["relevance >= ?"]
    params = [min_relevance]
    
    if source_filter and len(source_filter) > 0:
        placeholders = ",".join(["?"] * len(source_filter))
        conditions.append(f"source IN ({placeholders})")
        params.extend(source_filter)
    
    if search_term:
        conditions.append("(title LIKE ? OR abstract LIKE ?)")
        search_pattern = f"%{search_term}%"
        params.extend([search_pattern, search_pattern])
    
    where_clause = " AND ".join(conditions)
    sort_direction = "DESC" if sort_desc else "ASC"
    
    # Always add created_at as secondary sort for consistency
    if sort_by != "created_at":
        order_by = f"{sort_by} {sort_direction}, created_at DESC"
    else:
        order_by = f"{sort_by} {sort_direction}"
    
    query = f"""
        SELECT source, source_id, doi, pmid, title, abstract, 
               relevance, extraction_confidence, accrual_ease, total_score, reasoning, created_at
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY {order_by}
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    
    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    
    return [
        {
            "source": r[0],
            "source_id": r[1],
            "doi": r[2],
            "pmid": r[3],
            "title": r[4],
            "abstract": r[5],
            "relevance": r[6],
            "extraction_confidence": r[7],
            "accrual_ease": r[8],
            "total_score": r[9],
            "reasoning": r[10],
            "created_at": r[11],
        }
        for r in rows
    ]


# --- Paper-Trial Matches Functions ---

def init_paper_trial_matches_table(model_preset: str):
    """Initialize the paper-trial matches table for a model preset.
    
    Creates the table if it doesn't exist.
    """
    _ensure_paper_trial_matches_table(model_preset)
    logger.info(f"Initialized paper_trial_matches table for {model_preset}")


def archive_paper_trial_matches_table(model_preset: str) -> str:
    """Archive a model's paper-trial matches table to archive.db.
    
    Copies the table to archive.db with a timestamp, then drops the original.
    
    Args:
        model_preset: Model preset to archive
        
    Returns:
        The archived table name in archive.db (e.g., paper_trial_matches_local_archive_20260122_143022)
        
    Raises:
        ValueError: If table doesn't exist
    """
    table_name = _get_paper_trial_matches_table_name(model_preset)
    archive_name = _archive_table_to_archive_db(table_name, model_preset)
    return archive_name


def save_paper_trial_match(
    paper_source: str,
    paper_source_id: str,
    paper_title: str,
    trial_nct_id: str,
    trial_title: str,
    match_score: int,
    match_reasoning: str,
    applicability_score: int,
    change_directive_quote: str,
    change_rationale_quote: str,
    model_preset: str,
) -> bool:
    """Save a paper-trial match to the database.
    
    Args:
        paper_source: Source of the paper (e.g., "pubmed", "s2")
        paper_source_id: Source-specific ID (e.g., PMID, S2 ID)
        paper_title: Title of the paper
        trial_nct_id: NCT ID of the clinical trial
        trial_title: Title of the clinical trial
        match_score: Overall match score (1-5)
        match_reasoning: LLM explanation of match quality
        applicability_score: How easily applicable the change is (1-5)
        change_directive_quote: Verbatim quote showing what/how to change
        change_rationale_quote: Verbatim quote showing why to change
        model_preset: Model preset used for evaluation
    
    Returns:
        True if match was saved (new), False if it already existed
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    with _get_conn() as conn:
        cursor = conn.execute(f"""
            INSERT OR IGNORE INTO {table_name} (
                paper_source, paper_source_id, paper_title,
                trial_nct_id, trial_title,
                match_score, match_reasoning,
                applicability_score, change_directive_quote, change_rationale_quote
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_source, paper_source_id, paper_title,
            trial_nct_id, trial_title,
            match_score, match_reasoning,
            applicability_score, change_directive_quote, change_rationale_quote
        ))
        return cursor.rowcount > 0


def update_paper_trial_match(
    paper_source: str,
    paper_source_id: str,
    trial_nct_id: str,
    match_score: int,
    match_reasoning: str,
    applicability_score: int,
    change_directive_quote: str,
    change_rationale_quote: str,
    model_preset: str,
    paper_title: str | None = None,
    trial_title: str | None = None,
) -> bool:
    """Update or insert a paper-trial match with new scores and reasoning.
    
    Uses INSERT OR REPLACE to handle cases where the match doesn't exist yet
    (e.g., after archiving when using --re-review-own).
    
    Args:
        paper_source: Source of the paper
        paper_source_id: Source-specific ID
        trial_nct_id: NCT ID of the clinical trial
        match_score: Updated match score (1-5)
        match_reasoning: Updated LLM explanation
        applicability_score: Updated applicability score (1-5)
        change_directive_quote: Updated directive quote
        change_rationale_quote: Updated rationale quote
        model_preset: Model preset
        paper_title: Title of the paper (required if match doesn't exist)
        trial_title: Title of the trial (required if match doesn't exist)
    
    Returns:
        True if match was updated/inserted, False if required fields are missing
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    # Clamp scores to valid range
    match_score = max(1, min(5, match_score))
    applicability_score = max(1, min(5, applicability_score))
    
    # If paper_title or trial_title not provided, try to get them from existing match
    if paper_title is None or trial_title is None:
        existing_match = get_paper_trial_matches_for_paper(
            paper_source, paper_source_id, model_preset, min_match_score=1
        )
        # Find the specific match for this trial
        for m in existing_match:
            if m["trial_nct_id"] == trial_nct_id:
                if paper_title is None:
                    paper_title = m.get("paper_title", "")
                if trial_title is None:
                    trial_title = m.get("trial_title", "")
                break
    
    # Default to empty strings if still missing (INSERT OR REPLACE will work)
    if paper_title is None:
        paper_title = ""
    if trial_title is None:
        trial_title = ""
    
    with _get_conn() as conn:
        cursor = conn.execute(f"""
            INSERT OR REPLACE INTO {table_name} (
                paper_source, paper_source_id, paper_title,
                trial_nct_id, trial_title,
                match_score, match_reasoning,
                applicability_score, change_directive_quote, change_rationale_quote,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            paper_source, paper_source_id, paper_title,
            trial_nct_id, trial_title,
            match_score, match_reasoning,
            applicability_score, change_directive_quote, change_rationale_quote
        ))
        return cursor.rowcount > 0


def has_paper_trial_match(
    paper_source: str,
    paper_source_id: str,
    trial_nct_id: str,
    model_preset: str,
) -> bool:
    """Check if a paper-trial match already exists.
    
    Args:
        paper_source: Source of the paper
        paper_source_id: Source-specific ID
        trial_nct_id: NCT ID of the clinical trial
        model_preset: Model preset to check
    
    Returns:
        True if match exists, False otherwise
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    with _get_conn() as conn:
        row = conn.execute(
            f"SELECT 1 FROM {table_name} WHERE paper_source=? AND paper_source_id=? AND trial_nct_id=?",
            (paper_source, paper_source_id, trial_nct_id)
        ).fetchone()
    return row is not None


def get_paper_trial_matches_for_paper(
    paper_source: str,
    paper_source_id: str,
    model_preset: str,
    min_match_score: int = 1,
) -> list[dict]:
    """Get all matches for a specific paper.
    
    Args:
        paper_source: Source of the paper
        paper_source_id: Source-specific ID
        model_preset: Model preset to query
        min_match_score: Minimum match score to include (default: 1)
    
    Returns:
        List of match dicts with all fields
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT paper_source, paper_source_id, paper_title,
                   trial_nct_id, trial_title,
                   match_score, match_reasoning,
                   applicability_score, change_directive_quote, change_rationale_quote,
                   created_at, updated_at
            FROM {table_name}
            WHERE paper_source=? AND paper_source_id=? AND match_score >= ?
            ORDER BY match_score DESC, applicability_score DESC
        """, (paper_source, paper_source_id, min_match_score)).fetchall()
    
    return [
        {
            "paper_source": r[0],
            "paper_source_id": r[1],
            "paper_title": r[2],
            "trial_nct_id": r[3],
            "trial_title": r[4],
            "match_score": r[5],
            "match_reasoning": r[6],
            "applicability_score": r[7],
            "change_directive_quote": r[8],
            "change_rationale_quote": r[9],
            "created_at": r[10],
            "updated_at": r[11],
        }
        for r in rows
    ]


def get_paper_trial_matches_for_trial(
    trial_nct_id: str,
    model_preset: str,
    min_match_score: int = 1,
) -> list[dict]:
    """Get all matches for a specific trial.
    
    Args:
        trial_nct_id: NCT ID of the clinical trial
        model_preset: Model preset to query
        min_match_score: Minimum match score to include (default: 1)
    
    Returns:
        List of match dicts with all fields
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT paper_source, paper_source_id, paper_title,
                   trial_nct_id, trial_title,
                   match_score, match_reasoning,
                   applicability_score, change_directive_quote, change_rationale_quote,
                   created_at, updated_at
            FROM {table_name}
            WHERE trial_nct_id=? AND match_score >= ?
            ORDER BY match_score DESC, applicability_score DESC
        """, (trial_nct_id, min_match_score)).fetchall()
    
    return [
        {
            "paper_source": r[0],
            "paper_source_id": r[1],
            "paper_title": r[2],
            "trial_nct_id": r[3],
            "trial_title": r[4],
            "match_score": r[5],
            "match_reasoning": r[6],
            "applicability_score": r[7],
            "change_directive_quote": r[8],
            "change_rationale_quote": r[9],
            "created_at": r[10],
            "updated_at": r[11],
        }
        for r in rows
    ]


def get_all_paper_trial_matches(
    model_preset: str,
    min_match_score: int = 1,
    limit: int = None,
) -> list[dict]:
    """Get all paper-trial matches from the database.
    
    Args:
        model_preset: Model preset to query
        min_match_score: Minimum match score to include (default: 1)
        limit: Maximum number of matches to return (None = no limit)
    
    Returns:
        List of match dicts with all fields
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    limit_clause = ""
    if limit is not None:
        limit_clause = f"LIMIT {limit}"
    
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT paper_source, paper_source_id, paper_title,
                   trial_nct_id, trial_title,
                   match_score, match_reasoning,
                   applicability_score, change_directive_quote, change_rationale_quote,
                   created_at, updated_at
            FROM {table_name}
            WHERE match_score >= ?
            ORDER BY match_score DESC, applicability_score DESC
            {limit_clause}
        """, (min_match_score,)).fetchall()
    
    return [
        {
            "paper_source": r[0],
            "paper_source_id": r[1],
            "paper_title": r[2],
            "trial_nct_id": r[3],
            "trial_title": r[4],
            "match_score": r[5],
            "match_reasoning": r[6],
            "applicability_score": r[7],
            "change_directive_quote": r[8],
            "change_rationale_quote": r[9],
            "created_at": r[10],
            "updated_at": r[11],
        }
        for r in rows
    ]


def get_document_by_source_id(
    model_preset: str,
    paper_source: str,
    paper_source_id: str,
) -> dict | None:
    """Get a document from the documents table by source and source_id.
    
    Args:
        model_preset: Model preset to query (or list of presets to search)
        paper_source: Source of the paper
        paper_source_id: Source-specific ID
    
    Returns:
        Document dict with all fields, or None if not found
    """
    # Try current preset first
    _ensure_table(model_preset)
    table_name = _get_table_name(model_preset)
    
    with _get_conn() as conn:
        row = conn.execute(f"""
            SELECT source, source_id, doi, pmid, title, abstract,
                   relevance, extraction_confidence, accrual_ease, total_score, reasoning, created_at, updated_at
            FROM {table_name}
            WHERE source=? AND source_id=?
        """, (paper_source, paper_source_id)).fetchone()
    
    if row:
        return {
            "source": row[0],
            "source_id": row[1],
            "doi": row[2],
            "pmid": row[3],
            "title": row[4],
            "abstract": row[5],
            "relevance": row[6],
            "extraction_confidence": row[7],
            "accrual_ease": row[8],
            "total_score": row[9],
            "reasoning": row[10],
            "created_at": row[11],
            "updated_at": row[12],
        }
    
    # If not found in current preset, try other presets
    for preset in get_all_model_presets():
        if preset == model_preset:
            continue
        _ensure_table(preset)
        other_table = _get_table_name(preset)
        with _get_conn() as conn:
            row = conn.execute(f"""
                SELECT source, source_id, doi, pmid, title, abstract,
                       relevance, extraction_confidence, accrual_ease, total_score, reasoning, created_at, updated_at
                FROM {other_table}
                WHERE source=? AND source_id=?
            """, (paper_source, paper_source_id)).fetchone()
        
        if row:
            return {
                "source": row[0],
                "source_id": row[1],
                "doi": row[2],
                "pmid": row[3],
                "title": row[4],
                "abstract": row[5],
                "relevance": row[6],
                "extraction_confidence": row[7],
                "accrual_ease": row[8],
                "total_score": row[9],
                "reasoning": row[10],
                "created_at": row[11],
                "updated_at": row[12],
            }
    
    return None


def get_papers_with_matches(model_preset: str) -> set[tuple[str, str]]:
    """Get set of (paper_source, paper_source_id) tuples that have at least one match.
    
    Useful for resume functionality - skip papers that already have matches.
    
    Args:
        model_preset: Model preset to query
    
    Returns:
        Set of (paper_source, paper_source_id) tuples
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT DISTINCT paper_source, paper_source_id
            FROM {table_name}
        """).fetchall()
    
    return {(r[0], r[1]) for r in rows}


def get_paper_trial_matches_stats(model_preset: str) -> dict:
    """Get statistics about paper-trial matches.
    
    Args:
        model_preset: Model preset to query
    
    Returns:
        Dict with statistics: total_matches, unique_papers, unique_trials,
        avg_match_score, avg_applicability_score, high_quality_matches
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    with _get_conn() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        unique_papers = conn.execute(f"""
            SELECT COUNT(DISTINCT paper_source || paper_source_id) FROM {table_name}
        """).fetchone()[0]
        unique_trials = conn.execute(f"""
            SELECT COUNT(DISTINCT trial_nct_id) FROM {table_name}
        """).fetchone()[0]
        avg_match = conn.execute(f"""
            SELECT AVG(match_score) FROM {table_name} WHERE match_score IS NOT NULL
        """).fetchone()[0] or 0
        avg_applicability = conn.execute(f"""
            SELECT AVG(applicability_score) FROM {table_name} WHERE applicability_score IS NOT NULL
        """).fetchone()[0] or 0
        high_quality = conn.execute(f"""
            SELECT COUNT(*) FROM {table_name} WHERE match_score >= 4 AND applicability_score >= 4
        """).fetchone()[0]
    
    return {
        "total_matches": total,
        "unique_papers": unique_papers,
        "unique_trials": unique_trials,
        "avg_match_score": round(avg_match, 2),
        "avg_applicability_score": round(avg_applicability, 2),
        "high_quality_matches": high_quality,
    }


def get_papers_with_match_stats(
    model_preset: str,
    min_match_score: int = 1,
    sort_by: str = "high_quality_count",
    sort_desc: bool = True,
) -> list[dict]:
    """Get papers with aggregated match statistics.
    
    Aggregates matches by paper and returns statistics for sorting/filtering.
    
    Args:
        model_preset: Model preset to query
        min_match_score: Minimum match score to include (default: 1)
        sort_by: Field to sort by: high_quality_count, mean_match_score, max_match_score, 
                 match_count, avg_applicability_score (default: high_quality_count)
        sort_desc: Whether to sort descending (default: True)
    
    Returns:
        List of dicts with paper info and aggregated stats:
        - paper_source, paper_source_id, paper_title
        - match_count: Total number of matches
        - mean_match_score: Average match_score
        - max_match_score: Highest match_score
        - high_quality_count: Count where match_score >= 4 AND applicability_score >= 4
        - avg_applicability_score: Average applicability_score
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    # Validate sort_by field
    valid_sort_fields = {
        "high_quality_count": "high_quality_count",
        "mean_match_score": "mean_match_score",
        "max_match_score": "max_match_score",
        "match_count": "match_count",
        "avg_applicability_score": "avg_applicability_score",
    }
    sort_field = valid_sort_fields.get(sort_by, "high_quality_count")
    sort_direction = "DESC" if sort_desc else "ASC"
    
    with _get_conn() as conn:
        # Get all match scores for median calculation (using GROUP_CONCAT)
        rows = conn.execute(f"""
            SELECT paper_source, paper_source_id, paper_title,
                   COUNT(*) as match_count,
                   AVG(match_score) as mean_match_score,
                   MAX(match_score) as max_match_score,
                   AVG(applicability_score) as avg_applicability_score,
                   SUM(CASE WHEN match_score >= 4 AND applicability_score >= 4 THEN 1 ELSE 0 END) as high_quality_count,
                   GROUP_CONCAT(match_score) as match_scores
            FROM {table_name}
            WHERE match_score >= ?
            GROUP BY paper_source, paper_source_id, paper_title
            ORDER BY {sort_field} {sort_direction}
        """, (min_match_score,)).fetchall()
    
    # Calculate median for each paper
    result = []
    for row in rows:
        paper_source, paper_source_id, paper_title, match_count, mean_match_score, max_match_score, avg_applicability_score, high_quality_count, match_scores_str = row
        
        # Calculate median from GROUP_CONCAT string
        median_match_score = None
        if match_scores_str:
            try:
                scores = [int(s) for s in match_scores_str.split(",")]
                median_match_score = statistics.median(scores)
            except (ValueError, statistics.StatisticsError):
                median_match_score = mean_match_score  # Fallback to mean
        
        result.append({
            "paper_source": paper_source,
            "paper_source_id": paper_source_id,
            "paper_title": paper_title,
            "match_count": match_count,
            "mean_match_score": round(mean_match_score or 0, 2),
            "median_match_score": round(median_match_score or 0, 2) if median_match_score is not None else None,
            "max_match_score": max_match_score or 0,
            "high_quality_count": high_quality_count or 0,
            "avg_applicability_score": round(avg_applicability_score or 0, 2),
        })
    
    return result


def get_paper_trial_matches_for_ui(
    model_preset: str,
    paper_source: str,
    paper_source_id: str,
    min_match_score: int = 1,
    min_applicability_score: int = 1,
    limit: int = None,
    offset: int = 0,
) -> list[dict]:
    """Get matches for a paper with UI-specific filtering and pagination.
    
    Extends get_paper_trial_matches_for_paper() with additional filtering and pagination.
    
    Args:
        model_preset: Model preset to query
        paper_source: Source of the paper
        paper_source_id: Source-specific ID
        min_match_score: Minimum match score to include (default: 1)
        min_applicability_score: Minimum applicability score to include (default: 1)
        limit: Maximum number of results (None = no limit)
        offset: Offset for pagination (default: 0)
    
    Returns:
        List of match dicts with all fields (same format as get_paper_trial_matches_for_paper())
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    # Build query - extend existing query structure
    conditions = ["paper_source = ?", "paper_source_id = ?", "match_score >= ?", "applicability_score >= ?"]
    params = [paper_source, paper_source_id, min_match_score, min_applicability_score]
    
    where_clause = " AND ".join(conditions)
    order_by = "ORDER BY match_score DESC, applicability_score DESC"
    
    limit_clause = ""
    if limit is not None:
        limit_clause = f"LIMIT {limit} OFFSET {offset}"
    
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT paper_source, paper_source_id, paper_title,
                   trial_nct_id, trial_title,
                   match_score, match_reasoning,
                   applicability_score, change_directive_quote, change_rationale_quote,
                   created_at, updated_at
            FROM {table_name}
            WHERE {where_clause}
            {order_by}
            {limit_clause}
        """, params).fetchall()
    
    return [
        {
            "paper_source": r[0],
            "paper_source_id": r[1],
            "paper_title": r[2],
            "trial_nct_id": r[3],
            "trial_title": r[4],
            "match_score": r[5],
            "match_reasoning": r[6],
            "applicability_score": r[7],
            "change_directive_quote": r[8],
            "change_rationale_quote": r[9],
            "created_at": r[10],
            "updated_at": r[11],
        }
        for r in rows
    ]


def get_papers_scores_batch(
    model_preset: str,
    papers: list[tuple[str, str]],  # List of (paper_source, paper_source_id) tuples
) -> dict[tuple[str, str], dict]:
    """Get paper scores for multiple papers in a single query.
    
    Args:
        model_preset: Model preset to query
        papers: List of (paper_source, paper_source_id) tuples
    
    Returns:
        Dict mapping (paper_source, paper_source_id) -> scores dict
        Scores dict contains: relevance, extraction_confidence, accrual_ease, total_score
        Papers not found in database are not included in the returned dict
    """
    if not papers:
        return {}
    
    _ensure_table(model_preset)
    table_name = _get_table_name(model_preset)
    
    # Build WHERE clause with OR conditions (SQLite doesn't support tuple IN)
    # WHERE (source=? AND source_id=?) OR (source=? AND source_id=?) ...
    where_conditions = []
    params = []
    for paper_source, paper_source_id in papers:
        where_conditions.append("(source=? AND source_id=?)")
        params.extend([paper_source, paper_source_id])
    
    where_clause = " OR ".join(where_conditions)
    
    with _get_conn() as conn:
        rows = conn.execute(f"""
            SELECT source, source_id, relevance, extraction_confidence, accrual_ease, total_score
            FROM {table_name}
            WHERE {where_clause}
        """, params).fetchall()
    
    # Build result dict
    result = {}
    for row in rows:
        paper_key = (row[0], row[1])
        result[paper_key] = {
            "relevance": row[2],
            "extraction_confidence": row[3],
            "accrual_ease": row[4],
            "total_score": row[5],
        }
    
    return result


def filter_papers_by_quality(
    papers: list[dict],
    paper_scores_map: dict[tuple[str, str], dict],
    min_relevance: int = 1,
    min_extraction_confidence: int = 1,
    min_accrual_ease: int = 1,
) -> list[dict]:
    """Filter papers by quality scores and add scores to paper dicts.
    
    Args:
        papers: List of paper dicts with paper_source and paper_source_id
        paper_scores_map: Dict mapping (paper_source, paper_source_id) -> scores dict
        min_relevance: Minimum relevance score
        min_extraction_confidence: Minimum extraction_confidence score
        min_accrual_ease: Minimum accrual_ease score
    
    Returns:
        Filtered list of papers with paper scores added (paper_relevance, paper_extraction_confidence, 
        paper_accrual_ease, paper_total_score)
    """
    filtered = []
    for paper in papers:
        paper_key = (paper["paper_source"], paper["paper_source_id"])
        paper_scores = paper_scores_map.get(paper_key)
        
        if paper_scores:
            if (paper_scores["relevance"] >= min_relevance and
                paper_scores["extraction_confidence"] >= min_extraction_confidence and
                paper_scores["accrual_ease"] >= min_accrual_ease):
                # Add paper scores to the paper dict for display
                paper["paper_relevance"] = paper_scores["relevance"]
                paper["paper_extraction_confidence"] = paper_scores["extraction_confidence"]
                paper["paper_accrual_ease"] = paper_scores["accrual_ease"]
                paper["paper_total_score"] = paper_scores["total_score"]
                filtered.append(paper)
    
    return filtered


def get_papers_with_matches_count(
    model_preset: str,
    min_match_score: int = 1,
) -> int:
    """Get count of unique papers that have matches.
    
    Args:
        model_preset: Model preset to query
        min_match_score: Minimum match score to include (default: 1)
    
    Returns:
        Count of unique papers with matches
    """
    _ensure_paper_trial_matches_table(model_preset)
    table_name = _get_paper_trial_matches_table_name(model_preset)
    
    with _get_conn() as conn:
        count = conn.execute(f"""
            SELECT COUNT(DISTINCT paper_source || paper_source_id)
            FROM {table_name}
            WHERE match_score >= ?
        """, (min_match_score,)).fetchone()[0]
    
    return count


def _merge_tables_from_archive(table_name_pattern: str, unique_key_fields: list[str], model_preset: str) -> list[dict]:
    """Generic function to merge all backup tables from archive.db for a given preset.
    
    This is a DRY function that handles both documents and matches tables.
    
    Args:
        table_name_pattern: Pattern to match table names (e.g., "documents_{preset}_archive_*")
        unique_key_fields: List of field names that form the unique key for deduplication
        model_preset: Model preset (used to construct pattern)
    
    Returns:
        List of merged items (dicts) with all fields, deduplicated by unique_key_fields
    """
    # Replace {preset} in pattern with actual preset (sanitized)
    sanitized_preset = model_preset.replace("-", "_")
    pattern = table_name_pattern.replace("{preset}", sanitized_preset)
    # Convert * to % for SQL LIKE pattern
    like_pattern = pattern.replace("*", "%")
    
    # Get all matching tables from archive.db
    all_items = []
    try:
        with _get_archive_conn() as archive_conn:
            # Get all tables matching the pattern
            cursor = archive_conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE ?
            """, (like_pattern,))
            archive_tables = [row[0] for row in cursor.fetchall()]
            
            if not archive_tables:
                logger.debug(f"No archive tables found matching pattern: {like_pattern}")
                return []
            
            # Sort by timestamp (most recent first) - tables end with _archive_YYYYMMDD_HHMMSS
            archive_tables.sort(reverse=True)
            
            # Collect all items from all archive tables
            for table_name in archive_tables:
                try:
                    # Get all columns dynamically (use parameterized query for table name)
                    # SQLite doesn't support parameterized table names, so we need to validate
                    # the table name is safe (alphanumeric, underscore, no SQL injection)
                    if not all(c.isalnum() or c == '_' for c in table_name):
                        logger.warning(f"Skipping suspicious table name: {table_name}")
                        continue
                    
                    cursor = archive_conn.execute(f'PRAGMA table_info("{table_name}")')
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    if not columns:
                        continue
                    
                    # Select all columns (quote table name for safety)
                    columns_str = ",".join([f'"{col}"' for col in columns])
                    cursor = archive_conn.execute(f'SELECT {columns_str} FROM "{table_name}"')
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        item = dict(zip(columns, row))
                        all_items.append(item)
                except sqlite3.OperationalError as e:
                    logger.warning(f"Skipping archive table {table_name} due to error: {e}")
                    continue
    except sqlite3.OperationalError:
        # Archive.db might not exist yet
        logger.debug("Archive.db does not exist or is not accessible")
        return []
    
    if not all_items:
        return []
    
    # Deduplicate by unique_key_fields, keeping most recent entry
    # Build a dict keyed by unique_key tuple
    seen = {}
    for item in all_items:
        # Build unique key tuple
        key_tuple = tuple(item.get(field) for field in unique_key_fields)
        
        # Skip if any key field is None
        if None in key_tuple:
            continue
        
        # Get timestamp for comparison (prefer updated_at, fallback to created_at)
        timestamp_str = item.get("updated_at") or item.get("created_at")
        if not timestamp_str:
            # If no timestamp, keep first occurrence
            if key_tuple not in seen:
                seen[key_tuple] = item
            continue
        
        # Compare with existing entry
        if key_tuple not in seen:
            seen[key_tuple] = item
        else:
            # Compare timestamps
            existing_timestamp = seen[key_tuple].get("updated_at") or seen[key_tuple].get("created_at")
            if existing_timestamp and timestamp_str > existing_timestamp:
                seen[key_tuple] = item
    
    return list(seen.values())


def merge_documents_from_archive(model_preset: str) -> list[dict]:
    """Merge all archived documents tables for a preset.
    
    Args:
        model_preset: Model preset to merge archives for
    
    Returns:
        List of merged document dicts, deduplicated by (source, source_id)
    """
    return _merge_tables_from_archive(
        "documents_{preset}_archive_*",
        ["source", "source_id"],
        model_preset
    )


def merge_matches_from_archive(model_preset: str) -> list[dict]:
    """Merge all archived matches tables for a preset.
    
    Args:
        model_preset: Model preset to merge archives for
    
    Returns:
        List of merged match dicts, deduplicated by (paper_source, paper_source_id, trial_nct_id)
    """
    return _merge_tables_from_archive(
        "paper_trial_matches_{preset}_archive_*",
        ["paper_source", "paper_source_id", "trial_nct_id"],
        model_preset
    )


def filter_completed_within_hours(items: list[dict], hours: float, table_type: str) -> tuple[list[dict], list[dict]]:
    """Filter items by completion status and time window.
    
    Args:
        items: List of item dicts (documents or matches)
        hours: Number of hours for time window (0 = process all)
        table_type: Either "documents" or "matches"
    
    Returns:
        Tuple of (completed_within_window, to_review):
        - completed_within_window: Items that are completed and within time window (skip re-review)
        - to_review: Items outside window or incomplete (need re-review)
    """
    if hours == 0:
        # Process everything
        return ([], items)
    
    # Define required fields for "completed" status
    if table_type == "documents":
        required_fields = ["relevance", "extraction_confidence", "accrual_ease", "reasoning"]
    elif table_type == "matches":
        required_fields = ["match_score", "match_reasoning", "applicability_score", 
                          "change_directive_quote", "change_rationale_quote"]
    else:
        raise ValueError(f"Invalid table_type: {table_type}. Must be 'documents' or 'matches'")
    
    # Calculate cutoff time
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    completed_within_window = []
    to_review = []
    
    for item in items:
        # Check if completed (all required fields present and not None)
        is_completed = all(
            item.get(field) is not None and item.get(field) != ""
            for field in required_fields
        )
        
        if not is_completed:
            # Incomplete entries always need review
            to_review.append(item)
            continue
        
        # Check timestamp (prefer updated_at, fallback to created_at)
        timestamp_str = item.get("updated_at") or item.get("created_at")
        if not timestamp_str:
            # No timestamp - needs review
            to_review.append(item)
            continue
        
        try:
            # Parse timestamp (SQLite format: YYYY-MM-DD HH:MM:SS)
            item_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            
            if item_time >= cutoff_time:
                # Within window and completed - skip re-review
                completed_within_window.append(item)
            else:
                # Outside window - needs review
                to_review.append(item)
        except (ValueError, TypeError):
            # Invalid timestamp - needs review
            logger.warning(f"Invalid timestamp format for item: {timestamp_str}")
            to_review.append(item)
    
    return (completed_within_window, to_review)
