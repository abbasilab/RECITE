"""Database schema and helpers for RECITE benchmark."""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


# Default database path
def _default_recite_db_path() -> Path:
    from recite.utils.path_loader import get_local_db_dir
    return get_local_db_dir() / "recite.db"


def get_db_path() -> Path:
    """
    Get default database path, creating data directory if needed.
    
    Returns:
        Path to default database (LOCAL_DB_DIR/recite.db, default data/dev)
    """
    db_path = _default_recite_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def backup_database(db_path: Optional[Path] = None) -> Optional[Path]:
    """
    Backup existing database to data/legacy/ directory.
    
    Args:
        db_path: Path to database file (default: LOCAL_DB_DIR/recite.db)
        
    Returns:
        Path to backup file, or None if no database existed
    """
    if db_path is None:
        db_path = _default_recite_db_path()

    if not db_path.exists():
        logger.info(f"No existing database at {db_path} to backup")
        return None
    
    # Create legacy directory
    legacy_dir = db_path.parent / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = legacy_dir / f"{db_path.stem}_{timestamp}.db"
    
    # Copy database
    logger.info(f"Backing up database: {db_path} -> {backup_path}")
    shutil.copy2(db_path, backup_path)
    logger.info(f"Database backed up to {backup_path}")
    
    return backup_path


def init_database(db_path: Optional[Path] = None, force: bool = False) -> sqlite3.Connection:
    """
    Initialize RECITE database with schema.
    
    Creates tables:
    1. discovered_trials - All discovered NCT IDs
    2. trial_metadata - Trial metadata (year, conditions, phases, etc.)
    3. trials_with_versions - Trials with multiple versions
    4. trial_versions - Raw version data with ECs
    5. ec_changes - Identified EC changes with evidence
    6. recite - Cleaned RECITE benchmark format
    
    Args:
        db_path: Path to database file (default: LOCAL_DB_DIR/recite.db)
        force: If True, backup existing database and create fresh one
        
    Returns:
        Database connection
    """
    if db_path is None:
        db_path = _default_recite_db_path()
    else:
        # Ensure parent directory exists even if path is provided
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing database if force flag is set
    if force and db_path.exists():
        backup_database(db_path)
        logger.info(f"Removing existing database: {db_path}")
        db_path.unlink()
    
    logger.info(f"Initializing RECITE database at {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    
    # Create tables
    _create_discovered_trials_table(conn)
    _create_trial_metadata_table(conn)
    _create_trials_with_versions_table(conn)
    _create_trial_versions_table(conn)
    _create_protocol_texts_table(conn)
    _create_ec_changes_table(conn)
    _create_recite_table(conn)
    _create_strict_view(conn)
    
    conn.commit()
    logger.info("Database schema initialized")
    
    return conn


def _create_discovered_trials_table(conn: sqlite3.Connection):
    """Create discovered_trials table for tracking all discovered NCT IDs."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS discovered_trials (
            instance_id TEXT PRIMARY KEY,
            discovery_method TEXT,
            discovered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            version_count INTEGER,
            version_check_at TEXT
        )
        """
    )
    
    # Create index
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_discovered_trials_method 
        ON discovered_trials(discovery_method)
        """
    )


def _create_trial_metadata_table(conn: sqlite3.Connection):
    """Create trial_metadata table for trial-level metadata from XML."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trial_metadata (
            instance_id TEXT PRIMARY KEY,
            year INTEGER,
            conditions TEXT,
            keywords TEXT,
            phases TEXT,
            locations TEXT,
            study_type TEXT,
            enrollment INTEGER,
            enrollment_type TEXT,
            brief_title TEXT,
            start_date TEXT,
            overall_status TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (instance_id) REFERENCES discovered_trials(instance_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trial_metadata_year
        ON trial_metadata(year)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trial_metadata_study_type
        ON trial_metadata(study_type)
        """
    )


def ensure_trial_metadata_columns(conn: sqlite3.Connection) -> None:
    """Add brief_title and enrollment_type to trial_metadata if missing (e.g. existing DBs)."""
    try:
        rows = conn.execute("PRAGMA table_info(trial_metadata)").fetchall()
        names = [r[1] for r in rows]
    except sqlite3.OperationalError:
        return
    for col, typ in [("brief_title", "TEXT"), ("enrollment_type", "TEXT")]:
        if col not in names:
            conn.execute(f"ALTER TABLE trial_metadata ADD COLUMN {col} {typ}")
            conn.commit()
            logger.info("Added trial_metadata.%s", col)


def _create_trials_with_versions_table(conn: sqlite3.Connection):
    """Create trials_with_versions table for tracking trials with multiple versions."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trials_with_versions (
            instance_id TEXT PRIMARY KEY,
            version_count INTEGER NOT NULL,
            checked_at TEXT DEFAULT CURRENT_TIMESTAMP,
            versions_downloaded BOOLEAN DEFAULT 0,
            FOREIGN KEY (instance_id) REFERENCES discovered_trials(instance_id)
        )
        """
    )
    
    # Add new columns if they don't exist (for existing databases)
    new_columns = [
        ("has_eligibility_changes", "BOOLEAN DEFAULT 0"),
        ("eligibility_version_numbers", "TEXT"),
        ("module_labels_json", "TEXT"),
    ]
    
    for column_name, column_type in new_columns:
        try:
            conn.execute(f"ALTER TABLE trials_with_versions ADD COLUMN {column_name} {column_type}")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
    
    # Create index
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trials_with_versions_downloaded 
        ON trials_with_versions(versions_downloaded)
        """
    )


def _create_trial_versions_table(conn: sqlite3.Connection):
    """Create trial_versions table."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trial_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            version_date TEXT,
            overall_status TEXT,
            eligibility_criteria TEXT,
            raw_data_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(instance_id, version_number)
        )
        """
    )
    
    # Add new column if it doesn't exist (for existing databases)
    try:
        conn.execute("ALTER TABLE trial_versions ADD COLUMN module_labels TEXT")
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass
    
    # Create index for faster lookups
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trial_versions_instance_id 
        ON trial_versions(instance_id)
        """
    )


def _create_protocol_texts_table(conn: sqlite3.Connection):
    """Create protocol_texts table: one row per trial with raw PDF text (no duplication per EC change)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS protocol_texts (
            instance_id TEXT PRIMARY KEY,
            pdf_path TEXT,
            raw_text TEXT,
            pdf_date TEXT,
            page_count INTEGER,
            extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_protocol_texts_instance_id
        ON protocol_texts(instance_id)
        """
    )


def _create_ec_changes_table(conn: sqlite3.Connection):
    """Create ec_changes table. Protocol text is in protocol_texts (keyed by instance_id)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ec_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            source_version INTEGER NOT NULL,
            target_version INTEGER NOT NULL,
            source_version_date TEXT,
            target_version_date TEXT,
            ec_before TEXT NOT NULL,
            ec_after TEXT NOT NULL,
            ec_change_type TEXT,
            evidence_source TEXT,
            evidence_source_path TEXT,
            quality_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(instance_id, source_version, target_version)
        )
        """
    )
    
    # Create indexes
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ec_changes_instance_id 
        ON ec_changes(instance_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ec_changes_quality 
        ON ec_changes(quality_score)
        """
    )


def _create_recite_table(conn: sqlite3.Connection):
    """Create recite table for cleaned benchmark instances."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recite (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            source_version INTEGER NOT NULL,
            target_version INTEGER NOT NULL,
            source_text TEXT NOT NULL,
            evidence TEXT NOT NULL,
            reference_text TEXT NOT NULL,
            ec_change_id INTEGER,
            quality_score REAL,
            validated BOOLEAN DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ec_change_id) REFERENCES ec_changes(id)
        )
        """
    )
    
    # Create indexes
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_recite_instance_id 
        ON recite(instance_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_recite_validated 
        ON recite(validated)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_recite_quality 
        ON recite(quality_score)
        """
    )
    
    # Add new columns if they don't exist (for existing databases)
    new_columns = [
        ("evidence_extraction_level", "TEXT"),
        ("evidence_extraction_score", "INTEGER"),
    ]
    
    for column_name, column_type in new_columns:
        try:
            conn.execute(f"ALTER TABLE recite ADD COLUMN {column_name} {column_type}")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass


def _create_strict_view(conn: sqlite3.Connection):
    """Create recite_strict view for high-confidence samples. Drops existing view so definition stays current after schema changes."""
    conn.execute("DROP VIEW IF EXISTS recite_strict")
    conn.execute(
        """
        CREATE VIEW recite_strict AS
        SELECT * FROM recite
        WHERE evidence_extraction_level = 'parsed_ec_justification'
        AND evidence_extraction_score = 3
        AND quality_score IS NOT NULL
        ORDER BY quality_score DESC, evidence_extraction_score DESC
        """
    )


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get database connection, initializing if needed.
    
    Args:
        db_path: Path to database file (default: LOCAL_DB_DIR/recite.db)
        
    Returns:
        Database connection
    """
    if db_path is None:
        db_path = _default_recite_db_path()
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Ensure parent directory exists even if path is provided
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize if database doesn't exist
    if not db_path.exists():
        return init_database(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Ensure recite_strict view matches current schema (e.g. after evidence_* column removal)
    _create_strict_view(conn)
    return conn
