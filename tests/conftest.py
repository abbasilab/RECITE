"""Shared test fixtures for RECITE smoke tests."""

import json
import sqlite3
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def trial_record():
    """Load the synthetic trial record fixture."""
    with open(FIXTURES_DIR / "trial_record.json") as f:
        return json.load(f)


@pytest.fixture
def paper_abstract():
    """Load the synthetic paper abstract fixture."""
    with open(FIXTURES_DIR / "paper_abstract.json") as f:
        return json.load(f)


@pytest.fixture
def protocol_text():
    """Load the synthetic protocol amendment text fixture."""
    return (FIXTURES_DIR / "protocol_amendment.txt").read_text()


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database with RECITE schema."""
    db_path = tmp_path / "test_recite.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ec_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nct_id TEXT NOT NULL,
            version_from INTEGER,
            version_to INTEGER,
            ec_before TEXT,
            ec_after TEXT,
            change_type TEXT,
            similarity REAL,
            change_ratio REAL,
            quality_score REAL,
            evidence_source_path TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS protocol_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nct_id TEXT NOT NULL,
            raw_text TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recite (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nct_id TEXT NOT NULL,
            version_from INTEGER,
            version_to INTEGER,
            preamended_text TEXT,
            evidence TEXT,
            amended_text TEXT,
            ec_change_id INTEGER,
            evidence_extraction_level TEXT,
            evidence_extraction_score INTEGER
        )
    """)
    conn.commit()
    yield conn
    conn.close()
