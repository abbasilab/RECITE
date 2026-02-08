"""
Builders module for RECITE benchmark.

Handles building final RECITE benchmark instances.
"""

import sqlite3
from typing import Optional

from loguru import logger

from recite.benchmark.utils import clean_text


def create_recite_instances(
    max_trials: Optional[int],
    conn: sqlite3.Connection,
):
    """Create cleaned RECITE benchmark instances from ec_changes + protocol_texts."""
    cursor = conn.cursor()
    
    query = """
        SELECT ec.id, ec.instance_id, ec.source_version, ec.target_version,
               ec.ec_before, ec.ec_after, pt.raw_text AS protocol_raw_text
        FROM ec_changes ec
        LEFT JOIN protocol_texts pt ON ec.instance_id = pt.instance_id
        WHERE ec.evidence_source_path IS NOT NULL
    """
    if max_trials:
        query += f" LIMIT {max_trials}"
    
    changes = cursor.execute(query).fetchall()
    
    total = len(changes)
    logger.info(f"Creating RECITE instances from {total} EC changes")
    
    # Check if merge_source column exists (added by merge_data_dbs.py)
    recite_cols = [r[1] for r in cursor.execute("PRAGMA table_info(recite)").fetchall()]
    has_merge_source = "merge_source" in recite_cols
    
    # Check if id needs to be explicitly provided (merged DBs have composite PK)
    table_sql = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='recite'"
    ).fetchone()[0]
    needs_explicit_id = "AUTOINCREMENT" not in table_sql and "PRIMARY KEY (id, merge_source)" in table_sql
    
    # Get next id if needed
    next_id = 1
    if needs_explicit_id:
        max_id = cursor.execute("SELECT MAX(id) FROM recite WHERE merge_source = 'local'").fetchone()[0]
        next_id = (max_id or 0) + 1
    
    created = 0
    skipped = 0
    
    for i, change in enumerate(changes, 1):
        # Log progress every 100 changes or every 10% of total
        if i % 100 == 0 or (total > 1000 and i % (total // 10) == 0):
            logger.info(f"  Progress: {i}/{total} ({i*100//total}%), created: {created}, skipped: {skipped}")
        # Evidence from protocol_texts (raw text only)
        raw_pdf_text = change["protocol_raw_text"] if change["protocol_raw_text"] else ""
        if raw_pdf_text and len(raw_pdf_text.strip()) > 100:
            evidence = raw_pdf_text
            extraction_level = "raw_pdf_text_only"
            extraction_score = 1
        else:
            evidence = ""  # PDF exists but no text extracted yet or minimal
            extraction_level = "pdf_document_only"
            extraction_score = 0
        
        # Check if already exists
        existing = cursor.execute(
            """
            SELECT id FROM recite
            WHERE instance_id = ? AND source_version = ? AND target_version = ?
            """,
            (change["instance_id"], change["source_version"], change["target_version"]),
        ).fetchone()
        
        if not existing:
            # Clean text for benchmark
            source_text = clean_text(change["ec_before"])
            reference_text = clean_text(change["ec_after"])
            evidence_cleaned = clean_text(evidence) if evidence else ""
            
            if needs_explicit_id:
                cursor.execute(
                    """
                    INSERT INTO recite
                    (id, instance_id, source_version, target_version, source_text, evidence, reference_text, 
                     ec_change_id, evidence_extraction_level, evidence_extraction_score, merge_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        next_id,
                        change["instance_id"],
                        change["source_version"],
                        change["target_version"],
                        source_text,
                        evidence_cleaned,
                        reference_text,
                        change["id"],
                        extraction_level,
                        extraction_score,
                        "local",
                    ),
                )
                next_id += 1
            elif has_merge_source:
                cursor.execute(
                    """
                    INSERT INTO recite
                    (instance_id, source_version, target_version, source_text, evidence, reference_text, 
                     ec_change_id, evidence_extraction_level, evidence_extraction_score, merge_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        change["instance_id"],
                        change["source_version"],
                        change["target_version"],
                        source_text,
                        evidence_cleaned,
                        reference_text,
                        change["id"],
                        extraction_level,
                        extraction_score,
                        "local",
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO recite
                    (instance_id, source_version, target_version, source_text, evidence, reference_text, 
                     ec_change_id, evidence_extraction_level, evidence_extraction_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        change["instance_id"],
                        change["source_version"],
                        change["target_version"],
                        source_text,
                        evidence_cleaned,
                        reference_text,
                        change["id"],
                        extraction_level,
                        extraction_score,
                    ),
                )
            created += 1
        else:
            skipped += 1
        
        # Commit periodically (every 100 instances) to preserve progress
        if i % 100 == 0:
            conn.commit()
            logger.debug(f"  Committed {i} instances so far...")
    
    # Final commit
    conn.commit()
    logger.info(f"RECITE instances created: {created} created, {skipped} skipped")
