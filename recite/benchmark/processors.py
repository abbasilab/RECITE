"""
Processors module for RECITE benchmark.

Handles EC change detection and evidence extraction.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from recite.benchmark.ec_detector import detect_ec_changes
from recite.benchmark.protocol_parser import (
    extract_amendment_table,
    extract_ec_changes_from_amendment,
    extract_pdf_version_info,
    extract_protocol_sections,
    filter_amendments_by_version,
    filter_raw_pdf_text_by_version,
    match_ec_to_amendment,
)


def _parse_date(date_str: str) -> Optional[Any]:
    """Parse various date formats."""
    from datetime import datetime
    
    if not date_str:
        return None
    formats = [
        "%d %B %Y",  # "30 March 2021"
        "%d %b %Y",  # "30 Mar 2021"
        "%B %d, %Y",  # "March 30, 2021"
        "%b %d, %Y",  # "Mar 30, 2021"
        "%Y-%m-%d",  # "2021-03-30"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def validate_pdf_version(
    pdf_version_info: Dict[str, Any],
    target_version: int,
    target_version_date: Optional[str],
    date_tolerance_days: int = 7,
) -> tuple[bool, str]:
    """
    Validate that PDF version/amendment matches or precedes the EC change version.
    
    Args:
        pdf_version_info: Version info from extract_pdf_version_info()
        target_version: Target version number from EC change
        target_version_date: Target version date from EC change
        date_tolerance_days: Tolerance for date lag (default: 7 days)
        
    Returns:
        Tuple of (is_valid, match_type)
        - is_valid: True if PDF version ≤ target_version (no data leakage)
        - match_type: How the match was determined
    """
    if not pdf_version_info:
        return False, "no_version_info"
    
    from datetime import timedelta
    
    # Strategy 1: Compare dates (PRIMARY - most reliable)
    # Match specific amendment date to target_version_date (with tolerance)
    pdf_date = pdf_version_info.get("pdf_date")
    amendment_dates = pdf_version_info.get("amendment_dates", [])
    
    if target_version_date and amendment_dates:
        version_date = _parse_date(target_version_date)
        if version_date:
            tolerance = timedelta(days=date_tolerance_days)
            # Check if any amendment date matches or is before target_version_date (with tolerance)
            for amendment_info in amendment_dates:
                amendment_date_str = amendment_info.get("date")
                if amendment_date_str:
                    amendment_date = _parse_date(amendment_date_str)
                    if amendment_date:
                        amendment_num = amendment_info.get("amendment")
                        # If amendment date <= target_version_date + tolerance, it's valid
                        # (allows small lag where PDF uploaded slightly after version)
                        if amendment_date <= (version_date + tolerance):
                            # Also check if this amendment number <= target_version (double check)
                            if amendment_num and amendment_num <= target_version:
                                return True, "date_based"
                        # If amendment date > target_version_date + tolerance, potential leakage
                        elif amendment_date > (version_date + tolerance):
                            logger.warning(
                                f"Amendment {amendment_num} date {amendment_date_str} > target_version_date {target_version_date} "
                                f"(exceeds {date_tolerance_days}-day tolerance)"
                            )
    
    # Fallback: Compare PDF date to target_version_date (with tolerance)
    if pdf_date and target_version_date:
        pdf_dt = _parse_date(pdf_date)
        version_dt = _parse_date(target_version_date)
        if pdf_dt and version_dt:
            tolerance = timedelta(days=date_tolerance_days)
            # Allow PDF date to be slightly after version date (within tolerance)
            if pdf_dt <= (version_dt + tolerance):
                return True, "date_based_pdf"
            else:
                logger.warning(
                    f"PDF date {pdf_date} > target_version_date {target_version_date} "
                    f"(exceeds {date_tolerance_days}-day tolerance) - potential data leakage"
                )
                return False, "date_mismatch"
    
    # Strategy 2: Compare amendment number to version number (FALLBACK)
    # Note: This assumes Amendment N ≈ Version N, but may not always be true
    # Use with caution - date-based matching is preferred
    max_amendment = pdf_version_info.get("max_amendment")
    if max_amendment is not None:
        if max_amendment <= target_version:
            return True, "amendment_number"
        else:
            logger.warning(
                f"PDF amendment {max_amendment} > target_version {target_version} - potential data leakage"
            )
            return False, "amendment_number_mismatch"
    
    # Strategy 3: Explicit version string (FALLBACK)
    explicit_version = pdf_version_info.get("explicit_version")
    if explicit_version:
        try:
            pdf_version_num = float(explicit_version)
            if pdf_version_num <= target_version:
                return True, "explicit_version"
        except (ValueError, TypeError):
            pass
    
    # If we can't determine, be conservative and flag as unvalidated
    return False, "unknown"


def identify_amendments(
    max_trials: Optional[int],
    conn: sqlite3.Connection,
):
    """Identify trials with EC amendments and store in ec_changes table."""
    cursor = conn.cursor()
    
    # Get all trials with multiple versions
    query = """
        SELECT instance_id, COUNT(*) as version_count
        FROM trial_versions
        GROUP BY instance_id
        HAVING version_count > 1
        ORDER BY instance_id
    """
    
    if max_trials:
        query += f" LIMIT {max_trials}"
    
    trials = cursor.execute(query).fetchall()
    
    total = len(trials)
    logger.info(f"Processing {total} trials for EC amendments")
    
    changes_found = 0
    trials_processed = 0
    
    for i, trial in enumerate(trials, 1):
        # Log progress every 10 trials or every 10% of total
        if i % 10 == 0 or (total > 100 and i % (total // 10) == 0):
            logger.info(f"  Progress: {i}/{total} ({i*100//total}%), processed: {trials_processed}, changes found: {changes_found}")
        else:
            logger.debug(f"  Processing {trial['instance_id']} ({i}/{total})")
        instance_id = trial["instance_id"]
        
        # Get all versions for this trial
        versions = cursor.execute(
            """
            SELECT version_number, version_date, eligibility_criteria
            FROM trial_versions
            WHERE instance_id = ?
            ORDER BY version_number
            """,
            (instance_id,),
        ).fetchall()
        
        # Compare consecutive versions
        for i in range(len(versions) - 1):
            v1 = versions[i]
            v2 = versions[i + 1]
            
            ec1 = v1["eligibility_criteria"] or ""
            ec2 = v2["eligibility_criteria"] or ""
            
            # Detect changes
            change_result = detect_ec_changes(ec1, ec2)
            
            if change_result["has_change"]:
                # Check if already exists
                existing = cursor.execute(
                    """
                    SELECT id FROM ec_changes
                    WHERE instance_id = ? AND source_version = ? AND target_version = ?
                    """,
                    (instance_id, v1["version_number"], v2["version_number"]),
                ).fetchone()
                
                if not existing:
                    cursor.execute(
                        """
                        INSERT INTO ec_changes
                        (instance_id, source_version, target_version, source_version_date, target_version_date,
                         ec_before, ec_after, ec_change_type, quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            instance_id,
                            v1["version_number"],
                            v2["version_number"],
                            v1["version_date"],
                            v2["version_date"],
                            ec1,
                            ec2,
                            change_result["change_type"],
                            change_result["similarity"],
                        ),
                    )
                    changes_found += 1
                    logger.debug(
                        f"  Found EC change: {instance_id} v{v1['version_number']} -> v{v2['version_number']}"
                    )
        
        trials_processed += 1
        conn.commit()
    
    logger.info(f"EC amendment detection complete: {trials_processed} trials processed, {changes_found} changes found")


def _extract_pdf_date(pdf_path: Path) -> Optional[str]:
    """Try to get PDF date from metadata or return None."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata or {}
        doc.close()
        return meta.get("creationDate") or meta.get("modDate")
    except Exception:
        return None


def extract_protocol_texts(
    conn: sqlite3.Connection,
    instance_ids: Optional[List[str]] = None,
    max_trials: Optional[int] = None,
) -> int:
    """Extract raw text from protocol PDFs into protocol_texts table (one row per trial).

    Only trials that have evidence_source_path set and are not yet in protocol_texts are processed.
    Returns number of trials inserted into protocol_texts.
    """
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT instance_id, evidence_source_path
        FROM ec_changes
        WHERE evidence_source = 'protocol_pdf'
        AND evidence_source_path IS NOT NULL
        AND instance_id NOT IN (SELECT instance_id FROM protocol_texts)
    """
    params: List[object] = []
    if instance_ids is not None and len(instance_ids) > 0:
        placeholders = ",".join("?" * len(instance_ids))
        query += f" AND instance_id IN ({placeholders})"
        params.extend(instance_ids)
    if max_trials:
        query += f" LIMIT {max_trials}"

    trials = cursor.execute(query, params).fetchall()
    total = len(trials)
    if not total:
        logger.info("No protocol PDFs to extract (all trials already in protocol_texts)")
        return 0

    logger.info(f"Extracting raw text from {total} protocol PDFs into protocol_texts")
    extracted = 0
    failed = 0
    for i, row in enumerate(trials, 1):
        instance_id = row["instance_id"]
        pdf_path = Path(row["evidence_source_path"])
        if i % 10 == 0 or (total > 100 and i % (total // 10) == 0):
            logger.info(f"  Progress: {i}/{total} ({i*100//total}%), extracted: {extracted}, failed: {failed}")
        if not pdf_path.exists():
            logger.warning(f"Protocol PDF not found: {pdf_path}")
            failed += 1
            continue
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            raw_text = "\n".join([page.get_text() for page in doc])
            page_count = len(doc)
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to extract PDF text from {pdf_path}: {e}")
            failed += 1
            continue
        pdf_date = _extract_pdf_date(pdf_path)
        cursor.execute(
            """
            INSERT OR REPLACE INTO protocol_texts (instance_id, pdf_path, raw_text, pdf_date, page_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (instance_id, str(pdf_path), raw_text, pdf_date, page_count),
        )
        extracted += 1
    conn.commit()
    logger.info(f"Protocol text extraction complete: {extracted} inserted, {failed} failed")
    return extracted


def extract_evidence(
    max_trials: Optional[int],
    conn: sqlite3.Connection,
    instance_ids: Optional[List[str]] = None,
    ec_change_ids: Optional[List[int]] = None,
):
    """Extract raw protocol text into protocol_texts table (one row per trial).

    Optional instance_ids or ec_change_ids restrict to a subset. When both are None,
    processes all trials with protocol PDFs not yet in protocol_texts.
    """
    # ec_change_ids: resolve to instance_ids for protocol_texts (we key by trial)
    if ec_change_ids is not None and len(ec_change_ids) > 0:
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(ec_change_ids))
        rows = cursor.execute(
            f"SELECT DISTINCT instance_id FROM ec_changes WHERE id IN ({placeholders})",
            ec_change_ids,
        ).fetchall()
        instance_ids = [r["instance_id"] for r in rows]
    extract_protocol_texts(conn, instance_ids=instance_ids, max_trials=max_trials)
