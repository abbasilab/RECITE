"""
Utility functions for RECITE benchmark.
"""

import re
import sqlite3
from typing import Any, List, Optional


def normalize_instance_id(instance_id: str) -> Optional[str]:
    """
    Normalize NCT ID format.
    
    Args:
        instance_id: NCT ID in various formats
        
    Returns:
        Normalized NCT ID (NCT########) or None if invalid
    """
    # Remove whitespace and convert to uppercase
    instance_id = instance_id.strip().upper()
    
    # Remove common prefixes/suffixes
    instance_id = re.sub(r"^NCT\s*", "", instance_id)
    instance_id = re.sub(r"\s+", "", instance_id)
    
    # Validate format (8 digits)
    if re.match(r"^\d{8}$", instance_id):
        return f"NCT{instance_id}"
    
    # If already in correct format
    if re.match(r"^NCT\d{8}$", instance_id):
        return instance_id
    
    return None


def parse_eligibility_sections(ec_text: str) -> dict:
    """
    Parse eligibility criteria into inclusion and exclusion sections.
    
    Args:
        ec_text: Full eligibility criteria text
        
    Returns:
        Dictionary with 'inclusion' and 'exclusion' keys
    """
    result = {
        "inclusion": "",
        "exclusion": "",
        "full_text": ec_text,
    }
    
    if not ec_text:
        return result
    
    # Try to split by common section headers
    text_lower = ec_text.lower()
    
    # Find inclusion section
    inclusion_patterns = [
        r"inclusion\s+criteria[:\s]*(.*?)(?=exclusion|$)",
        r"inclusion[:\s]*(.*?)(?=exclusion|$)",
    ]
    
    for pattern in inclusion_patterns:
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            result["inclusion"] = match.group(1).strip()
            break
    
    # Find exclusion section
    exclusion_patterns = [
        r"exclusion\s+criteria[:\s]*(.*?)(?=inclusion|$)",
        r"exclusion[:\s]*(.*?)(?=inclusion|$)",
    ]
    
    for pattern in exclusion_patterns:
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            result["exclusion"] = match.group(1).strip()
            break
    
    return result


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", "", text)
    
    # Normalize line breaks
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\r", "\n", text)
    
    # Remove multiple consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def extract_protocol_digits(instance_id: str) -> str:
    """
    Extract digits for protocol URL pattern.
    
    Args:
        instance_id: NCT ID
        
    Returns:
        First 4 digits (or available digits) for URL pattern
    """
    # Extract digits from NCT ID
    digits = re.sub(r"[^\d]", "", instance_id)
    
    # Return first 4 digits (or all if less than 4)
    return digits[:4] if len(digits) >= 4 else digits


def get_trials_with_versions(
    conn: sqlite3.Connection,
    max_trials: Optional[int] = None,
) -> List[str]:
    """
    Get list of NCT IDs that have multiple versions.
    
    Args:
        conn: Database connection
        max_trials: Maximum number of trials to return
        
    Returns:
        List of NCT IDs
    """
    cursor = conn.cursor()
    query = """
        SELECT instance_id FROM trials_with_versions
        WHERE versions_downloaded = 0
        ORDER BY checked_at DESC
    """
    if max_trials:
        query += f" LIMIT {max_trials}"
    
    rows = cursor.execute(query).fetchall()
    return [row["instance_id"] for row in rows]


def get_trials_with_ec_changes(
    conn: sqlite3.Connection,
    max_trials: Optional[int] = None,
) -> List[str]:
    """
    Get list of NCT IDs that have EC changes.
    
    Args:
        conn: Database connection
        max_trials: Maximum number of trials to return
        
    Returns:
        List of NCT IDs
    """
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT instance_id FROM ec_changes
        ORDER BY created_at DESC
    """
    if max_trials:
        query += f" LIMIT {max_trials}"
    
    rows = cursor.execute(query).fetchall()
    return [row["instance_id"] for row in rows]


def get_trials_with_protocols(
    conn: sqlite3.Connection,
    max_trials: Optional[int] = None,
) -> List[str]:
    """
    Get list of NCT IDs that have protocol PDFs.
    
    Args:
        conn: Database connection
        max_trials: Maximum number of trials to return
        
    Returns:
        List of NCT IDs
    """
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT instance_id FROM ec_changes
        WHERE evidence_source_path IS NOT NULL
        ORDER BY created_at DESC
    """
    if max_trials:
        query += f" LIMIT {max_trials}"
    
    rows = cursor.execute(query).fetchall()
    return [row["instance_id"] for row in rows]


def get_trials_ready_for_recite(
    conn: sqlite3.Connection,
    max_trials: Optional[int] = None,
) -> List[str]:
    """
    Get list of NCT IDs ready for RECITE instance creation.
    
    Args:
        conn: Database connection
        max_trials: Maximum number of trials to return
        
    Returns:
        List of NCT IDs
    """
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT instance_id FROM ec_changes
        WHERE evidence_source_path IS NOT NULL
        ORDER BY created_at DESC
    """
    if max_trials:
        query += f" LIMIT {max_trials}"
    
    rows = cursor.execute(query).fetchall()
    return [row["instance_id"] for row in rows]


def execute_batched_in_query(
    cursor: sqlite3.Cursor,
    base_query_template: str,
    items: List[str],
    batch_size: int = 250,
    fixed_params: Optional[List[Any]] = None,
) -> List[sqlite3.Row]:
    """
    Execute SQL query with batched IN clause to avoid SQLite variable limit.
    
    Follows DRY: Single implementation used by all modules.
    
    Args:
        cursor: Database cursor
        base_query_template: SQL query with {placeholders} for IN clause
                           Example: "SELECT * FROM table WHERE id IN ({placeholders})"
        items: List of items for IN clause
        batch_size: Maximum items per batch (default: 250)
        fixed_params: Additional fixed parameters (e.g., discovery_method)
        
    Returns:
        Combined results from all batches
    """
    if not items:
        return []
    
    all_results = []
    fixed_params = fixed_params or []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        
        # Replace {placeholders} in template
        query = base_query_template.format(placeholders=placeholders)
        
        # Combine fixed params with batch items
        params = fixed_params + batch
        
        # Execute query
        batch_results = cursor.execute(query, params).fetchall()
        all_results.extend(batch_results)
    
    return all_results


def should_skip_trial_version_check(
    cursor: sqlite3.Cursor,
    instance_id: str,
) -> bool:
    """
    Check if trial should be skipped (already checked).
    
    Uses optimized single query with LEFT JOIN.
    
    Args:
        cursor: Database cursor
        instance_id: NCT ID to check
        
    Returns:
        True if trial should be skipped (already checked)
    """
    # First check if trial exists in discovered_trials
    dt_result = cursor.execute(
        "SELECT version_count FROM discovered_trials WHERE instance_id = ?",
        (instance_id,),
    ).fetchone()
    
    # If not in discovered_trials, check if in trials_with_versions
    if not dt_result:
        twv_result = cursor.execute(
            "SELECT instance_id FROM trials_with_versions WHERE instance_id = ?",
            (instance_id,),
        ).fetchone()
        return twv_result is not None
    
    # Use optimized query with LEFT JOIN
    result = cursor.execute(
        """
        SELECT 
            dt.version_count,
            CASE WHEN twv.instance_id IS NOT NULL THEN 1 ELSE 0 END as in_trials_with_versions
        FROM discovered_trials dt
        LEFT JOIN trials_with_versions twv ON dt.instance_id = twv.instance_id
        WHERE dt.instance_id = ?
        """,
        (instance_id,),
    ).fetchone()
    
    if not result:
        return False
    
    # Skip if has version_count OR already in trials_with_versions
    should_skip = (
        result["version_count"] is not None
        or result["in_trials_with_versions"] == 1
    )
    
    return should_skip
