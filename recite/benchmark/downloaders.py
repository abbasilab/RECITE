"""Download trial versions and protocol PDFs."""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from loguru import logger

from recite.benchmark.api_client import (
    extract_eligibility_criteria,
    fetch_version_data,
    fetch_version_history,
)
from recite.crawler.adapters import ClinicalTrialsGovAdapter


def download_versions(
    instance_ids: Optional[List[str]],
    max_trials: Optional[int],
    conn: sqlite3.Connection,
    use_expedited: bool = True,
):
    """
    Download trial versions and store in database.
    
    If instance_ids is None, queries trials_with_versions table for candidates.
    
    Args:
        instance_ids: Optional list of specific NCT IDs to process
        max_trials: Maximum number of trials to process
        conn: Database connection
        use_expedited: If True, only download versions with Eligibility in moduleLabels
    """
    import json
    
    from recite.benchmark.module_labels import get_versions_to_download
    
    adapter = ClinicalTrialsGovAdapter()
    cursor = conn.cursor()
    
    # Get NCT IDs to process and eligibility data
    eligibility_data = {}
    
    if instance_ids is None:
        # Query from filter table
        query = """
            SELECT instance_id, eligibility_version_numbers FROM trials_with_versions
            WHERE versions_downloaded = 0
            ORDER BY checked_at DESC
        """
        if max_trials:
            query += f" LIMIT {max_trials}"
        rows = cursor.execute(query).fetchall()
        instance_ids = [row["instance_id"] for row in rows]
        # Store eligibility versions for each trial
        eligibility_data = {row["instance_id"]: row["eligibility_version_numbers"] for row in rows}
    else:
        if max_trials:
            instance_ids = instance_ids[:max_trials]
        
        # Query eligibility versions for provided NCT IDs
        # Batch queries to avoid SQLite's variable limit (typically 999)
        if instance_ids:
            batch_size = 500  # Safe batch size well below SQLite's limit
            for i in range(0, len(instance_ids), batch_size):
                batch_instance_ids = instance_ids[i:i + batch_size]
                placeholders = ",".join("?" * len(batch_instance_ids))
                batch_rows = cursor.execute(
                    f"SELECT instance_id, eligibility_version_numbers FROM trials_with_versions WHERE instance_id IN ({placeholders})",
                    batch_instance_ids,
                ).fetchall()
                for row in batch_rows:
                    eligibility_data[row["instance_id"]] = row["eligibility_version_numbers"]
    
    total = len(instance_ids)
    logger.info(f"Downloading versions for {total} trials")
    logger.info(f"Expedited mode: {use_expedited}")
    
    downloaded = 0
    skipped = 0
    failed = 0
    versions_saved = 0  # Track total versions saved by expedited mode
    pending_updates = []  # Track pending updates for batched commits
    commit_batch_size = 250  # Commit every 250 trials (matching SQL batch size)
    
    for i, instance_id in enumerate(instance_ids, 1):
        # Log progress every 10 trials or every 10% of total
        if i % 10 == 0 or (total > 100 and i % (total // 10) == 0):
            logger.info(f"  Progress: {i}/{total} ({i*100//total}%), downloaded: {downloaded}, skipped: {skipped}, failed: {failed}")
        else:
            logger.debug(f"Processing {instance_id} ({i}/{total})")
        
        # Check if already downloaded
        existing = cursor.execute(
            "SELECT COUNT(*) FROM trial_versions WHERE instance_id = ?", (instance_id,)
        ).fetchone()[0]
        
        if existing > 0:
            skipped += 1
            logger.debug(f"  Skipping {instance_id} (already in database)")
            # Track pending update (don't commit yet)
            pending_updates.append(instance_id)
            
            # Batch commit every commit_batch_size trials
            if len(pending_updates) >= commit_batch_size:
                for pending_instance_id in pending_updates:
                    cursor.execute(
                        "UPDATE trials_with_versions SET versions_downloaded = 1 WHERE instance_id = ?",
                        (pending_instance_id,),
                    )
                conn.commit()
                logger.debug(f"  Batched commit: updated {len(pending_updates)} trials")
                pending_updates.clear()
            continue
        
        # Fetch version history
        history_data = fetch_version_history(instance_id, adapter=adapter)
        
        if not history_data:
            failed += 1
            logger.warning(f"  Failed to fetch history for {instance_id}")
            continue
        
        # Extract version list
        all_versions = _extract_versions_from_history(history_data)
        
        # Determine which versions to download
        if use_expedited:
            # Get eligibility versions from database
            eligibility_versions_json = eligibility_data.get(instance_id)
            if eligibility_versions_json:
                try:
                    eligibility_versions = json.loads(eligibility_versions_json)
                    if eligibility_versions:  # Only filter if we have eligibility versions
                        versions_to_download_nums = get_versions_to_download(eligibility_versions)
                        # Filter versions list to only include those we need
                        versions = [
                            v for v in all_versions 
                            if v["version_number"] in versions_to_download_nums
                        ]
                        skipped_versions = len(all_versions) - len(versions)
                        versions_saved += skipped_versions
                        logger.info(f"Downloading {len(versions)} versions for {instance_id} (expedited mode, skipped {skipped_versions})")
                        logger.debug(f"  Eligibility versions: {eligibility_versions}")
                        logger.debug(f"  Versions to download: {versions_to_download_nums}")
                    else:
                        # Empty eligibility list, fallback to all versions
                        logger.warning(f"  Empty eligibility_version_numbers for {instance_id}, downloading all versions (fallback)")
                        versions = all_versions
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"  Failed to parse eligibility_version_numbers for {instance_id}: {e}, downloading all versions (fallback)")
                    versions = all_versions
            else:
                # No eligibility data, fallback to downloading all
                logger.warning(f"  No eligibility_version_numbers for {instance_id}, downloading all versions (fallback)")
                versions = all_versions
        else:
            # Download all versions
            versions = all_versions
        
        # Download each version
        for version_info in versions:
            version_num = version_info.get("version_number", 0)
            version_date = version_info.get("version_date")
            status = version_info.get("overall_status")
            
            # Fetch version data
            version_data = fetch_version_data(instance_id, version_num, adapter=adapter)
            
            if not version_data:
                logger.warning(f"  Failed to fetch version {version_num} for {instance_id}")
                continue
            
            # Extract eligibility criteria
            ec = extract_eligibility_criteria(version_data)
            
            # Store moduleLabels if available
            module_labels = version_info.get("module_labels", [])
            module_labels_json = json.dumps(module_labels) if module_labels else None
            
            # Store in database
            cursor.execute(
                """
                INSERT OR REPLACE INTO trial_versions 
                (instance_id, version_number, version_date, overall_status, eligibility_criteria, raw_data_json, module_labels)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    instance_id,
                    version_num,
                    version_date,
                    status,
                    ec,
                    json.dumps(version_data),
                    module_labels_json,
                ),
            )
        
        downloaded += 1
        logger.debug(f"  Stored {len(versions)} versions for {instance_id}")
        
        # Update versions_downloaded flag
        cursor.execute(
            "UPDATE trials_with_versions SET versions_downloaded = 1 WHERE instance_id = ?",
            (instance_id,),
        )
        
        # Batch commit every commit_batch_size trials
        if downloaded % commit_batch_size == 0:
            conn.commit()
            logger.debug(f"  Batched commit: saved {commit_batch_size} trials")
    
    # Final commit for any remaining pending updates
    if pending_updates:
        for pending_instance_id in pending_updates:
            cursor.execute(
                "UPDATE trials_with_versions SET versions_downloaded = 1 WHERE instance_id = ?",
                (pending_instance_id,),
            )
        conn.commit()
        logger.debug(f"  Final batched commit: updated {len(pending_updates)} skipped trials")
    
    # Final commit for any remaining downloaded trials
    if downloaded % commit_batch_size != 0:
        conn.commit()
    
    # Log performance metrics
    if use_expedited and versions_saved > 0:
        logger.info(f"Expedited mode: Saved {versions_saved:,} version downloads across {downloaded} trials")


def download_ecs(
    instance_ids: List[str],
    max_trials: Optional[int],
    conn: sqlite3.Connection,
):
    """Download eligibility criteria for trials."""
    # This is essentially the same as download_versions but focused on EC
    # For now, EC is extracted as part of download_versions
    download_versions(instance_ids, max_trials, conn)


def check_protocol_availability(instance_id: str, adapter: ClinicalTrialsGovAdapter) -> dict:
    """
    Check if a trial has a protocol document via API metadata (no download).
    
    Returns dict with:
        - has_protocol: bool
        - protocol_info: dict with filename, date, size if available
    """
    import requests
    
    url = f"https://clinicaltrials.gov/api/v2/studies/{instance_id}"
    try:
        adapter._rate_limit()
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return {"has_protocol": False, "error": f"HTTP {resp.status_code}"}
        
        data = resp.json()
        doc_section = data.get("documentSection", {})
        large_doc_module = doc_section.get("largeDocumentModule", {})
        large_docs = large_doc_module.get("largeDocs", [])
        
        # Find protocol documents
        for doc in large_docs:
            if doc.get("hasProtocol"):
                return {
                    "has_protocol": True,
                    "protocol_info": {
                        "filename": doc.get("filename"),
                        "date": doc.get("date"),
                        "size": doc.get("size"),
                        "label": doc.get("label"),
                    },
                }
        
        return {"has_protocol": False}
    except Exception as e:
        return {"has_protocol": False, "error": str(e)}


def download_protocols(
    instance_ids: Optional[List[str]],
    max_trials: Optional[int],
    conn: sqlite3.Connection,
    output_dir: Path = Path("data/benchmarks/protocols"),
):
    """
    Check and download protocol PDFs for trials.
    
    First checks API metadata to see if protocol exists, then downloads only if available.
    Records result in ec_changes.evidence_source:
        - 'protocol_pdf' with path if downloaded
        - 'no_protocol_available' if API says no protocol
    
    If instance_ids is None, queries ec_changes table for trials that need checking.
    """
    from recite.benchmark.evidence_downloader import find_protocol_pdfs
    
    adapter = ClinicalTrialsGovAdapter()
    cursor = conn.cursor()
    
    # Get trials that need protocol check
    if instance_ids:
        # Process specific NCT IDs
        trials_to_process = instance_ids[:max_trials] if max_trials else instance_ids
    else:
        # Get from database - trials with EC changes not yet checked
        query = """
            SELECT DISTINCT instance_id
            FROM ec_changes
            WHERE evidence_source IS NULL
        """
        if max_trials:
            query += f" LIMIT {max_trials}"
        trials = cursor.execute(query).fetchall()
        trials_to_process = [trial["instance_id"] for trial in trials]
    
    total = len(trials_to_process)
    logger.info(f"Checking/downloading protocols for {total} trials")
    
    checked = 0
    has_protocol = 0
    downloaded = 0
    no_protocol = 0
    
    for i, instance_id in enumerate(trials_to_process, 1):
        # Log progress every 10 trials or every 10% of total
        if i % 10 == 0 or (total > 100 and i % (total // 10) == 0):
            logger.info(f"  Progress: {i}/{total} ({i*100//total}%), has_protocol: {has_protocol}, downloaded: {downloaded}, no_protocol: {no_protocol}")
        else:
            logger.debug(f"  Checking protocol for {instance_id} ({i}/{total})")
        
        # First check API metadata for protocol availability
        availability = check_protocol_availability(instance_id, adapter)
        checked += 1
        
        if not availability.get("has_protocol"):
            # No protocol available - record and skip download
            no_protocol += 1
            cursor.execute(
                """
                UPDATE ec_changes
                SET evidence_source = 'no_protocol_available'
                WHERE instance_id = ?
                AND evidence_source IS NULL
                """,
                (instance_id,),
            )
            conn.commit()
            logger.debug(f"    No protocol available for {instance_id}")
            continue
        
        # Protocol exists - try to download
        has_protocol += 1
        logger.debug(f"    Protocol exists for {instance_id}, downloading...")
        
        protocol_result = find_protocol_pdfs(instance_id, adapter, output_dir, protocol_info=availability.get("protocol_info"))
        
        if protocol_result.get("success"):
            downloaded += 1
            cursor.execute(
                """
                UPDATE ec_changes
                SET evidence_source = 'protocol_pdf',
                    evidence_source_path = ?
                WHERE instance_id = ? 
                AND (evidence_source_path IS NULL OR evidence_source_path = '')
                AND (evidence_source IS NULL OR evidence_source != 'protocol_pdf')
                """,
                (protocol_result["file_path"], instance_id),
            )
            conn.commit()
            logger.info(f"    Downloaded protocol: {protocol_result['file_path']}")
        else:
            # API said protocol exists but download failed - mark for retry
            logger.warning(f"    Protocol download failed for {instance_id}: {protocol_result.get('error')}")
    
    logger.info(f"Protocol check complete: {checked} checked, {has_protocol} have protocols, {downloaded} downloaded, {no_protocol} no protocol")


def _extract_versions_from_history(history_data: dict) -> List[dict]:
    """Extract version list from history API response."""
    versions = []
    
    if "history" in history_data and "changes" in history_data["history"]:
        changes = history_data["history"]["changes"]
        for change in changes:
            versions.append({
                "version_number": change.get("version", 0),
                "version_date": change.get("date"),
                "overall_status": change.get("status"),
                "module_labels": change.get("moduleLabels", []),  # NEW
            })
    
    return versions
