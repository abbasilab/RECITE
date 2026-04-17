"""Trial discovery from ClinicalTrials.gov."""

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET

from loguru import logger

from recite.benchmark.api_client import fetch_version_history
from recite.benchmark.ctg_adapter import ClinicalTrialsGovAdapter


def discover_all_instance_ids(
    method: str = "bulk_xml",
    max_results: Optional[int] = None,
) -> Iterator[str]:
    """
    Discover all NCT IDs from ClinicalTrials.gov.
    
    Args:
        method: Discovery method ('bulk_xml' or 'api_pagination')
        max_results: Maximum number of NCT IDs to return (None for all)
        
    Yields:
        NCT IDs (strings)
    """
    logger.debug(f"Discovering NCT IDs using method: {method}, max_results: {max_results}")
    if method == "bulk_xml":
        logger.debug("Using bulk XML discovery method")
        yield from discover_all_instance_ids_bulk_xml(max_results=max_results)
    elif method == "api_pagination":
        logger.debug("Using API pagination discovery method")
        yield from discover_all_instance_ids_via_api(max_results=max_results)
    else:
        logger.error(f"Unknown discovery method: {method}")
        raise ValueError(f"Unknown discovery method: {method}")


def discover_all_instance_ids_bulk_xml(
    max_results: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Iterator[str]:
    """
    Discover all NCT IDs by downloading and parsing AllPublicXML.zip.
    
    Args:
        max_results: Maximum number of NCT IDs to return (None for all)
        cache_dir: Directory to cache the zip file (default: data/)
        
    Yields:
        NCT IDs (strings)
    """
    if cache_dir is None:
        cache_dir = Path("data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    xml_zip_url = "https://clinicaltrials.gov/AllPublicXML.zip"
    zip_path = cache_dir / "AllPublicXML.zip"
    
    # Check if zip file already exists
    if zip_path.exists():
        logger.info(f"Using cached bulk XML file: {zip_path}")
    else:
        logger.info(f"Downloading bulk XML from {xml_zip_url}")
        
        adapter = ClinicalTrialsGovAdapter()
        adapter._rate_limit()
        
        # Download the zip file
        resp = adapter._request_with_backoff("GET", xml_zip_url)
        
        if resp.status_code != 200:
            logger.error(f"Failed to download bulk XML: {resp.status_code}")
            return
        
        # Save to cache directory
        logger.info(f"Saving bulk XML to {zip_path}")
        zip_path.write_bytes(resp.content)
        logger.info(f"Downloaded and saved {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Extract to cache directory (reuse extracted files across runs)
    xml_cache_dir = cache_dir / "xml_cache"
    xml_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    xml_files = list(xml_cache_dir.rglob("*.xml"))
    
    if xml_files:
        logger.info(f"Using cached extracted XML files: {len(xml_files)} files in {xml_cache_dir}")
        count = 0
        processed_files = 0
        for xml_file in xml_files:
            if max_results and count >= max_results:
                break
            processed_files += 1
            if processed_files % 1000 == 0:
                logger.info(f"  Processing XML files: {processed_files}/{len(xml_files)} ({processed_files*100//len(xml_files)}%), extracted {count} NCT IDs so far")
            for instance_id in extract_instance_ids_from_xml(xml_file):
                yield instance_id
                count += 1
                if count % 10000 == 0:
                    logger.info(f"  Extracted {count} NCT IDs from {processed_files} XML files...")
                if max_results and count >= max_results:
                    break
        logger.info(f"  Completed: extracted {count} NCT IDs from {processed_files} XML files")
        return
    else:
        logger.info("Extracting XML files from zip archive...")
        logger.info(f"  Zip file size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
        logger.info(f"  Cache directory: {xml_cache_dir}")
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get total file count for progress tracking
            total_files = len(zip_ref.namelist())
            logger.info(f"  Archive contains {total_files} files")
            logger.info("  Extracting files (this may take several minutes)...")
            
            # Extract with progress logging
            extracted_count = 0
            for i, member in enumerate(zip_ref.namelist(), 1):
                zip_ref.extract(member, xml_cache_dir)
                extracted_count += 1
                
                # Log progress every 1000 files or every 10% of total
                if extracted_count % 1000 == 0 or (total_files > 10000 and extracted_count % (total_files // 10) == 0):
                    logger.info(f"  Extracted {extracted_count}/{total_files} files ({extracted_count*100//total_files}%)...")
            
            logger.info(f"  Extraction complete: {extracted_count} files extracted to {xml_cache_dir}")
        
        # Find all XML files
        logger.info("Scanning for XML files...")
        xml_files = list(xml_cache_dir.rglob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML files")
        
        count = 0
        processed_files = 0
        for xml_file in xml_files:
            if max_results and count >= max_results:
                break
            
            processed_files += 1
            # Log progress every 1000 files or every 10000 NCT IDs
            if processed_files % 1000 == 0:
                logger.info(f"  Processing XML files: {processed_files}/{len(xml_files)} ({processed_files*100//len(xml_files)}%), extracted {count} NCT IDs so far")
            
            for instance_id in extract_instance_ids_from_xml(xml_file):
                yield instance_id
                count += 1
                # Log progress every 10000 NCT IDs
                if count % 10000 == 0:
                    logger.info(f"  Extracted {count} NCT IDs from {processed_files} XML files...")
                if max_results and count >= max_results:
                    break
        
        logger.info(f"  Completed: extracted {count} NCT IDs from {processed_files} XML files")


def extract_instance_ids_from_xml(xml_path: Path) -> Iterator[str]:
    """
    Extract NCT IDs from a single XML file.
    
    Args:
        xml_path: Path to XML file
        
    Yields:
        NCT IDs found in the XML file
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # ClinicalTrials.gov XML uses <instance_id> tag
        # Try different possible namespaces
        import re
        
        # First try: direct instance_id tag (no namespace)
        instance_id_elements = root.findall(".//instance_id")
        for elem in instance_id_elements:
            instance_id = elem.text
            if instance_id and instance_id.strip():
                yield instance_id.strip()
                return  # Only yield once per file
        
        # Second try: with namespace
        # ClinicalTrials.gov XML might use namespaces
        for elem in root.iter():
            tag = elem.tag
            # Remove namespace prefix if present
            if "}" in tag:
                tag = tag.split("}")[-1]
            if tag == "instance_id" and elem.text:
                instance_id = elem.text.strip()
                if instance_id:
                    yield instance_id
                    return
        
        # Fallback: search in all text content
        text_content = ET.tostring(root, encoding="unicode", method="text")
        matches = re.findall(r"NCT\d{8}", text_content)
        if matches:
            yield matches[0]  # Return first match
                    
    except Exception as e:
        logger.warning(f"Failed to parse XML file {xml_path}: {e}")


def discover_all_instance_ids_via_api(
    max_results: Optional[int] = None,
) -> Iterator[str]:
    """
    Discover NCT IDs via API pagination (fallback method).
    
    Args:
        max_results: Maximum number of NCT IDs to return (None for all)
        
    Yields:
        NCT IDs (strings)
    """
    adapter = ClinicalTrialsGovAdapter()
    
    # Use very broad query to get all trials
    query = "*"  # Broad query to match all trials
    
    logger.info("Discovering NCT IDs via API pagination...")
    
    count = 0
    for doc in adapter.search_all_pages(query, max_results=max_results):
        if doc.source_id:
            yield doc.source_id
            count += 1
            if max_results and count >= max_results:
                break


def check_trial_has_versions(
    instance_id: str,
    adapter: Optional[ClinicalTrialsGovAdapter] = None,
    use_expedited: bool = True,
) -> Tuple[bool, int, bool, List[int], Dict[int, List[str]]]:
    """
    Check if a trial has multiple versions via API and extract moduleLabels information.
    
    Args:
        instance_id: NCT ID to check
        adapter: Optional adapter instance
        use_expedited: If True, only return has_multiple=True if Eligibility in moduleLabels
        
    Returns:
        Tuple of (has_multiple_versions, version_count, has_eligibility_changes, 
                 eligibility_versions, module_labels_dict)
    """
    from recite.benchmark.api_client import extract_module_labels
    from recite.benchmark.module_labels import (
        get_eligibility_versions,
        has_eligibility_changes,
    )
    
    if adapter is None:
        adapter = ClinicalTrialsGovAdapter()
    
    logger.debug(f"  Checking version history for {instance_id}...")
    history_data = fetch_version_history(instance_id, adapter=adapter)
    
    if not history_data:
        logger.debug(f"  No version history found for {instance_id}")
        return False, 0, False, [], {}
    
    # Extract version count from history
    version_count = 0
    if "history" in history_data and "changes" in history_data["history"]:
        changes = history_data["history"]["changes"]
        version_count = len(changes) if isinstance(changes, list) else 0
    
    # Extract moduleLabels
    module_labels_dict = extract_module_labels(history_data)
    logger.debug(f"  Extracted moduleLabels for {instance_id}: {len(module_labels_dict)} versions")
    
    # Check for Eligibility in moduleLabels
    has_eligibility = has_eligibility_changes(module_labels_dict)
    eligibility_versions = get_eligibility_versions(module_labels_dict)
    
    if eligibility_versions:
        logger.debug(f"  Versions with 'Eligibility': {eligibility_versions}")
    
    # Determine if has multiple versions
    # If expedited mode, only return True if has Eligibility changes
    if use_expedited:
        has_multiple = version_count >= 2 and has_eligibility
        if version_count >= 2 and not has_eligibility:
            logger.debug(f"  {instance_id}: {version_count} versions but no Eligibility in moduleLabels (filtered out)")
    else:
        has_multiple = version_count >= 2
    
    logger.debug(f"  {instance_id}: {version_count} versions, has_eligibility={has_eligibility}, has_multiple={has_multiple}")
    return has_multiple, version_count, has_eligibility, eligibility_versions, module_labels_dict


def get_chunked_instance_ids(
    instance_ids: List[str],
    chunk_index: int,
    total_chunks: int,
    seed: int = 42,
) -> List[str]:
    """
    Divide NCT IDs into chunks using seeded shuffle.
    
    Args:
        instance_ids: List of all NCT IDs to process
        chunk_index: Which chunk to return (0-indexed, e.g., 0 for first chunk)
        total_chunks: Total number of chunks to divide into
        seed: Random seed for reproducible shuffling
        
    Returns:
        List of NCT IDs for the specified chunk
    """
    import random
    
    if not instance_ids:
        return []
    
    if total_chunks < 1:
        raise ValueError("total_chunks must be at least 1")
    
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise ValueError(f"chunk_index must be between 0 and {total_chunks - 1}")
    
    shuffled = instance_ids.copy()
    random.Random(seed).shuffle(shuffled)
    
    chunk_size = len(shuffled) // total_chunks
    start_idx = chunk_index * chunk_size
    end_idx = start_idx + chunk_size if chunk_index < total_chunks - 1 else len(shuffled)
    
    return shuffled[start_idx:end_idx]


def check_trial_versions_batch(
    instance_ids: List[str],
    adapter: Optional[ClinicalTrialsGovAdapter] = None,
    conn=None,
    discovery_method: str = "bulk_xml",
    use_expedited: bool = True,
    chunk_index: Optional[int] = None,
    total_chunks: Optional[int] = None,
) -> None:
    """
    Check version counts for a batch of NCT IDs and store in database.
    
    Args:
        instance_ids: List of NCT IDs to check
        adapter: Optional adapter instance
        conn: Database connection (if provided, stores results)
        discovery_method: Method used for discovery
        use_expedited: If True, filter by moduleLabels (only trials with Eligibility)
        chunk_index: Optional chunk index for incremental processing
        total_chunks: Optional total chunks (required if chunk_index provided)
    """
    import time
    
    if adapter is None:
        adapter = ClinicalTrialsGovAdapter()
    
    # Handle chunking (happens FIRST for consistency - same chunks every time)
    # Skip logic will handle already-checked trials within each chunk
    if chunk_index is not None:
        if total_chunks is None:
            raise ValueError("total_chunks must be provided when chunk_index is specified")
        # Chunk ALL trials first (for consistency - seeded shuffle ensures same chunks)
        instance_ids = get_chunked_instance_ids(instance_ids, chunk_index, total_chunks)
        logger.info(f"Processing chunk {chunk_index + 1}/{total_chunks} ({len(instance_ids):,} trials)")
        # Estimate time: ~4.79 trials/second (will be less if many are skipped)
        estimated_hours = len(instance_ids) / (4.79 * 3600)
        logger.info(f"  Estimated time: {estimated_hours:.1f} hours (may be less if trials are skipped)")
    
    if conn is None:
        # Just check without storing
        for instance_id in instance_ids:
            has_versions, count, has_elig, elig_versions, _ = check_trial_has_versions(
                instance_id, adapter, use_expedited=use_expedited
            )
            logger.debug(f"{instance_id}: {count} versions (has_multiple={has_versions}, has_eligibility={has_elig})")
        return
    
    cursor = conn.cursor()
    
    total = len(instance_ids)
    logger.info(f"  Checking version counts for {total} trials...")
    logger.info(f"  Expedited mode: {use_expedited}")
    logger.info(f"  This may take a while - checking via API with rate limiting...")
    
    checked = 0
    skipped = 0
    with_versions = 0
    filtered_out = 0  # Trials filtered by expedited mode
    failed = 0
    start_time = None
    
    for i, instance_id in enumerate(instance_ids, 1):
        if start_time is None:
            start_time = time.time()
        
        # Check if already checked using optimized single query with LEFT JOIN
        from recite.benchmark.utils import should_skip_trial_version_check
        should_skip = should_skip_trial_version_check(cursor, instance_id)
        
        if should_skip:
            # Already checked, skip (granular per-trial skip)
            skipped += 1
            if i % 100 == 0:
                elapsed = time.time() - start_time if start_time else 0
                rate = checked / elapsed if elapsed > 0 else 0
                eta_seconds = (total - i) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                logger.debug(f"  Progress: {i}/{total} ({i*100//total}%), checked: {checked}, skipped: {skipped}, with_versions: {with_versions}, failed: {failed}, rate: {rate:.1f}/s, ETA: {eta_minutes:.1f}m")
            continue
        
        try:
            has_versions_flag, version_count, has_eligibility, eligibility_versions, module_labels_dict = check_trial_has_versions(
                instance_id, adapter, use_expedited=use_expedited
            )
            checked += 1
            
            # Use transaction for multi-step inserts to prevent race conditions
            conn.execute("BEGIN TRANSACTION")
            try:
                # Insert or update discovered_trials
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO discovered_trials 
                    (instance_id, discovery_method, version_count, version_check_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (instance_id, discovery_method, version_count),
                )
                
                # If has multiple versions (and passes expedited filter if enabled)
                if has_versions_flag:
                    with_versions += 1
                    
                    # Prepare moduleLabels data for storage
                    eligibility_versions_json = json.dumps(eligibility_versions) if eligibility_versions else None
                    module_labels_json_str = json.dumps(module_labels_dict) if module_labels_dict else None
                    
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO trials_with_versions
                        (instance_id, version_count, checked_at, versions_downloaded,
                         has_eligibility_changes, eligibility_version_numbers, module_labels_json)
                        VALUES (?, ?, CURRENT_TIMESTAMP, 0, ?, ?, ?)
                        """,
                        (instance_id, version_count, bool(has_eligibility), eligibility_versions_json, module_labels_json_str),
                    )
                    logger.debug(f"  ✓ {instance_id}: {version_count} versions, has_eligibility={has_eligibility} (added to trials_with_versions)")
                elif version_count >= 2 and use_expedited:
                    # Trial has multiple versions but was filtered out by expedited mode
                    filtered_out += 1
                    logger.debug(f"  ✗ {instance_id}: {version_count} versions but no Eligibility in moduleLabels (filtered out)")
                
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
        except Exception as e:
            failed += 1
            logger.warning(f"  ✗ Failed to check {instance_id}: {e}")
            # Still mark as checked (with version_count=None) to avoid retrying
            cursor.execute(
                """
                INSERT OR REPLACE INTO discovered_trials 
                (instance_id, discovery_method, version_count, version_check_at)
                VALUES (?, ?, NULL, CURRENT_TIMESTAMP)
                """,
                (instance_id, discovery_method),
            )
        
        # Log progress every 100 trials or every 10% of total
        if i % 100 == 0 or (total > 1000 and i % (total // 10) == 0):
            elapsed = time.time() - start_time if start_time else 0
            rate = checked / elapsed if elapsed > 0 else 0
            eta_seconds = (total - i) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            logger.info(f"  Progress: {i}/{total} ({i*100//total}%), checked: {checked}, skipped: {skipped}, with_versions: {with_versions}, filtered: {filtered_out}, failed: {failed}, rate: {rate:.1f}/s, ETA: {eta_minutes:.1f}m")
        
        # Commit periodically (every 50 trials) to avoid long transactions
        if i % 50 == 0:
            conn.commit()
            logger.debug(f"  Committed progress: {i}/{total} trials processed")
    
    conn.commit()
    elapsed_total = time.time() - start_time if start_time else 0
    avg_rate = checked / elapsed_total if elapsed_total > 0 else 0
    logger.info(f"  Version check complete:")
    logger.info(f"    - Checked: {checked} trials")
    logger.info(f"    - Skipped: {skipped} trials (already checked)")
    logger.info(f"    - Failed: {failed} trials")
    logger.info(f"    - With multiple versions: {with_versions} trials")
    if use_expedited and filtered_out > 0:
        logger.info(f"    - Filtered out (no Eligibility): {filtered_out} trials")
        logger.info(f"    - Expedited mode: Filtered {filtered_out} trials, saved processing time")
    logger.info(f"    - Total time: {elapsed_total/60:.1f} minutes")
    logger.info(f"    - Average rate: {avg_rate:.2f} trials/second")
