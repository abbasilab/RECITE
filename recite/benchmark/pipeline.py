"""Pipeline orchestrator for RECITE benchmark."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from recite.benchmark.builders import create_recite_instances
from recite.benchmark.discovery import (
    check_trial_versions_batch,
    discover_all_instance_ids,
)
from recite.benchmark.downloaders import download_protocols, download_versions
from recite.benchmark.processors import extract_evidence, identify_amendments
from recite.benchmark.utils import (
    execute_batched_in_query,
    get_trials_ready_for_recite,
    get_trials_with_ec_changes,
    get_trials_with_protocols,
    get_trials_with_versions,
)
from recite.crawler.adapters import ClinicalTrialsGovAdapter


def run_e2e_pipeline(
    discovery_method: str = "bulk_xml",
    max_trials: Optional[int] = None,
    db_path: Optional[Path] = None,
    force: bool = False,
    use_expedited: bool = True,
    chunk_index: Optional[int] = None,
    total_chunks: Optional[int] = None,
    stop_after: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full E2E pipeline with progressive filtering.
    
    Args:
        discovery_method: Method to discover NCT IDs ('bulk_xml' or 'api_pagination')
        max_trials: Maximum number of trials to process (None for all)
        db_path: Path to database file (None for default)
        force: If True, backup existing database and run from scratch (no skipping)
        use_expedited: If True, use moduleLabels filtering to expedite processing
        chunk_index: Optional chunk index for incremental processing (0-indexed)
        total_chunks: Optional total chunks (required if chunk_index provided)
        stop_after: Optional stage to stop after (e.g. 'metadata' to stop after metadata population)
        
    Returns:
        Dictionary with pipeline statistics
    """
    from recite.benchmark.db import get_connection, init_database
    
    logger.info("Starting E2E RECITE benchmark pipeline")
    logger.info(f"Discovery method: {discovery_method}")
    logger.info(f"Expedited mode: {use_expedited}")
    if chunk_index is not None:
        logger.info(f"Chunking: Processing chunk {chunk_index + 1}/{total_chunks}")
    if force:
        logger.info("Force mode: Backing up existing database and running from scratch")
    
    # Initialize database (will backup if force=True)
    conn = init_database(db_path, force=force)
    adapter = ClinicalTrialsGovAdapter()
    cursor = conn.cursor()
    
    stats = {
        "discovered": 0,
        "with_versions": 0,
        "versions_downloaded": 0,
        "with_ec_changes": 0,
        "with_protocols": 0,
        "with_evidence": 0,
        "recite_instances": 0,
    }
    
    try:
        # Stage 1: Discover all NCT IDs
        logger.info("Stage 1: Discovering NCT IDs...")
        
        # Always discover, but skip individual NCT IDs that already exist (unless force=True)
        discovered_count = 0
        instance_ids_to_check = []
        
        # Get existing NCT IDs to avoid duplicates (unless force mode)
        existing_instance_ids = set()
        if not force:
            existing_instance_ids = set(
                row["instance_id"] 
                for row in cursor.execute(
                    "SELECT instance_id FROM discovered_trials WHERE discovery_method = ?",
                    (discovery_method,),
                ).fetchall()
            )
            if existing_instance_ids:
                logger.info(f"  Found {len(existing_instance_ids)} already discovered NCT IDs (will skip duplicates)")
        
        new_instance_ids_found = False
        for instance_id in discover_all_instance_ids(method=discovery_method, max_results=max_trials):
            # Skip if already in database (unless force mode)
            if not force and instance_id in existing_instance_ids:
                logger.debug(f"  Skipping {instance_id} (already discovered)")
                continue
            
            # Store in discovered_trials table
            cursor.execute(
                """
                INSERT OR IGNORE INTO discovered_trials 
                (instance_id, discovery_method, discovered_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (instance_id, discovery_method),
            )
            conn.commit()
            discovered_count += 1
            instance_ids_to_check.append(instance_id)
            existing_instance_ids.add(instance_id)
            new_instance_ids_found = True
            
            if max_trials and discovered_count >= max_trials:
                break
        
        # If no new NCT IDs were discovered but we have existing ones, use them
        if not new_instance_ids_found and existing_instance_ids and not force:
            logger.info(f"  All NCT IDs already discovered, using existing {len(existing_instance_ids)} NCT IDs")
            # Use existing NCT IDs for processing
            # Get ALL discovered trials (chunking happens later for consistency)
            # Skip logic will handle already-checked trials within each chunk
            query = """
                SELECT instance_id FROM discovered_trials 
                WHERE discovery_method = ?
                ORDER BY discovered_at DESC
            """
            if max_trials:
                query += f" LIMIT {max_trials}"
            rows = cursor.execute(query, (discovery_method,)).fetchall()
            instance_ids_to_check = [row["instance_id"] for row in rows]
            discovered_count = len(instance_ids_to_check)
        
        # If we need more and have existing ones (and not force mode), add them
        elif not force and max_trials and len(instance_ids_to_check) < max_trials:
            needed = max_trials - len(instance_ids_to_check)
            
            # Use batched query to avoid SQLite variable limit
            if instance_ids_to_check:
                # For NOT IN with large lists, we need to get all candidates
                # and filter out excluded ones using batched IN checks
                # Get all candidates first
                all_candidates = cursor.execute(
                    """
                    SELECT instance_id FROM discovered_trials 
                    WHERE discovery_method = ?
                    ORDER BY discovered_at DESC
                    """,
                    (discovery_method,),
                ).fetchall()
                
                candidate_ids = [row["instance_id"] for row in all_candidates]
                exclude_set = set(instance_ids_to_check)
                
                # Filter out excluded IDs
                filtered_ids = [instance_id for instance_id in candidate_ids if instance_id not in exclude_set]
                
                # Take only what we need
                instance_ids_to_check.extend(filtered_ids[:needed])
            else:
                # No existing IDs, just query normally
                query = """
                    SELECT instance_id FROM discovered_trials 
                    WHERE discovery_method = ?
                    ORDER BY discovered_at DESC
                    LIMIT ?
                """
                rows = cursor.execute(query, (discovery_method, needed)).fetchall()
                instance_ids_to_check.extend([row["instance_id"] for row in rows])
        
        stats["discovered"] = len(instance_ids_to_check) if instance_ids_to_check else discovered_count
        logger.info(f"  Total NCT IDs available: {len(instance_ids_to_check) if instance_ids_to_check else discovered_count}")
        
        if not instance_ids_to_check:
            logger.warning("No NCT IDs available. Pipeline will exit early.")
            return stats
        
        # Early metadata stage: Populate trial_metadata table from XML (only for bulk_xml)
        # Skip in tests or when xml_cache_dir doesn't exist (fast path)
        if discovery_method == "bulk_xml":
            import os
            # Skip metadata in test environment (faster tests)
            if os.getenv("SKIP_METADATA_IN_TESTS", "").lower() in ("1", "true", "yes"):
                stats["metadata_populated"] = 0
            else:
                from recite.benchmark.metadata import populate_trial_metadata_table
                
                # Use same cache_dir convention as discovery (default: data/)
                cache_dir = Path("data")
                xml_cache_dir = cache_dir / "xml_cache"
                
                # Fast path: skip if directory doesn't exist (common in tests)
                if xml_cache_dir.exists() and any(xml_cache_dir.iterdir()):
                    logger.info("Early metadata stage: Populating trial_metadata table...")
                    metadata_count = populate_trial_metadata_table(
                        conn, xml_cache_dir, instance_ids=instance_ids_to_check
                    )
                    stats["metadata_populated"] = metadata_count
                    logger.info(f"  Populated metadata for {metadata_count} trials")
                else:
                    # Skip silently in tests (no warning needed)
                    stats["metadata_populated"] = 0
        else:
            stats["metadata_populated"] = 0
        
        # Check if we should stop after metadata
        if stop_after == "metadata":
            logger.info("Stopping after metadata population (--stop-after metadata)")
            return stats
        
        # Stage 2: Check version counts and filter
        logger.info("Stage 2: Checking version counts...")
        check_trial_versions_batch(
            instance_ids_to_check, 
            adapter, 
            conn, 
            discovery_method=discovery_method,
            use_expedited=use_expedited,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )
        
        # Count trials with versions
        with_versions_count = cursor.execute(
            "SELECT COUNT(DISTINCT instance_id) FROM trials_with_versions"  # deduped across merge sources
        ).fetchone()[0]
        stats["with_versions"] = with_versions_count
        logger.info(f"  Found {with_versions_count} trials with multiple versions")
        
        # Check if we should stop after versions check
        if stop_after == "versions":
            logger.info("Stopping after version check (--stop-after versions)")
            return stats
        
        # Stage 3: Download versions
        logger.info("Stage 3: Downloading trial versions...")
        trials_with_versions = get_trials_with_versions(conn, max_trials)
        if trials_with_versions:
            download_versions(trials_with_versions, max_trials, conn, use_expedited=use_expedited)
        else:
            logger.warning("No trials with multiple versions found")
        
        # Update versions_downloaded flag (only if we actually downloaded)
        if trials_with_versions:
            for instance_id in trials_with_versions:
                cursor.execute(
                    """
                    UPDATE trials_with_versions
                    SET versions_downloaded = 1
                    WHERE instance_id = ?
                    """,
                    (instance_id,),
                )
            conn.commit()
        
        versions_downloaded_count = cursor.execute(
            "SELECT COUNT(DISTINCT instance_id) FROM trials_with_versions WHERE versions_downloaded = 1"  # deduped across merge sources
        ).fetchone()[0]
        stats["versions_downloaded"] = versions_downloaded_count
        logger.info(f"  Downloaded versions for {versions_downloaded_count} trials")
        
        # Stage 4: Identify EC changes
        logger.info("Stage 4: Identifying EC changes...")
        identify_amendments(max_trials, conn)
        
        with_ec_changes_count = len(get_trials_with_ec_changes(conn, max_trials))
        stats["with_ec_changes"] = with_ec_changes_count
        logger.info(f"  Found {with_ec_changes_count} trials with EC changes")
        
        # Stage 5: Download protocols
        logger.info("Stage 5: Downloading protocol PDFs...")
        download_protocols(None, max_trials, conn)
        
        with_protocols_count = len(get_trials_with_protocols(conn, max_trials))
        stats["with_protocols"] = with_protocols_count
        logger.info(f"  Downloaded protocols for {with_protocols_count} trials")
        
        # Stage 6: Extract evidence
        logger.info("Stage 6: Extracting evidence from PDFs...")
        extract_evidence(max_trials, conn)
        
        # Count trials with evidence (has protocol PDF)
        with_evidence_count = cursor.execute(
            """
            SELECT COUNT(DISTINCT instance_id) 
            FROM ec_changes 
            WHERE evidence_source_path IS NOT NULL
            """
        ).fetchone()[0]
        stats["with_evidence"] = with_evidence_count
        logger.info(f"  Extracted evidence for {with_evidence_count} trials")
        
        # Stage 7: Build RECITE instances
        logger.info("Stage 7: Building RECITE instances...")
        create_recite_instances(max_trials, conn)
        
        recite_count = cursor.execute("SELECT COUNT(*) FROM recite").fetchone()[0]
        stats["recite_instances"] = recite_count
        logger.info(f"  Created {recite_count} RECITE instances")
        
        logger.info("Pipeline complete!")
        logger.info(f"Final statistics: {stats}")
        
        # Save pipeline statistics report
        _save_pipeline_report(stats, db_path)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise
    finally:
        conn.close()
    
    return stats


def _save_pipeline_report(stats: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    """
    Save pipeline statistics report to reports/ directory.
    
    Args:
        stats: Pipeline statistics dictionary
        db_path: Database path (for report naming)
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_name = db_path.stem if db_path else "recite"
    report_path = reports_dir / f"pipeline_stats_{db_name}_{timestamp}.json"
    
    # Add metadata
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "database": str(db_path) if db_path else "data/dev/recite.db",
        "statistics": stats,
    }
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    logger.info(f"Pipeline statistics report saved to {report_path}")


def get_pipeline_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Get statistics for each pipeline stage.
    
    Args:
        conn: Database connection
        
    Returns:
        Dictionary with counts at each stage
    """
    cursor = conn.cursor()
    
    # Handle case where tables might not exist yet
    try:
    
        stats = {
            "discovered": cursor.execute("SELECT COUNT(DISTINCT instance_id) FROM discovered_trials").fetchone()[0],  # deduped across merge sources
            "with_versions": cursor.execute("SELECT COUNT(DISTINCT instance_id) FROM trials_with_versions").fetchone()[0],  # deduped across merge sources
            "versions_downloaded": cursor.execute(
                "SELECT COUNT(DISTINCT instance_id) FROM trials_with_versions WHERE versions_downloaded = 1"  # deduped across merge sources
            ).fetchone()[0],
            "with_ec_changes": cursor.execute(
                "SELECT COUNT(DISTINCT instance_id) FROM ec_changes"
            ).fetchone()[0],
            "with_protocols": cursor.execute(
                "SELECT COUNT(DISTINCT instance_id) FROM ec_changes WHERE evidence_source_path IS NOT NULL"
            ).fetchone()[0],
            "with_evidence": cursor.execute(
                "SELECT COUNT(DISTINCT instance_id) FROM ec_changes WHERE evidence_source_path IS NOT NULL"
            ).fetchone()[0],
            "recite_instances": cursor.execute("SELECT COUNT(*) FROM recite").fetchone()[0],
        }
    except sqlite3.OperationalError:
        # Tables don't exist yet, return zeros
        stats = {
            "discovered": 0,
            "with_versions": 0,
            "versions_downloaded": 0,
            "with_ec_changes": 0,
            "with_protocols": 0,
            "with_evidence": 0,
            "recite_instances": 0,
        }
    
    return stats
