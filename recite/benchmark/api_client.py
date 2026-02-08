"""
API client functions for fetching ClinicalTrials.gov version history.

Standalone functions that can reuse existing adapter methods if straightforward.
"""

import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from recite.crawler.adapters import ClinicalTrialsGovAdapter


def fetch_version_history(
    instance_id: str,
    method: str = "auto",
    adapter: Optional[ClinicalTrialsGovAdapter] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch version history for a trial.
    
    Based on exploration results, internal_api_history is recommended (100% success rate,
    includes version history and eligibility criteria).
    
    Args:
        instance_id: NCT ID
        method: Method to use ('auto', 'internal_api', 'v2_api')
        adapter: Optional adapter instance (will create one if not provided)
        
    Returns:
        Version history data or None if failed
    """
    if adapter is None:
        adapter = ClinicalTrialsGovAdapter()
    
    # Try internal API first (recommended based on exploration - 100% success rate)
    if method in ("auto", "internal_api"):
        try:
            adapter._rate_limit()
            url = f"https://clinicaltrials.gov/api/int/studies/{instance_id}?history=true"
            resp = adapter._request_with_backoff("GET", url)
            
            if resp.status_code == 200:
                data = resp.json()
                # Check if it has history
                if "history" in data or "changes" in data.get("history", {}):
                    return data
        except Exception as e:
            logger.warning(f"Internal API failed for {instance_id}: {e}")
            if method != "auto":
                return None
    
    # Fallback to v2 API (doesn't have version history, but has current version)
    if method in ("auto", "v2_api"):
        try:
            adapter._rate_limit()
            url = f"https://clinicaltrials.gov/api/v2/studies/{instance_id}"
            resp = adapter._request_with_backoff("GET", url)
            
            if resp.status_code == 200:
                data = resp.json()
                return data
        except Exception as e:
            logger.warning(f"V2 API failed for {instance_id}: {e}")
    
    return None


def fetch_version_data(
    instance_id: str,
    version_number: int,
    method: str = "auto",
    adapter: Optional[ClinicalTrialsGovAdapter] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch specific version data for a trial.
    
    Args:
        instance_id: NCT ID
        version_number: Version number (0-based)
        method: Method to use ('auto', 'internal_api', 'v2_api')
        adapter: Optional adapter instance
        
    Returns:
        Version data or None if failed
    """
    if adapter is None:
        adapter = ClinicalTrialsGovAdapter()
    
    # Try internal API first
    if method in ("auto", "internal_api"):
        try:
            adapter._rate_limit()
            url = f"https://clinicaltrials.gov/api/int/studies/{instance_id}/history/{version_number}"
            resp = adapter._request_with_backoff("GET", url)
            
            if resp.status_code == 200:
                data = resp.json()
                return data
        except Exception as e:
            logger.warning(f"Internal API failed for {instance_id} version {version_number}: {e}")
            if method != "auto":
                return None
    
    # Fallback: would need to parse from v2 API or other method
    # For now, return None if internal API fails
    return None


def extract_eligibility_criteria(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract eligibility criteria from API response.
    
    Args:
        data: API response data
        
    Returns:
        Eligibility criteria text or None
    """
    # Try various paths based on API structure
    paths = [
        ["protocolSection", "eligibilityModule", "eligibilityCriteria"],
        ["study", "protocolSection", "eligibilityModule", "eligibilityCriteria"],
        ["eligibilityCriteria"],
    ]
    
    for path in paths:
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
                if current and isinstance(current, str) and len(current.strip()) > 0:
                    return current.strip()
            else:
                break
    
    return None


def extract_version_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract version information from API response.
    
    Args:
        data: API response data
        
    Returns:
        Dictionary with version information
    """
    info = {
        "version_date": None,
        "overall_status": None,
        "version_number": None,
    }
    
    # Try to extract from various possible structures
    if "history" in data:
        history = data["history"]
        if "changes" in history and isinstance(history["changes"], list):
            # Get latest change
            if history["changes"]:
                latest = history["changes"][-1]
                info["version_date"] = latest.get("date")
                info["overall_status"] = latest.get("status")
                info["version_number"] = latest.get("version")
    
    # Also check status module
    if "protocolSection" in data:
        status_module = data["protocolSection"].get("statusModule", {})
        if not info["overall_status"]:
            info["overall_status"] = status_module.get("overallStatus")
    
    return info


def extract_module_labels(history_data: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Extract moduleLabels for each version from history API response.
    
    This function delegates to the shared utility to maintain DRY principles.
    
    Args:
        history_data: Version history API response
        
    Returns:
        Dictionary mapping version_number -> list of module labels
        Example: {4: ["Study Status", "Eligibility"], 5: ["Eligibility"]}
    """
    from recite.benchmark.module_labels import extract_module_labels_from_history
    
    return extract_module_labels_from_history(history_data)
