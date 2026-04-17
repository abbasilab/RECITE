"""Module labels extraction and processing."""

import json
from typing import Any, Dict, List

from loguru import logger


def extract_module_labels_from_history(history_data: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Extract moduleLabels for each version from history API response.
    
    This is the single source of truth for moduleLabels extraction.
    
    Args:
        history_data: Version history API response
        
    Returns:
        Dictionary mapping version_number -> list of module labels
        Example: {4: ["Study Status", "Eligibility"], 5: ["Eligibility"]}
    """
    module_labels_dict = {}
    
    if not history_data:
        return module_labels_dict
    
    if "history" in history_data and "changes" in history_data["history"]:
        changes = history_data["history"]["changes"]
        if isinstance(changes, list):
            for change in changes:
                version_num = change.get("version")
                module_labels = change.get("moduleLabels", [])
                
                if version_num is not None:
                    # Ensure module_labels is a list
                    if isinstance(module_labels, list):
                        module_labels_dict[version_num] = module_labels
                    elif module_labels:
                        # If it's a string or other type, convert to list
                        module_labels_dict[version_num] = [str(module_labels)]
                    else:
                        module_labels_dict[version_num] = []
    
    return module_labels_dict


def get_eligibility_versions(module_labels_dict: Dict[int, List[str]]) -> List[int]:
    """
    Get list of version numbers that have 'Eligibility' in moduleLabels.
    
    Args:
        module_labels_dict: Dictionary mapping version_number -> list of module labels
        
    Returns:
        Sorted list of version numbers with 'Eligibility' in moduleLabels
    """
    eligibility_versions = []
    
    for version_num, labels in module_labels_dict.items():
        # Check if 'Eligibility' is in any of the labels (case-insensitive)
        if any("eligibility" in str(label).lower() for label in labels):
            eligibility_versions.append(version_num)
    
    return sorted(eligibility_versions)


def get_versions_to_download(eligibility_versions: List[int]) -> List[int]:
    """
    Calculate which versions to download for expedited mode.
    
    Downloads:
    - Versions with 'Eligibility' in moduleLabels
    - Previous version for each (to get "before" state for comparison)
    
    Args:
        eligibility_versions: List of version numbers with 'Eligibility' in moduleLabels
        
    Returns:
        Sorted list of version numbers to download
    """
    versions_to_download = set()
    
    for version_num in eligibility_versions:
        # Add the version with Eligibility
        versions_to_download.add(version_num)
        
        # Add previous version (if not version 0)
        if version_num > 0:
            versions_to_download.add(version_num - 1)
    
    return sorted(list(versions_to_download))


def has_eligibility_changes(module_labels_dict: Dict[int, List[str]]) -> bool:
    """
    Check if any version has 'Eligibility' in moduleLabels.
    
    Args:
        module_labels_dict: Dictionary mapping version_number -> list of module labels
        
    Returns:
        True if any version has 'Eligibility' in moduleLabels
    """
    return len(get_eligibility_versions(module_labels_dict)) > 0
