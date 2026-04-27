"""Eligibility criteria change detection."""

import difflib
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


def detect_ec_changes(
    version1_ec: str,
    version2_ec: str,
    min_change_threshold: float = 0.05,
    similarity_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Detect changes between two eligibility criteria texts.
    
    Args:
        version1_ec: Original eligibility criteria text
        version2_ec: Modified eligibility criteria text
        min_change_threshold: Minimum ratio of changed content to consider it meaningful
        similarity_threshold: Minimum similarity to consider texts related
        
    Returns:
        Dictionary with change detection results
    """
    if not version1_ec or not version2_ec:
        return {
            "has_change": False,
            "change_type": None,
            "similarity": 0.0,
            "change_ratio": 0.0,
            "reason": "Missing EC text",
        }
    
    # Normalize whitespace
    ec1_normalized = _normalize_text(version1_ec)
    ec2_normalized = _normalize_text(version2_ec)
    
    # Check if identical after normalization
    if ec1_normalized == ec2_normalized:
        return {
            "has_change": False,
            "change_type": None,
            "similarity": 1.0,
            "change_ratio": 0.0,
            "reason": "Texts are identical after normalization",
        }
    
    # Calculate similarity
    similarity = _calculate_similarity(ec1_normalized, ec2_normalized)
    
    # Calculate change ratio
    diff = list(difflib.unified_diff(
        ec1_normalized.splitlines(keepends=True),
        ec2_normalized.splitlines(keepends=True),
        lineterm="",
    ))
    
    # Count changed lines
    changed_lines = sum(1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    total_lines = max(len(ec1_normalized.splitlines()), len(ec2_normalized.splitlines()))
    change_ratio = changed_lines / total_lines if total_lines > 0 else 0.0
    
    # Determine if change is meaningful
    has_change = (
        similarity < similarity_threshold
        or change_ratio >= min_change_threshold
    )
    
    # Detect change type (inclusion, exclusion, or both)
    change_type = _detect_change_type(ec1_normalized, ec2_normalized)
    
    # Generate diff preview
    diff_preview = _generate_diff_preview(ec1_normalized, ec2_normalized)
    
    return {
        "has_change": has_change,
        "change_type": change_type,
        "similarity": similarity,
        "change_ratio": change_ratio,
        "changed_lines": changed_lines,
        "total_lines": total_lines,
        "diff_preview": diff_preview,
        "reason": "Change detected" if has_change else "Change below threshold",
    }


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra whitespace
    lines = [line.strip() for line in text.splitlines()]
    # Remove empty lines
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher."""
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def _detect_change_type(ec1: str, ec2: str) -> Optional[str]:
    """
    Detect if change affects inclusion, exclusion, or both criteria.
    
    Returns:
        'inclusion', 'exclusion', 'both', or None
    """
    ec1_lower = ec1.lower()
    ec2_lower = ec2.lower()
    
    # Check for inclusion/exclusion sections
    has_inclusion_1 = "inclusion" in ec1_lower
    has_exclusion_1 = "exclusion" in ec1_lower
    has_inclusion_2 = "inclusion" in ec2_lower
    has_exclusion_2 = "exclusion" in ec2_lower
    
    # Simple heuristic: check if inclusion or exclusion sections changed
    inclusion_changed = has_inclusion_1 != has_inclusion_2 or (
        has_inclusion_1
        and has_inclusion_2
        and _section_changed(ec1_lower, ec2_lower, "inclusion")
    )
    
    exclusion_changed = has_exclusion_1 != has_exclusion_2 or (
        has_exclusion_1
        and has_exclusion_2
        and _section_changed(ec1_lower, ec2_lower, "exclusion")
    )
    
    if inclusion_changed and exclusion_changed:
        return "both"
    elif inclusion_changed:
        return "inclusion"
    elif exclusion_changed:
        return "exclusion"
    else:
        return None


def _section_changed(text1: str, text2: str, section: str) -> bool:
    """Check if a specific section (inclusion/exclusion) changed."""
    # Extract section text
    section1 = _extract_section(text1, section)
    section2 = _extract_section(text2, section)
    
    if not section1 or not section2:
        return section1 != section2
    
    similarity = _calculate_similarity(section1, section2)
    return similarity < 0.95  # Threshold for section change


def _extract_section(text: str, section: str) -> Optional[str]:
    """Extract a specific section from EC text."""
    text_lower = text.lower()
    section_lower = section.lower()
    
    # Find section start
    start_keywords = [f"{section_lower} criteria", f"{section_lower}:"]
    start_idx = None
    
    for keyword in start_keywords:
        idx = text_lower.find(keyword)
        if idx != -1:
            start_idx = idx
            break
    
    if start_idx is None:
        return None
    
    # Find section end (next section or end of text)
    end_keywords = ["exclusion", "inclusion"]
    end_idx = len(text)
    
    for keyword in end_keywords:
        if keyword != section_lower:
            idx = text_lower.find(keyword, start_idx + 1)
            if idx != -1 and idx < end_idx:
                end_idx = idx
    
    return text[start_idx:end_idx].strip()


def _generate_diff_preview(text1: str, text2: str, max_lines: int = 10) -> List[str]:
    """Generate a preview of differences between texts."""
    diff = list(
        difflib.unified_diff(
            text1.splitlines(keepends=True),
            text2.splitlines(keepends=True),
            lineterm="",
            n=3,  # Context lines
        )
    )
    
    # Limit preview
    return diff[:max_lines]


def test_ec_detection(
    sample_pairs: List[Tuple[str, str]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """
    Test EC change detection on sample pairs.
    
    Args:
        sample_pairs: List of (ec1, ec2) tuples to test
        thresholds: Dictionary of threshold values to test
        
    Returns:
        Test results with accuracy metrics
    """
    results = {
        "total_pairs": len(sample_pairs),
        "detections": [],
        "summary": {},
    }
    
    for i, (ec1, ec2) in enumerate(sample_pairs):
        detection = detect_ec_changes(
            ec1,
            ec2,
            min_change_threshold=thresholds.get("min_change_threshold", 0.05),
            similarity_threshold=thresholds.get("similarity_threshold", 0.8),
        )
        results["detections"].append({
            "pair_id": i,
            "has_change": detection["has_change"],
            "change_type": detection["change_type"],
            "similarity": detection["similarity"],
            "change_ratio": detection["change_ratio"],
        })
    
    # Calculate summary statistics
    total_changes = sum(1 for d in results["detections"] if d["has_change"])
    results["summary"] = {
        "total_changes_detected": total_changes,
        "change_detection_rate": total_changes / len(sample_pairs) if sample_pairs else 0,
        "avg_similarity": (
            sum(d["similarity"] for d in results["detections"]) / len(results["detections"])
            if results["detections"]
            else 0
        ),
        "avg_change_ratio": (
            sum(d["change_ratio"] for d in results["detections"]) / len(results["detections"])
            if results["detections"]
            else 0
        ),
    }
    
    return results
