"""Evidence source downloader."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from recite.benchmark.ctg_adapter import ClinicalTrialsGovAdapter


def download_evidence_for_trial(
    nct_id: str,
    version_from: int,
    version_to: int,
    sources: List[str],
    adapter: Optional[ClinicalTrialsGovAdapter] = None,
    output_dir: Path = Path("data/benchmarks/protocols"),
) -> Dict[str, Any]:
    """
    Download evidence sources for a specific EC change.
    
    Prioritizes protocol PDFs (contain rationales for EC changes).
    Falls back to API fields if protocol PDFs are not available.
    
    Args:
        nct_id: NCT ID
        version_from: Source version number
        version_to: Target version number
        sources: List of source types to try ('protocol_pdf', 'api_fields', etc.)
        adapter: Optional adapter instance
        output_dir: Directory to save downloaded files
        
    Returns:
        Dictionary with download results
    """
    if adapter is None:
        adapter = ClinicalTrialsGovAdapter()
    
    results = {
        "nct_id": nct_id,
        "version_from": version_from,
        "version_to": version_to,
        "sources_attempted": sources,
        "sources_downloaded": {},
        "evidence_fields": {},
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = output_dir / nct_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Prioritize protocol PDFs (primary evidence source with rationales)
    for source in sources:
        if source == "protocol_pdf":
            pdf_result = find_protocol_pdfs(nct_id, adapter, trial_dir)
            results["sources_downloaded"]["protocol_pdf"] = pdf_result
        
        elif source == "api_fields":
            # Fallback to API fields if protocol not available
            api_result = _extract_api_evidence_fields(nct_id, version_to, adapter)
            results["evidence_fields"].update(api_result)
    
    return results


def find_protocol_pdfs(
    nct_id: str,
    adapter: ClinicalTrialsGovAdapter,
    output_dir: Path,
    protocol_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Find and download protocol PDFs for a trial.
    
    Uses API metadata to get correct filename, then downloads from:
    https://clinicaltrials.gov/ProvidedDocs/{last_2_digits}/{nct_id}/{filename}
    
    Args:
        nct_id: NCT ID
        adapter: Adapter instance
        output_dir: Directory to save PDFs
        protocol_info: Optional dict with filename from API (from check_protocol_availability)
        
    Returns:
        Dictionary with download results
    """
    result = {
        "success": False,
        "url": None,
        "file_path": None,
        "error": None,
    }
    
    # Extract last 2 digits of NCT ID
    # NCT04424641 -> 41
    last_2_digits = nct_id[-2:] if len(nct_id) >= 2 else nct_id
    
    # Get filename from protocol_info if provided, otherwise try common patterns
    filenames_to_try = []
    if protocol_info and protocol_info.get("filename"):
        filenames_to_try.append(protocol_info["filename"])
    # Fallback patterns if no info provided
    filenames_to_try.extend(["Prot_000.pdf", "Prot_001.pdf", "Prot_ICF_000.pdf", "Prot_ICF_001.pdf"])
    
    # Use ProvidedDocs URL (correct pattern)
    base_url = f"https://clinicaltrials.gov/ProvidedDocs/{last_2_digits}/{nct_id}"
    
    for filename in filenames_to_try:
        url = f"{base_url}/{filename}"
        
        try:
            adapter._rate_limit()
            resp = adapter._request_with_backoff("GET", url)
            
            if resp.status_code == 200:
                output_dir.mkdir(parents=True, exist_ok=True)
                trial_dir = output_dir / nct_id
                trial_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = trial_dir / filename
                with open(file_path, "wb") as f:
                    f.write(resp.content)
                
                result["success"] = True
                result["url"] = url
                result["file_path"] = str(file_path)
                return result
            elif resp.status_code != 404:
                # Non-404 error, record it
                result["error"] = f"HTTP {resp.status_code}"
        except Exception as e:
            result["error"] = str(e)
            logger.warning(f"Failed to download protocol PDF for {nct_id}: {e}")
    
    if not result["error"]:
        result["error"] = "HTTP 404"
    return result


def _download_protocol_pdf(
    nct_id: str,
    version: int,
    adapter: ClinicalTrialsGovAdapter,
    output_dir: Path,
) -> Dict[str, Any]:
    """Download protocol PDF for a specific version (legacy - use find_protocol_pdfs instead)."""
    # Use new function (version parameter ignored - Prot_000.pdf contains all amendments)
    return find_protocol_pdfs(nct_id, adapter, output_dir)


def _extract_api_evidence_fields(
    nct_id: str,
    version: int,
    adapter: ClinicalTrialsGovAdapter,
) -> Dict[str, Any]:
    """Extract evidence fields from API response."""
    from recite.benchmark.api_client import fetch_version_data
    
    result = {}
    
    version_data = fetch_version_data(nct_id, version, adapter=adapter)
    
    if version_data:
        # Search for amendment-related fields (exploration found these in statusModule)
        amendment_keywords = ["amendment", "justification", "change", "modification", "update", "submit"]
        
        # Check status module specifically (where exploration found evidence)
        if isinstance(version_data, dict):
            protocol_section = version_data.get("study", {}).get("protocolSection", {}) or version_data.get("protocolSection", {})
            status_module = protocol_section.get("statusModule", {})
            
            # Extract relevant fields from status module
            for key, value in status_module.items():
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in amendment_keywords):
                    if value:
                        if isinstance(value, dict):
                            result[key] = json.dumps(value)[:500]
                        elif isinstance(value, str) and len(value.strip()) > 0:
                            result[key] = value[:500]
            
            # Also check references module for publications
            references_module = protocol_section.get("referencesModule", {})
            if references_module:
                result["references"] = json.dumps(references_module.get("references", []))[:1000]
    
    return result
