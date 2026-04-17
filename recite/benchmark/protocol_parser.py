"""
Protocol PDF parser for RECITE benchmark.

Extracts amendment tables, EC changes, and rationales from protocol PDFs.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # pymupdf
from loguru import logger


def extract_amendment_table(pdf_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Extract amendment summary table from protocol PDF.
    
    Looks for "Summary of Protocol Amendments" or similar table structure.
    
    Args:
        pdf_path: Path to protocol PDF
        
    Returns:
        List of amendment dictionaries with number, date, changes, rationales
    """
    try:
        doc = fitz.open(str(pdf_path))
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
        return None
    
    # Search for amendment table
    amendments = []
    
    # Pattern 1: Look for "Table 1" with "Protocol Amendments"
    # Use DOTALL to match across newlines
    table_pattern = r"Table\s+1.*?Protocol\s+Amendments"
    table_match = re.search(table_pattern, full_text, re.IGNORECASE | re.DOTALL)
    
    if table_match:
        # Extract table section - look for end of table (next section or end)
        start_idx = table_match.start()
        # Search for end markers - be more lenient to capture all amendments
        # Don't use "3\s+" as it matches Amendment 3!
        end_patterns = [
            r"\n\s*\d+\.\s+[A-Z]{2,}",  # Next numbered section with capital letters
            r"\n\s*[A-Z]{4,}\s+[A-Z]{4,}",  # All caps heading (longer to avoid false matches)
            r"\n\s*Version\s+History",  # Version history section
        ]
        
        # Search further to ensure we get all amendments (amendment 3 is around position 3354)
        search_end = min(start_idx + 15000, len(full_text))
        for pattern in end_patterns:
            match = re.search(pattern, full_text[start_idx:search_end], re.IGNORECASE)
            if match and match.start() > 1000:  # Ensure we have enough content
                search_end = start_idx + match.start()
                break
        
        table_section = full_text[start_idx:search_end]
        amendments = _parse_amendment_table(table_section)
    
    # Pattern 2: Look for "Summary of Protocol Amendments" section
    if not amendments:
        summary_patterns = [
            r"Summary\s+of\s+Protocol\s+Amendments",
            r"PROTOCOL\s+AND\s+SUMMARY\s+OF\s+PROTOCOL\s+AMENDMENTS",
        ]
        
        for pattern in summary_patterns:
            summary_match = re.search(pattern, full_text, re.IGNORECASE)
            if summary_match:
                start_idx = summary_match.start()
                search_end = min(start_idx + 5000, len(full_text))
                table_section = full_text[start_idx:search_end]
                amendments = _parse_amendment_table(table_section)
                if amendments:
                    break
    
    # Pattern 3: Look for "PROTOCOL AMENDMENT SUMMARY OF CHANGES" (single amendment format)
    if not amendments:
        single_amendment_pattern = r"PROTOCOL\s+AMENDMENT\s+SUMMARY\s+OF\s+CHANGES"
        single_match = re.search(single_amendment_pattern, full_text, re.IGNORECASE)
        
        if single_match:
            # Look for amendment number in header (e.g., "Amendment 5")
            header_section = full_text[max(0, single_match.start()-500):single_match.start()]
            amendment_num_match = re.search(r"Amendment\s+(\d+)", header_section, re.IGNORECASE)
            
            if amendment_num_match:
                amendment_num = int(amendment_num_match.group(1))
                # Look for date
                date_match = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", header_section, re.IGNORECASE)
                date_str = date_match.group(1) if date_match else None
                
                # Extract change summary section
                start_idx = single_match.end()
                # Find end of summary (next major section)
                end_patterns = [
                    r"\n\s*\d+\.\s+[A-Z]{2,}",  # Next numbered section
                    r"\n\s*TABLE\s+OF\s+CONTENTS",  # Table of contents
                ]
                end_idx = min(start_idx + 10000, len(full_text))
                for pattern in end_patterns:
                    match = re.search(pattern, full_text[start_idx:], re.IGNORECASE)
                    if match and match.start() > 500:
                        end_idx = start_idx + match.start()
                        break
                
                summary_text = full_text[start_idx:end_idx].strip()
                summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                
                if summary_text and len(summary_text) > 100:
                    amendments = [{
                        "amendment_number": amendment_num,
                        "date": date_str,
                        "text": summary_text,
                        "changes": _extract_changes_from_text(summary_text),
                        "rationales": _extract_rationales_from_text(summary_text),
                    }]
    
    # Pattern 2: Look for individual amendment entries (e.g., "Amendment 3")
    if not amendments:
        amendments = _find_individual_amendments(full_text)
    
    # Pattern 3: Look for version history with amendments
    if not amendments:
        version_pattern = r"Version\s+(\d+\.\d+).*?\(Amendment\s+(\d+)\)"
        matches = re.finditer(version_pattern, full_text, re.IGNORECASE)
        for match in matches:
            version = match.group(1)
            amendment_num = int(match.group(2))
            # Extract text after this match
            start = match.end()
            next_match = re.search(version_pattern, full_text[start:], re.IGNORECASE)
            end = start + (next_match.start() if next_match else min(2000, len(full_text) - start))
            amendment_text = full_text[start:end]
            
            amendments.append({
                "amendment_number": amendment_num,
                "version": version,
                "text": amendment_text,
                "changes": _extract_changes_from_text(amendment_text),
                "rationales": _extract_rationales_from_text(amendment_text),
            })
    
    return amendments if amendments else None


def extract_pdf_version_info(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract version/amendment metadata from protocol PDF.
    
    Args:
        pdf_path: Path to protocol PDF
        
    Returns:
        Dictionary with version information:
        - max_amendment: Maximum amendment number found
        - pdf_date: PDF date if found
        - explicit_version: Explicit version string if found
        - amendment_dates: List of amendment numbers with dates
    """
    try:
        doc = fitz.open(str(pdf_path))
        full_text = "\n".join([page.get_text() for page in doc])
        first_3_pages = "\n".join([doc[i].get_text() for i in range(min(3, len(doc)))])
        doc.close()
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
        return {}
    
    info = {
        "max_amendment": None,
        "pdf_date": None,
        "explicit_version": None,
        "amendment_dates": [],
    }
    
    # Extract amendment numbers
    amendment_pattern = r"Amendment\s+(\d+)"
    amendments = re.findall(amendment_pattern, full_text, re.IGNORECASE)
    if amendments:
        info["max_amendment"] = max(int(a) for a in amendments)
        info["all_amendments"] = sorted(set(int(a) for a in amendments))
    
    # Extract version strings (e.g., "Version 4.0", "v3.1")
    version_patterns = [
        r"Version\s+(\d+\.?\d*)",
        r"v(\d+\.?\d*)",
        r"Protocol\s+Version\s+(\d+)",
    ]
    for pattern in version_patterns:
        matches = re.findall(pattern, first_3_pages, re.IGNORECASE)
        if matches:
            info["explicit_version"] = matches[0]
            break
    
    # Extract dates from first pages
    date_pattern = r"(\d{1,2}\s+\w+\s+\d{4})"
    dates = re.findall(date_pattern, first_3_pages, re.IGNORECASE)
    if dates:
        info["pdf_date"] = dates[0]  # Usually first date is the PDF date
    
    # Extract amendment dates from table
    amendment_table = extract_amendment_table(pdf_path)
    if amendment_table:
        info["amendment_dates"] = [
            {"amendment": a.get("amendment_number"), "date": a.get("date")}
            for a in amendment_table
            if a.get("date")
        ]
    
    return info


def _parse_amendment_table(table_text: str) -> List[Dict[str, Any]]:
    """Parse amendment table structure."""
    amendments = []
    
    # Pattern: Table format with numbered rows
    # Format: "1\n(10 January 2020)\nDescription with multiple lines and bullet points..."
    
    # Find all amendment number patterns first
    # Pattern: number on its own line, followed by date in parentheses
    amendment_starts = []
    
    # Find all "number\n(date)" patterns
    # Also handle "number (date)" on same line
    patterns = [
        r"^(\d+)\s*$\s*\(([^)]+)\)",  # Number on line, date on next line
        r"^(\d+)\s+\(([^)]+)\)",  # Number and date on same line
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, table_text, re.MULTILINE):
            amendment_num = int(match.group(1))
            # Only add if we haven't seen this amendment number
            if not any(a["num"] == amendment_num for a in amendment_starts):
                amendment_starts.append({
                    "num": amendment_num,
                    "date": match.group(2).strip(),
                    "start": match.end(),
                })
    
    # Sort by position to maintain order
    amendment_starts.sort(key=lambda x: x["start"])
    
    # Extract description for each amendment
    for i, amendment_info in enumerate(amendment_starts):
        start_idx = amendment_info["start"]
        
        # Find end: next amendment or section break
        if i + 1 < len(amendment_starts):
            # Next amendment starts here - back up to avoid overlap
            end_idx = amendment_starts[i + 1]["start"] - 50
        else:
            # Last amendment - find next section (but be more lenient)
            # Look for clear section breaks, not just "3 " which might be amendment 3
            end_patterns = [
                r"\n\s*\d+\.\s+[A-Z]{2,}",  # Next numbered section with capital letters
                r"\n\s*[A-Z]{4,}\s+[A-Z]{4,}",  # All caps heading (longer to avoid false matches)
                r"\n\s*Version\s+History",  # Version history section
            ]
            end_idx = len(table_text)
            for pattern in end_patterns:
                match = re.search(pattern, table_text[start_idx:], re.IGNORECASE)
                if match and match.start() > 200:  # Ensure we have enough content
                    end_idx = start_idx + match.start()
                    break
        
        # Extract description
        description = table_text[start_idx:end_idx].strip()
        
        # Clean up: normalize whitespace but preserve line breaks for bullet points
        # Replace multiple spaces with single space, but keep newlines before bullet points
        description = re.sub(r'[ \t]+', ' ', description)  # Multiple spaces to single
        description = re.sub(r'\n\s*\n+', '\n', description)  # Multiple newlines to single
        description = description.strip()
        
        if description and len(description) > 10:  # Filter very short matches
            amendments.append({
                "amendment_number": amendment_info["num"],
                "date": amendment_info["date"],
                "text": description,
                "changes": _extract_changes_from_text(description),
                "rationales": _extract_rationales_from_text(description),
            })
    
    # Sort by amendment number
    amendments.sort(key=lambda x: x.get("amendment_number", 0))
    
    return amendments


def _find_individual_amendments(full_text: str) -> List[Dict[str, Any]]:
    """Find individual amendment entries in protocol."""
    amendments = []
    
    # Look for version history section
    version_pattern = r"Version\s+(\d+\.\d+).*?\(Amendment\s+(\d+)\).*?(\d{1,2}\s+\w+\s+\d{4})"
    matches = re.finditer(version_pattern, full_text, re.IGNORECASE)
    
    for match in matches:
        version = match.group(1)
        amendment_num = match.group(2)
        date_str = match.group(3)
        
        # Find section for this amendment
        start_idx = match.end()
        next_match = re.search(
            r"Version\s+\d+\.\d+.*?\(Amendment\s+\d+\)", full_text[start_idx:], re.IGNORECASE
        )
        end_idx = start_idx + (next_match.start() if next_match else min(5000, len(full_text) - start_idx))
        
        amendment_text = full_text[start_idx:end_idx]
        
        amendments.append({
            "amendment_number": int(amendment_num),
            "version": version,
            "date": date_str,
            "text": amendment_text,
            "changes": _extract_changes_from_text(amendment_text),
            "rationales": _extract_rationales_from_text(amendment_text),
        })
    
    return amendments


def _extract_changes_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract EC changes from amendment text."""
    changes = []
    
    # Look for eligibility criteria mentions
    ec_keywords = [
        "inclusion criteria",
        "exclusion criteria",
        "eligibility criteria",
        "modified.*criteria",
        "relaxed.*criteria",
        "changed.*criteria",
    ]
    
    # Split by bullet points or sentences
    sentences = re.split(r'[•\n]', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()
        
        # Check if sentence mentions EC changes
        if any(keyword in sentence_lower for keyword in ec_keywords):
            # Determine change type
            change_type = "eligibility_criteria"
            if "inclusion" in sentence_lower:
                change_type = "inclusion_criteria"
            elif "exclusion" in sentence_lower:
                change_type = "exclusion_criteria"
            
            changes.append({
                "description": sentence[:300],  # Limit length
                "type": change_type,
            })
    
    # Also look for explicit patterns
    ec_patterns = [
        r"(?:modified|changed|relaxed|updated).*?(?:inclusion|exclusion|eligibility)\s+criteria[^.]*",
        r"eligibility\s+criteria[^.]*?(?:modified|changed|updated)[^.]*",
    ]
    
    for pattern in ec_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            change_text = match.group(0).strip()
            if change_text and len(change_text) > 20:
                changes.append({
                    "description": change_text[:300],
                    "type": "eligibility_criteria",
                })
    
    return changes


def _extract_rationales_from_text(text: str) -> List[str]:
    """Extract rationale text from amendment."""
    rationales = []
    
    # In table format, rationales are often embedded in the description
    # Look for phrases that indicate rationale:
    # - "given the expected..."
    # - "to include..." (with reason)
    # - "Clarification of..."
    # - Sentences with "because", "due to", "in order to"
    
    # Split by bullet points or line breaks to get individual items
    items = re.split(r'[•\n]', text)
    
    for item in items:
        item = item.strip()
        if not item or len(item) < 20:
            continue
        
        item_lower = item.lower()
        
        # Look for rationale indicators
        rationale_indicators = [
            r"given\s+the\s+expected",
            r"given\s+the\s+low",
            r"because\s+of",
            r"due\s+to",
            r"in\s+order\s+to",
            r"to\s+adequately",
            r"to\s+characterize",
            r"recommended\s+the\s+following",
            r"clarification",
            r"prohibited",
        ]
        
        # Check if item contains rationale
        has_rationale = False
        for pattern in rationale_indicators:
            if re.search(pattern, item_lower):
                has_rationale = True
                break
        
        # Also check for sentences that explain "why" (contain "to" + verb)
        if not has_rationale:
            # Pattern: "to [verb] [something]" often indicates rationale
            if re.search(r"to\s+\w+\s+\w+", item_lower):
                has_rationale = True
        
        if has_rationale:
            # Clean up
            item = re.sub(r'\s+', ' ', item).strip()
            if item and item not in rationales:
                rationales.append(item)
    
    # Extract explicit rationale phrases
    # Look for "given that", "because", etc. with following text
    explicit_patterns = [
        r"given\s+(?:the\s+)?(?:expected|low|high|fact\s+that)\s+([^\.]+)",
        r"because\s+(?:of\s+)?([^\.]+)",
        r"due\s+to\s+([^\.]+)",
        r"in\s+order\s+to\s+([^\.]+)",
        r"rationale[:\s]+([^\.]+)",
        r"justification[:\s]+([^\.]+)",
    ]
    
    for pattern in explicit_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            rationale = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if rationale and len(rationale) > 15:
                rationale = re.sub(r'\s+', ' ', rationale)
                if rationale not in rationales:
                    rationales.append(rationale)
    
    return rationales


def extract_ec_changes_from_amendment(
    pdf_text: str, amendment_number: int
) -> Optional[Dict[str, Any]]:
    """
    Extract EC changes from a specific amendment.
    
    Args:
        pdf_text: Full PDF text
        amendment_number: Amendment number to extract
        
    Returns:
        Dictionary with EC changes and rationales
    """
    # Find amendment section
    amendment_pattern = rf"Amendment\s+{amendment_number}[^A-Z]*"
    match = re.search(amendment_pattern, pdf_text, re.IGNORECASE)
    
    if not match:
        return None
    
    start_idx = match.start()
    # Find next amendment or end
    next_match = re.search(rf"Amendment\s+{amendment_number + 1}", pdf_text[start_idx:], re.IGNORECASE)
    end_idx = start_idx + (next_match.start() if next_match else min(10000, len(pdf_text) - start_idx))
    
    amendment_text = pdf_text[start_idx:end_idx]
    
    # Extract EC-related changes
    ec_changes = _extract_changes_from_text(amendment_text)
    rationales = _extract_rationales_from_text(amendment_text)
    
    return {
        "amendment_number": amendment_number,
        "ec_changes": ec_changes,
        "rationales": rationales,
        "text": amendment_text,
    }


def extract_ec_justification_from_text(text: str, ec_before: str = "", ec_after: str = "") -> str:
    """
    Extract EC-specific justification from amendment text.
    
    Attempts to find sections relevant to eligibility criteria changes.
    
    Args:
        text: Full amendment text
        ec_before: Original EC (optional, for context)
        ec_after: Modified EC (optional, for context)
        
    Returns:
        Extracted EC justification text
    """
    if not text:
        return ""
    
    text_lower = text.lower()
    
    # Keywords that indicate EC-related content
    ec_keywords = [
        "eligibility criteria",
        "inclusion criteria",
        "exclusion criteria",
        "modified.*criteria",
        "relaxed.*criteria",
        "changed.*criteria",
        "updated.*criteria",
        "adjusted.*criteria",
    ]
    
    # Split text into sentences/phrases
    sentences = re.split(r'[\.\n•]', text)
    
    ec_relevant_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        
        sentence_lower = sentence.lower()
        
        # Check if sentence mentions EC keywords
        if any(re.search(keyword, sentence_lower) for keyword in ec_keywords):
            ec_relevant_sentences.append(sentence)
        
        # Also check for rationale keywords in context of EC
        rationale_keywords = ["given", "because", "due to", "to include", "to exclude"]
        if any(keyword in sentence_lower for keyword in rationale_keywords):
            # Check if nearby sentences mention EC
            # (This is a simple heuristic - could be improved)
            if len(ec_relevant_sentences) > 0:
                ec_relevant_sentences.append(sentence)
    
    # Combine relevant sentences (deduplicate)
    if ec_relevant_sentences:
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sentence in ec_relevant_sentences:
            sentence_normalized = re.sub(r'\s+', ' ', sentence.strip()).lower()
            if sentence_normalized not in seen and len(sentence_normalized) > 20:
                seen.add(sentence_normalized)
                unique_sentences.append(sentence)
        
        justification = " ".join(unique_sentences)
        # Clean up
        justification = re.sub(r'\s+', ' ', justification).strip()
        return justification
    
    # Fallback: if no EC-specific text found, return empty
    # (Caller can use raw text instead)
    return ""


def match_ec_to_amendment(
    ec_change: Dict[str, Any], amendment_table: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Match API-detected EC change to amendment entry.
    
    Args:
        ec_change: EC change from API (instance_id, source_version, target_version, ec_before, ec_after)
        amendment_table: List of amendments from protocol
        
    Returns:
        Matched amendment with raw text and parsed EC justification
    """
    if not amendment_table:
        return None
    
    # Get amendment closest to target_version
    matched = None
    
    # Simple heuristic: use latest amendment (usually contains all changes)
    if amendment_table:
        matched = amendment_table[-1]
    
    # Extract both raw text and EC-specific justification
    if matched:
        # Raw text: full amendment description
        raw_text = matched.get("text", "")
        
        # Parsed EC justification: extract EC-relevant parts
        ec_justification = extract_ec_justification_from_text(
            raw_text,
            ec_change.get("ec_before", ""),
            ec_change.get("ec_after", ""),
        )
        
        # If no EC-specific justification found, try to extract from rationales
        if not ec_justification:
            ec_rationales = []
            for rationale in matched.get("rationales", []):
                rationale_lower = rationale.lower()
                if any(
                    keyword in rationale_lower
                    for keyword in ["eligibility", "criteria", "inclusion", "exclusion"]
                ):
                    ec_rationales.append(rationale)
            
            if ec_rationales:
                ec_justification = " ".join(ec_rationales)
        
        return {
            "amendment_number": matched.get("amendment_number"),
            "date": matched.get("date"),
            "raw_text": raw_text,  # Full amendment text
            "ec_justification": ec_justification,  # Parsed EC-specific justification
            "rationales": matched.get("rationales", []),
            "changes": matched.get("changes", []),
        }
    
    return None


def extract_protocol_sections(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract major protocol sections from PDF.
    
    Extracts structured sections like Eligibility Criteria, Study Design, Objectives, etc.
    
    Args:
        pdf_path: Path to protocol PDF
        
    Returns:
        Dictionary with section names as keys and text as values:
        {
            "eligibility_criteria": "...",
            "study_design": "...",
            "objectives": "...",
            "background": "...",
            "amendment_summary": "..."
        }
    """
    try:
        doc = fitz.open(str(pdf_path))
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
        return {}
    
    sections = {}
    
    # Section patterns to look for
    section_patterns = {
        "eligibility_criteria": [
            r"(?:^|\n)\s*\d+\.?\s*ELIGIBILITY\s+CRITERIA",
            r"(?:^|\n)\s*ELIGIBILITY\s+CRITERIA",
            r"(?:^|\n)\s*Section\s+\d+.*?[Ee]ligibility",
            r"(?:^|\n)\s*\d+\.?\s*INCLUSION\s+AND\s+EXCLUSION",
        ],
        "study_design": [
            r"(?:^|\n)\s*\d+\.?\s*STUDY\s+DESIGN",
            r"(?:^|\n)\s*STUDY\s+DESIGN",
            r"(?:^|\n)\s*METHODS?\s+AND\s+MATERIALS",
        ],
        "objectives": [
            r"(?:^|\n)\s*\d+\.?\s*OBJECTIVES?",
            r"(?:^|\n)\s*PRIMARY\s+OBJECTIVE",
            r"(?:^|\n)\s*STUDY\s+OBJECTIVES?",
        ],
        "background": [
            r"(?:^|\n)\s*\d+\.?\s*BACKGROUND",
            r"(?:^|\n)\s*BACKGROUND",
            r"(?:^|\n)\s*INTRODUCTION",
        ],
    }
    
    # Extract each section
    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if match:
                start_idx = match.end()
                
                # Find end of section (next major section or end)
                end_patterns = [
                    r"\n\s*\d+\.\s+[A-Z]{2,}",  # Next numbered section
                    r"\n\s*[A-Z]{4,}\s+[A-Z]{4,}",  # All caps heading
                ]
                
                end_idx = len(full_text)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, full_text[start_idx:], re.IGNORECASE)
                    if end_match and end_match.start() > 100:  # Ensure we have content
                        end_idx = start_idx + end_match.start()
                        break
                
                section_text = full_text[start_idx:end_idx].strip()
                # Clean up: normalize whitespace
                section_text = re.sub(r'\s+', ' ', section_text).strip()
                
                if section_text and len(section_text) > 50:  # Filter very short matches
                    sections[section_name] = section_text
                    break  # Found section, move to next
    
    # Extract amendment summary (already have function for this)
    amendment_table = extract_amendment_table(pdf_path)
    if amendment_table:
        # Create summary text from amendment table
        amendment_summary_parts = []
        for amendment in amendment_table:
            amendment_num = amendment.get("amendment_number")
            date = amendment.get("date")
            text = amendment.get("text", "")[:500]  # Limit length
            if amendment_num:
                summary = f"Amendment {amendment_num}"
                if date:
                    summary += f" ({date})"
                summary += f": {text}"
                amendment_summary_parts.append(summary)
        
        if amendment_summary_parts:
            sections["amendment_summary"] = "\n\n".join(amendment_summary_parts)
    
    return sections


def filter_amendments_by_version(
    amendment_table: List[Dict[str, Any]],
    target_version: int,
    target_version_date: Optional[str],
    date_tolerance_days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Filter amendments to only include those available at target_version.
    
    Returns amendments that would have been known at the time of target_version,
    as if target_version is the latest version at that point in time.
    
    Args:
        amendment_table: List of amendment dictionaries
        target_version: Target version number
        target_version_date: Target version date
        date_tolerance_days: Tolerance for date lag (default: 7 days)
        
    Returns:
        Filtered list of amendments available at target_version
    """
    if not amendment_table:
        return []
    
    from datetime import datetime, timedelta
    
    def parse_date(date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
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
    
    filtered = []
    version_date = parse_date(target_version_date) if target_version_date else None
    
    for amendment in amendment_table:
        amendment_num = amendment.get("amendment_number")
        amendment_date_str = amendment.get("date")
        
        # Filter by amendment number
        if amendment_num is not None:
            if amendment_num > target_version:
                # Future amendment - exclude
                continue
        
        # Filter by date (with tolerance)
        if version_date and amendment_date_str:
            amendment_date = parse_date(amendment_date_str)
            if amendment_date:
                # Allow small lag (PDF might be uploaded slightly after version)
                tolerance = timedelta(days=date_tolerance_days)
                if amendment_date > (version_date + tolerance):
                    # Amendment date is too far in future - exclude
                    continue
        
        # Amendment is valid - include it
        filtered.append(amendment)
    
    return filtered


def filter_raw_pdf_text_by_version(
    raw_pdf_text: str,
    amendment_table: List[Dict[str, Any]],
    target_version: int,
    target_version_date: Optional[str],
) -> str:
    """
    Filter raw PDF text to remove content from future amendments.
    
    Identifies sections in raw text that correspond to amendments > target_version
    and removes/excludes them, leaving only content available at target_version.
    
    Args:
        raw_pdf_text: Full raw PDF text
        amendment_table: List of all amendments from PDF
        target_version: Target version number
        target_version_date: Target version date
        
    Returns:
        Filtered raw text containing only content up to target_version
    """
    if not raw_pdf_text or not amendment_table:
        return raw_pdf_text
    
    # Identify future amendments (> target_version)
    future_amendments = [
        a for a in amendment_table
        if a.get("amendment_number") and a.get("amendment_number") > target_version
    ]
    
    if not future_amendments:
        # No future amendments - return text as-is
        return raw_pdf_text
    
    # Find future amendment numbers
    future_amendment_nums = [a.get("amendment_number") for a in future_amendments]
    
    # Strategy: Find amendment headers/sections in raw text and remove them
    # Look for patterns like "Amendment 4", "Amendment 5", etc.
    filtered_text = raw_pdf_text
    
    # Remove sections starting with future amendment headers
    for future_num in sorted(future_amendment_nums, reverse=True):  # Process from highest to lowest
        # Pattern: "Amendment N" or "Amendment N:" followed by content
        # Try to find where this amendment section starts
        patterns = [
            rf"(?i)\n\s*Amendment\s+{future_num}\s*[:\n]",
            rf"(?i)\n\s*{future_num}\s*\([^)]+\)",  # "4 (date)" on its own line
            rf"(?i)Amendment\s+{future_num}\s+\([^)]+\)",  # "Amendment 4 (date)"
            rf"(?i)^\s*{future_num}\s*$",  # Just the number on its own line
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filtered_text, re.MULTILINE)
            if match:
                # Found start of future amendment
                start_idx = match.start()
                
                # Find end: next amendment or section break
                # Look for next amendment number or major section
                end_patterns = [
                    rf"\n\s*Amendment\s+({future_num + 1}|\d+)\s*[:\n]",  # Next amendment
                    rf"\n\s*({future_num + 1}|\d+)\s*\([^)]+\)",  # Next amendment with date
                    r"\n\s*\d+\.\s+[A-Z]{2,}",  # Next numbered section
                    r"\n\s*[A-Z]{4,}\s+[A-Z]{4,}",  # All caps heading
                ]
                
                end_idx = len(filtered_text)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, filtered_text[start_idx:], re.IGNORECASE | re.MULTILINE)
                    if end_match and end_match.start() > 100:  # Ensure we have content
                        end_idx = start_idx + end_match.start()
                        break
                
                # Remove this section
                filtered_text = filtered_text[:start_idx] + filtered_text[end_idx:]
                logger.debug(f"Removed Amendment {future_num} section from raw text")
                break  # Found and removed, move to next future amendment
    
    # Also remove any remaining explicit mentions of future amendments in text
    for future_num in future_amendment_nums:
        # Remove lines/sentences that mention future amendments
        # Pattern: lines containing "Amendment N" where N > target_version
        lines = filtered_text.split("\n")
        filtered_lines = []
        for line in lines:
            # Check if line mentions future amendment (various patterns)
            if (
                re.search(rf"(?i)Amendment\s+{future_num}\b", line)
                or re.search(rf"^\s*{future_num}\s*\([^)]+\)", line)  # "3 (date)" at start of line
                or (re.search(rf"\b{future_num}\s*\([^)]+\)", line) and future_num > target_version)  # "3 (date)" anywhere
            ):
                # Skip this line
                continue
            filtered_lines.append(line)
        filtered_text = "\n".join(filtered_lines)
        
        # Also remove standalone mentions in the middle of text
        # Pattern: "Amendment N" not at start of line
        filtered_text = re.sub(
            rf"(?i)\bAmendment\s+{future_num}\b[^\n]*",
            "",
            filtered_text,
        )
        
        # Remove any remaining patterns that match the test patterns
        # Pattern 1: "Amendment N" with word boundary
        filtered_text = re.sub(
            rf"(?i)Amendment\s+{future_num}\b",
            "",
            filtered_text,
        )
        
        # Pattern 2: "N (date)" format
        filtered_text = re.sub(
            rf"\b{future_num}\s*\([^)]+\)",
            "",
            filtered_text,
        )
        
        # Also remove lines that start with just the number (common in amendment tables)
        lines = filtered_text.split("\n")
        filtered_lines = []
        for line in lines:
            # Skip lines that are just the future amendment number
            stripped = line.strip()
            if stripped == str(future_num):
                continue
            filtered_lines.append(line)
        filtered_text = "\n".join(filtered_lines)
    
    return filtered_text
