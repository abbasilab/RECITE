"""
Parse LLM responses for directives and impact into paper_answers columns.
Handles JSON or fallback heuristics; partial answers and failures gracefully.
"""

import json
import re
from typing import Any, Dict, Optional

from loguru import logger

# Valid answer_location values
ANSWER_LOCATIONS = ("exact", "in_paper", "none")


def _normalize_location(s: Optional[str]) -> str:
    if not s or not s.strip():
        return "none"
    t = s.strip().lower()
    if t in ("exact", "in_paper", "in paper"):
        return "in_paper" if "paper" in t else "exact"
    if "paper" in t or "within" in t or "elsewhere" in t:
        return "in_paper"
    if "no" in t or "none" in t or "not" in t:
        return "none"
    return "exact" if "yes" in t or "present" in t else "none"


def _robust_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from LLM response (strip markdown, trailing commas)."""
    if not text or not text.strip():
        return None
    t = text.strip()
    # Strip markdown code block
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    # Remove trailing commas before ] or }
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    # Remove commas inside numbers (e.g. 4,851 -> 4851) so JSON is valid
    while "," in t and re.search(r"\d,\d", t):
        t = re.sub(r"(\d),(\d)", r"\1\2", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def parse_directives_response(raw_response: Optional[str]) -> Dict[str, Any]:
    """
    Parse LLM response for Question 1 (directives).
    Returns dict with: directives_answer_location, directives_has_answer, directives_exact_text, directives_count.
    """
    out = {
        "directives_answer_location": "none",
        "directives_has_answer": 0,
        "directives_exact_text": None,
        "directives_count": None,
    }
    if not raw_response or not raw_response.strip():
        return out

    data = _robust_json_parse(raw_response)
    if data is not None:
        loc = data.get("answer_location") or data.get("location") or data.get("directives_answer_location")
        out["directives_answer_location"] = _normalize_location(str(loc) if loc is not None else None)
        out["directives_has_answer"] = 1 if out["directives_answer_location"] in ("exact", "in_paper") else 0
        raw_text = data.get("directives_exact_text") or data.get("directive_text") or data.get("exact_text")
        if raw_text is not None:
            if isinstance(raw_text, list):
                out["directives_exact_text"] = ("\n".join(str(x).strip() for x in raw_text if x)).strip() or None
            elif isinstance(raw_text, str):
                out["directives_exact_text"] = raw_text.strip() or None
        cnt = data.get("directives_count") or data.get("directive_count")
        if cnt is not None:
            try:
                out["directives_count"] = int(cnt) if not isinstance(cnt, list) else (int(cnt[0]) if cnt else None)
            except (TypeError, ValueError, IndexError):
                pass
        return out

    # Heuristic: look for "exact", "in paper", "none" and directive list
    t = raw_response.strip().lower()
    if "in the paper" in t or "within the paper" in t or "exists within" in t:
        out["directives_answer_location"] = "in_paper"
        out["directives_has_answer"] = 1
    elif "exact" in t or "yes," in t or "following" in t or ":" in raw_response:
        out["directives_answer_location"] = "exact"
        out["directives_has_answer"] = 1
        # Use full response as fallback for exact text (user may have listed directives)
        if len(raw_response.strip()) > 50:
            out["directives_exact_text"] = raw_response.strip()[:10000]
    return out


def parse_impact_response(raw_response: Optional[str]) -> Dict[str, Any]:
    """
    Parse LLM response for Question 2 (impact).
    Returns dict with: impact_answer_location, impact_has_answer, impact_percent, impact_absolute, impact_unit, impact_qualitative.
    """
    out = {
        "impact_answer_location": "none",
        "impact_has_answer": 0,
        "impact_percent": None,
        "impact_absolute": None,
        "impact_unit": None,
        "impact_qualitative": None,
        "impact_evidence": None,
    }
    if not raw_response or not raw_response.strip():
        return out

    data = _robust_json_parse(raw_response)
    if data is not None:
        loc = data.get("answer_location") or data.get("location") or data.get("impact_answer_location")
        out["impact_answer_location"] = _normalize_location(str(loc) if loc is not None else None)
        out["impact_has_answer"] = 1 if out["impact_answer_location"] in ("exact", "in_paper") else 0
        pct = data.get("impact_percent")
        if pct is not None:
            try:
                if isinstance(pct, list):
                    pct = pct[0] if pct else None
                if pct is not None:
                    if isinstance(pct, str):
                        pct = pct.replace(",", "")
                    out["impact_percent"] = float(pct)
            except (TypeError, ValueError, IndexError):
                pass
        abs_val = data.get("impact_absolute")
        if abs_val is not None:
            try:
                if isinstance(abs_val, list):
                    abs_val = abs_val[0] if abs_val else None
                if abs_val is not None:
                    if isinstance(abs_val, str):
                        abs_val = abs_val.replace(",", "")
                    out["impact_absolute"] = float(abs_val)
            except (TypeError, ValueError, IndexError):
                pass
        raw_unit = data.get("impact_unit")
        if raw_unit is not None:
            if isinstance(raw_unit, list):
                out["impact_unit"] = str(raw_unit[0]).strip() if raw_unit else None
            elif isinstance(raw_unit, str):
                out["impact_unit"] = raw_unit.strip() or None
        raw_qual = data.get("impact_qualitative")
        if raw_qual is not None:
            if isinstance(raw_qual, list):
                out["impact_qualitative"] = ("\n".join(str(x) for x in raw_qual if x)).strip() or None
            elif isinstance(raw_qual, str):
                out["impact_qualitative"] = raw_qual.strip() or None
        raw_evidence = data.get("impact_evidence")
        if raw_evidence is not None:
            if isinstance(raw_evidence, list):
                out["impact_evidence"] = ("\n".join(str(x).strip() for x in raw_evidence if x)).strip() or None
            elif isinstance(raw_evidence, str):
                out["impact_evidence"] = raw_evidence.strip() or None
        return out

    # Heuristic: look for a percentage number (e.g. 15%, 15 percent)
    t = raw_response.strip()
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*percent", t, re.I)
    if pct_match:
        g = pct_match.group(1) or pct_match.group(2)
        try:
            out["impact_percent"] = float(g)
            out["impact_answer_location"] = "exact"
            out["impact_has_answer"] = 1
            out["impact_unit"] = "percent"
        except (TypeError, ValueError):
            pass
    # Absolute number (e.g. "50 more patients", "increase of 50")
    abs_match = re.search(r"(\d+)\s*(?:more|additional)\s*(?:patients|participants)?", t, re.I)
    if abs_match and out["impact_percent"] is None:
        try:
            out["impact_absolute"] = float(abs_match.group(1))
            out["impact_answer_location"] = "exact"
            out["impact_has_answer"] = 1
            out["impact_unit"] = "patients"
        except (TypeError, ValueError):
            pass
    if "in the paper" in t.lower() or "within the paper" in t.lower():
        out["impact_answer_location"] = "in_paper"
        out["impact_has_answer"] = 1
    return out
