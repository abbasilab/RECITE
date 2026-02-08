"""
Smoke tests for directive and impact extraction (parsing module).

Tests the core LLM-response parsing logic: JSON extraction, heuristic fallbacks,
normalization. No LLM calls — feeds synthetic LLM outputs directly.
"""

from recite.accrual.parsing import (
    parse_directives_response,
    parse_impact_response,
    _robust_json_parse,
    _normalize_location,
)


# ---------------------------------------------------------------------------
# JSON Parsing Helpers
# ---------------------------------------------------------------------------


class TestRobustJsonParse:
    """Test JSON extraction from messy LLM output."""

    def test_clean_json(self):
        result = _robust_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        result = _robust_json_parse('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_trailing_comma(self):
        result = _robust_json_parse('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}

    def test_comma_in_number(self):
        """Commas inside numbers (e.g. '4,851') should be stripped."""
        result = _robust_json_parse('{"count": 4851}')
        assert result["count"] == 4851

    def test_empty_input(self):
        assert _robust_json_parse("") is None
        assert _robust_json_parse(None) is None


class TestNormalizeLocation:
    """Test free-text location normalization."""

    def test_exact(self):
        assert _normalize_location("exact") == "exact"

    def test_in_paper_variants(self):
        assert _normalize_location("in paper") == "in_paper"
        assert _normalize_location("In the paper") == "in_paper"
        assert _normalize_location("within the paper") == "in_paper"

    def test_none_variants(self):
        assert _normalize_location("none") == "none"
        assert _normalize_location("not found") == "none"
        assert _normalize_location("") == "none"
        assert _normalize_location(None) == "none"


# ---------------------------------------------------------------------------
# Directive Parsing
# ---------------------------------------------------------------------------


class TestDirectivesParsing:
    """Test extraction of modification directives from LLM responses."""

    def test_json_response_with_directives(self):
        """Well-formed JSON with directives should parse correctly."""
        response = '''{
            "answer_location": "exact",
            "directives_exact_text": [
                "Expand age limit to 75 years",
                "Relax renal threshold to eGFR 45"
            ],
            "directives_count": 2
        }'''
        result = parse_directives_response(response)
        assert result["directives_answer_location"] == "exact"
        assert result["directives_has_answer"] == 1
        assert "Expand age limit" in result["directives_exact_text"]
        assert "Relax renal" in result["directives_exact_text"]
        assert result["directives_count"] == 2

    def test_json_response_no_directives(self):
        """JSON response indicating no directives found."""
        response = '{"answer_location": "none", "directives_exact_text": null}'
        result = parse_directives_response(response)
        assert result["directives_answer_location"] == "none"
        assert result["directives_has_answer"] == 0

    def test_markdown_wrapped_json(self):
        """JSON wrapped in markdown code fences should still parse."""
        response = '```json\n{"answer_location": "exact", "directives_exact_text": "Raise age to 75"}\n```'
        result = parse_directives_response(response)
        assert result["directives_has_answer"] == 1
        assert "Raise age" in result["directives_exact_text"]

    def test_heuristic_fallback(self):
        """Plain-text response should trigger heuristic parsing."""
        response = "Yes, the following directives are in the paper: raise age limits to 75."
        result = parse_directives_response(response)
        assert result["directives_has_answer"] == 1
        # Contains "in the paper" so heuristic categorizes as in_paper
        assert result["directives_answer_location"] in ("exact", "in_paper")

    def test_empty_response(self):
        result = parse_directives_response("")
        assert result["directives_has_answer"] == 0
        assert result["directives_answer_location"] == "none"

    def test_none_response(self):
        result = parse_directives_response(None)
        assert result["directives_has_answer"] == 0

    def test_string_directive_text(self):
        """Single-string directive_text field should work."""
        response = '{"answer_location": "exact", "directive_text": "Relax eGFR to 45"}'
        result = parse_directives_response(response)
        assert result["directives_exact_text"] == "Relax eGFR to 45"


# ---------------------------------------------------------------------------
# Impact Parsing
# ---------------------------------------------------------------------------


class TestImpactParsing:
    """Test extraction of accrual impact estimates from LLM responses."""

    def test_json_response_with_percent(self):
        """Well-formed JSON with percentage impact."""
        response = '''{
            "answer_location": "exact",
            "impact_percent": 35.0,
            "impact_unit": "percent",
            "impact_qualitative": "Substantial increase in eligible pool"
        }'''
        result = parse_impact_response(response)
        assert result["impact_has_answer"] == 1
        assert result["impact_percent"] == 35.0
        assert result["impact_unit"] == "percent"
        assert "Substantial" in result["impact_qualitative"]

    def test_json_response_with_absolute(self):
        """JSON with absolute patient count impact."""
        response = '{"answer_location": "exact", "impact_absolute": 120, "impact_unit": "patients"}'
        result = parse_impact_response(response)
        assert result["impact_has_answer"] == 1
        assert result["impact_absolute"] == 120.0

    def test_heuristic_percent_extraction(self):
        """Plain-text response with percentage should be extracted via regex."""
        response = "The study reported a 28% increase in enrollment after the amendment."
        result = parse_impact_response(response)
        assert result["impact_percent"] == 28.0
        assert result["impact_has_answer"] == 1

    def test_heuristic_absolute_extraction(self):
        """Plain-text with 'N more patients' should be extracted."""
        response = "The change yielded 50 more patients enrolled."
        result = parse_impact_response(response)
        assert result["impact_absolute"] == 50.0
        assert result["impact_has_answer"] == 1

    def test_empty_response(self):
        result = parse_impact_response("")
        assert result["impact_has_answer"] == 0
        assert result["impact_percent"] is None

    def test_none_response(self):
        result = parse_impact_response(None)
        assert result["impact_has_answer"] == 0

    def test_json_with_comma_in_number(self):
        """Numbers with commas (e.g., '4,851') should parse correctly."""
        response = '{"answer_location": "exact", "impact_absolute": "4,851", "impact_unit": "patients"}'
        result = parse_impact_response(response)
        assert result["impact_absolute"] == 4851.0

    def test_no_impact_found(self):
        """Response indicating no impact data should return zeros."""
        response = '{"answer_location": "none"}'
        result = parse_impact_response(response)
        assert result["impact_has_answer"] == 0
        assert result["impact_percent"] is None
        assert result["impact_absolute"] is None
