"""
Smoke tests for RECITE benchmark construction pipeline.

Tests the core algorithms: EC diff detection, text cleaning, triplet creation.
No LLM calls — exercises pure algorithmic logic on synthetic fixtures.
"""

import sqlite3

from recite.benchmark.ec_detector import (
    detect_ec_changes,
    _normalize_text,
    _calculate_similarity,
)
from recite.benchmark.builders import create_recite_instances
from recite.benchmark.utils import clean_text, normalize_nct_id


# ---------------------------------------------------------------------------
# EC Diff Detection
# ---------------------------------------------------------------------------


class TestECDetector:
    """Test eligibility-criteria change detection (difflib-based)."""

    def test_detects_meaningful_change(self, trial_record):
        """When age/BMI/eGFR thresholds change, detect_ec_changes should flag it."""
        result = detect_ec_changes(
            trial_record["original_ec"],
            trial_record["revised_ec"],
        )
        assert result["has_change"] is True
        assert result["similarity"] < 1.0
        assert result["change_ratio"] > 0
        # Per-section similarity >0.95 means change_type heuristic may return None
        # even when overall change is detected — this is correct behavior
        assert result["changed_lines"] > 0

    def test_identical_texts_no_change(self):
        """Identical texts should report no change."""
        ec = "Inclusion Criteria:\n- Age 18 to 65\n\nExclusion Criteria:\n- None"
        result = detect_ec_changes(ec, ec)
        assert result["has_change"] is False
        assert result["similarity"] == 1.0
        assert result["change_ratio"] == 0.0

    def test_empty_input(self):
        """Empty or missing EC text should return has_change=False."""
        result = detect_ec_changes("", "some text")
        assert result["has_change"] is False

    def test_whitespace_normalization(self):
        """Texts differing only in whitespace should be treated as identical."""
        ec1 = "Inclusion Criteria:\n  - Age 18 to 65\n\n"
        ec2 = "Inclusion Criteria:\n- Age 18 to 65"
        result = detect_ec_changes(ec1, ec2)
        assert result["has_change"] is False

    def test_inclusion_only_change(self):
        """Change only in inclusion section should be typed 'inclusion'."""
        ec1 = "Inclusion Criteria:\n- Age 18 to 65\n- BMI 20-30\n- HbA1c 6-9%\n\nExclusion Criteria:\n- Active cancer"
        ec2 = "Inclusion Criteria:\n- Age 18 to 85\n- BMI 18-40\n- HbA1c 5-12%\n\nExclusion Criteria:\n- Active cancer"
        result = detect_ec_changes(ec1, ec2)
        assert result["has_change"] is True
        assert result["change_type"] == "inclusion"

    def test_similarity_score_range(self, trial_record):
        """Similarity should be between 0 and 1."""
        result = detect_ec_changes(
            trial_record["original_ec"],
            trial_record["revised_ec"],
        )
        assert 0.0 <= result["similarity"] <= 1.0

    def test_below_threshold_not_flagged(self):
        """A tiny change below the threshold should not be flagged."""
        ec1 = "Inclusion Criteria:\n- Age 18 to 65\n- BMI 25-40\n- HbA1c 7-10%\n- Diabetes for 1 year\n- Able to consent\n- English speaking"
        # Change just one character (65 -> 66) — small change relative to total
        ec2 = "Inclusion Criteria:\n- Age 18 to 66\n- BMI 25-40\n- HbA1c 7-10%\n- Diabetes for 1 year\n- Able to consent\n- English speaking"
        result = detect_ec_changes(ec1, ec2, min_change_threshold=0.5, similarity_threshold=0.5)
        # With very high thresholds, a tiny change should not be flagged
        assert result["has_change"] is False


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------


class TestTextUtils:
    """Test text cleaning and normalization utilities."""

    def test_clean_text_strips_html(self):
        assert clean_text("<b>Bold</b> text") == "Bold text"

    def test_clean_text_normalizes_whitespace(self):
        assert clean_text("  too   many    spaces  ") == "too many spaces"

    def test_clean_text_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_normalize_nct_id_standard(self):
        assert normalize_nct_id("NCT00000001") == "NCT00000001"

    def test_normalize_nct_id_digits_only(self):
        assert normalize_nct_id("00000001") == "NCT00000001"

    def test_normalize_nct_id_with_spaces(self):
        assert normalize_nct_id(" NCT 00000001 ") == "NCT00000001"

    def test_normalize_nct_id_invalid(self):
        assert normalize_nct_id("INVALID") is None

    def test_normalize_text(self):
        """Internal normalize function collapses whitespace and strips empties."""
        assert _normalize_text("  hello  \n\n  world  ") == "hello\nworld"

    def test_calculate_similarity_identical(self):
        assert _calculate_similarity("abc", "abc") == 1.0

    def test_calculate_similarity_different(self):
        sim = _calculate_similarity("abc", "xyz")
        assert sim < 0.5


# ---------------------------------------------------------------------------
# Triplet Creation (Builder)
# ---------------------------------------------------------------------------


class TestBuilder:
    """Test RECITE instance (triplet) creation from EC changes + protocol texts."""

    def test_creates_instance_from_fixture(self, tmp_db, trial_record, protocol_text):
        """Insert fixture data into DB, run builder, verify a RECITE instance is created."""
        # Insert an EC change
        tmp_db.execute(
            """INSERT INTO ec_changes
               (nct_id, version_from, version_to, ec_before, ec_after,
                change_type, similarity, change_ratio, evidence_source_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trial_record["nct_id"],
                trial_record["version_from"],
                trial_record["version_to"],
                trial_record["original_ec"],
                trial_record["revised_ec"],
                "both",
                0.75,
                0.25,
                "protocols/NCT00000001_v3.pdf",
            ),
        )
        # Insert matching protocol text
        tmp_db.execute(
            "INSERT INTO protocol_texts (nct_id, raw_text) VALUES (?, ?)",
            (trial_record["nct_id"], protocol_text),
        )
        tmp_db.commit()

        # Run builder
        create_recite_instances(max_trials=10, conn=tmp_db)
        tmp_db.commit()

        # Verify
        rows = tmp_db.execute("SELECT * FROM recite").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["nct_id"] == "NCT00000001"
        assert len(row["preamended_text"]) > 0
        assert len(row["amended_text"]) > 0
        assert len(row["evidence"]) > 0
        assert row["evidence_extraction_level"] == "raw_pdf_text_only"
        assert row["evidence_extraction_score"] == 1

    def test_skips_duplicate(self, tmp_db, trial_record, protocol_text):
        """Running builder twice should not create duplicates."""
        tmp_db.execute(
            """INSERT INTO ec_changes
               (nct_id, version_from, version_to, ec_before, ec_after, evidence_source_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                trial_record["nct_id"],
                trial_record["version_from"],
                trial_record["version_to"],
                trial_record["original_ec"],
                trial_record["revised_ec"],
                "protocols/NCT00000001_v3.pdf",
            ),
        )
        tmp_db.execute(
            "INSERT INTO protocol_texts (nct_id, raw_text) VALUES (?, ?)",
            (trial_record["nct_id"], protocol_text),
        )
        tmp_db.commit()

        create_recite_instances(max_trials=10, conn=tmp_db)
        tmp_db.commit()
        create_recite_instances(max_trials=10, conn=tmp_db)
        tmp_db.commit()

        rows = tmp_db.execute("SELECT * FROM recite").fetchall()
        assert len(rows) == 1  # no duplicates

    def test_empty_protocol_text_scores_zero(self, tmp_db, trial_record):
        """When protocol text is too short, extraction_score should be 0."""
        tmp_db.execute(
            """INSERT INTO ec_changes
               (nct_id, version_from, version_to, ec_before, ec_after, evidence_source_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                trial_record["nct_id"],
                trial_record["version_from"],
                trial_record["version_to"],
                trial_record["original_ec"],
                trial_record["revised_ec"],
                "protocols/NCT00000001_v3.pdf",
            ),
        )
        tmp_db.execute(
            "INSERT INTO protocol_texts (nct_id, raw_text) VALUES (?, ?)",
            (trial_record["nct_id"], "short"),
        )
        tmp_db.commit()

        create_recite_instances(max_trials=10, conn=tmp_db)
        tmp_db.commit()

        row = tmp_db.execute("SELECT * FROM recite").fetchone()
        assert row["evidence_extraction_level"] == "pdf_document_only"
        assert row["evidence_extraction_score"] == 0
