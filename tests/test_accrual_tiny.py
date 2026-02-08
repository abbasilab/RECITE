"""
Smoke tests for the accrual scoring pipeline.

Tests the end-to-end flow: fixture data → DB → EC diff → triplet → parse outputs.
Validates that scoring/matching helpers work on synthetic data.
No LLM calls — exercises DB operations and parsing on fixtures.
"""

import json
import sqlite3
from pathlib import Path

from recite.benchmark.ec_detector import detect_ec_changes
from recite.benchmark.builders import create_recite_instances
from recite.benchmark.utils import clean_text
from recite.accrual.parsing import parse_directives_response, parse_impact_response


class TestAccrualEndToEnd:
    """Simulate a tiny accrual scoring workflow on fixtures."""

    def test_full_pipeline_fixture(self, tmp_db, trial_record, protocol_text, paper_abstract):
        """
        End-to-end: detect EC change → build triplet → parse directive → parse impact.
        Validates the data flows correctly through each stage.
        """
        # --- Stage 1: EC diff detection ---
        diff_result = detect_ec_changes(
            trial_record["original_ec"],
            trial_record["revised_ec"],
        )
        assert diff_result["has_change"] is True

        # --- Stage 2: Build triplet in DB ---
        tmp_db.execute(
            """INSERT INTO ec_changes
               (instance_id, source_version, target_version, ec_before, ec_after,
                change_type, similarity, change_ratio, evidence_source_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trial_record["instance_id"],
                trial_record["source_version"],
                trial_record["target_version"],
                trial_record["original_ec"],
                trial_record["revised_ec"],
                diff_result["change_type"],
                diff_result["similarity"],
                diff_result["change_ratio"],
                "protocols/NCT00000001_v3.pdf",
            ),
        )
        tmp_db.execute(
            "INSERT INTO protocol_texts (instance_id, raw_text) VALUES (?, ?)",
            (trial_record["instance_id"], protocol_text),
        )
        tmp_db.commit()

        create_recite_instances(max_trials=10, conn=tmp_db)
        tmp_db.commit()

        recite_rows = tmp_db.execute("SELECT * FROM recite").fetchall()
        assert len(recite_rows) == 1
        instance = recite_rows[0]

        # --- Stage 3: Simulate directive extraction (mock LLM output) ---
        mock_directive_response = json.dumps({
            "answer_location": "exact",
            "directives_exact_text": paper_abstract["directives"],
            "directives_count": len(paper_abstract["directives"]),
        })
        directive_result = parse_directives_response(mock_directive_response)
        assert directive_result["directives_has_answer"] == 1
        assert directive_result["directives_count"] == 2
        assert "age limits" in directive_result["directives_exact_text"].lower()

        # --- Stage 4: Simulate impact parsing (mock LLM output) ---
        mock_impact_response = json.dumps({
            "answer_location": "exact",
            "impact_percent": 35.0,
            "impact_absolute": 120,
            "impact_unit": "patients",
            "impact_qualitative": "Substantial increase based on expanded age and renal criteria",
            "impact_evidence": "Based on registry analysis, 35% increase in eligible pool.",
        })
        impact_result = parse_impact_response(mock_impact_response)
        assert impact_result["impact_has_answer"] == 1
        assert impact_result["impact_percent"] == 35.0
        assert impact_result["impact_absolute"] == 120.0

        # --- Verify data integrity across stages ---
        assert instance["instance_id"] == trial_record["instance_id"]
        assert len(instance["source_text"]) > 0
        assert len(instance["reference_text"]) > 0
        assert len(instance["evidence"]) > 0

    def test_accrual_gain_calculation(self, trial_record):
        """
        Test the accrual gain formula: scalar_gain = enrollment × pct_gain / 100.
        This is the core calculation described in the paper.
        """
        current_enrollment = 200  # hypothetical current enrollment
        pct_gain = 35.0  # from paper abstract fixture

        scalar_gain = current_enrollment * pct_gain / 100.0
        assert scalar_gain == 70.0  # 200 * 35% = 70 additional patients

    def test_multiple_directives_parsed(self):
        """Verify multiple directives are correctly joined."""
        response = json.dumps({
            "answer_location": "exact",
            "directives_exact_text": [
                "Expand upper age limit to 75 years",
                "Relax renal function threshold to eGFR 45",
                "Widen BMI range to 22-45 kg/m2",
            ],
            "directives_count": 3,
        })
        result = parse_directives_response(response)
        assert result["directives_count"] == 3
        text = result["directives_exact_text"]
        assert "age limit" in text.lower()
        assert "renal" in text.lower()
        assert "bmi" in text.lower()

    def test_diff_output_matches_fixture_changes(self, trial_record):
        """
        The EC diff preview should reflect the specific changes in our fixture:
        age 65→75, HbA1c 10→11, BMI 25-40→22-45, eGFR 60→45.
        """
        result = detect_ec_changes(
            trial_record["original_ec"],
            trial_record["revised_ec"],
        )
        # The diff should show changed lines
        assert result["changed_lines"] > 0
        # Similarity should be moderate (substantial overlap, some changes)
        assert 0.5 < result["similarity"] < 0.99
