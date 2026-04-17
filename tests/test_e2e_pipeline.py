"""
End-to-end integration test for the RECITE pipeline.

Runs all 4 stages on synthetic data with real DB + parquet outputs.
No LLM calls — mock responses simulate what the LLM would return.
Verifies: DB rows, parquet files, stats JSON, data integrity across stages.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from recite.benchmark.ec_detector import detect_ec_changes
from recite.benchmark.builders import create_recite_instances
from recite.benchmark.utils import clean_text
from recite.ec_impact.parsing import parse_directives_response, parse_impact_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Two synthetic trials with different amendment patterns
TRIALS = [
    {
        "instance_id": "NCT00000001",
        "title": "Phase III Diabetes Drug X Trial",
        "source_version": 2,
        "target_version": 3,
        "original_ec": (
            "Inclusion Criteria:\n"
            "- Age 18 to 65 years\n"
            "- Type 2 Diabetes diagnosed >= 1 year\n"
            "- HbA1c 7.0% to 10.0%\n"
            "- BMI 25-40 kg/m2\n"
            "\n"
            "Exclusion Criteria:\n"
            "- Severe hypoglycemia in past 6 months\n"
            "- eGFR < 60 mL/min/1.73m2\n"
            "- Active liver disease"
        ),
        "revised_ec": (
            "Inclusion Criteria:\n"
            "- Age 18 to 75 years\n"
            "- Type 2 Diabetes diagnosed >= 1 year\n"
            "- HbA1c 7.0% to 11.0%\n"
            "- BMI 22-45 kg/m2\n"
            "\n"
            "Exclusion Criteria:\n"
            "- Severe hypoglycemia in past 6 months\n"
            "- eGFR < 45 mL/min/1.73m2\n"
            "- Active liver disease"
        ),
        "protocol_text": (FIXTURES_DIR / "protocol_amendment.txt").read_text(),
    },
    {
        "instance_id": "NCT00000002",
        "title": "Phase II Oncology Immunotherapy Trial",
        "source_version": 1,
        "target_version": 2,
        "original_ec": (
            "Inclusion Criteria:\n"
            "- Age 18 to 70 years\n"
            "- ECOG performance status 0-1\n"
            "- Measurable disease per RECIST 1.1\n"
            "- Adequate organ function\n"
            "\n"
            "Exclusion Criteria:\n"
            "- Prior immunotherapy\n"
            "- Active autoimmune disease\n"
            "- Brain metastases"
        ),
        "revised_ec": (
            "Inclusion Criteria:\n"
            "- Age 18 to 80 years\n"
            "- ECOG performance status 0-2\n"
            "- Measurable disease per RECIST 1.1\n"
            "- Adequate organ function\n"
            "\n"
            "Exclusion Criteria:\n"
            "- More than 2 prior lines of immunotherapy\n"
            "- Active severe autoimmune disease requiring systemic treatment\n"
            "- Symptomatic brain metastases"
        ),
        "protocol_text": (
            "PROTOCOL AMENDMENT #2\n"
            "Study NCT00000002: Phase II Oncology Immunotherapy Trial\n"
            "Effective Date: June 1, 2024\n\n"
            "RATIONALE: Interim analysis showed acceptable toxicity in older patients and\n"
            "those with controlled autoimmune conditions. DSMB approved broadening.\n\n"
            "CHANGES:\n"
            "- Upper age raised from 70 to 80 to include elderly patients\n"
            "- ECOG 2 now permitted (ambulatory, capable of self-care)\n"
            "- Prior immunotherapy: changed from 'any prior' to '>2 prior lines'\n"
            "- Autoimmune: refined to 'severe requiring systemic treatment'\n"
            "- Brain mets: clarified to 'symptomatic' only\n\n"
            "EXPECTED IMPACT: ~40% increase in eligible pool.\n"
        ),
    },
]

# Mock LLM outputs for directive extraction (one per paper)
MOCK_PAPERS = [
    {
        "paper_id": "SYNTH-001",
        "directive_response": json.dumps({
            "answer_location": "exact",
            "directives_exact_text": [
                "Diabetes trials should raise the upper age limit to 75 years",
                "Renal exclusion thresholds should be relaxed to eGFR < 45",
            ],
            "directives_count": 2,
        }),
        "impact_response": json.dumps({
            "answer_location": "exact",
            "impact_percent": 35.0,
            "impact_absolute": 120,
            "impact_unit": "patients",
            "impact_qualitative": "Substantial increase from broadened age and renal criteria",
            "impact_evidence": "Registry analysis showed 35% increase in eligible pool.",
        }),
    },
    {
        "paper_id": "SYNTH-002",
        "directive_response": json.dumps({
            "answer_location": "exact",
            "directives_exact_text": [
                "Immunotherapy trials should permit ECOG 2 patients",
                "Prior immunotherapy should not be an absolute exclusion",
                "Asymptomatic brain metastases should be permitted",
            ],
            "directives_count": 3,
        }),
        "impact_response": json.dumps({
            "answer_location": "exact",
            "impact_percent": 42.0,
            "impact_absolute": 85,
            "impact_unit": "patients",
            "impact_qualitative": "Significant broadening of eligible pool",
            "impact_evidence": "Multi-site analysis showed 42% enrollment gain.",
        }),
    },
]


def _create_db(db_path: Path) -> sqlite3.Connection:
    """Create a fresh RECITE database with full schema."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE ec_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            source_version INTEGER,
            target_version INTEGER,
            ec_before TEXT,
            ec_after TEXT,
            change_type TEXT,
            similarity REAL,
            change_ratio REAL,
            quality_score REAL,
            evidence_source_path TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE protocol_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            raw_text TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE recite (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            source_version INTEGER,
            target_version INTEGER,
            source_text TEXT,
            evidence TEXT,
            reference_text TEXT,
            ec_change_id INTEGER,
            evidence_extraction_level TEXT,
            evidence_extraction_score INTEGER,
            quality_score REAL
        )
    """)
    conn.execute("""
        CREATE TABLE accrual_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            instance_id TEXT,
            directives_answer_location TEXT,
            directives_has_answer INTEGER,
            directives_exact_text TEXT,
            directives_count INTEGER,
            impact_has_answer INTEGER,
            impact_percent REAL,
            impact_absolute REAL,
            impact_unit TEXT,
            impact_qualitative TEXT,
            impact_evidence TEXT,
            scalar_gain REAL
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# E2E Test
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestEndToEnd:
    """Full pipeline: EC diff → triplet build → directive parse → impact score → outputs."""

    def test_full_pipeline_writes_db_and_parquet(self, tmp_path):
        """
        Run the complete 4-stage pipeline on 2 synthetic trials.
        Verify: DB tables populated, parquet written, stats file produced.
        """
        db_path = tmp_path / "recite_e2e.db"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        conn = _create_db(db_path)

        # ===== Stage 1: EC Diff Detection =====
        diffs = []
        for trial in TRIALS:
            result = detect_ec_changes(trial["original_ec"], trial["revised_ec"])
            assert result["has_change"] is True, f"Expected change for {trial['instance_id']}"
            diffs.append(result)

            conn.execute(
                """INSERT INTO ec_changes
                   (instance_id, source_version, target_version, ec_before, ec_after,
                    change_type, similarity, change_ratio, evidence_source_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trial["instance_id"],
                    trial["source_version"],
                    trial["target_version"],
                    trial["original_ec"],
                    trial["revised_ec"],
                    result["change_type"],
                    result["similarity"],
                    result["change_ratio"],
                    f"protocols/{trial['instance_id']}.pdf",
                ),
            )
            conn.execute(
                "INSERT INTO protocol_texts (instance_id, raw_text) VALUES (?, ?)",
                (trial["instance_id"], trial["protocol_text"]),
            )
        conn.commit()

        # Verify Stage 1 DB state
        ec_count = conn.execute("SELECT COUNT(*) FROM ec_changes").fetchone()[0]
        assert ec_count == 2
        pt_count = conn.execute("SELECT COUNT(*) FROM protocol_texts").fetchone()[0]
        assert pt_count == 2

        # ===== Stage 2: Triplet Building =====
        create_recite_instances(max_trials=None, conn=conn)
        conn.commit()

        recite_count = conn.execute("SELECT COUNT(*) FROM recite").fetchone()[0]
        assert recite_count == 2, f"Expected 2 RECITE instances, got {recite_count}"

        # Verify triplet content
        rows = conn.execute("SELECT * FROM recite ORDER BY instance_id").fetchall()
        for row in rows:
            assert len(row["source_text"]) > 0
            assert len(row["reference_text"]) > 0
            assert len(row["evidence"]) > 50  # protocol text should be substantial
            assert row["evidence_extraction_level"] == "raw_pdf_text_only"

        # ===== Stage 3: Directive Extraction (mock LLM) =====
        for paper in MOCK_PAPERS:
            directive_result = parse_directives_response(paper["directive_response"])
            assert directive_result["directives_has_answer"] == 1

            impact_result = parse_impact_response(paper["impact_response"])
            assert impact_result["impact_has_answer"] == 1

            # Calculate scalar gain (the core accrual formula)
            enrollment = 200  # hypothetical
            scalar_gain = enrollment * (impact_result["impact_percent"] or 0) / 100.0

            conn.execute(
                """INSERT INTO accrual_results
                   (paper_id, directives_answer_location, directives_has_answer,
                    directives_exact_text, directives_count,
                    impact_has_answer, impact_percent, impact_absolute,
                    impact_unit, impact_qualitative, impact_evidence, scalar_gain)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper["paper_id"],
                    directive_result["directives_answer_location"],
                    directive_result["directives_has_answer"],
                    directive_result["directives_exact_text"],
                    directive_result["directives_count"],
                    impact_result["impact_has_answer"],
                    impact_result["impact_percent"],
                    impact_result["impact_absolute"],
                    impact_result["impact_unit"],
                    impact_result["impact_qualitative"],
                    impact_result["impact_evidence"],
                    scalar_gain,
                ),
            )
        conn.commit()

        # Verify Stage 3 DB state
        accrual_count = conn.execute("SELECT COUNT(*) FROM accrual_results").fetchone()[0]
        assert accrual_count == 2

        # ===== Stage 4: Export to Parquet =====
        # Export RECITE instances
        recite_df = pd.read_sql_query("SELECT * FROM recite", conn)
        parquet_path = output_dir / "benchmark.parquet"
        recite_df.to_parquet(parquet_path, index=False)

        # Export accrual results
        accrual_df = pd.read_sql_query("SELECT * FROM accrual_results", conn)
        accrual_parquet = output_dir / "accrual_results.parquet"
        accrual_df.to_parquet(accrual_parquet, index=False)

        # Write combined stats
        stats = {
            "ec_changes": ec_count,
            "recite_instances": recite_count,
            "accrual_results": accrual_count,
            "avg_impact_percent": float(accrual_df["impact_percent"].mean()),
            "total_scalar_gain": float(accrual_df["scalar_gain"].sum()),
        }
        stats_path = output_dir / "pipeline_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))

        conn.close()

        # ===== Verification: Read back all outputs =====

        # 1. DB file exists and has data
        assert db_path.exists()
        assert db_path.stat().st_size > 0

        # 2. Parquet files are valid
        assert parquet_path.exists()
        pq_table = pq.read_table(parquet_path)
        assert pq_table.num_rows == 2
        assert "source_text" in pq_table.column_names
        assert "evidence" in pq_table.column_names
        assert "reference_text" in pq_table.column_names

        assert accrual_parquet.exists()
        accrual_table = pq.read_table(accrual_parquet)
        assert accrual_table.num_rows == 2
        assert "impact_percent" in accrual_table.column_names
        assert "scalar_gain" in accrual_table.column_names

        # 3. Stats file is valid JSON with correct values
        assert stats_path.exists()
        loaded_stats = json.loads(stats_path.read_text())
        assert loaded_stats["ec_changes"] == 2
        assert loaded_stats["recite_instances"] == 2
        assert loaded_stats["accrual_results"] == 2
        assert loaded_stats["avg_impact_percent"] == pytest.approx(38.5, abs=0.1)  # (35 + 42) / 2
        assert loaded_stats["total_scalar_gain"] == pytest.approx(154.0, abs=0.1)  # 70 + 84

        # 4. Cross-check: parquet data matches DB data
        bench_df = pd.read_parquet(parquet_path)
        assert set(bench_df["instance_id"].tolist()) == {"NCT00000001", "NCT00000002"}

        accrual_read = pd.read_parquet(accrual_parquet)
        assert set(accrual_read["paper_id"].tolist()) == {"SYNTH-001", "SYNTH-002"}
        assert all(accrual_read["directives_has_answer"] == 1)
        assert all(accrual_read["impact_has_answer"] == 1)

        # 5. Print summary (visible in pytest -v output)
        print(f"\n{'='*60}")
        print(f"E2E Pipeline Summary")
        print(f"{'='*60}")
        print(f"  EC changes detected:   {ec_count}")
        print(f"  RECITE instances:      {recite_count}")
        print(f"  Accrual results:       {accrual_count}")
        print(f"  Avg impact:            {loaded_stats['avg_impact_percent']:.1f}%")
        print(f"  Total scalar gain:     {loaded_stats['total_scalar_gain']:.0f} patients")
        print(f"  DB size:               {db_path.stat().st_size:,} bytes")
        print(f"  Benchmark parquet:     {parquet_path.stat().st_size:,} bytes")
        print(f"  Accrual parquet:       {accrual_parquet.stat().st_size:,} bytes")
        print(f"{'='*60}")
