"""Integration tests for the RECITE pipeline."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq
import pytest
from loguru import logger

from recite.benchmark.ec_detector import detect_ec_changes
from recite.benchmark.builders import create_recite_instances
from recite.benchmark.evaluator import default_evaluator
from recite.benchmark.results_db import (
    BENCHMARK_METRIC_COLUMNS,
    ensure_config,
    ensure_results_table,
    get_benchmark_summary_rows,
    get_connection,
    insert_result,
)
from recite.benchmark.utils import clean_text


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"

TRIALS = [
    {
        "nct_id": "NCT00000001",
        "version_from": 2,
        "version_to": 3,
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
        "nct_id": "NCT00000002",
        "version_from": 1,
        "version_to": 2,
        "original_ec": (
            "Inclusion Criteria:\n"
            "- Age 18 to 70 years\n"
            "- ECOG performance status 0-1\n"
            "- Measurable disease per RECIST 1.1\n"
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
            "RATIONALE: Interim analysis showed acceptable toxicity in older patients.\n"
            "CHANGES:\n"
            "- Upper age raised from 70 to 80\n"
            "- ECOG 2 now permitted\n"
            "- Prior immunotherapy refined to >2 prior lines\n"
            "EXPECTED IMPACT: ~40% increase in eligible pool.\n"
        ),
    },
]

def _create_pipeline_db(db_path: Path) -> sqlite3.Connection:
    """Create a RECITE pipeline database with full schema."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE ec_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nct_id TEXT NOT NULL,
            version_from INTEGER,
            version_to INTEGER,
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
            nct_id TEXT NOT NULL,
            raw_text TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE recite (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nct_id TEXT NOT NULL,
            version_from INTEGER,
            version_to INTEGER,
            preamended_text TEXT,
            evidence TEXT,
            amended_text TEXT,
            ec_change_id INTEGER,
            evidence_extraction_level TEXT,
            evidence_extraction_score INTEGER,
            quality_score REAL
        )
    """)
    conn.commit()
    return conn


def _seed_ec_changes_and_protocols(conn: sqlite3.Connection) -> None:
    """Insert synthetic trials into ec_changes and protocol_texts."""
    for trial in TRIALS:
        result = detect_ec_changes(trial["original_ec"], trial["revised_ec"])
        conn.execute(
            """INSERT INTO ec_changes
               (nct_id, version_from, version_to, ec_before, ec_after,
                change_type, similarity, change_ratio, evidence_source_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trial["nct_id"],
                trial["version_from"],
                trial["version_to"],
                trial["original_ec"],
                trial["revised_ec"],
                result["change_type"],
                result["similarity"],
                result["change_ratio"],
                f"protocols/{trial['nct_id']}.pdf",
            ),
        )
        conn.execute(
            "INSERT INTO protocol_texts (nct_id, raw_text) VALUES (?, ?)",
            (trial["nct_id"], trial["protocol_text"]),
        )
    conn.commit()


def _build_mock_predictions(
    samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create mock prediction results with realistic fields."""
    predictions = []
    for sample in samples:
        # Simulate a prediction that's close but not identical to ground truth
        ground_truth = sample["amended_text"]
        prediction = ground_truth.replace("75", "74").replace("80", "79")
        metrics = default_evaluator(ground_truth, prediction)
        predictions.append({
            "id": sample["id"],
            "split_name": "benchmark",
            "nct_id": sample["nct_id"],
            "version_from": sample["version_from"],
            "version_to": sample["version_to"],
            "preamended_text": sample["preamended_text"],
            "evidence": sample["evidence"],
            "amended_text": ground_truth,
            "prediction": prediction,
            "predicted_at": "2026-04-16T00:00:00Z",
            **metrics,
        })
    return predictions


def _mock_judge_scores(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach mock judge scores to prediction results."""
    for pred in predictions:
        pred["llm_judge_binary"] = 1.0 if pred["edit_similarity"] > 0.9 else 0.0
        pred["llm_judge_score"] = round(pred["edit_similarity"] * 4.0, 2)
        pred["llm_judge_normalized"] = round(pred["edit_similarity"], 4)
        pred["llm_judge_raw_response"] = json.dumps({
            "score": pred["llm_judge_score"],
            "reasoning": "Mock judge: prediction closely matches ground truth.",
        })
    return predictions


# ---------------------------------------------------------------------------
# Test 1: Data Loading (ClinicalTrials.gov live API)
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestDataLoading:
    """Pull a small batch from ClinicalTrials.gov and verify valid NCT records."""

    def test_fetch_trials_with_eligibility_criteria(self):
        """Fetch a small batch of trials and verify they have eligibility criteria."""
        from recite.benchmark.ctg_adapter import ClinicalTrialsGovAdapter

        adapter = ClinicalTrialsGovAdapter(requests_per_second=3.0)
        docs = list(adapter.search("diabetes eligibility criteria", max_results=5))

        assert len(docs) >= 1, "Expected at least 1 trial from ClinicalTrials.gov"
        for doc in docs:
            assert doc.source == "ctg"
            assert doc.source_id.startswith("NCT")
            assert len(doc.source_id) == 11  # NCTxxxxxxxx
            assert doc.title, f"Trial {doc.source_id} missing title"
            # Abstract includes eligibility criteria (appended by adapter)
            assert doc.abstract is not None, f"Trial {doc.source_id} missing abstract"


# ---------------------------------------------------------------------------
# Test 2: Data Processing (DB + Parquet)
# ---------------------------------------------------------------------------


class TestDataProcessing:
    """Load trials into sqlite, build triplets, export to parquet."""

    def test_pipeline_db_to_parquet(self, tmp_path):
        """EC diff → triplet build → parquet export with schema validation."""
        db_path = tmp_path / "pipeline.db"
        conn = _create_pipeline_db(db_path)
        _seed_ec_changes_and_protocols(conn)

        # Verify ec_changes populated
        ec_count = conn.execute("SELECT COUNT(*) FROM ec_changes").fetchone()[0]
        assert ec_count == 2

        # Build triplets
        create_recite_instances(max_trials=None, conn=conn)
        conn.commit()

        recite_rows = conn.execute("SELECT * FROM recite ORDER BY nct_id").fetchall()
        assert len(recite_rows) == 2

        for row in recite_rows:
            assert len(row["preamended_text"]) > 0
            assert len(row["amended_text"]) > 0
            assert len(row["evidence"]) > 50
            assert row["evidence_extraction_level"] == "raw_pdf_text_only"
            assert row["evidence_extraction_score"] == 1

        # Export to parquet
        recite_df = pd.read_sql_query("SELECT * FROM recite", conn)
        parquet_path = tmp_path / "benchmark.parquet"
        recite_df.to_parquet(parquet_path, index=False)
        conn.close()

        # Validate parquet schema
        pq_table = pq.read_table(parquet_path)
        assert pq_table.num_rows == 2
        required_cols = {"nct_id", "preamended_text", "evidence", "amended_text",
                         "version_from", "version_to", "evidence_extraction_level"}
        assert required_cols.issubset(set(pq_table.column_names))

        # Round-trip: read back and verify content integrity
        df_back = pd.read_parquet(parquet_path)
        assert set(df_back["nct_id"].tolist()) == {"NCT00000001", "NCT00000002"}
        assert all(df_back["evidence_extraction_score"] == 1)


# ---------------------------------------------------------------------------
# Test 3: Prediction Generation (mocked LLM)
# ---------------------------------------------------------------------------


class TestPredictionGeneration:
    """Load benchmark samples, generate predictions with mocked model."""

    def test_mock_predictions_stored_correctly(self, tmp_path):
        """Format prompts, run mocked model, verify output fields."""
        # Build pipeline DB with triplets
        db_path = tmp_path / "pipeline.db"
        conn = _create_pipeline_db(db_path)
        _seed_ec_changes_and_protocols(conn)
        create_recite_instances(max_trials=None, conn=conn)
        conn.commit()

        samples = [dict(r) for r in conn.execute("SELECT * FROM recite").fetchall()]
        conn.close()

        assert len(samples) == 2

        # Generate mock predictions (uses default_evaluator for BLEU/ROUGE/edit)
        predictions = _build_mock_predictions(samples)
        assert len(predictions) == 2

        # Store in results DB
        results_db_path = tmp_path / "results.db"
        results_conn = get_connection(results_db_path)

        config_meta = {
            "model_id": "mock-model",
            "top_k": 3,
            "no_rag": False,
            "parquet_paths": {"benchmark": str(tmp_path / "benchmark.parquet")},
            "prompts_file": "config/benchmark_prompts.json",
            "evaluator_type": "default",
            "evaluator_config": None,
        }
        config_id = ensure_config(results_conn, config_meta)

        for pred in predictions:
            insert_result(results_conn, config_id, pred)
        results_conn.commit()

        # Verify stored results
        table_name = f"results_{config_id}"
        stored = results_conn.execute(f"SELECT * FROM {table_name}").fetchall()
        assert len(stored) == 2

        for row in stored:
            row_d = dict(row)
            # Required fields present
            assert row_d["prediction"] is not None
            assert len(row_d["prediction"]) > 0
            assert row_d["preamended_text"] is not None
            assert row_d["amended_text"] is not None
            # Default metrics computed
            assert row_d["edit_distance"] is not None
            assert 0.0 <= row_d["edit_similarity"] <= 1.0
            assert 0.0 <= row_d["bleu"] <= 1.0
            assert 0.0 <= row_d["rouge_l"] <= 1.0
            # Prediction is different from ground truth (our mock changes numbers)
            assert row_d["binary_correct"] == 0.0
            assert row_d["edit_similarity"] > 0.8  # close but not exact

        results_conn.close()


# ---------------------------------------------------------------------------
# Test 4: Benchmark Execution — Judging (mocked judge)
# ---------------------------------------------------------------------------


class TestBenchmarkJudging:
    """Run judge on predictions, verify scores in DB."""

    def test_mock_judge_scores_in_results_db(self, tmp_path):
        """Load predictions, run mocked judge, verify binary + ordinal columns."""
        # Build pipeline DB
        db_path = tmp_path / "pipeline.db"
        conn = _create_pipeline_db(db_path)
        _seed_ec_changes_and_protocols(conn)
        create_recite_instances(max_trials=None, conn=conn)
        conn.commit()

        samples = [dict(r) for r in conn.execute("SELECT * FROM recite").fetchall()]
        conn.close()

        # Generate predictions then attach mock judge scores
        predictions = _build_mock_predictions(samples)
        predictions = _mock_judge_scores(predictions)

        # Store in results DB
        results_db_path = tmp_path / "results.db"
        results_conn = get_connection(results_db_path)

        config_meta = {
            "model_id": "mock-model-judged",
            "top_k": 3,
            "no_rag": False,
            "parquet_paths": {"benchmark": str(tmp_path / "benchmark.parquet")},
            "prompts_file": "config/benchmark_prompts.json",
            "evaluator_type": "llm_judge",
            "evaluator_config": {"api_type": "mock", "model": "mock-judge"},
        }
        config_id = ensure_config(results_conn, config_meta)

        for pred in predictions:
            insert_result(results_conn, config_id, pred)
        results_conn.commit()

        # Verify judge columns
        table_name = f"results_{config_id}"
        stored = results_conn.execute(f"SELECT * FROM {table_name}").fetchall()
        assert len(stored) == 2

        for row in stored:
            row_d = dict(row)
            # Judge fields populated
            assert row_d["llm_judge_binary"] is not None
            assert row_d["llm_judge_binary"] in (0.0, 1.0)
            assert row_d["llm_judge_score"] is not None
            assert 0.0 <= row_d["llm_judge_score"] <= 4.0
            assert row_d["llm_judge_normalized"] is not None
            assert 0.0 <= row_d["llm_judge_normalized"] <= 1.0
            assert row_d["llm_judge_raw_response"] is not None
            # Default metrics still present
            assert row_d["bleu"] is not None
            assert row_d["rouge_l"] is not None

        results_conn.close()


# ---------------------------------------------------------------------------
# Test 5: Results & Stats — summary table with metrics
# ---------------------------------------------------------------------------


class TestResultsAndStats:
    """Compute metrics summary and verify output table."""

    def test_summary_table_output(self, tmp_path):
        """Load all results, compute aggregate metrics, verify summary."""
        # Build pipeline DB
        db_path = tmp_path / "pipeline.db"
        conn = _create_pipeline_db(db_path)
        _seed_ec_changes_and_protocols(conn)
        create_recite_instances(max_trials=None, conn=conn)
        conn.commit()

        samples = [dict(r) for r in conn.execute("SELECT * FROM recite").fetchall()]

        # Generate predictions with judge scores
        predictions = _mock_judge_scores(_build_mock_predictions(samples))

        # Store in results DB
        results_db_path = tmp_path / "results.db"
        results_conn = get_connection(results_db_path)

        config_meta = {
            "model_id": "mock-summary-model",
            "top_k": 3,
            "no_rag": False,
            "parquet_paths": {"benchmark": str(tmp_path / "benchmark.parquet")},
            "prompts_file": "config/benchmark_prompts.json",
            "evaluator_type": "llm_judge",
            "evaluator_config": {"api_type": "mock", "model": "mock-judge"},
        }
        config_id = ensure_config(results_conn, config_meta)
        for pred in predictions:
            insert_result(results_conn, config_id, pred)
        results_conn.commit()

        # --- Compute summary via get_benchmark_summary_rows ---
        summary_rows = get_benchmark_summary_rows(results_conn)
        assert len(summary_rows) >= 1

        row = summary_rows[0]
        # Expected columns in summary
        assert row["model_id"] == "mock-summary-model"
        assert row["split_name"] == "benchmark"
        assert row["n"] == 2

        # All metric means should be present and in valid ranges
        for col in BENCHMARK_METRIC_COLUMNS:
            mean_key = f"{col}_mean"
            val = row[mean_key]
            if val is not None:
                assert 0.0 <= val <= 4.0 or col == "llm_judge_score", (
                    f"{mean_key}={val} out of range"
                )

        # Specific metric checks
        assert row["edit_similarity_mean"] is not None
        assert row["edit_similarity_mean"] > 0.8  # mock preds are close
        assert row["bleu_mean"] is not None
        # BLEU may be 0.0 if nltk is not installed (graceful fallback in evaluator)
        assert row["bleu_mean"] >= 0.0
        assert row["llm_judge_binary_mean"] is not None
        assert row["llm_judge_normalized_mean"] is not None

        results_conn.close()

        # --- Export combined stats as JSON (md-compatible) ---
        recite_df = pd.read_sql_query("SELECT * FROM recite", conn)
        conn.close()

        stats = {
            "ec_changes": 2,
            "recite_instances": len(recite_df),
            "benchmark_summary": [
                {
                    "model": row["model_id"],
                    "n": row["n"],
                    "edit_similarity": row["edit_similarity_mean"],
                    "bleu": row["bleu_mean"],
                    "rouge_l": row["rouge_l_mean"],
                    "judge_binary": row["llm_judge_binary_mean"],
                    "judge_normalized": row["llm_judge_normalized_mean"],
                }
                for row in summary_rows
            ],
        }

        # Write JSON
        stats_path = tmp_path / "pipeline_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        assert stats_path.exists()

        loaded = json.loads(stats_path.read_text())
        assert loaded["recite_instances"] == 2
        assert len(loaded["benchmark_summary"]) >= 1
        assert loaded["benchmark_summary"][0]["n"] == 2

        # Write markdown summary table
        md_path = tmp_path / "summary.md"
        lines = ["| Model | N | Edit Sim | BLEU | Judge Binary | Judge Norm |",
                 "|-------|---|----------|------|--------------|------------|"]
        for s in loaded["benchmark_summary"]:
            lines.append(
                f"| {s['model']} | {s['n']} "
                f"| {s['edit_similarity']:.3f} "
                f"| {s['bleu']:.3f} "
                f"| {s['judge_binary']:.3f} "
                f"| {s['judge_normalized']:.3f} |"
            )
        md_path.write_text("\n".join(lines))
        assert md_path.exists()
        content = md_path.read_text()
        assert "mock-summary-model" in content
        assert "|" in content
