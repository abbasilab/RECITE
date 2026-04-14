#!/usr/bin/env python3
"""
RECITE rebuttal analysis — recreate paper Table 1 + truncation-subset head-to-head.

Outputs:
  1. Table 1 recreation: all models, binary equiv + normalized score (v1 mini judge)
  2. Truncation subset: restrict to samples where amended EC is present in 4k window,
     then head-to-head comparison across models.

Usage:
    uv run python scripts/recite_rebuttal_analysis.py
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger

logger.add("logs/recite_rebuttal_analysis.log", rotation="10 MB", retention="7 days")

ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DB = ROOT / "data" / "prod" / "benchmark_results.db"
RECITE_DB = ROOT / "data" / "dev" / "recite.db"
OUT_DIR = ROOT / "data" / "rebuttal" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical config IDs for the 8 original paper models (one per model+rag combo)
CANONICAL_CONFIGS = {
    ("local-gemma2-2b", 0): "783a0330de928e71913c0f05",
    ("local-gemma2-2b", 1): "9919b9efea552663a5029a76",
    ("local-gemma2-9b", 0): "b602b83dc65b3c677974cbf9",
    ("local-gemma2-9b", 1): "be5060d19cd1f0e7424afbb0",
    ("local-longctx-7b", 0): "97ef508facb8b047f3d1b4ae",
    ("local-longctx-7b", 1): "1fa16bf2ad873d803624dde8",
    ("local-qwen-0_5b", 0): "8ae5df04b909d4c5eff94bdd",
    ("local-qwen-0_5b", 1): "83525e11a31ca0dabb6e48c7",
    ("local-qwen-3b", 0): "2ed8bbee8162109e1d7ebb72",
    ("local-qwen-3b", 1): "1988ec418928af1290853c4a",
    ("local-qwen-7b", 0): "98d87e89859b6f9d79be8741",
    ("local-qwen-7b", 1): "5bb7e5de4490ea5d71a57caa",
    ("versa-4o", 0): "46987023a1086f5f9072bae4",
    ("versa-4o", 1): "9fa974044430151d492dc6c3",
    ("versa-4o-mini", 0): "f37e1a948e0085d497871277",
    ("versa-4o-mini", 1): "c61f176d6d55c87d6318590e",
}

# Rebuttal model config IDs (in judge_scores table)
REBUTTAL_CONFIGS = {
    ("Qwen/Qwen2.5-72B-Instruct", 1): "json_qwen25-72b_no_rag",
    ("google/gemma-2-27b-it", 1): "json_gemma2-27b_no_rag",
}

# Display names for models
DISPLAY_NAMES = {
    "local-qwen-0_5b": "Qwen2.5-0.5B",
    "local-qwen-3b": "Qwen2.5-3B",
    "local-qwen-7b": "Qwen2.5-7B",
    "local-gemma2-2b": "Gemma2-2B",
    "local-gemma2-9b": "Gemma2-9B",
    "local-longctx-7b": "LongCtx-7B",
    "versa-4o": "GPT-4o",
    "versa-4o-mini": "GPT-4o-mini",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "google/gemma-2-27b-it": "Gemma2-27B",
}

# Model ordering for tables
MODEL_ORDER = [
    "local-longctx-7b",
    "local-qwen-0_5b",
    "local-qwen-3b",
    "local-qwen-7b",
    "local-gemma2-2b",
    "local-gemma2-9b",
    "Qwen/Qwen2.5-72B-Instruct",
    "google/gemma-2-27b-it",
    "versa-4o-mini",
    "versa-4o",
]


def get_truncation_evidence() -> dict[int, dict[str, Any]]:
    """Load truncation evidence keyed by recite_id."""
    conn = sqlite3.connect(str(RECITE_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT recite_id, reference_text_present, reference_text_confidence "
        "FROM truncation_evidence_41mini WHERE reference_text_present IS NOT NULL"
    ).fetchall()
    conn.close()
    return {r["recite_id"]: dict(r) for r in rows}


def get_original_model_scores(merge_source: str = "cluster1") -> dict[tuple[str, int], dict[int, dict]]:
    """Get original GPT-4o judge scores from results tables for the 8 paper models.

    Returns: {(model_id, no_rag): {sample_id: {binary, normalized, split}}}
    """
    conn = sqlite3.connect(str(BENCHMARK_DB))
    conn.row_factory = sqlite3.Row
    result = {}
    for (model_id, no_rag), config_id in CANONICAL_CONFIGS.items():
        table = f"results_{config_id}"
        try:
            rows = conn.execute(
                f"SELECT id, split_name, llm_judge_binary, llm_judge_normalized "
                f"FROM \"{table}\" WHERE merge_source = ?",
                (merge_source,),
            ).fetchall()
        except sqlite3.OperationalError:
            # Table might not have merge_source — try without
            rows = conn.execute(
                f"SELECT id, split_name, llm_judge_binary, llm_judge_normalized "
                f"FROM \"{table}\""
            ).fetchall()
        scores = {}
        for r in rows:
            scores[r["id"]] = {
                "binary": r["llm_judge_binary"],
                "normalized": r["llm_judge_normalized"],
                "split": r["split_name"],
            }
        result[(model_id, no_rag)] = scores
    conn.close()
    return result


def get_v1_mini_scores() -> dict[tuple[str, int], dict[int, dict]]:
    """Get v1 mini judge scores from judge_scores table.

    Returns: {(model_id, no_rag): {sample_id: {binary, normalized}}}
    """
    conn = sqlite3.connect(str(BENCHMARK_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT config_id, sample_id, model_id, no_rag, binary_score, normalized_score "
        "FROM judge_scores"
    ).fetchall()
    conn.close()
    result: dict[tuple[str, int], dict[int, dict]] = {}
    for r in rows:
        key = (r["model_id"], r["no_rag"])
        if key not in result:
            result[key] = {}
        result[key][r["sample_id"]] = {
            "binary": r["binary_score"],
            "normalized": r["normalized_score"],
        }
    return result


def compute_metrics(scores: dict[int, dict], sample_ids: set[int] | None = None) -> dict:
    """Compute binary equivalence and normalized score for a set of samples."""
    if sample_ids is not None:
        filtered = {sid: s for sid, s in scores.items() if sid in sample_ids}
    else:
        filtered = scores
    if not filtered:
        return {"n": 0, "binary_equiv": 0.0, "normalized": 0.0}
    n = len(filtered)
    binary_sum = sum(s["binary"] for s in filtered.values())
    norm_sum = sum(s["normalized"] for s in filtered.values())
    return {
        "n": n,
        "binary_equiv": binary_sum / n,
        "normalized": norm_sum / n,
    }


def format_table(rows: list[dict], title: str) -> str:
    """Format a table as markdown."""
    lines = [f"\n## {title}\n"]
    if not rows:
        lines.append("No data.\n")
        return "\n".join(lines)

    # Header
    cols = list(rows[0].keys())
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    logger.info("Starting RECITE rebuttal analysis")

    # Load data
    trunc_evidence = get_truncation_evidence()
    original_scores = get_original_model_scores()
    v1_mini_scores = get_v1_mini_scores()

    # Sample IDs where evidence IS present in truncated 4k window
    evidence_present_ids = {sid for sid, info in trunc_evidence.items() if info["reference_text_present"] == 1}
    evidence_absent_ids = {sid for sid, info in trunc_evidence.items() if info["reference_text_present"] == 0}
    all_sample_ids = set(trunc_evidence.keys())

    logger.info(f"Truncation evidence: {len(evidence_present_ids)} present, {len(evidence_absent_ids)} absent, {len(all_sample_ids)} total")

    # ── Table 1: Full benchmark results (original GPT-4o judge, test split) ──
    table1_rows = []
    for model_id in MODEL_ORDER:
        for no_rag in [1, 0]:  # no_rag first, then rag
            key = (model_id, no_rag)
            display = DISPLAY_NAMES.get(model_id, model_id)
            rag_label = "No RAG" if no_rag else "RAG"

            # Original judge scores (all splits available)
            if key in original_scores:
                # Test split only
                test_scores = {sid: s for sid, s in original_scores[key].items() if s["split"] == "test"}
                orig_metrics = compute_metrics(test_scores)
            else:
                orig_metrics = {"n": 0, "binary_equiv": 0.0, "normalized": 0.0}

            # V1 mini scores
            if key in v1_mini_scores:
                v1_metrics = compute_metrics(v1_mini_scores[key])
            else:
                v1_metrics = {"n": 0, "binary_equiv": 0.0, "normalized": 0.0}

            table1_rows.append({
                "Model": display,
                "RAG": rag_label,
                "N (orig)": orig_metrics["n"],
                "Binary% (orig)": orig_metrics["binary_equiv"] * 100,
                "NormScore (orig)": orig_metrics["normalized"],
                "N (v1 mini)": v1_metrics["n"],
                "Binary% (v1)": v1_metrics["binary_equiv"] * 100,
                "NormScore (v1)": v1_metrics["normalized"],
            })

    # ── Table 2: Truncation subset — evidence present (all splits, original judge) ──
    table2_rows = []
    for model_id in MODEL_ORDER:
        for no_rag in [1, 0]:
            key = (model_id, no_rag)
            display = DISPLAY_NAMES.get(model_id, model_id)
            rag_label = "No RAG" if no_rag else "RAG"

            if key in original_scores:
                all_metrics = compute_metrics(original_scores[key], all_sample_ids)
                present_metrics = compute_metrics(original_scores[key], evidence_present_ids)
                absent_metrics = compute_metrics(original_scores[key], evidence_absent_ids)
            elif key in v1_mini_scores:
                all_metrics = compute_metrics(v1_mini_scores[key], all_sample_ids)
                present_metrics = compute_metrics(v1_mini_scores[key], evidence_present_ids)
                absent_metrics = compute_metrics(v1_mini_scores[key], evidence_absent_ids)
            else:
                continue

            delta = present_metrics["normalized"] - absent_metrics["normalized"] if absent_metrics["n"] > 0 else 0.0

            table2_rows.append({
                "Model": display,
                "RAG": rag_label,
                "N (all)": all_metrics["n"],
                "NormScore (all)": all_metrics["normalized"],
                "N (evidence)": present_metrics["n"],
                "NormScore (evidence)": present_metrics["normalized"],
                "N (no evidence)": absent_metrics["n"],
                "NormScore (no evidence)": absent_metrics["normalized"],
                "Δ (ev - no_ev)": delta,
            })

    # ── Table 3: Head-to-head on evidence-present subset (v1 mini where available, else original) ──
    table3_rows = []
    for model_id in MODEL_ORDER:
        for no_rag in [1, 0]:
            key = (model_id, no_rag)
            display = DISPLAY_NAMES.get(model_id, model_id)
            rag_label = "No RAG" if no_rag else "RAG"

            # Prefer v1 mini scores, fall back to original
            if key in v1_mini_scores:
                source = "v1 mini"
                scores = v1_mini_scores[key]
            elif key in original_scores:
                source = "GPT-4o"
                scores = original_scores[key]
            else:
                continue

            present_metrics = compute_metrics(scores, evidence_present_ids)
            if present_metrics["n"] == 0:
                continue

            table3_rows.append({
                "Model": display,
                "RAG": rag_label,
                "Judge": source,
                "N": present_metrics["n"],
                "Binary%": present_metrics["binary_equiv"] * 100,
                "NormScore": present_metrics["normalized"],
            })

    # Sort table 3 by NormScore descending for ranking
    table3_rows.sort(key=lambda r: r["NormScore"], reverse=True)

    # ── Output ──
    output = []
    output.append("# RECITE Rebuttal Analysis")
    output.append(f"\nGenerated: 2026-04-13")
    output.append(f"\nTruncation evidence (4k window): {len(evidence_present_ids)} present ({len(evidence_present_ids)/len(all_sample_ids)*100:.1f}%), {len(evidence_absent_ids)} absent ({len(evidence_absent_ids)/len(all_sample_ids)*100:.1f}%)")
    output.append(format_table(table1_rows, "Table 1: Benchmark Results (Test Split — Original GPT-4o Judge vs V1 Mini)"))
    output.append(format_table(table2_rows, "Table 2: Performance by Truncation Evidence Availability (All Splits, Original Judge)"))
    output.append(format_table(table3_rows, "Table 3: Head-to-Head on Evidence-Present Subset (Ranked by NormScore)"))

    report = "\n".join(output)
    out_path = OUT_DIR / "rebuttal_analysis.md"
    out_path.write_text(report)
    logger.info(f"Analysis written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
