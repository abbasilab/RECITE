#!/usr/bin/env python3
"""Compare reproduced predictions against original benchmark results DB.

Joins reproduced JSONL predictions with original DB results on
(instance_id, source_version, target_version) and reports exact-match rates
and character-level similarity.

Usage:
    uv run python reproduce/compare_predictions.py \
        --original-db /path/to/benchmark_results.db \
        --predictions-dir data/benchmark_predictions \
        [--models local-qwen-0_5b,local-qwen-3b,...] \
        [--run-pattern "*_no_rag"]
"""

import argparse
import json
import sqlite3
import sys
from difflib import SequenceMatcher
from pathlib import Path


# Map model_id to config hash in the original DB (no_rag, train split)
ORIGINAL_CONFIG_MAP = {
    "local-qwen-0_5b": "83525e11a31ca0dabb6e48c7",
    "local-qwen-3b": "1988ec418928af1290853c4a",
    "local-gemma2-2b": "9919b9efea552663a5029a76",
    "local-gemma2-9b": "be5060d19cd1f0e7424afbb0",
    "local-qwen-7b": "5bb7e5de4490ea5d71a57caa",
    "local-longctx-7b": "1fa16bf2ad873d803624dde8",
    "versa-4o": "9fa974044430151d492dc6c3",
    "versa-4o-mini": "d108166165cb614412a4bb20",
}


def load_original_predictions(db_path: str, config_id: str) -> dict:
    """Load original predictions from DB, keyed by (instance_id, vfrom, vto)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"SELECT instance_id, source_version, target_version, prediction "
        f"FROM results_{config_id} WHERE split_name='train'"
    ).fetchall()
    conn.close()
    preds = {}
    for instance_id, vfrom, vto, prediction in rows:
        key = (instance_id, int(vfrom), int(vto))
        preds[key] = prediction
    return preds


def load_reproduced_predictions(pred_dir: Path) -> dict:
    """Load reproduced predictions from JSONL, keyed by (instance_id, vfrom, vto)."""
    preds = {}
    jsonl = pred_dir / "predictions_train_checkpoint.jsonl"
    if not jsonl.exists():
        jsonl = pred_dir / "predictions_train.jsonl"
    if not jsonl.exists():
        return preds
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            key = (d["instance_id"], int(d["source_version"]), int(d["target_version"]))
            preds[key] = d.get("prediction", "")
    return preds


def compare(original: dict, reproduced: dict) -> dict:
    """Compare predictions and return stats."""
    common_keys = set(original) & set(reproduced)
    if not common_keys:
        return {"matched": 0, "total_repro": len(reproduced), "total_orig": len(original), "exact": 0, "exact_pct": 0.0, "avg_similarity": 0.0, "mismatches": []}

    exact = 0
    similarities = []
    mismatches = []
    for key in sorted(common_keys):
        orig = original[key] or ""
        repro = reproduced[key] or ""
        if orig.strip() == repro.strip():
            exact += 1
            similarities.append(1.0)
        else:
            sim = SequenceMatcher(None, orig, repro).ratio()
            similarities.append(sim)
            mismatches.append((key, sim, orig[:200], repro[:200]))

    return {
        "matched": len(common_keys),
        "total_repro": len(reproduced),
        "total_orig": len(original),
        "exact": exact,
        "exact_pct": exact / len(common_keys) * 100 if common_keys else 0,
        "avg_similarity": sum(similarities) / len(similarities),
        "mismatches": mismatches[:5],  # First 5 mismatches for inspection
    }


def find_latest_run(model_dir: Path, pattern: str = "*_no_rag") -> Path | None:
    """Find the latest run directory matching pattern."""
    runs = sorted(model_dir.glob(f"run_{pattern}"))
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-db",
        default="data/results/benchmark_results_cluster1.db",
        help="Path to original benchmark_results.db",
    )
    parser.add_argument(
        "--predictions-dir",
        default="data/benchmark_predictions",
        help="Directory containing reproduced prediction subdirs",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model IDs (default: all available)",
    )
    parser.add_argument(
        "--run-pattern",
        default="*_no_rag",
        help="Glob pattern for run directories",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=3,
        help="Number of mismatched examples to show per model",
    )
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    if not pred_dir.is_absolute():
        pred_dir = Path.cwd() / pred_dir

    models = args.models.split(",") if args.models else list(ORIGINAL_CONFIG_MAP.keys())

    print("=" * 70)
    print("RECITE Reproduction Comparison Report")
    print("=" * 70)
    print(f"Original DB: {args.original_db}")
    print(f"Predictions: {pred_dir}")
    print()

    results = {}
    for model_id in models:
        model_dir = pred_dir / model_id
        if not model_dir.exists():
            continue

        config_id = ORIGINAL_CONFIG_MAP.get(model_id)
        if not config_id:
            print(f"[SKIP] {model_id}: no original config mapping")
            continue

        run_dir = find_latest_run(model_dir, args.run_pattern)
        if not run_dir:
            print(f"[SKIP] {model_id}: no run matching {args.run_pattern}")
            continue

        original = load_original_predictions(args.original_db, config_id)
        reproduced = load_reproduced_predictions(run_dir)
        stats = compare(original, reproduced)
        results[model_id] = stats

        status = "PASS" if stats["exact_pct"] == 100.0 else "DIFF"
        print(f"[{status}] {model_id}:")
        print(f"  Samples compared: {stats['matched']}/{stats['total_repro']} reproduced, {stats['total_orig']} original")
        print(f"  Exact match: {stats['exact']}/{stats['matched']} ({stats['exact_pct']:.1f}%)")
        print(f"  Avg char similarity: {stats['avg_similarity']:.3f}")

        if stats.get("mismatches") and args.show_mismatches > 0:
            print(f"  Top mismatches:")
            for key, sim, orig_snip, repro_snip in stats["mismatches"][: args.show_mismatches]:
                nct, vf, vt = key
                print(f"    {nct} v{vf}→v{vt} (sim={sim:.3f}):")
                print(f"      ORIG:  {orig_snip[:80]}...")
                print(f"      REPRO: {repro_snip[:80]}...")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"{'Model':<20} {'Compared':>8} {'Exact':>8} {'Match%':>8} {'Similarity':>10}")
    print("-" * 70)
    for model_id, stats in results.items():
        print(
            f"{model_id:<20} {stats['matched']:>8} {stats['exact']:>8} "
            f"{stats['exact_pct']:>7.1f}% {stats['avg_similarity']:>9.3f}"
        )
    print("=" * 70)

    # Exit with error code if any model has non-100% match
    all_pass = all(s["exact_pct"] == 100.0 for s in results.values() if s["matched"] > 0)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
