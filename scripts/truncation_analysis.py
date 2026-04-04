#!/usr/bin/env python3
"""
Truncation/Coverage Statistics for RECITE Rebuttal (Reviewer DVUB W1).

Computes per-model truncation statistics for all 3,116 benchmark samples,
length-controlled re-analysis, and documents the exact pipeline.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken

# ---------------------------------------------------------------------------
# Configuration: models and their effective evidence token budgets
# ---------------------------------------------------------------------------
# no_rag_max_tokens as computed by the pipeline:
#   For python_gpu: min(ctx - 4096, ctx - 2560)  →  ctx - 4096 for ctx >= 4096
#   For API (no_rag): same formula from config_loader: ctx - 4096
MODELS = {
    "GPT-4o": {
        "context_window": 128_000,
        "no_rag_max_tokens": 123_904,
        "api_type": "ucsf_versa",
        "tokenizer": "cl100k_base",
        "config_id_norag": "9fa974044430151d492dc6c3",
    },
    "GPT-4o-mini": {
        "context_window": 128_000,
        "no_rag_max_tokens": 123_904,
        "api_type": "ucsf_versa",
        "tokenizer": "cl100k_base",
        "config_id_norag": "c61f176d6d55c87d6318590e",
    },
    "Qwen 2.5 0.5B": {
        "context_window": 32_768,
        "no_rag_max_tokens": 28_672,
        "api_type": "python_gpu",
        "tokenizer": "cl100k_base",  # proxy; actual is Qwen BPE (~similar ratio)
        "config_id_norag": "83525e11a31ca0dabb6e48c7",
    },
    "Qwen 2.5 3B": {
        "context_window": 32_768,
        "no_rag_max_tokens": 28_672,
        "api_type": "python_gpu",
        "tokenizer": "cl100k_base",
        "config_id_norag": "1988ec418928af1290853c4a",
    },
    "Qwen 2.5 7B": {
        "context_window": 32_768,
        "no_rag_max_tokens": 28_672,
        "api_type": "python_gpu",
        "tokenizer": "cl100k_base",
        "config_id_norag": "5bb7e5de4490ea5d71a57caa",
    },
    "Gemma 2 2B": {
        "context_window": 8_192,
        "no_rag_max_tokens": 4_096,
        "api_type": "python_gpu",
        "tokenizer": "cl100k_base",  # proxy
        "config_id_norag": "9919b9efea552663a5029a76",
    },
    "Gemma 2 9B": {
        "context_window": 8_192,
        "no_rag_max_tokens": 4_096,
        "api_type": "python_gpu",
        "tokenizer": "cl100k_base",
        "config_id_norag": "be5060d19cd1f0e7424afbb0",
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "context_window": 128_000,
        "no_rag_max_tokens": 123_904,
        "api_type": "python_gpu",
        "tokenizer": "cl100k_base",
        "config_id_norag": "1fa16bf2ad873d803624dde8",
    },
}

# Benchmark results DB
RESULTS_DB = Path(os.environ.get("RESULTS_DB", "data/dev/results.db"))


def tokenize_evidence(evidence_series: pd.Series) -> np.ndarray:
    """Tokenize all evidence strings with tiktoken cl100k_base, return array of token counts."""
    enc = tiktoken.get_encoding("cl100k_base")
    counts = []
    for text in evidence_series:
        if pd.isna(text) or not str(text).strip():
            counts.append(0)
        else:
            counts.append(len(enc.encode(str(text))))
    return np.array(counts)


def compute_truncation_stats(token_counts: np.ndarray, max_tokens: int) -> dict:
    """Compute truncation statistics for a given token budget."""
    n = len(token_counts)
    truncated_mask = token_counts > max_tokens
    n_truncated = int(truncated_mask.sum())
    pct_truncated = 100.0 * n_truncated / n

    # Content retained (fraction of tokens kept)
    retained = np.minimum(token_counts, max_tokens) / np.maximum(token_counts, 1)
    lost = token_counts - np.minimum(token_counts, max_tokens)

    # Stats for truncated samples only
    if n_truncated > 0:
        trunc_retained = retained[truncated_mask]
        trunc_lost = lost[truncated_mask]
    else:
        trunc_retained = np.array([1.0])
        trunc_lost = np.array([0])

    return {
        "n_samples": n,
        "n_truncated": n_truncated,
        "pct_truncated": pct_truncated,
        "mean_retained_frac": float(retained.mean()),
        "median_retained_frac": float(np.median(retained)),
        "p5_retained_frac": float(np.percentile(retained, 5)),
        "trunc_mean_retained_frac": float(trunc_retained.mean()),
        "trunc_median_lost_tokens": int(np.median(trunc_lost)),
        "trunc_mean_lost_tokens": int(trunc_lost.mean()),
        "trunc_max_lost_tokens": int(trunc_lost.max()) if n_truncated > 0 else 0,
    }


def length_controlled_analysis(
    df: pd.DataFrame,
    token_counts: np.ndarray,
    config_id: str,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """Break samples into length buckets and compute per-bucket performance."""
    table = f"results_{config_id}"
    try:
        results = pd.read_sql_query(f'SELECT * FROM [{table}]', conn)
    except Exception as e:
        print(f"  Warning: could not load results from {table}: {e}")
        return pd.DataFrame()

    # Merge token counts into results
    # Results have 'id' column matching benchmark 'id'
    df_with_tokens = df[["id"]].copy()
    df_with_tokens["evidence_tokens"] = token_counts

    # Results may have train/val/test splits, merge on id
    results = results.merge(df_with_tokens, on="id", how="left")
    results = results.dropna(subset=["evidence_tokens"])
    results["evidence_tokens"] = results["evidence_tokens"].astype(int)

    if len(results) == 0:
        return pd.DataFrame()

    # Define buckets
    bins = [0, 1000, 4096, 8192, 16384, 32768, 65536, 128000, float("inf")]
    labels = ["<1K", "1K-4K", "4K-8K", "8K-16K", "16K-32K", "32K-64K", "64K-128K", ">128K"]
    results["length_bucket"] = pd.cut(results["evidence_tokens"], bins=bins, labels=labels, right=True)

    # Compute metrics per bucket
    metrics_cols = ["edit_similarity", "bleu", "rouge_l"]
    judge_cols = ["llm_judge_binary", "llm_judge_score", "llm_judge_normalized"]

    # Use whatever columns are available
    available_metrics = [c for c in metrics_cols + judge_cols if c in results.columns]

    if not available_metrics:
        return pd.DataFrame()

    grouped = results.groupby("length_bucket", observed=False)
    bucket_stats = []
    for bucket_name, group in grouped:
        row = {"bucket": bucket_name, "n": len(group)}
        for col in available_metrics:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else None
        bucket_stats.append(row)

    return pd.DataFrame(bucket_stats)


def main():
    # Load benchmark data
    parquet_path = Path("data/benchmark_splits/benchmark.parquet")
    print(f"Loading benchmark data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} samples")

    # Tokenize evidence
    print("Tokenizing evidence documents with tiktoken cl100k_base...")
    token_counts = tokenize_evidence(df["evidence"])
    char_counts = df["evidence"].str.len().values

    # Also compute source_text tokens (part of prompt, reduces available space)
    print("Tokenizing source_text (prompt overhead)...")
    source_tokens = tokenize_evidence(df["source_text"])

    # Estimate prompt overhead: system + user template + source_text
    # System prompt: ~35 tokens, user template: ~80 tokens, formatting: ~20 tokens
    PROMPT_OVERHEAD_ESTIMATE = 135  # conservative estimate excluding evidence and source_text

    print("\n" + "=" * 80)
    print("SECTION 1: EVIDENCE DOCUMENT LENGTH STATISTICS")
    print("=" * 80)

    print(f"\nAll {len(df)} samples:")
    print(f"  Characters:  mean={char_counts.mean():.0f}  median={np.median(char_counts):.0f}  p95={np.percentile(char_counts, 95):.0f}  max={char_counts.max():.0f}")
    print(f"  Tokens:      mean={token_counts.mean():.0f}  median={np.median(token_counts):.0f}  p95={np.percentile(token_counts, 95):.0f}  max={token_counts.max():.0f}")
    print(f"  Source:  mean={source_tokens.mean():.0f}  median={np.median(source_tokens):.0f}  p95={np.percentile(source_tokens, 95):.0f}  max={source_tokens.max():.0f}")

    # Percentile table
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    print("\n  Token count percentiles:")
    for p in percentiles:
        print(f"    p{p:02d}: {np.percentile(token_counts, p):.0f} tokens ({np.percentile(char_counts, p):.0f} chars)")

    print("\n" + "=" * 80)
    print("SECTION 2: PER-MODEL TRUNCATION STATISTICS (no_rag mode)")
    print("=" * 80)

    all_model_stats = {}
    for model_name, model_cfg in MODELS.items():
        max_tok = model_cfg["no_rag_max_tokens"]
        ctx = model_cfg["context_window"]
        stats = compute_truncation_stats(token_counts, max_tok)
        all_model_stats[model_name] = stats

        print(f"\n--- {model_name} (ctx={ctx:,}, evidence_budget={max_tok:,} tokens) ---")
        print(f"  Samples truncated: {stats['n_truncated']}/{stats['n_samples']} ({stats['pct_truncated']:.1f}%)")
        print(f"  Mean content retained: {stats['mean_retained_frac']:.1%}")
        print(f"  Median content retained: {stats['median_retained_frac']:.1%}")
        print(f"  P5 content retained: {stats['p5_retained_frac']:.1%}")
        if stats["n_truncated"] > 0:
            print(f"  [Truncated samples only]:")
            print(f"    Mean retained: {stats['trunc_mean_retained_frac']:.1%}")
            print(f"    Median tokens lost: {stats['trunc_median_lost_tokens']:,}")
            print(f"    Max tokens lost: {stats['trunc_max_lost_tokens']:,}")

    # Summary table
    print("\n\n--- SUMMARY TABLE ---")
    print(f"{'Model':<30} {'Context':>8} {'Budget':>8} {'Truncated':>12} {'Mean Retained':>14} {'P5 Retained':>12}")
    print("-" * 90)
    for model_name, model_cfg in MODELS.items():
        s = all_model_stats[model_name]
        ctx = model_cfg["context_window"]
        budget = model_cfg["no_rag_max_tokens"]
        print(f"{model_name:<30} {ctx:>8,} {budget:>8,} {s['n_truncated']:>5}/{s['n_samples']:<5} ({s['pct_truncated']:>5.1f}%) {s['mean_retained_frac']:>13.1%} {s['p5_retained_frac']:>11.1%}")

    print("\n" + "=" * 80)
    print("SECTION 3: LENGTH-CONTROLLED RE-ANALYSIS")
    print("=" * 80)

    # Connect to results DB
    if RESULTS_DB.exists():
        conn = sqlite3.connect(str(RESULTS_DB))
        for model_name, model_cfg in MODELS.items():
            config_id = model_cfg["config_id_norag"]
            print(f"\n--- {model_name} (no_rag, config={config_id[:12]}...) ---")
            bucket_df = length_controlled_analysis(df, token_counts, config_id, conn)
            if len(bucket_df) > 0:
                # Print nicely
                for _, row in bucket_df.iterrows():
                    if row["n"] == 0:
                        continue
                    metrics_str = ""
                    for col in bucket_df.columns:
                        if col.endswith("_mean") and row[col] is not None and not pd.isna(row[col]):
                            short = col.replace("_mean", "").replace("llm_judge_", "judge_")
                            metrics_str += f"  {short}={row[col]:.3f}"
                    print(f"  {row['bucket']:<10} n={row['n']:>5}{metrics_str}")
            else:
                print("  No results available")
        conn.close()
    else:
        print(f"\n  Results DB not found at {RESULTS_DB}")

    # ---------------------------------------------------------------------------
    # Save structured output
    # ---------------------------------------------------------------------------
    output = {
        "dataset": {
            "n_samples": len(df),
            "splits": df["split"].value_counts().to_dict(),
            "evidence_chars": {
                "mean": float(char_counts.mean()),
                "median": float(np.median(char_counts)),
                "p95": float(np.percentile(char_counts, 95)),
                "max": int(char_counts.max()),
            },
            "evidence_tokens_cl100k": {
                "mean": float(token_counts.mean()),
                "median": float(np.median(token_counts)),
                "p95": float(np.percentile(token_counts, 95)),
                "max": int(token_counts.max()),
                "percentiles": {str(p): float(np.percentile(token_counts, p)) for p in percentiles},
            },
        },
        "per_model_truncation": {
            model_name: {
                "context_window": model_cfg["context_window"],
                "no_rag_max_tokens": model_cfg["no_rag_max_tokens"],
                "api_type": model_cfg["api_type"],
                **all_model_stats[model_name],
            }
            for model_name, model_cfg in MODELS.items()
        },
    }

    out_path = Path("data/truncation_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nStructured output saved to {out_path}")


if __name__ == "__main__":
    main()
