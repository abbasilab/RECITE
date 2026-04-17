"""Export RECITE benchmark data to parquet files."""

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from recite.benchmark.utils import clean_text

# Default name for the combined benchmark parquet
COMBINED_PARQUET_NAME = "benchmark.parquet"


def export_to_parquet_combined(
    conn: sqlite3.Connection,
    output_dir: Path,
    output_name: str = COMBINED_PARQUET_NAME,
    min_quality_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Export all RECITE data to a single combined parquet (no train/val/test split).
    Use this for the standard ~3k benchmark set.

    Args:
        conn: Database connection.
        output_dir: Directory to write the parquet and stats.
        output_name: Filename (default: benchmark.parquet).
        min_quality_score: Optional minimum quality_score to filter samples.

    Returns:
        Dict with total_samples and path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    logger.info("Exporting combined benchmark to {}", out_path)

    query = """
        SELECT
            r.id,
            r.nct_id,
            r.version_from,
            r.version_to,
            r.preamended_text,
            r.evidence,
            r.amended_text,
            r.quality_score,
            r.evidence_extraction_level,
            r.evidence_extraction_score,
            tm.year,
            tm.conditions,
            tm.keywords,
            tm.phases,
            tm.locations,
            tm.study_type,
            tm.enrollment,
            tm.start_date,
            tm.overall_status
        FROM recite r
        LEFT JOIN trial_metadata tm ON r.nct_id = tm.nct_id
    """
    if min_quality_score is not None:
        query += f" WHERE r.quality_score >= {min_quality_score}"

    df = pd.read_sql_query(query, conn)
    if df.empty:
        logger.warning("No data found in database")
        return {"total_samples": 0, "path": str(out_path)}

    logger.info("Loaded {} samples from database", len(df))

    for col in ["conditions", "keywords", "phases", "locations"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if x and pd.notna(x) else [])

    if "evidence" in df.columns:
        df["evidence_cleaned"] = df["evidence"].apply(
            lambda x: clean_text(str(x)) if x is not None and pd.notna(x) else ""
        )

    initial_count = len(df)
    df = df.dropna(subset=["preamended_text", "evidence", "amended_text"])
    if len(df) < initial_count:
        logger.info("Dropped {} samples with missing essential fields", initial_count - len(df))

    df.to_parquet(out_path, index=False)
    logger.info("Wrote {} samples to {}", len(df), out_path)

    stats = {"total_samples": len(df), "path": str(out_path)}
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def export_final_test_to_parquet(
    conn: sqlite3.Connection,
    output_dir: Path,
    output_name: str = "final_test.parquet",
    num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Export eval set from local samples (merge_source='local') with evidence.
    No shuffling; order is ORDER BY r.id. Optionally limit to first num_samples.

    Args:
        conn: Database connection (recite.db with merge_source column).
        output_dir: Directory to write the parquet.
        output_name: Filename (default: final_test.parquet).
        num_samples: If set, write only first N rows; else all.

    Returns:
        Dict with total_samples and path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    logger.info("Exporting final_test (merge_source=local) to {}", out_path)
    if num_samples is not None:
        logger.debug("Limiting final_test to first {} samples", num_samples)

    query = """
        SELECT
            r.id,
            r.nct_id,
            r.version_from,
            r.version_to,
            r.preamended_text,
            r.evidence,
            r.amended_text,
            r.quality_score,
            r.evidence_extraction_level,
            r.evidence_extraction_score,
            tm.year,
            tm.conditions,
            tm.keywords,
            tm.phases,
            tm.locations,
            tm.study_type,
            tm.enrollment,
            tm.start_date,
            tm.overall_status
        FROM recite r
        LEFT JOIN trial_metadata tm ON r.nct_id = tm.nct_id
        WHERE r.merge_source = 'local'
          AND r.evidence IS NOT NULL
          AND TRIM(COALESCE(r.evidence, '')) != ''
        ORDER BY r.id
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        logger.warning("No local samples with evidence found in recite table")
        return {"total_samples": 0, "path": str(out_path)}

    logger.info("Loaded {} local samples with evidence", len(df))
    n = len(df) if num_samples is None else min(num_samples, len(df))
    if num_samples is not None and n < len(df):
        logger.info("Writing first {} of {} samples (--num-final-test)", n, len(df))
    out_df = df.iloc[:n]

    for col in ["conditions", "keywords", "phases", "locations"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].apply(
                lambda x: json.loads(x) if x and pd.notna(x) else []
            )

    out_df.to_parquet(out_path, index=False)
    logger.info("Wrote {} samples to {}", n, out_path)
    return {"total_samples": n, "path": str(out_path)}


def export_to_parquet_splits(
    conn: sqlite3.Connection,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_quality_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Export RECITE data to parquet files with train/val/test splits.
    
    Args:
        conn: Database connection
        output_dir: Directory to write parquet files and statistics
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        seed: Random seed for reproducible shuffling
        min_quality_score: Optional minimum quality_score to filter samples
        
    Returns:
        Dictionary with export statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading data from database...")
    
    # Build query to join recite with trial_metadata
    query = """
        SELECT 
            r.id,
            r.nct_id,
            r.version_from,
            r.version_to,
            r.preamended_text,
            r.evidence,
            r.amended_text,
            r.quality_score,
            r.evidence_extraction_level,
            r.evidence_extraction_score,
            tm.year,
            tm.conditions,
            tm.keywords,
            tm.phases,
            tm.locations,
            tm.study_type,
            tm.enrollment,
            tm.start_date,
            tm.overall_status
        FROM recite r
        LEFT JOIN trial_metadata tm ON r.nct_id = tm.nct_id
    """
    
    if min_quality_score is not None:
        query += f" WHERE r.quality_score >= {min_quality_score}"
    
    # Load into DataFrame
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        logger.warning("No data found in database")
        return {
            "total_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
        }
    
    logger.info(f"Loaded {len(df)} samples from database")
    
    # Parse JSON columns (conditions, keywords, phases, locations)
    for col in ["conditions", "keywords", "phases", "locations"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if x and pd.notna(x) else [])
    
    # Add cleaned evidence column (simple text cleaning: whitespace, HTML, line breaks)
    if "evidence" in df.columns:
        df["evidence_cleaned"] = df["evidence"].apply(
            lambda x: clean_text(str(x)) if x is not None and pd.notna(x) else ""
        )
    
    # Clean: drop rows with missing essential fields
    initial_count = len(df)
    df = df.dropna(subset=["preamended_text", "evidence", "amended_text"])
    if len(df) < initial_count:
        logger.info(f"Dropped {initial_count - len(df)} samples with missing essential fields")
    
    # Shuffle with seed
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split sizes
    total = len(df)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # Remaining goes to test
    
    # Split
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Write parquet files
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    logger.info(f"Wrote parquet files to {output_dir}")
    
    # Compute and write statistics
    stats = compute_split_statistics(train_df, val_df, test_df)
    stats["total_samples"] = total
    stats["train_samples"] = len(train_df)
    stats["val_samples"] = len(val_df)
    stats["test_samples"] = len(test_df)
    
    # Write statistics JSON
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Wrote statistics to {stats_path}")
    
    return stats


def compute_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute statistics for train/val/test splits.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    # Helper to compute distribution for a column
    def compute_distribution(df: pd.DataFrame, col: str, split_name: str) -> Dict[str, Any]:
        if col not in df.columns:
            return {}
        
        result = {}
        
        # For list columns (conditions, keywords, phases, locations), flatten and count
        if df[col].dtype == "object" and df[col].notna().any():
            # Check if it's a list column
            sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            # Handle numpy arrays and lists
            import numpy as np
            is_list_like = isinstance(sample_val, (list, tuple, np.ndarray))
            
            if is_list_like:
                # Flatten lists/arrays and count
                all_items = []
                for items in df[col].dropna():
                    if isinstance(items, (list, tuple, np.ndarray)):
                        # Convert numpy array to list if needed
                        items_list = list(items) if isinstance(items, np.ndarray) else items
                        all_items.extend(items_list)
                
                counter = Counter(all_items)
                result[f"{split_name}_{col}_top_10"] = dict(counter.most_common(10))
                result[f"{split_name}_{col}_unique_count"] = len(counter)
                result[f"{split_name}_{col}_total_count"] = sum(counter.values())
            else:
                # Regular categorical column
                value_counts = df[col].value_counts().head(10)
                result[f"{split_name}_{col}_top_10"] = value_counts.to_dict()
                result[f"{split_name}_{col}_unique_count"] = df[col].nunique()
        
        return result
    
    # Compute distributions for metadata fields
    metadata_fields = ["conditions", "keywords", "phases", "locations", "study_type", "overall_status"]
    
    for field in metadata_fields:
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            dist = compute_distribution(split_df, field, split_name)
            stats.update(dist)
    
    # Year distribution
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "year" in split_df.columns:
            year_counts = split_df["year"].value_counts().sort_index()
            stats[f"{split_name}_year_distribution"] = year_counts.to_dict()
            stats[f"{split_name}_year_min"] = int(split_df["year"].min()) if split_df["year"].notna().any() else None
            stats[f"{split_name}_year_max"] = int(split_df["year"].max()) if split_df["year"].notna().any() else None
            stats[f"{split_name}_year_mean"] = float(split_df["year"].mean()) if split_df["year"].notna().any() else None
    
    # Quality score statistics
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "quality_score" in split_df.columns:
            stats[f"{split_name}_quality_score_mean"] = float(split_df["quality_score"].mean()) if split_df["quality_score"].notna().any() else None
            stats[f"{split_name}_quality_score_min"] = float(split_df["quality_score"].min()) if split_df["quality_score"].notna().any() else None
            stats[f"{split_name}_quality_score_max"] = float(split_df["quality_score"].max()) if split_df["quality_score"].notna().any() else None
    
    return stats
