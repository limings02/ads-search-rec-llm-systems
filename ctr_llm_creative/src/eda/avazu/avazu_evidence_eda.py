"""
Evidence-driven EDA for Avazu CTR dataset.

This module performs exploratory data analysis to provide quantified evidence for:
1. FeatureMap design decisions (hash bucket size, vocab size, normalization parameters)
2. Model architecture selection (interaction strength → MultiTask vs DeepFM)
3. Data processing pipelines (train/test split strategy, OOV handling, drift mitigation)

Key design principles:
- Stream-based processing: pandas.read_csv(chunksize=...) to avoid full data loading
- Evidence-driven: Decision → Metric → Artifact → Code
- Reproducible: All outputs saved to fixed directory structure with logging

Output structure:
    data/interim/avazu/
    ├── eda/            (raw EDA artifacts)
    ├── featuremap/     (featuremap_spec.yml for downstream use)
    ├── model_plan/     (model_plan.yml with architecture recommendations)
    └── reports/        (markdown decision chains)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import Counter
import warnings
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
import yaml

# ============================================================================
# Utility Classes & Functions
# ============================================================================

class TopKCounter:
    """
    Top-K value counter with memory-efficient pruning.
    
    Maintains a fixed-size Counter that automatically discards lowest-count items
    when size exceeds max_size, preventing unbounded memory growth.
    
    Usage:
        counter = TopKCounter(max_size=10000)
        for val in large_dataset:
            counter.update(val)
        top_k = counter.most_common(100)
    """
    
    def __init__(self, max_size: int = 100000):
        """
        Initialize counter.
        
        Args:
            max_size: Maximum number of unique items to track (default: 100000)
                      When exceeded, lowest-count items are pruned.
        """
        self.max_size = max_size
        self.counter: Counter = Counter()
        self.pruned_count = 0  # Track total pruned items
    
    def update(self, items: List[str]) -> None:
        """
        Update counter with items (like Counter.update).
        
        Args:
            items: List of values to count
        """
        self.counter.update(items)
        
        # Prune if exceeds max_size
        if len(self.counter) > self.max_size:
            # Remove 10% of lowest-count items
            num_to_remove = int(self.max_size * 0.1)
            
            # Get items sorted by count (ascending)
            sorted_items = self.counter.most_common()[::-1]  # Reverse to get lowest first
            
            for item, count in sorted_items[:num_to_remove]:
                self.pruned_count += count
                del self.counter[item]
    
    def most_common(self, n: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Return top-n most common items.
        
        Args:
            n: Number of top items to return (default: None = all)
        
        Returns:
            List of (value, count) tuples sorted by count descending
        """
        if n is None:
            return self.counter.most_common()
        return self.counter.most_common(n)
    
    def __len__(self) -> int:
        """Return number of tracked items (after pruning)."""
        return len(self.counter)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get counter statistics.
        
        Returns:
            Dict with keys: 'tracked_items', 'total_count', 'pruned_count'
        """
        return {
            'tracked_items': len(self.counter),
            'total_count': sum(self.counter.values()),
            'pruned_count': self.pruned_count,
        }


def hash_value(val: str, max_buckets: int = 1000000) -> int:
    """
    Hash a string value to bucket index.
    
    Used for approximate unique counting via HyperLogLog-like approach.
    
    Args:
        val: Value to hash
        max_buckets: Number of buckets (default: 1M)
    
    Returns:
        Bucket index (0 to max_buckets-1)
    """
    hash_obj = hashlib.md5(str(val).encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int % max_buckets


def estimate_entropy(top_k_counts: List[int], total_count: int) -> float:
    """
    Estimate Shannon entropy using top-K counts + OTHER bucket.
    
    Formula:
        H = -Σ (p_i * log2(p_i))
    
    where p_i = count_i / total_count
    
    Note: This is an APPROXIMATION. The true entropy may be higher
    if there are many rare items not in top-K. The OTHER bucket
    represents all items beyond top-K with equal probability assumption.
    
    Args:
        top_k_counts: List of counts for top-K items
        total_count: Total number of items
    
    Returns:
        Approximate entropy in bits (0 to log2(num_items))
    """
    if total_count == 0:
        return 0.0
    
    # Calculate probability for each top-K item
    entropy = 0.0
    for count in top_k_counts:
        if count > 0:
            p = count / total_count
            entropy -= p * np.log2(p)
    
    # Add contribution from OTHER bucket (all items not in top-K)
    other_count = total_count - sum(top_k_counts)
    if other_count > 0:
        # Assume OTHER items are uniformly distributed
        p_other = other_count / total_count
        # For OTHER bucket, we assume all items have equal probability
        # This is a simplification; entropy is maximized when all items are equiprobable
        entropy -= p_other * np.log2(p_other)
    
    return entropy


def estimate_hhi(top_k_counts: List[int], total_count: int) -> float:
    """
    Estimate Herfindahl-Hirschman Index (HHI).
    
    Formula:
        HHI = Σ (p_i)^2
    
    Range: [0, 1], where:
        - 0 = perfect competition (many equal items)
        - 1 = monopoly (one item dominates)
    
    Note: This is an APPROXIMATION using top-K. True HHI may be lower
    if there are many rare items not in top-K.
    
    Args:
        top_k_counts: List of counts for top-K items
        total_count: Total number of items
    
    Returns:
        Approximate HHI (0 to 1)
    """
    if total_count == 0:
        return 0.0
    
    hhi = 0.0
    for count in top_k_counts:
        if count > 0:
            p = count / total_count
            hhi += p ** 2
    
    return hhi


def wilson_ci(clicks: int, impressions: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Compute Wilson score interval for CTR confidence interval.
    
    Better than Clopper-Pearson for proportions, especially at extremes.
    
    Formula:
        ci_lower = (p_hat + z^2/(2n) - z*sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2))) / (1 + z^2/n)
        ci_upper = (p_hat + z^2/(2n) + z*sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2))) / (1 + z^2/n)
    
    where p_hat = clicks / impressions, z = 1.96 for 95% CI
    
    Args:
        clicks: Number of clicks (successes)
        impressions: Total impressions (trials)
        z: Z-score for confidence level (default: 1.96 for 95%)
    
    Returns:
        Tuple of (ci_lower, ci_upper) as floats [0, 1]
    """
    if impressions == 0:
        return 0.0, 0.0
    
    p_hat = clicks / impressions
    z_sq = z ** 2
    
    denom = 1 + z_sq / impressions
    center = p_hat + z_sq / (2 * impressions)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / impressions + z_sq / (4 * impressions ** 2))
    
    ci_lower = (center - margin) / denom
    ci_upper = (center + margin) / denom
    
    # Clip to [0, 1]
    ci_lower = max(0.0, ci_lower)
    ci_upper = min(1.0, ci_upper)
    
    return ci_lower, ci_upper


# ============================================================================
# Configuration & Setup
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace with fields:
        - input_csv: str, path to Avazu train.csv
        - out_root: str, output root directory (data/interim/avazu)
        - chunksize: int, chunk size for streaming (default: 500000)
        - topk: int, number of top-K values to extract (default: 20)
        - top_values: int, max top values for label analysis (default: 10)
        - min_support: int, minimum support for interaction pairs (default: 100)
        - sample_rows: int, sample size for MI computation (default: 200000)
        - seed: int, random seed (default: 42)
        - train_days: int, number of days for training split (default: 7)
        - test_days: int, number of days for test split (default: 2)
        - verbose: bool, verbose logging (default: False)
    """
    parser = argparse.ArgumentParser(
        description="Evidence-driven EDA for Avazu CTR dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to Avazu train.csv (e.g., data/raw/avazu/train.csv)"
    )
    parser.add_argument(
        "--out_root",
        required=True,
        help="Output root directory (e.g., data/interim/avazu)"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500000,
        help="Chunk size for streaming (default: 500000)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of top-K values to extract (default: 20)"
    )
    parser.add_argument(
        "--top_values",
        type=int,
        default=10,
        help="Max top values for label analysis (default: 10)"
    )
    parser.add_argument(
        "--min_support",
        type=int,
        default=100,
        help="Minimum support for interaction pairs (default: 100)"
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=200000,
        help="Sample size for MI computation (default: 200000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--train_days",
        type=int,
        default=7,
        help="Number of days for training split (default: 7)"
    )
    parser.add_argument(
        "--test_days",
        type=int,
        default=2,
        help="Number of days for test split (default: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_logger(log_dir: str, verbose: bool = False) -> logging.Logger:
    """
    Set up logging to both console and file.
    
    Args:
        log_dir: Directory to save log files
        verbose: If True, set level to DEBUG; else INFO
    
    Returns:
        Configured logger instance
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("AVAZU_EDA")
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s %(levelname)s: [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_dir_path / f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_fmt = logging.Formatter(
        "%(asctime)s %(levelname)s: [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    return logger


def create_output_dirs(out_root: str) -> Dict[str, Path]:
    """
    Create output directory structure and return path dict.
    
    Args:
        out_root: Root output directory (e.g., data/interim/avazu)
    
    Returns:
        Dict with keys: 'eda', 'topk', 'time', 'split', 'drift', 'label', 
                        'hash', 'interactions', 'leakage', 'featuremap', 
                        'model_plan', 'reports'
    """
    root_path = Path(out_root)
    
    # Define all subdirectories
    subdirs = {
        'eda': root_path / 'eda',
        'topk': root_path / 'eda' / 'topk',
        'time': root_path / 'eda' / 'time',
        'split': root_path / 'eda' / 'split',
        'drift': root_path / 'eda' / 'drift',
        'label': root_path / 'eda' / 'label',
        'hash': root_path / 'eda' / 'hash',
        'interactions': root_path / 'eda' / 'interactions',
        'leakage': root_path / 'eda' / 'leakage',
        'featuremap': root_path / 'featuremap',
        'model_plan': root_path / 'model_plan',
        'reports': root_path / 'reports',
    }
    
    # Create all directories
    for dir_path in subdirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return subdirs


# ============================================================================
# Pass 1: Schema & Basic Statistics
# ============================================================================

def detect_schema(csv_path: str, sample_rows: int = 1000) -> Dict[str, Any]:
    """
    Detect schema (column names, data types, null counts) from sample.
    
    Args:
        csv_path: Path to CSV file
        sample_rows: Number of rows to sample for type detection
    
    Returns:
        Dict with keys:
        - 'columns': list of column names
        - 'dtypes': dict {col: inferred_type}
        - 'label_col': name of label column (assumed: 'click' or last col)
        - 'feature_cols': list of feature columns (excluding label)
        - 'categorical_cols': list of detected categorical columns
        - 'numerical_cols': list of detected numerical columns
        - 'timestamp_col': name of timestamp column (assumed: 'hour')
        - 'sample_size': number of rows sampled
        - 'timestamp': schema detection timestamp
    """
    # Read sample to detect schema
    try:
        df_sample = pd.read_csv(csv_path, nrows=sample_rows)
    except Exception as e:
        raise IOError(f"Failed to read CSV file {csv_path}: {e}")
    
    columns = df_sample.columns.tolist()
    
    # Infer dtypes
    dtypes = {}
    categorical_cols = []
    numerical_cols = []
    
    for col in columns:
        # Check if column is numerical
        try:
            pd.to_numeric(df_sample[col].dropna(), errors='coerce')
            non_null_numeric = pd.to_numeric(df_sample[col], errors='coerce').notna().sum()
            if non_null_numeric > 0.8 * len(df_sample):
                dtypes[col] = 'numerical'
                numerical_cols.append(col)
            else:
                dtypes[col] = 'categorical'
                categorical_cols.append(col)
        except:
            dtypes[col] = 'categorical'
            categorical_cols.append(col)
    
    # Identify special columns
    label_col = 'click' if 'click' in columns else columns[-1]
    timestamp_col = 'hour' if 'hour' in columns else None
    feature_cols = [col for col in columns if col != label_col]
    
    return {
        'columns': columns,
        'dtypes': dtypes,
        'label_col': label_col,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'timestamp_col': timestamp_col,
        'sample_size': len(df_sample),
        'timestamp': datetime.now().isoformat()
    }


def compute_schema_and_overview(input_csv: str, chunksize: int, out_root: str, sample_rows: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute schema and overview statistics using streaming (chunked reading).
    
    Pass 1: Read all chunks to accumulate:
    - Total row count
    - Total clicks and global CTR
    - Min/max hour values
    - Null counts per column
    - Data type validation (click must be 0/1, hour must be 10-digit)
    
    Generates two files:
    - eda/schema.json: Column metadata and validation anomalies
    - eda/overview.json: Global statistics
    
    Args:
        input_csv: Path to input CSV file
        chunksize: Chunk size for streaming
        out_root: Output root directory
    
    Returns:
        Tuple of (schema_dict, overview_dict)
    """
    output_dirs = create_output_dirs(out_root)
    
    logger = logging.getLogger("AVAZU_EDA")
    logger.info(f"[SCHEMA & OVERVIEW] Starting schema detection and overview computation...")
    
    # Step 1: Detect schema from sample
    logger.info(f"[SCHEMA & OVERVIEW] Detecting schema from sample...")
    schema = detect_schema(input_csv, sample_rows=1000)
    columns = schema['columns']
    label_col = schema['label_col']
    timestamp_col = schema['timestamp_col']
    feature_cols = schema['feature_cols']
    
    logger.info(f"[SCHEMA & OVERVIEW] Detected {len(columns)} columns: {columns}")
    logger.info(f"[SCHEMA & OVERVIEW] Label: {label_col}, Timestamp: {timestamp_col}")
    
    # Step 2: Stream through all chunks to accumulate statistics
    logger.info(f"[SCHEMA & OVERVIEW] Streaming through data (chunksize={chunksize})...")
    
    # Initialize accumulators
    total_rows = 0
    total_clicks = 0
    null_counts = {col: 0 for col in columns}
    anomalies = {col: [] for col in columns}
    
    hour_values = []  # Collect hour values for min/max
    
    # Limit rows for dry-run if sample_rows is specified
    nrows_limit = sample_rows if sample_rows is not None else None
    if nrows_limit:
        logger.info(f"[SCHEMA & OVERVIEW] [DRY-RUN] Limiting to first {nrows_limit} rows for quick validation")
    
    # Stream through chunks
    chunk_idx = 0
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str, nrows=nrows_limit):
        chunk_idx += 1
        total_rows += len(chunk)
        
        # Process each column
        for col in columns:
            # Count nulls
            null_counts[col] += chunk[col].isna().sum()
            
            # Validate label column (must be 0 or 1)
            if col == label_col:
                invalid_labels = chunk[~chunk[col].isin(['0', '1', np.nan])][col].unique()
                if len(invalid_labels) > 0:
                    anomalies[col].append(f"Non-binary values: {invalid_labels.tolist()[:5]}")
                
                # Count clicks
                clicks_in_chunk = (chunk[col] == '1').sum()
                total_clicks += clicks_in_chunk
            
            # Validate timestamp column (must be 8-digit YYYYMMDD)
            if col == timestamp_col:
                invalid_hours = chunk[~chunk[col].str.match(r'^\d{8}$', na=False)][col].unique()
                if len(invalid_hours) > 0:
                    anomalies[col].append(f"Non-8-digit hours (expected YYYYMMDD): {invalid_hours.tolist()[:5]}")

                # Collect hour/day values for min/max (use YYYYMMDD)
                valid_hours = chunk[chunk[col].str.match(r'^\d{8}$', na=False)][col].dropna()
                if len(valid_hours) > 0:
                    hour_values.extend(valid_hours.tolist())
        
        if chunk_idx % 5 == 0:
            logger.debug(f"[SCHEMA & OVERVIEW] Processed {chunk_idx} chunks ({total_rows} rows)...")
    
    # Step 3: Post-process accumulated data
    logger.info(f"[SCHEMA & OVERVIEW] Processed {total_rows} total rows across {chunk_idx} chunks")
    
    # Compute global CTR
    global_ctr = total_clicks / total_rows if total_rows > 0 else 0.0
    
    # Compute hour min/max
    if hour_values:
        hour_values_sorted = sorted(hour_values)
        hour_min = hour_values_sorted[0]
        hour_max = hour_values_sorted[-1]
    else:
        hour_min = None
        hour_max = None
    
    # Clean up anomalies (only keep non-empty entries)
    anomalies = {col: anomalies[col] for col in columns if anomalies[col]}
    
    # Step 4: Build schema JSON
    schema_json = {
        'timestamp': datetime.now().isoformat(),
        'total_rows': int(total_rows),
        'columns': [],
        'anomalies': anomalies,
    }
    
    for col in columns:
        col_info = {
            'name': col,
            'dtype': schema['dtypes'][col],
            'null_count': int(null_counts[col]),
            'null_rate': float(null_counts[col] / total_rows) if total_rows > 0 else 0.0,
        }
        schema_json['columns'].append(col_info)
    
    # Step 5: Build overview JSON
    overview_json = {
        'timestamp': datetime.now().isoformat(),
        'total_rows': int(total_rows),
        'total_clicks': int(total_clicks),
        'global_ctr': float(global_ctr),
        'hour_min': hour_min,
        'hour_max': hour_max,
        'num_columns': len(columns),
        'num_features': len(feature_cols),
        'label_column': label_col,
        'timestamp_column': timestamp_col,
        'column_nulls': {col: {'count': int(null_counts[col]), 'rate': float(null_counts[col] / total_rows) if total_rows > 0 else 0.0} for col in columns},
    }
    
    # Step 6: Save to files
    schema_file = output_dirs['eda'] / 'schema.json'
    overview_file = output_dirs['eda'] / 'overview.json'
    
    try:
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(schema_json, f, indent=2, ensure_ascii=False)
        logger.info(f"[SCHEMA & OVERVIEW] Schema saved to {schema_file}")
    except Exception as e:
        logger.error(f"[SCHEMA & OVERVIEW] Failed to save schema.json: {e}")
        raise
    
    try:
        with open(overview_file, 'w', encoding='utf-8') as f:
            json.dump(overview_json, f, indent=2, ensure_ascii=False)
        logger.info(f"[SCHEMA & OVERVIEW] Overview saved to {overview_file}")
    except Exception as e:
        logger.error(f"[SCHEMA & OVERVIEW] Failed to save overview.json: {e}")
        raise
    
    logger.info(f"[SCHEMA & OVERVIEW] ✓ Schema & Overview Complete")
    logger.info(f"  - Total rows: {total_rows:,}")
    logger.info(f"  - Total clicks: {total_clicks:,}")
    logger.info(f"  - Global CTR: {global_ctr:.4f}")
    logger.info(f"  - Hour range: {hour_min} ~ {hour_max}")
    if anomalies:
        logger.warning(f"  - Anomalies detected in {len(anomalies)} columns")
    
    return schema_json, overview_json


def compute_columns_profile(input_csv: str, chunksize: int, topk: int, 
                           out_root: str, max_unique_set: int = 200000, sample_rows: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-column profiling statistics: missing rate, approximate cardinality,
    top-K ratios, entropy/HHI approximation, ID-likeness score.
    
    Uses stream-based processing with TopKCounter to manage memory.
    Generates:
    - eda/columns_profile.csv: summary statistics for all columns
    - eda/topk/topk_{col}.csv: top-K values for each column
    
    Args:
        input_csv: Path to input CSV file
        chunksize: Chunk size for streaming
        topk: Number of top-K values to track and extract
        out_root: Output root directory
        max_unique_set: Maximum unique values to track for cardinality estimation (default: 200000)
    
    Returns:
        Dict {col_name: {
            'missing_rate': float,
            'nunique_approx': int (approximate),
            'nunique_truncated': int (after pruning),
            'top1_ratio': float,
            'top10_ratio': float,
            'top50_ratio': float,
            'entropy_approx': float,
            'hhi_approx': float,
            'id_like_score': float,
        }}
    """
    output_dirs = create_output_dirs(out_root)
    logger = logging.getLogger("AVAZU_EDA")
    
    logger.info(f"[COLUMNS PROFILE] Starting per-column profiling (topk={topk})...")
    
    # Read first chunk to detect schema
    df_sample = pd.read_csv(input_csv, nrows=1000)
    columns = df_sample.columns.tolist()
    
    # Exclude id and click columns from analysis
    skip_cols = {'id', 'click', 'ID', 'Click'}
    profile_cols = [col for col in columns if col not in skip_cols]
    
    logger.info(f"[COLUMNS PROFILE] Profiling {len(profile_cols)} columns (skipped: {skip_cols & set(columns)})")
    
    # Initialize per-column accumulators
    counters = {col: TopKCounter(max_size=topk * 10) for col in profile_cols}  # Keep 10x topk for accuracy
    null_counts = {col: 0 for col in profile_cols}
    unique_sets = {col: set() for col in profile_cols}  # For cardinality estimation (truncated)
    total_rows = 0
    
    # Stream through data
    logger.info(f"[COLUMNS PROFILE] Streaming through data (chunksize={chunksize})...")
    chunk_idx = 0
    
    # Limit rows for dry-run if sample_rows is specified
    nrows_limit = sample_rows if sample_rows is not None else None
    if nrows_limit:
        logger.info(f"[COLUMNS PROFILE] [DRY-RUN] Limiting to first {nrows_limit} rows for quick validation")
    
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str, nrows=nrows_limit):
        chunk_idx += 1
        total_rows += len(chunk)
        
        for col in profile_cols:
            # Count nulls
            null_counts[col] += chunk[col].isna().sum()
            
            # Get non-null values
            valid_vals = chunk[~chunk[col].isna()][col].tolist()
            
            # Update counter with values
            if valid_vals:
                counters[col].update(valid_vals)
            
            # Track unique values for cardinality (with truncation to avoid OOM)
            if len(unique_sets[col]) < max_unique_set:
                valid_vals_set = set(valid_vals)
                unique_sets[col].update(valid_vals_set)
                
                # Truncate if exceeds max
                if len(unique_sets[col]) > max_unique_set:
                    unique_sets[col] = set(list(unique_sets[col])[:max_unique_set])
        
        if chunk_idx % 5 == 0:
            logger.debug(f"[COLUMNS PROFILE] Processed {chunk_idx} chunks ({total_rows} rows)...")
    
    logger.info(f"[COLUMNS PROFILE] Processed {total_rows} rows in {chunk_idx} chunks")
    
    # Compute profile for each column
    profile_results = {}
    profile_rows = []
    
    for col in profile_cols:
        # Basic stats
        missing_rate = null_counts[col] / total_rows if total_rows > 0 else 0.0
        
        # Approximate unique count (estimate using set with truncation)
        # If we hit the max_unique_set limit, we can estimate with HLL-like approach
        nunique_actual = len(unique_sets[col])
        
        # Approximate cardinality: if we truncated, extrapolate
        if nunique_actual == max_unique_set:
            # We truncated - estimate total unique as proportional
            # This is a rough estimate; ideally use HyperLogLog
            nunique_approx = int(nunique_actual * (total_rows - null_counts[col]) / (total_rows - null_counts[col]))
            # For now, use actual as lower bound
            nunique_approx = nunique_actual
        else:
            nunique_approx = nunique_actual
        
        # Get top-K items
        top_items = counters[col].most_common(topk)
        top_counts = [count for val, count in top_items]
        
        # Compute top-K ratios
        top1_count = top_counts[0] if len(top_counts) > 0 else 0
        top10_count = sum(top_counts[:10]) if len(top_counts) >= 10 else sum(top_counts)
        top50_count = sum(top_counts[:50]) if len(top_counts) >= 50 else sum(top_counts)
        
        non_null_count = total_rows - null_counts[col]
        top1_ratio = top1_count / non_null_count if non_null_count > 0 else 0.0
        top10_ratio = top10_count / non_null_count if non_null_count > 0 else 0.0
        top50_ratio = top50_count / non_null_count if non_null_count > 0 else 0.0
        
        # Compute entropy and HHI (approximate using top-K + OTHER)
        entropy_approx = estimate_entropy(top_counts, non_null_count) if non_null_count > 0 else 0.0
        hhi_approx = estimate_hhi(top_counts, non_null_count) if non_null_count > 0 else 0.0
        
        # ID-likeness score: approximate cardinality / total rows
        id_like_score = nunique_approx / total_rows if total_rows > 0 else 0.0
        
        # Store results
        profile_results[col] = {
            'missing_rate': float(missing_rate),
            'nunique_approx': int(nunique_approx),
            'nunique_truncated': int(nunique_actual),
            'top1_ratio': float(top1_ratio),
            'top10_ratio': float(top10_ratio),
            'top50_ratio': float(top50_ratio),
            'entropy_approx': float(entropy_approx),
            'hhi_approx': float(hhi_approx),
            'id_like_score': float(id_like_score),
        }
        
        # Add row for CSV output
        profile_rows.append({
            'column': col,
            'missing_rate': f"{missing_rate:.6f}",
            'nunique_approx': nunique_approx,
            'nunique_truncated': nunique_actual,
            'top1_ratio': f"{top1_ratio:.6f}",
            'top10_ratio': f"{top10_ratio:.6f}",
            'top50_ratio': f"{top50_ratio:.6f}",
            'entropy_approx': f"{entropy_approx:.4f}",
            'hhi_approx': f"{hhi_approx:.6f}",
            'id_like_score': f"{id_like_score:.6f}",
        })
        
        # Save top-K for this column
        topk_file = output_dirs['topk'] / f"topk_{col}.csv"
        topk_rows = [
            {'value': val, 'count': int(count), 'pct': f"{count / non_null_count:.6f}"}
            for val, count in top_items
        ]
        
        try:
            df_topk = pd.DataFrame(topk_rows)
            df_topk.to_csv(topk_file, index=False)
        except Exception as e:
            logger.warning(f"[COLUMNS PROFILE] Failed to save topk_{col}.csv: {e}")
    
    # Save columns_profile.csv
    profile_file = output_dirs['eda'] / 'columns_profile.csv'
    try:
        df_profile = pd.DataFrame(profile_rows)
        df_profile.to_csv(profile_file, index=False)
        logger.info(f"[COLUMNS PROFILE] Columns profile saved to {profile_file}")
    except Exception as e:
        logger.error(f"[COLUMNS PROFILE] Failed to save columns_profile.csv: {e}")
        raise
    
    # Log summary
    logger.info(f"[COLUMNS PROFILE] ✓ Columns Profile Complete")
    logger.info(f"  - Columns profiled: {len(profile_results)}")
    logger.info(f"  - Top-K files generated: {len(profile_results)}")
    
    # Show top ID-like columns
    id_like_cols = sorted(profile_results.items(), key=lambda x: x[1]['id_like_score'], reverse=True)[:5]
    logger.info(f"  - Top ID-like columns (highest cardinality ratio):")
    for col, stats in id_like_cols:
        logger.info(f"    * {col}: id_like_score={stats['id_like_score']:.4f}, nunique={stats['nunique_approx']}")
    
    return profile_results


def compute_column_stats(csv_path: str, chunksize: int) -> Dict[str, Dict[str, Any]]:
    """
    Compute basic statistics for each column using streaming.
    
    Performs: column-wise stats via multiple passes (null count, 
    unique values, value distribution summary)
    
    Args:
        csv_path: Path to CSV file
        chunksize: Chunk size for streaming
    
    Returns:
        Dict {col_name: {
            'null_count': int,
            'null_pct': float,
            'unique_count': int,
            'dtype': str,
            'min_val': any (if numeric),
            'max_val': any (if numeric),
            'mean': float (if numeric),
            'median': float (if numeric),
            'std': float (if numeric)
        }}
    """
    ...


def analyze_sparsity_and_longtail(csv_path: str, chunksize: int) -> Dict[str, Dict[str, Any]]:
    """
    Analyze sparsity (% of zeros) and long-tail characteristics.
    
    For categorical: compute value frequency distribution (gini coeff, top-10% volume)
    For numerical: compute sparsity, log-scale variance
    
    Args:
        csv_path: Path to CSV file
        chunksize: Chunk size for streaming
    
    Returns:
        Dict {col_name: {
            'sparsity': float (% zeros for numerical),
            'gini_coefficient': float (0-1, higher = more imbalanced),
            'top_10_pct_volume': float (% volume in top 10% values),
            'is_longtail': bool (gini > 0.6 or top_10_pct > 50%),
            'cardinality': int (unique value count),
            'most_common_pct': float (% of most frequent value)
        }}
    """
    ...


def compute_topk_values(csv_path: str, chunksize: int, topk: int, 
                        output_dirs: Dict[str, Path]) -> None:
    """
    Extract top-K value frequencies for each categorical column.
    
    Saves files: output_dirs['topk']/topk_{col}.csv
    Format: CSV with columns ['value', 'count', 'pct']
    
    Args:
        csv_path: Path to CSV file
        chunksize: Chunk size for streaming
        topk: Number of top values to extract
        output_dirs: Dict of output directories
    """
    ...


def extract_time_features(hour_value: str) -> Tuple[int, int, int]:
    """
    Parse hour field (format: YYYYMMDD) into components.

    Args:
        hour_value: Date string in YYYYMMDD (e.g., '20141021')

    Returns:
        Tuple of (day_of_week: 0-6, hour_of_day: int, is_weekend: 0/1)
        - `day_of_week`: 0 = Monday
        - `hour_of_day`: set to 0 when only date is available (no hourly granularity)
        - `is_weekend`: 1 if Saturday/Sunday else 0

    Notes:
        The original implementation assumed hourly granularity (YYYYMMDDHH).
        For Avazu `train.csv` the `hour` field is YYYYMMDD (8 digits). We parse
        day information and set `hour_of_day` to 0 to keep interfaces stable.
    """
    if not hour_value or not isinstance(hour_value, str):
        return (None, None, None)

    hv = hour_value.strip()
    try:
        # Expect YYYYMMDD (8 digits)
        if len(hv) >= 8 and hv[:8].isdigit():
            dt = datetime.strptime(hv[:8], "%Y%m%d")
            day_of_week = dt.weekday()
            hour_of_day = 0
            is_weekend = 1 if day_of_week >= 5 else 0
            return (day_of_week, hour_of_day, is_weekend)
    except Exception:
        pass

    return (None, None, None)


def compute_temporal_ctr(input_csv: str, chunksize: int, out_root: str, sample_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute daily CTR trends from 'hour' field (YYYYMMDD format).

    Streams through CSV, accumulates impressions/clicks by day (hour field
    contains YYYYMMDD), then writes sorted results.
    
    Saves files:
    - {out_root}/eda/time/time_ctr_hour.csv: columns [hour, impressions, clicks, ctr]
    - {out_root}/eda/time/time_ctr_day.csv: columns [day, impressions, clicks, ctr]
    
    Args:
        input_csv: Path to input CSV
        chunksize: Streaming chunk size
        out_root: Output root directory
    
    Returns:
        Dict with keys: 'hourly_ctr_trend' (daily buckets in this dataset),
        'daily_ctr_trend', 'peak_hour', 'low_hour'
    """
    logger = logging.getLogger(__name__)
    logger.info("[TEMPORAL_CTR] Starting hourly/daily CTR computation...")
    
    input_path = Path(input_csv)
    time_dir = Path(out_root) / 'eda' / 'time'
    time_dir.mkdir(parents=True, exist_ok=True)
    
    # Stream through data and accumulate by day (hour field is YYYYMMDD)
    # We keep hour_stats keyed by YYYYMMDD for compatibility, but hourly
    # granularity is not available in the dataset.
    hour_stats = {}  # day (YYYYMMDD) -> {'impressions': n, 'clicks': m}
    day_stats = {}   # day (YYYYMMDD) -> {'impressions': n, 'clicks': m}
    
    total_rows = 0
    
    # Limit rows for dry-run if sample_rows is specified
    nrows_limit = sample_rows if sample_rows is not None else None
    if nrows_limit:
        logger.info(f"[TEMPORAL_CTR] [DRY-RUN] Limiting to first {nrows_limit} rows for quick validation")
    
    try:
        for chunk in pd.read_csv(input_path, chunksize=chunksize, dtype=str, 
                                 usecols=['hour', 'click'], nrows=nrows_limit):
            total_rows += len(chunk)
            
            # Process each row
            for idx, row in chunk.iterrows():
                hour_str = row.get('hour', '')
                click_val = row.get('click', '0')
                
                # Parse hour/day (YYYYMMDD format expected)
                if len(hour_str) >= 8:
                    day = hour_str[:8]      # YYYYMMDD
                    hour = day             # use day as key since no hourly granularity

                    click_count = int(click_val) if click_val in ['0', '1'] else 0

                    # Update day/hour stats (both keyed by YYYYMMDD)
                    if hour not in hour_stats:
                        hour_stats[hour] = {'impressions': 0, 'clicks': 0}
                    hour_stats[hour]['impressions'] += 1
                    hour_stats[hour]['clicks'] += click_count

                    if day not in day_stats:
                        day_stats[day] = {'impressions': 0, 'clicks': 0}
                    day_stats[day]['impressions'] += 1
                    day_stats[day]['clicks'] += click_count
    
    except Exception as e:
        logger.error(f"[TEMPORAL_CTR] Error reading CSV: {e}")
        raise
    
    # Convert to DataFrame and compute CTR
    hour_records = []
    for hour in sorted(hour_stats.keys()):
        impr = hour_stats[hour]['impressions']
        clicks = hour_stats[hour]['clicks']
        ctr = clicks / impr if impr > 0 else 0.0
        hour_records.append({
            'hour': hour,
            'impressions': impr,
            'clicks': clicks,
            'ctr': ctr
        })
    
    day_records = []
    for day in sorted(day_stats.keys()):
        impr = day_stats[day]['impressions']
        clicks = day_stats[day]['clicks']
        ctr = clicks / impr if impr > 0 else 0.0
        day_records.append({
            'day': day,
            'impressions': impr,
            'clicks': clicks,
            'ctr': ctr
        })
    
    # Write to CSV
    hour_df = pd.DataFrame(hour_records)
    day_df = pd.DataFrame(day_records)
    
    # Ensure hour/day columns are strings (YYYYMMDDHH / YYYYMMDD format)
    hour_df['hour'] = hour_df['hour'].astype(str)
    day_df['day'] = day_df['day'].astype(str)
    
    hour_csv = time_dir / 'time_ctr_hour.csv'
    day_csv = time_dir / 'time_ctr_day.csv'
    
    hour_df.to_csv(hour_csv, index=False)
    day_df.to_csv(day_csv, index=False)
    
    # Compute peak and low hours
    hour_df_sorted = hour_df.sort_values('ctr', ascending=False)
    peak_hour = hour_df_sorted.iloc[0]['hour'] if len(hour_df_sorted) > 0 else None
    low_hour = hour_df_sorted.iloc[-1]['hour'] if len(hour_df_sorted) > 0 else None
    
    avg_ctr = sum(h['clicks'] for h in hour_stats.values()) / total_rows if total_rows > 0 else 0.0
    
    logger.info(f"[TEMPORAL_CTR] ✓ Complete")
    logger.info(f"  - Hourly records: {len(hour_records)}")
    logger.info(f"  - Daily records: {len(day_records)}")
    logger.info(f"  - Peak hour: {peak_hour} (ctr={hour_df_sorted.iloc[0]['ctr']:.4f})")
    logger.info(f"  - Low hour: {low_hour} (ctr={hour_df_sorted.iloc[-1]['ctr']:.4f})")
    logger.info(f"  - Global CTR: {avg_ctr:.4f}")
    
    return {
        'hourly_ctr_trend': hour_records,
        'daily_ctr_trend': day_records,
        'peak_hour': peak_hour,
        'low_hour': low_hour,
        'avg_ctr': avg_ctr,
        'hour_csv': str(hour_csv),
        'day_csv': str(day_csv)
    }


def compute_ctr_by_group(input_csv: str, chunksize: int, out_root: str, topk: int, sample_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute CTR distribution for top-K values of each feature (with CI & significance).
    
    Streams through CSV, accumulates impressions/clicks by feature value,
    computes Wilson CI and z-test significance vs global CTR.
    
    Saves files: {out_root}/eda/label/ctr_by_{col}.csv
    Format: CSV with columns [
        'feature', 'value', 'impressions', 'clicks', 'ctr', 
        'ci_lower', 'ci_upper', 'zscore', 'is_significant'
    ]
    
    Confidence interval: Wilson score interval (95%)
    Significance: z-test vs global CTR (p < 0.05)
    
    Args:
        input_csv: Path to input CSV file
        chunksize: Chunk size for streaming
        out_root: Output root directory
        topk: Number of top values per feature to analyze and output
    
    Returns:
        Dict with keys: 'total_rows', 'global_ctr', 'features_analyzed',
                       'ctr_by_feature': {col_name: [group_stats]}
    """
    output_dirs = create_output_dirs(out_root)
    logger = logging.getLogger("AVAZU_EDA")
    
    logger.info(f"[CTR_BY_GROUP] Starting feature-wise CTR computation (topk={topk})...")
    
    # Read schema to get feature columns
    df_sample = pd.read_csv(input_csv, nrows=1000)
    columns = df_sample.columns.tolist()
    
    # Exclude label and ID columns
    skip_cols = {'id', 'click', 'ID', 'Click', 'hour'}  # hour handled separately
    profile_cols = [col for col in columns if col not in skip_cols]
    
    logger.info(f"[CTR_BY_GROUP] Analyzing {len(profile_cols)} features")
    
    # Initialize accumulators
    feature_stats = {col: {} for col in profile_cols}  # col -> {value: {'impr': n, 'clicks': m}}
    total_rows = 0
    total_clicks = 0
    
    # Stream through data
    logger.info(f"[CTR_BY_GROUP] Streaming through data (chunksize={chunksize})...")
    chunk_idx = 0
    
    # Limit rows for dry-run if sample_rows is specified
    nrows_limit = sample_rows if sample_rows is not None else None
    if nrows_limit:
        logger.info(f"[CTR_BY_GROUP] [DRY-RUN] Limiting to first {nrows_limit} rows for quick validation")
    
    try:
        for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str, nrows=nrows_limit):
            chunk_idx += 1
            total_rows += len(chunk)
            
            # Count global clicks
            total_clicks += (chunk['click'] == '1').sum()
            
            # Accumulate by feature value
            for col in profile_cols:
                click_col = (chunk['click'] == '1').astype(int)
                
                for val in chunk[col].unique():
                    mask = chunk[col] == val
                    n_impr = mask.sum()
                    n_clicks = click_col[mask].sum()
                    
                    if val not in feature_stats[col]:
                        feature_stats[col][val] = {'impr': 0, 'clicks': 0}
                    
                    feature_stats[col][val]['impr'] += n_impr
                    feature_stats[col][val]['clicks'] += n_clicks
            
            if chunk_idx % 5 == 0:
                logger.debug(f"[CTR_BY_GROUP] Processed {chunk_idx} chunks ({total_rows} rows)...")
    
    except Exception as e:
        logger.error(f"[CTR_BY_GROUP] Error reading CSV: {e}")
        raise
    
    logger.info(f"[CTR_BY_GROUP] Processed {total_rows} rows, {total_clicks} total clicks")
    
    # Compute global CTR
    global_ctr = total_clicks / total_rows if total_rows > 0 else 0.0
    logger.info(f"[CTR_BY_GROUP] Global CTR: {global_ctr:.6f}")
    
    # Compute per-feature CTR with CI and significance
    results = {
        'total_rows': total_rows,
        'total_clicks': total_clicks,
        'global_ctr': global_ctr,
        'features_analyzed': len(profile_cols),
        'ctr_by_feature': {}
    }
    
    for col in profile_cols:
        # Sort by impressions (most frequent first)
        sorted_values = sorted(
            feature_stats[col].items(),
            key=lambda x: x[1]['impr'],
            reverse=True
        )
        
        # Keep only top-K
        top_values = sorted_values[:topk]
        
        # Compute statistics for each value
        group_rows = []
        for val, stats in top_values:
            impr = stats['impr']
            clicks = stats['clicks']
            ctr = clicks / impr if impr > 0 else 0.0
            
            # Wilson CI
            ci_lower, ci_upper = wilson_ci(clicks, impr)
            
            # Z-test: z = (ctr - global_ctr) / sqrt(global_ctr * (1 - global_ctr) / impr)
            if impr > 0 and global_ctr > 0 and global_ctr < 1:
                se = np.sqrt(global_ctr * (1 - global_ctr) / impr)
                zscore = (ctr - global_ctr) / se if se > 0 else 0.0
            else:
                zscore = 0.0
            
            # Significance: |z| > 1.96 for p < 0.05 (two-tailed)
            is_significant = abs(zscore) > 1.96
            
            group_rows.append({
                'feature': col,
                'value': val,
                'impressions': impr,
                'clicks': clicks,
                'ctr': ctr,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'zscore': zscore,
                'is_significant': is_significant
            })
        
        results['ctr_by_feature'][col] = group_rows
        
        # Save to CSV
        output_file = output_dirs['label'] / f"ctr_by_{col}.csv"
        try:
            df_group = pd.DataFrame(group_rows)
            df_group.to_csv(output_file, index=False)
        except Exception as e:
            logger.warning(f"[CTR_BY_GROUP] Failed to save ctr_by_{col}.csv: {e}")
    
    logger.info(f"[CTR_BY_GROUP] ✓ Complete")
    logger.info(f"  - Features analyzed: {len(profile_cols)}")
    logger.info(f"  - Top-K per feature: {topk}")
    
    # Show sample: feature with highest variance in CTR
    max_variance_col = None
    max_variance = -1
    for col in profile_cols:
        if results['ctr_by_feature'][col]:
            group_ctrs = [r['ctr'] for r in results['ctr_by_feature'][col]]
            variance = np.var(group_ctrs) if len(group_ctrs) > 1 else 0
            if variance > max_variance:
                max_variance = variance
                max_variance_col = col
    
    if max_variance_col:
        logger.info(f"  - Highest CTR variance: {max_variance_col} (var={max_variance:.6f})")
        top_group = results['ctr_by_feature'][max_variance_col][0]
        logger.info(f"    * Top value: {top_group['value']} (CTR={top_group['ctr']:.4f}, impr={top_group['impressions']})")
    
    return results


def compute_ctr_by_top_values(input_csv: str, chunksize: int,
                              top_values: int, min_support: int,
                              out_root: str, sample_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute CTR for top-K values with enhanced filtering and metrics.
    
    Improvements over compute_ctr_by_group():
    1. Uses pre-computed topK definitions from Stage 3 (topk_{col}.csv)
    2. Filters by min_support to remove small-sample artifacts ("样本少虚高" risk)
    3. Computes lift_vs_global (CTR ratio compared to baseline)
    4. More precise z-score with continuity correction
    5. Warns about unreliable estimates (CI width > CTR value)
    
    Saves files: {out_root}/eda/label/ctr_by_{col}_top.csv
    Format: CSV with columns [
        'value', 'impressions', 'clicks', 'ctr', 
        'ci_lower', 'ci_upper', 'lift_vs_global', 'z_score',
        'is_significant', 'sample_size_risk'
    ]
    
    Key metrics:
    - Wilson CI: Precise binomial confidence interval (95%)
    - Lift: ctr / global_ctr (>1 = higher than avg, <1 = lower)
    - Z-score: (ctr - global_ctr) / SE with continuity correction
    - Sample Size Risk: Flag when CI_width > CTR (unreliable estimate)
    
    Risk Warning ("样本少虚高"):
    When impressions < min_support or CI_width > CTR value:
    - CTR estimate is unreliable due to small sample
    - Confidence interval is wider than point estimate
    - Recommendation: Increase min_support or use regularization
    
    Args:
        input_csv: Path to input CSV file
        chunksize: Chunk size for streaming
        top_values: Number of top values per feature to analyze (e.g., 200)
        min_support: Minimum impressions to include in output (filter < min_support)
        out_root: Output root directory
    
    Returns:
        Dict with keys: 'total_rows', 'global_ctr', 'features_analyzed',
                       'min_support', 'ctr_by_feature', 'high_risk_features'
    """
    output_dirs = create_output_dirs(out_root)
    logger = logging.getLogger("AVAZU_EDA")
    
    logger.info(f"[CTR_BY_TOP_VALUES] Starting enhanced CTR analysis...")
    logger.info(f"  - Top values per feature: {top_values}")
    logger.info(f"  - Min support filter: {min_support} impressions")
    
    topk_dir = Path(out_root) / 'eda' / 'topk'
    
    # Step 1: Read schema and identify features
    df_sample = pd.read_csv(input_csv, nrows=1000)
    columns = df_sample.columns.tolist()
    skip_cols = {'id', 'click', 'ID', 'Click', 'hour'}
    profile_cols = [col for col in columns if col not in skip_cols]
    
    logger.info(f"[CTR_BY_TOP_VALUES] Analyzing {len(profile_cols)} features")
    
    # Step 2: Read topK definitions for each column
    topk_values_by_col = {}
    for col in profile_cols:
        topk_file = topk_dir / f'topk_{col}.csv'
        if topk_file.exists():
            try:
                df_topk = pd.read_csv(topk_file)
                topk_values_by_col[col] = df_topk['value'].astype(str).head(top_values).tolist()
            except Exception as e:
                logger.warning(f"[CTR_BY_TOP_VALUES] Failed to read {topk_file}: {e}")
                topk_values_by_col[col] = []
        else:
            logger.warning(f"[CTR_BY_TOP_VALUES] Missing {topk_file}")
            topk_values_by_col[col] = []
    
    # Step 3: Stream through data and accumulate statistics (Pass 1)
    logger.info(f"[CTR_BY_TOP_VALUES] Pass 1: Accumulating statistics...")
    
    feature_stats = {col: {} for col in profile_cols}  # col -> {value: {impr, clicks}}
    total_rows = 0
    total_clicks = 0
    chunk_idx = 0
    
    try:
        for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str):
            chunk_idx += 1
            total_rows += len(chunk)
            
            # Count global clicks
            total_clicks += (chunk['click'] == '1').sum()
            
            # Accumulate by feature value (only for topK values)
            for col in profile_cols:
                click_col = (chunk['click'] == '1').astype(int)
                target_values = set(topk_values_by_col[col])  # Only analyze topK
                
                for val in chunk[col].unique():
                    # Skip if not in top-K
                    if val not in target_values:
                        continue
                    
                    mask = chunk[col].astype(str) == val
                    n_impr = mask.sum()
                    n_clicks = click_col[mask].sum()
                    
                    if val not in feature_stats[col]:
                        feature_stats[col][val] = {'impr': 0, 'clicks': 0}
                    
                    feature_stats[col][val]['impr'] += n_impr
                    feature_stats[col][val]['clicks'] += n_clicks
            
            if chunk_idx % 5 == 0:
                logger.debug(f"[CTR_BY_TOP_VALUES] Processed {chunk_idx} chunks ({total_rows} rows)...")
    
    except Exception as e:
        logger.error(f"[CTR_BY_TOP_VALUES] Error reading CSV: {e}")
        raise
    
    logger.info(f"[CTR_BY_TOP_VALUES] Processed {total_rows} rows, {total_clicks} total clicks")
    
    # Compute global CTR
    global_ctr = total_clicks / total_rows if total_rows > 0 else 0.0
    logger.info(f"[CTR_BY_TOP_VALUES] Global CTR: {global_ctr:.6f}")
    
    # Step 4: Compute enhanced metrics for each value
    results = {
        'total_rows': total_rows,
        'total_clicks': total_clicks,
        'global_ctr': global_ctr,
        'features_analyzed': len(profile_cols),
        'min_support': min_support,
        'ctr_by_feature': {},
        'high_risk_features': []
    }
    
    for col in profile_cols:
        group_rows = []
        filtered_count = 0
        high_risk_count = 0
        
        # Get values in order from topk file
        target_values = topk_values_by_col[col]
        
        for val in target_values:
            if val not in feature_stats[col]:
                continue
            
            stats = feature_stats[col][val]
            impr = stats['impr']
            clicks = stats['clicks']
            
            # Filter by min_support
            if impr < min_support:
                filtered_count += 1
                continue
            
            ctr = clicks / impr if impr > 0 else 0.0
            
            # Wilson confidence interval (95%)
            ci_lower, ci_upper = wilson_ci(clicks, impr, z=1.96)
            ci_width = ci_upper - ci_lower
            
            # Lift vs global CTR
            if global_ctr > 0:
                lift = ctr / global_ctr
            else:
                lift = 0.0
            
            # Z-score with continuity correction
            if impr > 0 and global_ctr > 0 and global_ctr < 1:
                # Standard error for binomial proportion
                se = np.sqrt(global_ctr * (1 - global_ctr) / impr)
                
                # Continuity correction: adjust observed proportion
                p_corrected = (clicks + 0.5) / (impr + 1)
                zscore = (p_corrected - global_ctr) / se if se > 0 else 0.0
            else:
                zscore = 0.0
            
            # Significance test: |z| > 1.96 for p < 0.05 (two-tailed)
            is_significant = abs(zscore) > 1.96
            
            # Sample size risk: CI width > CTR value (unreliable estimate)
            # This indicates confidence interval is wider than the point estimate
            sample_size_risk = ci_width > ctr if ctr > 0 else ci_width > 0.05
            
            if sample_size_risk:
                high_risk_count += 1
            
            group_rows.append({
                'value': val,
                'impressions': impr,
                'clicks': clicks,
                'ctr': f"{ctr:.6f}",
                'ci_lower': f"{ci_lower:.6f}",
                'ci_upper': f"{ci_upper:.6f}",
                'lift_vs_global': f"{lift:.4f}",
                'z_score': f"{zscore:.4f}",
                'is_significant': is_significant,
                'sample_size_risk': sample_size_risk
            })
        
        results['ctr_by_feature'][col] = group_rows
        
        # Save to CSV
        output_file = output_dirs['label'] / f'ctr_by_{col}_top.csv'
        try:
            df_group = pd.DataFrame(group_rows)
            if len(df_group) > 0:
                df_group.to_csv(output_file, index=False)
                logger.debug(f"[CTR_BY_TOP_VALUES] Saved {col} ({len(group_rows)} values, {filtered_count} filtered)")
        except Exception as e:
            logger.warning(f"[CTR_BY_TOP_VALUES] Failed to save ctr_by_{col}_top.csv: {e}")
        
        # Track high-risk features
        if len(group_rows) > 0:
            risk_ratio = high_risk_count / len(group_rows)
            if risk_ratio > 0.3:  # >30% are risky
                results['high_risk_features'].append({
                    'column': col,
                    'high_risk_count': high_risk_count,
                    'total_count': len(group_rows),
                    'risk_ratio': risk_ratio
                })
    
    # Summary statistics
    logger.info(f"[CTR_BY_TOP_VALUES] Complete")
    logger.info(f"  - Features analyzed: {len(profile_cols)}")
    logger.info(f"  - Top values per feature: {top_values}")
    logger.info(f"  - Min support filter: {min_support} impressions")
    
    # Show statistics
    total_values = sum(len(v) for v in results['ctr_by_feature'].values())
    total_filtered = sum(len(topk_values_by_col[col]) - len(results['ctr_by_feature'][col]) 
                         for col in profile_cols)
    logger.info(f"  - Total values retained: {total_values}")
    logger.info(f"  - Total values filtered out: {total_filtered}")
    
    # High-risk features warning
    if results['high_risk_features']:
        logger.warning(f"[CTR_BY_TOP_VALUES] High-risk features (>30% with sample size risk):")
        for feature_info in results['high_risk_features'][:3]:
            col = feature_info['column']
            risky = feature_info['high_risk_count']
            total = feature_info['total_count']
            logger.warning(f"  - {col}: {risky}/{total} values ({risky/total*100:.1f}%)")
        logger.warning(f"  WARNING: These features have unreliable CTR estimates due to small sample sizes.")
        logger.warning(f"  RECOMMENDATION: Increase min_support or use regularization in model training.")
    
    return results


def detect_leakage_signals(csv_path: str, chunksize: int, 
                           output_dirs: Dict[str, Path]) -> Dict[str, List[str]]:
    """
    Detect strong identifier features (high cardinality + high correlation with CTR).
    
    Heuristics:
    - Cardinality > 10M or cardinality/total_rows > 0.8
    - High CTR variance across values (gini > 0.7 after grouping)
    - Feature name contains: 'ip', 'id', 'device_id', 'fingerprint', etc.
    
    Saves file: output_dirs['leakage']/leakage_signals.csv
    Format: CSV with columns ['feature', 'risk_type', 'signal', 'severity']
    
    Args:
        csv_path: Path to CSV file
        chunksize: Chunk size for streaming
        output_dirs: Dict of output directories
    
    Returns:
        Dict with keys: 'strong_identifiers', 'high_variance_features', 'risky_features'
    """
    ...


# ============================================================================
# Pass 2: Temporal Splitting, OOV Detection, Drift Analysis
# ============================================================================

def temporal_split_info(csv_path: str, chunksize: int, 
                       train_days: int, test_days: int) -> Tuple[str, str, str, str]:
    """
    Determine train/test temporal boundaries based on data.
    
    Logic: Assume hours are in ascending order; extract min/max datetime.
    Train: last (train_days) days; Test: following (test_days) days.
    
    Args:
        csv_path: Path to CSV file
        chunksize: Chunk size for streaming
        train_days: Number of days for training split
        test_days: Number of days for test split
    
    Returns:
        Tuple of (train_start: str, train_end: str, test_start: str, test_end: str)
        where each is date string (YYYYMMDD format)
    """
    ...


def compute_oov_rate_time_split(input_csv: str, chunksize: int, 
                                 train_days: int, test_days: int, 
                                 out_root: str,
                                 max_vocab_size: int = 100000,
                                 sample_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute OOV (out-of-vocabulary) rates using time-based train/test split.
    
    Logic:
    1. Read time_ctr_day.csv to get ordered day list
    2. Split: train = first train_days, test = next test_days
    3. Pass 1: Collect vocab (unique values) from train period
    4. Pass 2: Count OOV in test period (values not in train vocab)
    5. Use hash-set with truncation to manage memory
    
    Saves file: {out_root}/eda/split/oov_rate_train_test.csv
    Columns: [col, test_oov_rate, test_oov_count, test_total_count, truncated, notes]
    
    Args:
        input_csv: Path to input CSV file
        chunksize: Chunk size for streaming
        train_days: Number of days for training
        test_days: Number of days for testing
        out_root: Output root directory
        max_vocab_size: Max unique values to track in memory (default: 100000)
    
    Returns:
        Dict with keys: 'train_period', 'test_period', 'oov_stats', 'split_summary'
    """
    output_dirs = create_output_dirs(out_root)
    logger = logging.getLogger("AVAZU_EDA")
    
    logger.info(f"[OOV_TIME_SPLIT] Starting time-split OOV analysis...")
    logger.info(f"  - Train days: {train_days}, Test days: {test_days}")
    
    # Step 1: Read time_ctr_day.csv to get day boundaries
    time_ctr_day_file = Path(out_root) / 'eda' / 'time' / 'time_ctr_day.csv'
    if not time_ctr_day_file.exists():
        logger.error(f"[OOV_TIME_SPLIT] time_ctr_day.csv not found: {time_ctr_day_file}")
        raise FileNotFoundError(f"Missing {time_ctr_day_file}. Run Stage 4 first.")
    
    try:
        day_df = pd.read_csv(time_ctr_day_file)
        all_days = day_df['day'].astype(str).tolist()
    except Exception as e:
        logger.error(f"[OOV_TIME_SPLIT] Failed to read time_ctr_day.csv: {e}")
        raise
    
    logger.info(f"  - Total days in data: {len(all_days)}")
    
    # Determine train/test boundaries
    if len(all_days) < train_days + test_days:
        logger.warning(f"  - Not enough days. Data has {len(all_days)}, need {train_days + test_days}")
        # Adjust: use what we have
        train_size = max(1, len(all_days) - test_days)
    else:
        train_size = train_days
    
    test_size = test_days
    
    train_days_list = all_days[:train_size]
    test_days_list = all_days[train_size:train_size + test_size]
    
    train_period = (train_days_list[0], train_days_list[-1]) if train_days_list else (None, None)
    test_period = (test_days_list[0], test_days_list[-1]) if test_days_list else (None, None)
    
    logger.info(f"  - Train period: {train_period[0]} ~ {train_period[1]} ({len(train_days_list)} days)")
    logger.info(f"  - Test period: {test_period[0]} ~ {test_period[1]} ({len(test_days_list)} days)")
    
    # Step 2: Read schema
    df_sample = pd.read_csv(input_csv, nrows=1000)
    columns = df_sample.columns.tolist()
    skip_cols = {'id', 'click', 'ID', 'Click', 'hour'}
    profile_cols = [col for col in columns if col not in skip_cols]
    
    logger.info(f"  - Analyzing {len(profile_cols)} features")
    
    # Step 3: Pass 1 - Collect train vocabulary
    logger.info(f"[OOV_TIME_SPLIT] Pass 1: Building train vocabulary...")
    
    train_vocab = {col: set() for col in profile_cols}  # col -> set of values
    train_vocab_truncated = {col: False for col in profile_cols}  # Track if truncated
    train_vocab_size = {col: 0 for col in profile_cols}  # Actual count if truncated
    
    train_days_set = set(train_days_list)
    
    # Limit rows for dry-run if sample_rows is specified
    nrows_limit = sample_rows if sample_rows is not None else None
    if nrows_limit:
        logger.info(f"  - [DRY-RUN] Limiting to first {nrows_limit} rows for quick validation")
    
    try:
        for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str, nrows=nrows_limit):
            # Filter to train period only
            if 'hour' not in chunk.columns:
                logger.warning("[OOV_TIME_SPLIT] 'hour' column not found")
                continue
            
            # Extract day from hour (YYYYMMDDHH -> YYYYMMDD)
            chunk_days = chunk['hour'].str[:8]
            train_mask = chunk_days.isin(train_days_set)
            train_chunk = chunk[train_mask]
            
            # Accumulate vocabulary
            for col in profile_cols:
                for val in train_chunk[col].dropna().unique():
                    val_str = str(val)
                    
                    # Only add if under size limit
                    if len(train_vocab[col]) < max_vocab_size:
                        train_vocab[col].add(val_str)
                    else:
                        # Mark as truncated and count total
                        if not train_vocab_truncated[col]:
                            train_vocab_truncated[col] = True
                        train_vocab_size[col] += 1
    
    except Exception as e:
        logger.error(f"[OOV_TIME_SPLIT] Error in Pass 1: {e}")
        raise
    
    # Log train vocab stats
    logger.info(f"[OOV_TIME_SPLIT] Train vocabulary sizes:")
    for col in sorted(profile_cols)[:5]:
        truncated_mark = " [TRUNCATED]" if train_vocab_truncated[col] else ""
        logger.info(f"  - {col}: {len(train_vocab[col])}{truncated_mark}")
    
    # Step 4: Pass 2 - Compute OOV in test period
    logger.info(f"[OOV_TIME_SPLIT] Pass 2: Computing test OOV rates...")
    
    test_oov_stats = {}  # col -> {'oov_count': n, 'total_count': m, 'truncated': bool}
    
    test_days_set = set(test_days_list)
    
    try:
        for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str):
            # Filter to test period only
            if 'hour' not in chunk.columns:
                continue
            
            chunk_days = chunk['hour'].str[:8]
            test_mask = chunk_days.isin(test_days_set)
            test_chunk = chunk[test_mask]
            
            # Count OOV
            for col in profile_cols:
                if col not in test_oov_stats:
                    test_oov_stats[col] = {'oov_count': 0, 'total_count': 0, 'truncated': train_vocab_truncated[col]}
                
                col_values = test_chunk[col].dropna()
                test_oov_stats[col]['total_count'] += len(col_values)
                
                # Count OOV (not in train vocab)
                for val in col_values:
                    val_str = str(val)
                    if val_str not in train_vocab[col]:
                        test_oov_stats[col]['oov_count'] += 1
    
    except Exception as e:
        logger.error(f"[OOV_TIME_SPLIT] Error in Pass 2: {e}")
        raise
    
    # Step 5: Compute rates and create output
    logger.info(f"[OOV_TIME_SPLIT] Computing OOV rates...")
    
    output_rows = []
    for col in profile_cols:
        stats = test_oov_stats.get(col, {'oov_count': 0, 'total_count': 0, 'truncated': False})
        total = stats['total_count']
        oov_count = stats['oov_count']
        oov_rate = oov_count / total if total > 0 else 0.0
        truncated = stats['truncated']
        
        # Notes
        notes = []
        if truncated:
            notes.append(f"Train vocab truncated at {max_vocab_size}")
        if oov_rate > 0.1:
            notes.append(f"High OOV ({oov_rate:.1%})")
        if total == 0:
            notes.append("No test data")
        
        output_rows.append({
            'column': col,
            'test_oov_rate': oov_rate,
            'test_oov_count': oov_count,
            'test_total_count': total,
            'truncated': truncated,
            'notes': '; '.join(notes) if notes else ''
        })
    
    # Write to CSV
    output_file = output_dirs['split'] / 'oov_rate_train_test.csv'
    try:
        df_output = pd.DataFrame(output_rows)
        df_output.to_csv(output_file, index=False)
        logger.info(f"[OOV_TIME_SPLIT] Saved OOV report to {output_file}")
    except Exception as e:
        logger.error(f"[OOV_TIME_SPLIT] Failed to save output: {e}")
        raise
    
    # Summary
    high_oov_cols = [r for r in output_rows if r['test_oov_rate'] > 0.05]
    truncated_cols = [r for r in output_rows if r['truncated']]
    
    logger.info(f"[OOV_TIME_SPLIT] ✓ Complete")
    logger.info(f"  - Columns analyzed: {len(profile_cols)}")
    logger.info(f"  - High OOV (>5%): {len(high_oov_cols)} columns")
    if high_oov_cols:
        logger.info(f"    * {high_oov_cols[0]['column']}: OOV={high_oov_cols[0]['test_oov_rate']:.2%}")
    logger.info(f"  - Vocab truncated: {len(truncated_cols)} columns")
    
    return {
        'train_period': train_period,
        'test_period': test_period,
        'train_days_count': len(train_days_list),
        'test_days_count': len(test_days_list),
        'oov_stats': output_rows,
        'high_oov_cols': high_oov_cols,
        'truncated_cols': truncated_cols,
        'output_file': str(output_file)
    }


def compute_psi_train_test(input_csv: str, chunksize: int, 
                            out_root: str,
                            train_days: int, test_days: int,
                            sample_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute Population Stability Index (PSI) for train→test distribution shift.
    
    Logic:
    1. Read time_ctr_day.csv to get day boundaries (same as Stage 6)
    2. For each feature, read its topk_{col}.csv to get bin definitions
    3. Pass 1: Compute frequency of each bin in train period
    4. Pass 2: Compute frequency of each bin in test period
    5. Calculate PSI = Σ (p_test - p_train) × ln(p_test / p_train)
    6. Output: Summary CSV (col, psi) + detailed CSVs per feature (bin, p_train, p_test, psi_term)
    
    PSI Interpretation:
    - PSI < 0.1: Negligible change (safe to deploy model)
    - 0.1 ≤ PSI < 0.25: Small change (monitor but usually OK)
    - PSI ≥ 0.25: Significant change (retraining recommended)
    
    Saves files:
    - out_root/eda/drift/psi_train_test_summary.csv
    - out_root/eda/drift/psi_train_test_{col}.csv (detailed per-bin analysis)
    
    Args:
        input_csv: Path to input data file
        chunksize: Chunk size for streaming
        out_root: Output root directory
        train_days: Number of days for training period
        test_days: Number of days for test period
    
    Returns:
        Dict with keys: 'train_period', 'test_period', 'psi_summary', 'high_drift_cols'
    """
    output_dirs = create_output_dirs(out_root)
    logger = logging.getLogger("AVAZU_EDA")
    
    logger.info(f"[PSI_DRIFT] Starting PSI drift analysis...")
    logger.info(f"  - Train days: {train_days}, Test days: {test_days}")
    
    # Step 1: Read time boundaries (same as Stage 6)
    time_ctr_day_file = Path(out_root) / 'eda' / 'time' / 'time_ctr_day.csv'
    if not time_ctr_day_file.exists():
        logger.error(f"[PSI_DRIFT] time_ctr_day.csv not found. Run Stage 4 first.")
        raise FileNotFoundError(f"Missing {time_ctr_day_file}")
    
    try:
        day_df = pd.read_csv(time_ctr_day_file)
        all_days = day_df['day'].astype(str).tolist()
    except Exception as e:
        logger.error(f"[PSI_DRIFT] Failed to read time_ctr_day.csv: {e}")
        raise
    
    # Determine train/test split
    if len(all_days) < train_days + test_days:
        train_size = max(1, len(all_days) - test_days)
    else:
        train_size = train_days
    
    test_size = test_days
    train_days_list = all_days[:train_size]
    test_days_list = all_days[train_size:train_size + test_size]
    
    train_period = (train_days_list[0], train_days_list[-1]) if train_days_list else (None, None)
    test_period = (test_days_list[0], test_days_list[-1]) if test_days_list else (None, None)
    
    logger.info(f"  - Train period: {train_period[0]} ~ {train_period[1]} ({len(train_days_list)} days)")
    logger.info(f"  - Test period: {test_period[0]} ~ {test_period[1]} ({len(test_days_list)} days)")
    
    # Step 2: Determine columns and load topK bins
    df_sample = pd.read_csv(input_csv, nrows=1000)
    columns = df_sample.columns.tolist()
    skip_cols = {'id', 'click', 'ID', 'Click', 'hour'}
    profile_cols = [col for col in columns if col not in skip_cols]
    
    logger.info(f"  - Analyzing {len(profile_cols)} features")
    
    # Load topK definitions for each column
    topk_dir = Path(out_root) / 'eda' / 'topk'

    column_bins = {}  # col -> list of topK values + "OTHER"
    
    for col in profile_cols:
        topk_file = topk_dir / f'topk_{col}.csv'
        if topk_file.exists():
            try:
                df_topk = pd.read_csv(topk_file)
                topk_values = df_topk['value'].astype(str).tolist()
                # Add "OTHER" for values not in topK
                column_bins[col] = topk_values + ["OTHER"]
            except Exception as e:
                logger.warning(f"[PSI_DRIFT] Failed to load topk for {col}: {e}")
                # Fallback: use all observed values
                column_bins[col] = []
        else:
            logger.warning(f"[PSI_DRIFT] topk_{col}.csv not found")
            column_bins[col] = []
    
    # Step 3: Pass 1 - Compute train frequencies
    logger.info(f"[PSI_DRIFT] Pass 1: Computing train frequencies...")
    
    train_frequencies = {col: {} for col in profile_cols}  # col -> {bin: count}
    train_total = {col: 0 for col in profile_cols}  # col -> total count
    train_days_set = set(train_days_list)
    
    # Limit rows for dry-run if sample_rows is specified
    nrows_limit = sample_rows if sample_rows is not None else None
    if nrows_limit:
        logger.info(f"  - [DRY-RUN] Limiting to first {nrows_limit} rows for quick validation")
    
    try:
        for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str, nrows=nrows_limit):
            if 'hour' not in chunk.columns:
                continue
            
            chunk_days = chunk['hour'].str[:8]
            train_mask = chunk_days.isin(train_days_set)
            train_chunk = chunk[train_mask]
            
            for col in profile_cols:
                values = train_chunk[col].dropna().astype(str)
                train_total[col] += len(values)
                
                topk_list = column_bins.get(col, [])
                topk_set = set(topk_list[:-1]) if topk_list else set()  # Exclude "OTHER"
                
                for val in values:
                    if topk_set and val in topk_set:
                        bin_name = val
                    elif topk_set:
                        bin_name = "OTHER"
                    else:
                        bin_name = val
                    
                    train_frequencies[col][bin_name] = train_frequencies[col].get(bin_name, 0) + 1
    
    except Exception as e:
        logger.error(f"[PSI_DRIFT] Error in Pass 1: {e}")
        raise
    
    # Step 4: Pass 2 - Compute test frequencies
    logger.info(f"[PSI_DRIFT] Pass 2: Computing test frequencies...")
    
    test_frequencies = {col: {} for col in profile_cols}
    test_total = {col: 0 for col in profile_cols}
    test_days_set = set(test_days_list)
    
    try:
        for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str):
            if 'hour' not in chunk.columns:
                continue
            
            chunk_days = chunk['hour'].str[:8]
            test_mask = chunk_days.isin(test_days_set)
            test_chunk = chunk[test_mask]
            
            for col in profile_cols:
                values = test_chunk[col].dropna().astype(str)
                test_total[col] += len(values)
                
                topk_list = column_bins.get(col, [])
                topk_set = set(topk_list[:-1]) if topk_list else set()
                
                for val in values:
                    if topk_set and val in topk_set:
                        bin_name = val
                    elif topk_set:
                        bin_name = "OTHER"
                    else:
                        bin_name = val
                    
                    test_frequencies[col][bin_name] = test_frequencies[col].get(bin_name, 0) + 1
    
    except Exception as e:
        logger.error(f"[PSI_DRIFT] Error in Pass 2: {e}")
        raise
    
    # Step 5: Compute PSI
    logger.info(f"[PSI_DRIFT] Computing PSI values...")
    
    smoothing_factor = 1e-10  # Avoid log(0)
    summary_rows = []
    high_drift_cols = []
    
    for col in profile_cols:
        train_tot = train_total[col]
        test_tot = test_total[col]
        
        if train_tot == 0 or test_tot == 0:
            logger.warning(f"[PSI_DRIFT] {col}: No data in train or test")
            continue
        
        # Get all bins (union of both periods)
        all_bins = set(train_frequencies[col].keys()) | set(test_frequencies[col].keys())
        all_bins = sorted(list(all_bins))
        
        # Compute PSI
        psi = 0.0
        detail_rows = []
        
        for bin_name in all_bins:
            # Get frequencies
            train_count = train_frequencies[col].get(bin_name, 0)
            test_count = test_frequencies[col].get(bin_name, 0)
            
            # Convert to proportions
            p_train = (train_count + smoothing_factor) / (train_tot + smoothing_factor)
            p_test = (test_count + smoothing_factor) / (test_tot + smoothing_factor)
            
            # PSI contribution
            psi_term = (p_test - p_train) * np.log(p_test / p_train)
            psi += psi_term
            
            detail_rows.append({
                'bin': str(bin_name),
                'count_train': int(train_count),
                'count_test': int(test_count),
                'p_train': float(p_train),
                'p_test': float(p_test),
                'psi_term': float(psi_term)
            })
        
        # Save detailed analysis
        detail_file = output_dirs['drift'] / f'psi_train_test_{col}.csv'
        try:
            df_detail = pd.DataFrame(detail_rows)
            df_detail.to_csv(detail_file, index=False)
        except Exception as e:
            logger.warning(f"[PSI_DRIFT] Failed to save detail for {col}: {e}")
        
        # Determine drift level
        if psi >= 0.25:
            drift_level = "HIGH"
            high_drift_cols.append({'column': col, 'psi': psi, 'level': 'HIGH'})
        elif psi >= 0.1:
            drift_level = "MODERATE"
            high_drift_cols.append({'column': col, 'psi': psi, 'level': 'MODERATE'})
        else:
            drift_level = "LOW"
        
        summary_rows.append({
            'column': col,
            'psi': psi,
            'drift_level': drift_level,
            'train_unique_bins': len(train_frequencies[col]),
            'test_unique_bins': len(test_frequencies[col])
        })
    
    # Step 6: Save summary
    summary_file = output_dirs['drift'] / 'psi_train_test_summary.csv'
    try:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(summary_file, index=False)
        logger.info(f"[PSI_DRIFT] Saved PSI summary to {summary_file}")
    except Exception as e:
        logger.error(f"[PSI_DRIFT] Failed to save summary: {e}")
        raise
    
    # Summary statistics
    logger.info(f"[PSI_DRIFT] ✓ Complete")
    logger.info(f"  - Columns analyzed: {len(summary_rows)}")
    
    high_drift = [r for r in summary_rows if r['drift_level'] == 'HIGH']
    moderate_drift = [r for r in summary_rows if r['drift_level'] == 'MODERATE']
    
    logger.info(f"  - High drift (PSI ≥ 0.25): {len(high_drift)} columns")
    if high_drift:
        logger.info(f"    * {high_drift[0]['column']}: PSI={high_drift[0]['psi']:.4f}")
    
    logger.info(f"  - Moderate drift (0.1 ≤ PSI < 0.25): {len(moderate_drift)} columns")
    
    return {
        'train_period': train_period,
        'test_period': test_period,
        'train_days_count': len(train_days_list),
        'test_days_count': len(test_days_list),
        'psi_summary': summary_rows,
        'high_drift_cols': high_drift,
        'output_file': str(summary_file)
    }


def compute_oov_rate(csv_path: str, chunksize: int, 
                     output_dirs: Dict[str, Path],
                     train_start: str, train_end: str, 
                     test_start: str, test_end: str) -> Dict[str, float]:
    """
    Compute out-of-vocabulary (OOV) rates for train→test transition.
    
    Logic: Collect vocabulary (unique values) from training period;
    measure % of test values not seen in train, per feature.
    
    Saves file: output_dirs['split']/oov_rate_train_test.csv
    Format: CSV with columns ['feature', 'vocab_size_train', 'oov_rate', 'oov_count']
    
    Args:
        csv_path: Path to CSV file
        chunksize: Chunk size for streaming
        output_dirs: Dict of output directories
        train_start/end: Training period boundaries (YYYYMMDD)
        test_start/end: Test period boundaries (YYYYMMDD)
    
    Returns:
        Dict {col_name: oov_rate_float}
    """
    ...


def compute_psi(csv_path: str, chunksize: int, 
                output_dirs: Dict[str, Path],
                train_start: str, train_end: str, 
                test_start: str, test_end: str) -> Dict[str, float]:
    """
    Compute population stability index (PSI) for train→test shift detection.
    
    Formula: PSI = Σ (pct_test - pct_train) * ln(pct_test / pct_train)
    Threshold: PSI > 0.1 indicates meaningful drift; > 0.25 is high drift.
    
    Saves files:
    - output_dirs['drift']/psi_train_test_summary.csv
    - output_dirs['drift']/psi_train_test_{col}.csv (detailed bins)
    
    Args:
        csv_path: Path to CSV file
        chunksize: Chunk size for streaming
        output_dirs: Dict of output directories
        train_start/end: Training period boundaries
        test_start/end: Test period boundaries
    
    Returns:
        Dict {col_name: psi_value_float}
    """
    ...


# ============================================================================
# Hash Collision Analysis (No data reading required)
# ============================================================================

def compute_hash_collision_sim(input_csv: str, chunksize: int, 
                               sample_rows: int, out_root: str) -> Dict[str, Any]:
    """
    Compute hash collision rates by analyzing actual token distribution.
    
    Logic:
    1. Sample up to sample_rows from input data (前N行 approach for simplicity)
    2. Construct tokens: "{col}={value}" for all non-id, non-click columns
    3. For candidate bucket sizes [2^18, 2^19, 2^20, 2^21, 2^22]:
       - Hash each unique token to bucket index
       - Count distinct buckets occupied
       - Collision rate = 1 - (distinct_buckets / distinct_tokens)
    4. Track token set truncation if memory limit reached
    
    Token Truncation:
    - If unique tokens exceed max_unique_tokens (default: 10M), truncate and flag
    - This is a memory safety feature for very large datasets
    
    Saves file: {out_root}/eda/hash/hash_collision_sim.csv
    Columns: ['n_buckets', 'distinct_tokens', 'distinct_buckets', 'collision_rate', 'truncated', 'notes']
    
    Args:
        input_csv: Path to input CSV file
        chunksize: Chunk size for streaming (not strictly used in sampling, but kept for consistency)
        sample_rows: Number of rows to sample for analysis
        out_root: Output root directory
    
    Returns:
        Dict with keys: 'total_tokens', 'unique_tokens', 'truncated',
                       'collision_results': [
                           {'n_buckets': int, 'distinct_tokens': int, 'distinct_buckets': int,
                            'collision_rate': float, 'truncated': bool, 'notes': str}
                       ]
    """
    output_dirs = create_output_dirs(out_root)
    logger = logging.getLogger("AVAZU_EDA")
    
    logger.info(f"[HASH_COLLISION_SIM] Starting hash collision simulation...")
    logger.info(f"  - Sample rows: {sample_rows}")
    logger.info(f"  - Bucket sizes: 2^18 to 2^22 ({2**18} to {2**22})")
    
    # Step 1: Read schema
    df_sample = pd.read_csv(input_csv, nrows=1000)
    columns = df_sample.columns.tolist()
    
    # Exclude id and click columns
    skip_cols = {'id', 'click', 'ID', 'Click', 'hour'}
    profile_cols = [col for col in columns if col not in skip_cols]
    
    logger.info(f"[HASH_COLLISION_SIM] Token columns: {len(profile_cols)}")
    
    # Step 2: Sample data and collect tokens
    logger.info(f"[HASH_COLLISION_SIM] Sampling {sample_rows} rows...")
    
    tokens = set()  # Unique tokens
    total_token_count = 0
    truncated = False
    max_unique_tokens = 10_000_000  # Memory limit: 10M unique tokens
    
    try:
        rows_read = 0
        for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str):
            # Process chunk, but only up to sample_rows
            for idx, row in chunk.iterrows():
                if rows_read >= sample_rows:
                    break
                
                # Construct tokens for this row
                for col in profile_cols:
                    value = row[col]
                    if pd.isna(value) or value == '' or value == 'nan':
                        continue
                    
                    token = f"{col}={value}"
                    tokens.add(token)
                    total_token_count += 1
                    
                    # Check memory limit
                    if len(tokens) > max_unique_tokens:
                        truncated = True
                        logger.warning(f"[HASH_COLLISION_SIM] Token set exceeded {max_unique_tokens}. Truncating.")
                        break
                
                rows_read += 1
                
                if rows_read >= sample_rows or (truncated and len(tokens) > max_unique_tokens):
                    break
            
            if rows_read >= sample_rows or (truncated and len(tokens) > max_unique_tokens):
                break
    
    except Exception as e:
        logger.error(f"[HASH_COLLISION_SIM] Failed to sample data: {e}")
        raise
    
    logger.info(f"[HASH_COLLISION_SIM] Sampled {rows_read} rows")
    logger.info(f"[HASH_COLLISION_SIM] Total tokens: {total_token_count}")
    logger.info(f"[HASH_COLLISION_SIM] Unique tokens: {len(tokens)}")
    if truncated:
        logger.warning(f"[HASH_COLLISION_SIM] Token set truncated at {len(tokens)} unique tokens")
    
    # Step 3: Compute collision rates for different bucket sizes
    logger.info(f"[HASH_COLLISION_SIM] Computing collision rates...")
    
    bucket_sizes = [2**18, 2**19, 2**20, 2**21, 2**22]  # 256K to 4M
    collision_results = []
    
    for n_buckets in bucket_sizes:
        # Hash each token to bucket
        bucket_set = set()
        
        for token in tokens:
            # Use hashlib.md5 for hashing (consistent with hash_value function)
            hash_obj = hashlib.md5(token.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            bucket_idx = hash_int % n_buckets
            bucket_set.add(bucket_idx)
        
        # Compute collision rate
        distinct_tokens = len(tokens)
        distinct_buckets = len(bucket_set)
        collision_rate = 1.0 - (distinct_buckets / distinct_tokens) if distinct_tokens > 0 else 0.0
        
        # Determine note
        if collision_rate < 0.01:
            note = "Very low collision (excellent)"
        elif collision_rate < 0.05:
            note = "Low collision (good)"
        elif collision_rate < 0.10:
            note = "Moderate collision (acceptable)"
        elif collision_rate < 0.20:
            note = "High collision (suboptimal)"
        else:
            note = "Very high collision (increase bucket size)"
        
        result = {
            'n_buckets': n_buckets,
            'distinct_tokens': distinct_tokens,
            'distinct_buckets': distinct_buckets,
            'collision_rate': float(collision_rate),
            'truncated': truncated,
            'notes': note
        }
        
        collision_results.append(result)
        
        logger.info(f"  - 2^{int(np.log2(n_buckets))}: {collision_rate:.4f} collision "
                   f"({distinct_buckets}/{distinct_tokens} buckets)")
    
    # Step 4: Save results to CSV
    hash_dir = output_dirs['hash']
    output_file = hash_dir / 'hash_collision_sim.csv'
    
    try:
        collision_df = pd.DataFrame(collision_results)
        collision_df.to_csv(output_file, index=False)
        logger.info(f"[HASH_COLLISION_SIM] ✓ Saved to {output_file}")
    except Exception as e:
        logger.error(f"[HASH_COLLISION_SIM] Failed to save results: {e}")
        raise
    
    # Summary
    logger.info(f"[HASH_COLLISION_SIM] ✓ Complete")
    logger.info(f"  - Unique tokens analyzed: {len(tokens)}")
    logger.info(f"  - Truncated: {truncated}")
    logger.info(f"  - Recommended bucket size: {collision_results[2]['n_buckets']} (2^20, collision={collision_results[2]['collision_rate']:.4f})")
    
    return {
        'total_tokens': total_token_count,
        'unique_tokens': len(tokens),
        'truncated': truncated,
        'collision_results': collision_results,
        'output_file': str(output_file),
        'rows_sampled': rows_read
    }


def simulate_hash_collision(output_dirs: Dict[str, Path], 
                           num_buckets: int = 10000, 
                           seeds: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Simulate hash collision rates for typical categorical features.
    
    [Legacy function - compute_hash_collision_sim() is the main implementation]
    
    Logic: Assume uniform cardinality for high-cardinality features;
    estimate collision via birthday paradox approximation.
    
    Saves file: output_dirs['hash']/hash_collision_sim.csv
    Format: CSV with columns ['cardinality', 'num_buckets', 'seed', 'collision_rate']
    
    Args:
        output_dirs: Dict of output directories
        num_buckets: Number of hash buckets (typical: 10K)
        seeds: List of random seeds for simulation (default: [42, 123, 456])
    
    Returns:
        Dict {cardinality: expected_collision_rate}
    """
    ...


# ============================================================================
# Interaction Analysis (Pass 2 with optional sampling)
# ============================================================================

def compute_pairwise_interactions(csv_path: str, chunksize: int, 
                                  output_dirs: Dict[str, Path],
                                  sample_rows: int = 100000,
                                  top_pairs: int = 20) -> Dict[str, Any]:
    """
    Compute mutual information (MI) for top feature pairs using sampling.
    
    Logic: Sample up to sample_rows; discretize numerical features into bins;
    compute pairwise MI; sort by strength and save top pairs.
    
    Note: This is a proxy for true MI; can use histogram-based approximation.
    
    Saves file: output_dirs['interactions']/pair_mi_topbins.csv
    Format: CSV with columns ['feature_a', 'feature_b', 'mi_score', 'is_significant']
    
    Args:
        csv_path: Path to CSV file
        chunksize: Chunk size for streaming (used for sampling)
        output_dirs: Dict of output directories
        sample_rows: Max rows to sample for MI computation
        top_pairs: Number of top pairs to save
    
    Returns:
        Dict with keys: 'top_pairs', 'strong_interactions', 'weak_interactions'
    """
    # Placeholder for more advanced MI pipeline (kept for backward compatibility)
    return {
        'top_pairs': [],
        'strong_interactions': [],
        'weak_interactions': []
    }


def compute_pairwise_mi_topbins(input_csv: str, chunksize: int, out_root: str, sample_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute mutual information (MI) for selected feature pairs using top-N bins + OTHER.

    Strategy:
    - Select candidate columns from `out_root/eda/columns_profile.csv` when available.
      Fallback: use first N profile columns.
    - For each column, load top bins from `out_root/eda/topk/topk_{col}.csv` when present;
      otherwise compute top bins from a sample of the input CSV.
    - For a limited set of column pairs (<= 30 pairs by default), stream through data
      (cap rows at internal `max_rows` for performance, e.g. 200k) and build a 2D
      contingency table of (binA, binB) where values not in top bins are mapped to 'OTHER'.
    - Compute discrete mutual information (in nats) from the contingency table:
        MI = sum_{i,j} p(i,j) * log( p(i,j) / (p(i)*p(j)) )
    - Save results to `{out_root}/eda/interactions/pair_mi_topbins.csv` with columns
      ['col_a', 'col_b', 'mi', 'support_pairs', 'notes']

    Returns:
        Dict with keys: 'pairs_analyzed', 'output_file', 'rows_sampled', 'pairs'
    """
    logger = logging.getLogger("AVAZU_EDA")
    output_dirs = create_output_dirs(out_root)
    inter_dir = output_dirs['interactions']
    inter_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[PAIRWISE_MI] Starting pairwise MI (top-bins) analysis...")

    # Parameters
    top_n = 30
    max_pairs = 30
    max_rows = 200_000  # internal cap to bound work
    
    # Override max_rows with sample_rows for dry-run
    if sample_rows is not None:
        max_rows = sample_rows
        logger.info(f"  - [DRY-RUN] Limiting to {max_rows} rows for quick validation")

    # Read sample to get columns
    try:
        df_sample = pd.read_csv(input_csv, nrows=1000)
    except Exception:
        df_sample = pd.DataFrame()

    columns = df_sample.columns.tolist() if not df_sample.empty else []
    skip_cols = {'id', 'click', 'ID', 'Click', 'hour'}
    profile_cols = [c for c in columns if c not in skip_cols]

    # Try to select candidate columns from columns_profile.csv
    profile_file = Path(out_root) / 'eda' / 'columns_profile.csv'
    selected_cols: List[str] = []
    if profile_file.exists():
        try:
            dfp = pd.read_csv(profile_file)
            # Ensure expected columns exist
            if 'nunique_approx' in dfp.columns and 'top1_ratio' in dfp.columns:
                # prefer high cardinality and low top1_ratio
                dfp['score'] = dfp['nunique_approx'].fillna(0).astype(float) - (dfp['top1_ratio'].fillna(0).astype(float) * 1000)
                # 'feature' column expected to hold column names; fallback to index
                if 'feature' in dfp.columns:
                    dfp_sorted = dfp.sort_values('score', ascending=False)
                    selected_cols = [r for r in dfp_sorted['feature'].tolist() if r in profile_cols]
                else:
                    selected_cols = [c for c in profile_cols]
            else:
                selected_cols = [c for c in profile_cols]
        except Exception:
            selected_cols = [c for c in profile_cols]
    else:
        selected_cols = [c for c in profile_cols]

    # Cap number of columns/pairs
    if len(selected_cols) > 12:
        selected_cols = selected_cols[:12]

    # Build top bins per selected column
    topk_dir = Path(out_root) / 'eda' / 'topk'
    top_bins: Dict[str, set] = {}
    for col in selected_cols:
        top_set = set()
        top_file = topk_dir / f"topk_{col}.csv"
        if top_file.exists():
            try:
                df_top = pd.read_csv(top_file, nrows=top_n)
                if 'value' in df_top.columns:
                    top_set.update(df_top['value'].astype(str).tolist()[:top_n])
            except Exception:
                top_set = set()
        if not top_set:
            # Fallback: sample some rows to estimate top values
            try:
                df_vals = pd.read_csv(input_csv, usecols=[col], nrows=50000, dtype=str)
                top_vals = df_vals[col].value_counts().index.astype(str).tolist()[:top_n]
                top_set.update(top_vals)
            except Exception:
                top_set = set()
        top_bins[col] = set(top_set)

    # Build list of column pairs (combinations limited)
    pairs = []
    for i in range(len(selected_cols)):
        for j in range(i + 1, len(selected_cols)):
            pairs.append((selected_cols[i], selected_cols[j]))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    results = []
    rows_read = 0

    # For each pair, stream through data and build contingency counts
    for col_a, col_b in pairs:
        logger.info(f"[PAIRWISE_MI] Analyzing pair: {col_a} × {col_b}")
        counts = Counter()
        rows_read_pair = 0
        try:
            for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str, nrows=max_rows):
                for _, row in chunk.iterrows():
                    a = row.get(col_a)
                    b = row.get(col_b)
                    if pd.isna(a) or pd.isna(b):
                        continue
                    a = str(a)
                    b = str(b)
                    a_bin = a if a in top_bins.get(col_a, set()) else 'OTHER'
                    b_bin = b if b in top_bins.get(col_b, set()) else 'OTHER'
                    counts[(a_bin, b_bin)] += 1
                    rows_read += 1
                    rows_read_pair += 1
                    if rows_read >= max_rows:
                        break
                if rows_read >= max_rows:
                    break
        except Exception as e:
            logger.exception(e)

        support = sum(counts.values())
        mi = 0.0
        if support > 0:
            # Compute marginals
            joint = {k: v / support for k, v in counts.items()}
            pa = {}
            pb = {}
            for (a_bin, b_bin), p in joint.items():
                pa[a_bin] = pa.get(a_bin, 0.0) + p
                pb[b_bin] = pb.get(b_bin, 0.0) + p
            # MI (nats)
            for (a_bin, b_bin), p_ij in joint.items():
                p_i = pa.get(a_bin, 0.0)
                p_j = pb.get(b_bin, 0.0)
                if p_ij > 0 and p_i > 0 and p_j > 0:
                    mi += p_ij * np.log(p_ij / (p_i * p_j))

        note = f"top_bins={min(top_n, len(top_bins.get(col_a,[])))}|{min(top_n, len(top_bins.get(col_b,[])))}, sample_rows={rows_read_pair}"
        results.append({
            'col_a': col_a,
            'col_b': col_b,
            'mi': float(mi),
            'support_pairs': int(support),
            'notes': note
        })

    # Save CSV
    out_file = inter_dir / 'pair_mi_topbins.csv'
    try:
        df_out = pd.DataFrame(results)
        df_out = df_out.sort_values('mi', ascending=False)
        df_out.to_csv(out_file, index=False)
        logger.info(f"[PAIRWISE_MI] Saved results to {out_file}")
    except Exception as e:
        logger.exception(e)

    logger.info(f"[PAIRWISE_MI] ✓ Complete. Pairs analyzed: {len(results)}; Rows scanned: {rows_read}")
    return {
        'pairs_analyzed': len(results),
        'output_file': str(out_file),
        'rows_sampled': rows_read,
        'pairs': results
    }


# ============================================================================
# Report Generation
# ============================================================================

def generate_featuremap_evidence(all_results: Dict[str, Any], 
                                 output_dirs: Dict[str, Path]) -> None:
    """
    Generate featuremap_spec.yml based on EDA evidence.
    
    Logic: 
    - Cardinality & sparsity → determine vocab size, hash bucket size
    - Distribution shape → determine bucket boundaries for numerical features
    - Leakage signals → flag features to drop or handle
    - OOV rates → set embedding strategies
    
    Saves file: output_dirs['featuremap']/featuremap_spec.yml
    Format: YAML with FeatureMap structure
    
    Args:
        all_results: Dict containing all EDA results from previous steps
        output_dirs: Dict of output directories
    """
    ...


def generate_model_structure_evidence(all_results: Dict[str, Any], 
                                      output_dirs: Dict[str, Path]) -> None:
    """
    Generate model_plan.yml with architecture recommendations.
    
    Logic:
    - Strong interactions (top 10 pairs MI) → recommend MultiTask or multi-head architecture
    - Weak interactions → recommend DeepFM or simple factorization
    - Temporal drift → recommend batch normalization or instance normalization
    - Leakage signals → recommend feature filtering
    
    Saves file: output_dirs['model_plan']/model_plan.yml
    Format: YAML with model architecture and training strategy
    
    Args:
        all_results: Dict containing all EDA results
        output_dirs: Dict of output directories
    """
    ...


def generate_featuremap_evidence_markdown(all_results: Dict[str, Any], 
                                          output_dirs: Dict[str, Path],
                                          eda_config: Dict[str, Any]) -> None:
    """
    Generate featuremap_evidence.md with decision chain documentation.
    
    Format: Decision → Metric → Artifact → Code
    Example:
        ### Decision: Use hash embedding for site_id
        **Evidence**:
        - Cardinality: 4.7M (from overview.json)
        - OOV rate (train→test): 0.8% (from oov_rate_train_test.csv)
        - Top-10 values cover 15% of traffic (from topk/topk_site_id.csv)
        
        **Rationale**: High cardinality + low OOV suggests hash embedding with 
        bucket_size=2^13=8192 is appropriate.
        
        **Code**: See featuremap_spec.yml line 42
    
    Saves file: output_dirs['reports']/featuremap_evidence.md
    
    Args:
        all_results: Dict containing all EDA results
        output_dirs: Dict of output directories
        eda_config: Original EDA configuration (arguments)
    """
    ...


def generate_model_structure_evidence_markdown(all_results: Dict[str, Any], 
                                               output_dirs: Dict[str, Path],
                                               eda_config: Dict[str, Any]) -> None:
    """
    Generate model_structure_evidence.md with architecture justification.
    
    Format: Similar to featuremap_evidence.md, but focused on model choice.
    Example:
        ### Decision: Use DeepFM architecture
        **Evidence**:
        - Top feature pairs MI (from pair_mi_topbins.csv):
          * site_id ↔ app_id: MI=0.23 (moderate)
          * C14 ↔ C15: MI=0.08 (weak)
        - No very strong interactions detected
        - Temporal CTR trend shows clear diurnal pattern
        
        **Rationale**: Moderate interaction strength suggests DeepFM 
        (implicit FM) is sufficient; no need for multi-task.
        
        **Code**: See model_plan.yml, trainer config
    
    Saves file: output_dirs['reports']/model_structure_evidence.md
    
    Args:
        all_results: Dict containing all EDA results
        output_dirs: Dict of output directories
        eda_config: Original EDA configuration
    """
    logger = logging.getLogger("AVAZU_EDA")
    reports_dir = output_dirs.get('reports', Path('./reports'))
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_md = reports_dir / 'model_structure_evidence.md'

    lines = []
    lines.append('# Model Structure Evidence')
    lines.append('')
    lines.append('This document summarizes evidence used to recommend model architectures (e.g., DeepFM, DCN).')
    lines.append('')

    # Include top MI pairs if available
    try:
        mi_file = Path(output_dirs.get('interactions')) / 'pair_mi_topbins.csv'
        if mi_file.exists():
            df_mi = pd.read_csv(mi_file)
            lines.append('## Pairwise Mutual Information (Top bins)')
            lines.append('Top interacting feature pairs (by MI):')
            lines.append('')
            top_n = min(10, len(df_mi))
            for _, r in df_mi.head(top_n).iterrows():
                lines.append(f"- `{r['col_a']}` × `{r['col_b']}`: MI={r['mi']:.6f}, support={int(r['support_pairs'])}")
            lines.append('')

            # Recommendation logic based on MI
            avg_mi = float(df_mi['mi'].mean()) if 'mi' in df_mi.columns and len(df_mi) > 0 else 0.0
            strong_count = int((df_mi['mi'] > 0.01).sum()) if 'mi' in df_mi.columns else 0
            lines.append('## Recommendation')
            if avg_mi > 0.005 or strong_count >= 3:
                lines.append('- Evidence: non-trivial pairwise interactions exist. Recommend using a model that explicitly models feature interactions such as **DeepFM** (FM component + DNN) or **DCN** (cross network).')
                lines.append('- Rationale: FM component captures low-order feature interactions efficiently; DNN/Cross layers capture higher-order interactions. Strong MI pairs suggest including cross/interaction modeling to improve CTR prediction.')
            else:
                lines.append('- Evidence: interactions are weak overall. A plain DNN or linear + DNN may suffice; expensive cross modules are optional.')
            lines.append('')
        else:
            lines.append('## Pairwise Mutual Information')
            lines.append('Pairwise MI results not found. Run `compute_pairwise_mi_topbins` to generate interaction evidence and rerun this report.')
            lines.append('')
    except Exception as e:
        logger.exception(e)
        lines.append('Could not load pairwise MI results due to an error when reading the CSV.')

    # Add a short summary section linking to FeatureMap
    lines.append('## Notes')
    lines.append('- Use MI evidence together with FeatureMap hash collision and cardinality results when choosing embedding sizes and hash buckets.')
    lines.append('')

    try:
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        logger.info(f"[MODEL_STRUCT_MD] Wrote model structure evidence to {out_md}")
    except Exception as e:
        logger.exception(e)


# ============================================================================
# Main Orchestration
# ============================================================================

def main(args: argparse.Namespace) -> None:
    """
    Main orchestration function: run all EDA stages.
    
    Stages:
    1. Setup: Parse args, setup logger, create output dirs
    2. Schema detection & basic stats (Pass 1)
    3. Top-K value extraction (Pass 1)
    4. Temporal analysis (Pass 1)
    5. Group-wise CTR analysis (Pass 1)
    6. Leakage detection (Pass 1)
    7. Temporal split determination (Pass 2)
    8. OOV rate computation (Pass 2)
    9. PSI drift detection (Pass 2)
    10. Hash collision simulation (no data reading)
    11. Pairwise interaction analysis (Pass 2 with sampling)
    12. Report generation (FeatureMap + Model Plan + Evidence markdown)
    
    Args:
        args: Parsed command-line arguments
    """
    # ========== Stage 1: Setup ==========
    
    # Validate inputs
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        print(f"ERROR: Input CSV not found: {input_csv}", file=sys.stderr)
        sys.exit(1)
    
    out_root = Path(args.out_root)
    log_dir = out_root / 'logs'
    
    # Setup logger
    logger = setup_logger(str(log_dir), verbose=args.verbose)
    
    logger.info("=" * 80)
    logger.info("[AVAZU-EDA] Starting evidence-driven EDA...")
    logger.info("=" * 80)
    logger.info(f"Input: {input_csv.resolve()}")
    logger.info(f"Output Root: {out_root.resolve()}")
    logger.info(f"Chunksize: {args.chunksize}")
    logger.info(f"Config: topk={args.topk}, top_values={args.top_values}, sample_rows={args.sample_rows}")
    logger.info(f"Temporal Split: train_days={args.train_days}, test_days={args.test_days}")
    
    # Create output directories
    output_dirs = create_output_dirs(str(out_root))
    logger.info(f"[SETUP] Output directories created")
    
    # Store all results for report generation
    all_results = {
        'config': vars(args),
        'output_dirs': output_dirs,
    }
    
    # ========== Stage 2: Schema Detection & Overview ==========
    try:
        schema_json, overview_json = compute_schema_and_overview(
            str(input_csv), args.chunksize, str(out_root),
            sample_rows=args.sample_rows  # Limit rows for dry-run
        )
        all_results['schema'] = schema_json
        all_results['overview'] = overview_json
    except Exception as e:
        logger.error(f"[SCHEMA & OVERVIEW] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Stage 3: Columns Profile (Top-K, Missing, Entropy, etc.) ==========
    try:
        columns_profile = compute_columns_profile(
            str(input_csv), args.chunksize, args.topk, str(out_root), 
            max_unique_set=200000,
            sample_rows=args.sample_rows  # Limit rows for dry-run
        )
        all_results['columns_profile'] = columns_profile
    except Exception as e:
        logger.error(f"[COLUMNS PROFILE] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Stage 4: Temporal CTR Analysis ==========
    try:
        temporal_ctr = compute_temporal_ctr(
            str(input_csv), args.chunksize, str(out_root),
            sample_rows=args.sample_rows  # Limit rows for dry-run
        )
        all_results['temporal_ctr'] = temporal_ctr
    except Exception as e:
        logger.error(f"[TEMPORAL_CTR] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Stage 5: CTR by Group (Feature-wise Analysis) ==========
    try:
        ctr_by_group = compute_ctr_by_group(
            str(input_csv), args.chunksize, str(out_root), args.top_values,
            sample_rows=args.sample_rows  # Limit rows for dry-run
        )
        all_results['ctr_by_group'] = ctr_by_group
    except Exception as e:
        logger.error(f"[CTR_BY_GROUP] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Stage 5B: CTR by Top Values (Enhanced with min_support) ==========
    try:
        ctr_by_top = compute_ctr_by_top_values(
            str(input_csv), args.chunksize,
            args.top_values, args.min_support, str(out_root),
            sample_rows=args.sample_rows,
        )
        all_results['ctr_by_top_values'] = ctr_by_top
    except Exception as e:
        logger.error(f"[CTR_BY_TOP_VALUES] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Stage 6: OOV Rate with Time Split ==========
    try:
        oov_result = compute_oov_rate_time_split(
            str(input_csv), args.chunksize, 
            args.train_days, args.test_days, str(out_root),
            max_vocab_size=100000,
            sample_rows=args.sample_rows  # Limit rows for dry-run
        )
        all_results['oov_rate'] = oov_result
    except Exception as e:
        logger.error(f"[OOV_TIME_SPLIT] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Stage 7: PSI Drift Detection ==========
    try:
        psi_result = compute_psi_train_test(
            str(input_csv), args.chunksize, str(out_root),
            args.train_days, args.test_days,
            sample_rows=args.sample_rows  # Limit rows for dry-run
        )
        all_results['psi_drift'] = psi_result
    except Exception as e:
        logger.error(f"[PSI_DRIFT] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Stage 8: Hash Collision Simulation ==========
    try:
        hash_collision = compute_hash_collision_sim(
            str(input_csv), args.chunksize, 
            args.sample_rows, str(out_root)
        )
        all_results['hash_collision'] = hash_collision
    except Exception as e:
        logger.error(f"[HASH_COLLISION_SIM] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Stage 9: Pairwise MI (Top bins) ==========
    try:
        pairwise_mi = compute_pairwise_mi_topbins(
            str(input_csv), args.chunksize, str(out_root),
            sample_rows=args.sample_rows  # Limit rows for dry-run
        )
        all_results['pairwise_mi'] = pairwise_mi
    except Exception as e:
        logger.error(f"[PAIRWISE_MI] FAILED: {e}")
        logger.exception(e)
        sys.exit(1)
    
    # ========== Placeholder for remaining stages ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("[AVAZU-EDA] ✓ EDA Complete (Stages 1-9)")
    logger.info("=" * 80)
    logger.info("Remaining stages (9-12) to be implemented...")
    logger.info(f"All artifacts saved to: {out_root}")
    logger.info("=" * 80)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
