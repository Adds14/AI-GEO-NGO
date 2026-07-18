"""
Scoring utilities for the WASH Intervention Priority Engine.

Provides normalization, weighting, and composite scoring functions.
"""
import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Optional


def normalize_features(df: pd.DataFrame, columns: list, method: str = "minmax") -> pd.DataFrame:
    """
    Normalize feature columns to 0-1 scale.

    Args:
        df: Input DataFrame.
        columns: List of column names to normalize.
        method: Normalization method ('minmax' or 'zscore').

    Returns:
        DataFrame with normalized columns.
    """
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping.")
            continue
        if method == "minmax":
            col_min = result[col].min()
            col_max = result[col].max()
            if col_max - col_min == 0:
                result[col] = 0.0
            else:
                result[col] = (result[col] - col_min) / (col_max - col_min)
        elif method == "zscore":
            result[col] = (result[col] - result[col].mean()) / result[col].std()
    logger.info(f"Normalized {len(columns)} features using {method} method.")
    return result


def compute_weighted_score(features: Dict[str, float], weights: Dict[str, float], invert_keys: list = None) -> float:
    """
    Compute a weighted composite score from feature values.

    Args:
        features: Feature name to value mapping.
        weights: Feature name to weight mapping.
        invert_keys: Features where low value = high risk (will be inverted).

    Returns:
        Weighted composite score (0-1 scale).
    """
    invert_keys = invert_keys or []
    score = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        value = features.get(key, 0.0)
        if key in invert_keys:
            value = 1.0 - value
        score += weight * value
        total_weight += weight
    return score / total_weight if total_weight > 0 else 0.0


def classify_score(score: float, thresholds: Dict[str, tuple] = None) -> str:
    """
    Classify a score into a category based on thresholds.

    Args:
        score: The score to classify.
        thresholds: Dict mapping class name to (min, max) tuple.

    Returns:
        Classification string.
    """
    if thresholds is None:
        thresholds = {
            "Low Priority": (0, 40),
            "Medium Priority": (41, 70),
            "High Priority": (71, 100),
        }
    for class_name, (low, high) in thresholds.items():
        if low <= score <= high:
            return class_name
    return "Unknown"
