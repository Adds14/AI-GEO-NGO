"""
WASH Intervention Priority Engine.

Calculates a WASH Intervention Priority Index (0-100) for each region
using vulnerability scores and environmental indicators.

Priority Classes:
    0-40:  Low Priority
    41-70: Medium Priority
    71-100: High Priority
"""
from loguru import logger
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class WASHPriorityEngine:
    """Engine for computing WASH Intervention Priority scores."""

    # Default weights for priority calculation
    DEFAULT_WEIGHTS = {
        "vulnerability_score": 0.30,
        "mean_lst": 0.15,
        "mean_ndvi": 0.12,
        "mean_ndwi": 0.12,
        "mean_ndbi": 0.08,
        "vegetation_change": 0.08,
        "water_body_change": 0.05,
        "heat_stress_index": 0.05,
        "rainfall_anomaly": 0.05,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the WASH Priority Engine.

        Args:
            weights: Optional custom weights for priority calculation.
                     Keys are feature names, values are weights (should sum to 1.0).
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        logger.info(f"WASH Priority Engine initialized with {len(self.weights)} weighted factors.")

    def compute_priority_score(self, features: Dict[str, float], vulnerability_score: float) -> float:
        """
        Compute the WASH Intervention Priority Score (0-100) for a single region.

        Args:
            features: Dictionary of normalized feature values (0-1 scale).
            vulnerability_score: Climate vulnerability score (0-1 scale).

        Returns:
            Priority score between 0 and 100.
        """
        features["vulnerability_score"] = vulnerability_score

        score = 0.0
        for feature_name, weight in self.weights.items():
            value = features.get(feature_name, 0.0)
            if feature_name in ["mean_ndvi", "mean_ndwi"]:
                value = 1.0 - value
            score += weight * value

        priority_score = round(min(max(score * 100, 0), 100), 2)
        logger.debug(f"Computed priority score: {priority_score}")
        return priority_score

    def classify_priority(self, score: float) -> str:
        """
        Classify a priority score into Low, Medium, or High.

        Args:
            score: Priority score (0-100).

        Returns:
            Priority class string.
        """
        if score <= 40:
            return "Low Priority"
        elif score <= 70:
            return "Medium Priority"
        else:
            return "High Priority"

    def generate_explanation(self, features: Dict[str, float], priority_score: float, priority_class: str) -> str:
        """
        Generate a human-readable explanation for why a region received its priority.

        Args:
            features: Dictionary of feature values.
            priority_score: The computed priority score.
            priority_class: The priority classification.

        Returns:
            Human-readable explanation string.
        """
        contributions = []
        for feature_name, weight in self.weights.items():
            value = features.get(feature_name, 0.0)
            if feature_name in ["mean_ndvi", "mean_ndwi"]:
                value = 1.0 - value
            contribution = weight * value
            contributions.append((feature_name, contribution, features.get(feature_name, 0.0)))

        contributions.sort(key=lambda x: x[1], reverse=True)
        top_factors = contributions[:3]

        factor_descriptions = {
            "vulnerability_score": "overall climate vulnerability",
            "mean_lst": "high land surface temperature",
            "mean_ndvi": "low vegetation cover",
            "mean_ndwi": "limited water availability",
            "mean_ndbi": "high urban built-up density",
            "vegetation_change": "vegetation decline",
            "water_body_change": "water body reduction",
            "heat_stress_index": "heat stress conditions",
            "rainfall_anomaly": "rainfall irregularity",
        }

        reasons = [factor_descriptions.get(f[0], f[0]) for f in top_factors]
        explanation = (
            f"This region is classified as {priority_class} (score: {priority_score}/100). "
            f"The primary contributing factors are: {reasons[0]}, {reasons[1]}, and {reasons[2]}."
        )

        logger.debug(f"Generated explanation for score {priority_score}")
        return explanation

    def get_top_contributing_factors(self, features: Dict[str, float], top_n: int = 5) -> List[Dict]:
        """
        Get the top contributing factors for a region's priority score.

        Args:
            features: Dictionary of feature values.
            top_n: Number of top factors to return.

        Returns:
            List of dicts with feature name, weight, value, and contribution.
        """
        contributions = []
        for feature_name, weight in self.weights.items():
            value = features.get(feature_name, 0.0)
            effective_value = (1.0 - value) if feature_name in ["mean_ndvi", "mean_ndwi"] else value
            contribution = weight * effective_value
            contributions.append({
                "feature": feature_name,
                "weight": weight,
                "value": round(value, 4),
                "contribution": round(contribution, 4),
            })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions[:top_n]

    def compute_batch(self, feature_df: pd.DataFrame, vulnerability_scores: pd.Series) -> pd.DataFrame:
        """
        Compute WASH priority for all regions in a DataFrame.

        Args:
            feature_df: DataFrame with features (one row per region).
            vulnerability_scores: Series of vulnerability scores aligned with feature_df.

        Returns:
            DataFrame with priority_score, priority_class, and explanation columns added.
        """
        logger.info(f"Computing WASH priority for {len(feature_df)} regions...")

        results = []
        for idx, row in feature_df.iterrows():
            features = row.to_dict()
            vuln_score = vulnerability_scores.iloc[idx] if idx < len(vulnerability_scores) else 0.0

            score = self.compute_priority_score(features.copy(), vuln_score)
            priority_class = self.classify_priority(score)
            explanation = self.generate_explanation(features.copy(), score, priority_class)
            top_factors = self.get_top_contributing_factors(features.copy())

            results.append({
                "priority_score": score,
                "priority_class": priority_class,
                "explanation": explanation,
                "top_contributing_factors": top_factors,
                "vulnerability_score": vuln_score,
            })

        result_df = pd.DataFrame(results)
        logger.success(f"WASH priority computed for {len(results)} regions.")
        return result_df
