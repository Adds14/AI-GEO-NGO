"""
WASH Intervention Priority Engine.

Calculates a WASH Intervention Priority Index (0-100) for each region
using climate vulnerability scores and specific environmental indicators.

Priority Classes:
    0-40:  Low Priority
    41-70: Medium Priority
    71-100: High Priority
"""
from loguru import logger
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
import os


class WASHPriorityEngine:
    """Engine for computing WASH Intervention Priority scores."""

    # Updated weights to match Deliverable 9 explicitly requested inputs
    DEFAULT_WEIGHTS = {
        "vulnerability_score": 0.35,
        "lst": 0.15,
        "population_density": 0.10,
        "rainfall": 0.10,
        "ndvi": 0.08,
        "ndwi": 0.08,
        "ndbi": 0.07,
        "elevation": 0.07,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        logger.info(f"WASH Priority Engine initialized with {len(self.weights)} weighted factors.")

    def compute_priority_score(self, features: Dict[str, float], vulnerability_score: float) -> float:
        """
        Compute the WASH Intervention Priority Score (0-100) for a single region.
        Assumes features are normalized 0-1.
        """
        features["vulnerability_score"] = vulnerability_score

        score = 0.0
        for feature_name, weight in self.weights.items():
            value = features.get(feature_name, 0.0)
            # Invert beneficial features (higher NDVI/NDWI/Rainfall/Elevation = less risk)
            # Depending on context, high elevation might be hard to reach or safe from floods. 
            # We assume for this model: high NDVI, NDWI, and Rainfall = good (invert to make risk).
            if feature_name in ["ndvi", "ndwi", "rainfall"]:
                value = 1.0 - value
            score += weight * value

        priority_score = round(min(max(score * 100, 0), 100), 2)
        return priority_score

    def classify_priority(self, score: float) -> str:
        """Classify a priority score into Low, Medium, or High."""
        if score <= 40:
            return "Low Priority"
        elif score <= 70:
            return "Medium Priority"
        else:
            return "High Priority"

    def generate_explanation(self, features: Dict[str, float], priority_score: float, priority_class: str) -> str:
        """Generate a human-readable explanation."""
        contributions = []
        for feature_name, weight in self.weights.items():
            value = features.get(feature_name, 0.0)
            if feature_name in ["ndvi", "ndwi", "rainfall"]:
                value = 1.0 - value
            contribution = weight * value
            contributions.append((feature_name, contribution, features.get(feature_name, 0.0)))

        contributions.sort(key=lambda x: x[1], reverse=True)
        top_factors = contributions[:3]

        factor_descriptions = {
            "vulnerability_score": "overall climate vulnerability",
            "lst": "high land surface temperature",
            "ndvi": "low vegetation cover",
            "ndwi": "limited water availability",
            "ndbi": "high urban built-up density",
            "rainfall": "low rainfall/drought conditions",
            "population_density": "high population density",
            "elevation": "elevation-based access challenges",
        }

        reasons = [factor_descriptions.get(f[0], f[0]) for f in top_factors]
        explanation = (
            f"This region is classified as {priority_class} (score: {priority_score}/100). "
            f"The primary contributing factors are: {reasons[0]}, {reasons[1]}, and {reasons[2]}."
        )
        return explanation

    def get_top_contributing_factors(self, features: Dict[str, float], top_n: int = 5) -> Dict[str, float]:
        """Get the top contributing factors as a dictionary for easier serialization."""
        contributions = []
        for feature_name, weight in self.weights.items():
            value = features.get(feature_name, 0.0)
            effective_value = (1.0 - value) if feature_name in ["ndvi", "ndwi", "rainfall"] else value
            contribution = weight * effective_value
            contributions.append({
                "feature": feature_name,
                "contribution": round(contribution, 4),
            })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return {item['feature']: item['contribution'] for item in contributions[:top_n]}

    def compute_batch(self, feature_df: pd.DataFrame, vulnerability_scores: pd.Series) -> pd.DataFrame:
        """Compute WASH priority for all regions in a DataFrame."""
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
        
        # Merge results back with original feature IDs if available
        if 'geographic_id' in feature_df.columns:
            result_df.insert(0, 'geographic_id', feature_df['geographic_id'].values)
            
        logger.success(f"WASH priority computed for {len(results)} regions.")
        return result_df

    def export_results(self, priority_df: pd.DataFrame, output_dir: str, geometries_gdf: Optional[gpd.GeoDataFrame] = None):
        """
        Export priority results to CSV and GeoJSON.
        
        Args:
            priority_df: DataFrame containing the priority results.
            output_dir: Directory to save exports.
            geometries_gdf: Optional GeoDataFrame to join for GeoJSON export.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Export CSV
        csv_path = os.path.join(output_dir, "wash_priority_results.csv")
        priority_df.to_csv(csv_path, index=False)
        logger.info(f"Exported priority results to CSV: {csv_path}")
        
        # 2. Export GeoJSON if geometries provided
        if geometries_gdf is not None and 'geographic_id' in priority_df.columns:
            gdf_merged = geometries_gdf.merge(priority_df, on='geographic_id', how='left')
            geojson_path = os.path.join(output_dir, "wash_priority_results.geojson")
            
            # Convert dicts in 'top_contributing_factors' to string for GeoJSON compatibility
            if 'top_contributing_factors' in gdf_merged.columns:
                gdf_merged['top_contributing_factors'] = gdf_merged['top_contributing_factors'].astype(str)
                
            gdf_merged.to_file(geojson_path, driver='GeoJSON')
            logger.info(f"Exported priority results to GeoJSON: {geojson_path}")
