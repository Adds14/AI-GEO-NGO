"""
Prediction Engine Pipeline Module (Deliverable 10).

Orchestrates loading trained ML models, generating vulnerability predictions,
calculating the WASH Intervention Priority, and exporting final GIS-compatible datasets.
"""
import pandas as pd
import geopandas as gpd
from loguru import logger
from typing import Dict, List, Optional
import os

from src.ml.predict import Predictor
from src.priority.engine import WASHPriorityEngine


class PredictionEngine:
    """End-to-end Prediction and Prioritization Engine."""

    def __init__(self, ml_model_type: str = 'random_forest'):
        """
        Initialize the Prediction Engine.

        Args:
            ml_model_type: The ML model type ('random_forest', 'xgboost', 'kmeans').
        """
        self.predictor = Predictor(model_type=ml_model_type)
        self.priority_engine = WASHPriorityEngine()
        logger.debug(f"Initialized PredictionEngine with ML model: {ml_model_type}")

    def load_models(self, scaler_path: str, model_path: str, feature_cols: List[str]):
        """Load the underlying ML model and scaler artifacts."""
        logger.info("Loading ML artifacts into PredictionEngine.")
        self.predictor.load_artifacts(scaler_path, model_path, feature_cols)
        logger.success("Models loaded successfully.")

    def run_pipeline(self, 
                     feature_df: pd.DataFrame, 
                     output_dir: str, 
                     geometries_gdf: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
        """
        Run the full prediction and prioritization pipeline.

        Args:
            feature_df: DataFrame of new features to predict on.
            output_dir: Directory to export results.
            geometries_gdf: Optional GeoDataFrame for GIS exports.

        Returns:
            DataFrame containing the comprehensive results.
        """
        logger.info("Starting Prediction Pipeline...")
        
        # 1. Predict Climate Vulnerability
        logger.info("Step 1: Predicting Climate Vulnerability")
        ml_results = self.predictor.predict(feature_df)
        ml_df = pd.DataFrame(ml_results)
        
        # 2. Calculate WASH Intervention Priority
        logger.info("Step 2: Calculating WASH Intervention Priority")
        # Ensure we pass a Series of vulnerability scores aligned with feature_df
        vuln_scores = ml_df['vulnerability_score']
        priority_df = self.priority_engine.compute_batch(feature_df, vuln_scores)
        
        # Merge ML results with Priority results
        if 'geographic_id' in priority_df.columns:
            final_df = pd.merge(ml_df, priority_df, on=['geographic_id', 'vulnerability_score'], how='left')
        else:
            final_df = pd.concat([ml_df, priority_df.drop(columns=['vulnerability_score'])], axis=1)
            
        logger.success("Pipeline computation complete.")
        
        # 3. Save Outputs (Export)
        logger.info("Step 3: Exporting Results")
        self.export_outputs(final_df, output_dir, geometries_gdf)
        
        return final_df

    def export_outputs(self, final_df: pd.DataFrame, output_dir: str, geometries_gdf: Optional[gpd.GeoDataFrame] = None):
        """Export the final results to CSV, GeoJSON, and GIS-compatible formats."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert any dict columns to strings for safe export
        safe_df = final_df.copy()
        for col in ['feature_importance', 'top_contributing_factors']:
            if col in safe_df.columns:
                safe_df[col] = safe_df[col].astype(str)

        # 1. Export CSV
        csv_path = os.path.join(output_dir, "predictions_and_priorities.csv")
        safe_df.to_csv(csv_path, index=False)
        logger.info(f"Exported to CSV: {csv_path}")

        # 2. Export GeoJSON & GIS Shapefile
        if geometries_gdf is not None:
            # Ensure the join key exists
            join_key = 'geographic_id' if 'geographic_id' in safe_df.columns else 'region_id'
            geo_key = 'geographic_id' if 'geographic_id' in geometries_gdf.columns else 'region_id'
            
            if join_key in safe_df.columns and geo_key in geometries_gdf.columns:
                gdf_merged = geometries_gdf.merge(safe_df, left_on=geo_key, right_on=join_key, how='left')
                
                # GeoJSON Export
                geojson_path = os.path.join(output_dir, "predictions_and_priorities.geojson")
                gdf_merged.to_file(geojson_path, driver='GeoJSON')
                logger.info(f"Exported to GeoJSON: {geojson_path}")
                
                # Shapefile Export (GIS-compatible)
                try:
                    shp_dir = os.path.join(output_dir, "shapefile_export")
                    os.makedirs(shp_dir, exist_ok=True)
                    shp_path = os.path.join(shp_dir, "predictions.shp")
                    # Shapefiles have strict column name length limits (10 chars), so we might need to truncate
                    gdf_shp = gdf_merged.copy()
                    gdf_shp.columns = [c[:10] for c in gdf_shp.columns]
                    gdf_shp.to_file(shp_path, driver='ESRI Shapefile')
                    logger.info(f"Exported to ESRI Shapefile: {shp_path}")
                except Exception as e:
                    logger.warning(f"Shapefile export failed (usually due to column names/types): {e}")
            else:
                logger.warning(f"Could not merge geometries. Missing join keys: {join_key} or {geo_key}")
