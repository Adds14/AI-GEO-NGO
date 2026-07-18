"""
Feature Extraction Pipeline Module.

Extracts zonal statistics from environmental indicators over administrative
regions or regular grids, compiling them into a feature table for machine learning.
"""
import ee
import pandas as pd
import geopandas as gpd
from loguru import logger
from typing import Dict, List, Optional, Union
import datetime
import os


class FeatureExtractor:
    """Extract machine learning features from environmental indicators."""

    def __init__(self, gee_client=None):
        """
        Initialize the FeatureExtractor.
        
        Args:
            gee_client: Optional GEE client instance.
        """
        self.gee_client = gee_client
        logger.debug("Initialized FeatureExtractor.")

    def stack_indicators(self, indicators: Dict[str, ee.Image]) -> ee.Image:
        """
        Stack multiple single-band indicator images into a single multi-band image.
        
        Args:
            indicators (Dict[str, ee.Image]): Dictionary mapping band names to ee.Images.
            
        Returns:
            ee.Image: Multi-band image containing all indicators.
        """
        logger.info("Stacking indicator images.")
        stacked = None
        for name, image in indicators.items():
            # Ensure the band has the correct name
            img = image.select([image.bandNames().get(0)], [name])
            if stacked is None:
                stacked = img
            else:
                stacked = stacked.addBands(img)
        return stacked

    def extract_zonal_stats(self, 
                            stacked_image: ee.Image, 
                            regions: ee.FeatureCollection, 
                            scale: int = 30,
                            id_column: str = 'region_id',
                            timestamp: str = None) -> pd.DataFrame:
        """
        Calculate zonal statistics (mean, min, max, std) for regions.
        
        Args:
            stacked_image (ee.Image): Multi-band image of indicators.
            regions (ee.FeatureCollection): Polygons to compute stats over.
            scale (int): Scale in meters for reduction.
            id_column (str): The property name in regions to use as the geographic identifier.
            timestamp (str, optional): Timestamp to add to the features (e.g., '2023').
            
        Returns:
            pd.DataFrame: DataFrame containing the extracted features.
        """
        logger.info(f"Extracting zonal stats at {scale}m scale.")
        
        # Define a combined reducer for mean, min, max, and stdDev
        reducer = ee.Reducer.mean() \
            .combine(ee.Reducer.minMax(), sharedInputs=True) \
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            
        # Add centroid coordinates to each feature before reducing
        def add_centroid(feature):
            centroid = feature.geometry().centroid()
            coords = centroid.coordinates()
            return feature.set({
                'longitude': coords.get(0),
                'latitude': coords.get(1)
            })
            
        regions_with_coords = regions.map(add_centroid)

        # Reduce regions
        stats_fc = stacked_image.reduceRegions(
            collection=regions_with_coords,
            reducer=reducer,
            scale=scale,
            tileScale=4  # Helps prevent memory limits on GEE
        )
        
        # Fetch the results from Earth Engine (this is synchronous and might take time for large AOIs)
        logger.info("Fetching results from Earth Engine...")
        try:
            stats_list = stats_fc.getInfo()['features']
        except Exception as e:
            logger.error(f"Failed to fetch stats from Earth Engine: {e}")
            raise
            
        # Parse into a Pandas DataFrame
        logger.info("Parsing results into Pandas DataFrame.")
        rows = []
        for feature in stats_list:
            props = feature['properties']
            
            # Extract basic info
            row = {
                'geographic_id': props.get(id_column, 'unknown'),
                'latitude': props.get('latitude'),
                'longitude': props.get('longitude'),
                'timestamp': timestamp or datetime.datetime.now().isoformat()
            }
            
            # Extract indicator stats
            for key, val in props.items():
                if key not in [id_column, 'latitude', 'longitude'] and val is not None:
                    row[key] = val
                    
            rows.append(row)
            
        df = pd.DataFrame(rows)
        logger.info(f"Extracted {len(df)} feature rows.")
        return df

    def export_to_csv(self, df: pd.DataFrame, filepath: str):
        """
        Export the feature table to a CSV file.
        
        Args:
            df (pd.DataFrame): Feature DataFrame.
            filepath (str): Output CSV filepath.
        """
        logger.info(f"Exporting features to {filepath}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        df.to_csv(filepath, index=False)
        logger.success(f"Successfully exported feature table with shape {df.shape}")

    def run_pipeline(self, 
                     indicators: Dict[str, ee.Image], 
                     boundary_fc: ee.FeatureCollection, 
                     output_csv_path: str,
                     scale: int = 30,
                     id_column: str = 'region_id',
                     timestamp: str = None) -> pd.DataFrame:
        """
        Run the complete feature extraction pipeline.
        
        Args:
            indicators: Dictionary of indicator images.
            boundary_fc: GEE FeatureCollection of boundaries/grids.
            output_csv_path: Path to save the resulting CSV.
            scale: Reduction scale in meters.
            id_column: Column name for geographic identifier.
            timestamp: Timestamp string.
            
        Returns:
            pd.DataFrame: The generated feature table.
        """
        logger.info("Starting Feature Extraction Pipeline.")
        
        stacked = self.stack_indicators(indicators)
        df = self.extract_zonal_stats(
            stacked_image=stacked,
            regions=boundary_fc,
            scale=scale,
            id_column=id_column,
            timestamp=timestamp
        )
        self.export_to_csv(df, output_csv_path)
        
        return df
