"""
Automation Script: Fetch and Extract Google Earth Engine Data.
"""
import argparse
import sys
import os
import geopandas as gpd
from loguru import logger
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.boundaries import BoundaryLoader
from src.features.extractor import FeatureExtractor
from src.features.integrator import SpatialIntegrator
from config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Fetch and extract GEE satellite data.")
    parser.add_argument("--boundaries", type=str, required=True, 
                        help="Path to the input Shapefile or GeoJSON of the region.")
    parser.add_argument("--start", type=str, default="2023-01-01", 
                        help="Start date for satellite imagery (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="2023-12-31", 
                        help="End date for satellite imagery (YYYY-MM-DD).")
    parser.add_argument("--output", type=str, default="data/features/region_features.csv", 
                        help="Output path for the extracted features CSV.")
    args = parser.parse_args()

    logger.info("Initializing Data Fetching & Extraction Workflow")

    if not os.path.exists(args.boundaries):
        logger.error(f"Boundary file not found: {args.boundaries}")
        sys.exit(1)

    # 1. Load Boundaries
    logger.info(f"Loading boundaries from {args.boundaries}")
    boundary_loader = BoundaryLoader()
    gdf = boundary_loader.load_boundary(args.boundaries)
    
    if gdf.empty:
        logger.error("Loaded boundary GeoDataFrame is empty.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(gdf)} distinct sub-regions/grids.")

    # 2. Extract Features via Earth Engine
    # Note: Assumes GEE credentials are set in .env as per documentation
    logger.info("Connecting to Google Earth Engine and extracting zonal statistics...")
    
    extractor = FeatureExtractor(
        start_date=args.start,
        end_date=args.end,
        scale=500  # 500m resolution to save quotas
    )
    
    try:
        # Extract base indicators (NDVI, LST, NDWI)
        feature_df = extractor.extract_all(gdf)
        
        # 3. Save Output
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        feature_df.to_csv(args.output, index=False)
        logger.success(f"Successfully extracted features and saved to {args.output}")
        
    except Exception as e:
        logger.exception("Failed to extract features from Google Earth Engine. Ensure your GEE credentials are valid.")
        logger.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
