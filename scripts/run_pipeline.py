"""
Automation Script: End-to-End Execution of the AI-GEO-NGO Pipeline.
"""
import argparse
import sys
import os
import pandas as pd
import geopandas as gpd
from loguru import logger

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import PredictionEngine
from config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Run the AI-GEO-NGO prediction pipeline.")
    parser.add_argument("--data", type=str, default="data/features/region_features.csv", 
                        help="Path to the input feature CSV.")
    parser.add_argument("--boundaries", type=str, default=None, 
                        help="Optional: Path to the GeoJSON/Shapefile of region boundaries.")
    parser.add_argument("--output", type=str, default="data/predictions/", 
                        help="Directory to save the prediction results.")
    parser.add_argument("--model", type=str, default="random_forest", 
                        choices=["random_forest", "xgboost", "kmeans"],
                        help="Which ML model to use for vulnerability prediction.")
    args = parser.parse_args()

    logger.info("Initializing Pipeline Execution")

    # 1. Load Data
    if not os.path.exists(args.data):
        logger.error(f"Input data not found at {args.data}")
        sys.exit(1)
        
    logger.info(f"Loading feature dataset from {args.data}")
    feature_df = pd.read_csv(args.data)
    
    geometries_gdf = None
    if args.boundaries:
        if os.path.exists(args.boundaries):
            logger.info(f"Loading geographic boundaries from {args.boundaries}")
            geometries_gdf = gpd.read_file(args.boundaries)
        else:
            logger.warning(f"Boundary file not found at {args.boundaries}. GIS export will be skipped.")

    # 2. Run Pipeline
    engine = PredictionEngine(ml_model_type=args.model)
    
    # In a real scenario, you'd load trained artifacts here:
    # engine.load_models(scaler_path="models/scaler.pkl", model_path=f"models/{args.model}.joblib", feature_cols=[...])
    
    logger.info("Triggering PredictionEngine...")
    try:
        final_df = engine.run_pipeline(
            feature_df=feature_df,
            output_dir=args.output,
            geometries_gdf=geometries_gdf
        )
        logger.success(f"Pipeline completed successfully. Output saved to {args.output}")
        print("\n--- Pipeline Summary ---")
        print(f"Total Regions Processed: {len(final_df)}")
        if 'priority_class' in final_df.columns:
            print("WASH Priority Distribution:")
            print(final_df['priority_class'].value_counts())
            
    except Exception as e:
        logger.exception(f"Pipeline failed during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
