"""
Automation Script: Train ML Models for Climate Vulnerability.
"""
import argparse
import sys
import os
import pandas as pd
from loguru import logger
import joblib
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.train import ModelTrainer
from config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Train ML models for Vulnerability Prediction.")
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to the input feature CSV (must contain features and a target label).")
    parser.add_argument("--target", type=str, default="vulnerability_label", 
                        help="Name of the target column in the dataset (e.g., 'vulnerability_label' or 'risk_class').")
    parser.add_argument("--model", type=str, default="xgboost", 
                        choices=["random_forest", "xgboost"],
                        help="Algorithm to train.")
    parser.add_argument("--out_dir", type=str, default="models/", 
                        help="Directory to save the trained model and scaler.")
    args = parser.parse_args()

    logger.info(f"Initializing {args.model.upper()} Model Training Workflow")

    if not os.path.exists(args.data):
        logger.error(f"Training data not found: {args.data}")
        sys.exit(1)

    # 1. Load Data
    logger.info(f"Loading training data from {args.data}")
    df = pd.read_csv(args.data)
    
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in dataset. Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Separate features and target
    y = df[args.target].values
    
    # Drop non-feature columns (like IDs and target)
    ignore_cols = ['geographic_id', 'region_name', 'geometry', args.target]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X_raw = df[feature_cols].values
    logger.info(f"Training on {len(X_raw)} samples with {len(feature_cols)} features: {feature_cols}")

    # 2. Scale Features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Train Model
    trainer = ModelTrainer(model_type=args.model)
    try:
        metrics = trainer.train_model(
            X=X_scaled, 
            y=y, 
            feature_names=feature_cols,
            tune_hyperparameters=True  # Automatically find best parameters
        )
        
        logger.success("Model Training Complete!")
        print("\n--- Validation Metrics ---")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.3f}")
        print(f"F1 Score:  {metrics.get('f1_score', 0):.3f}")
        
        # 4. Save Artifacts
        os.makedirs(args.out_dir, exist_ok=True)
        
        # Save Scaler
        scaler_path = os.path.join(args.out_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved Feature Scaler to {scaler_path}")
        
        # Save Model
        model_path = os.path.join(args.out_dir, f"{args.model}.joblib")
        trainer.save_model(model_path)
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
