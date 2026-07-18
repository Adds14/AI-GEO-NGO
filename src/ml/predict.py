"""
Machine Learning Inference Module.

Handles loading trained models and generating predictions, 
vulnerability scores, and explanations for new regions.
"""
import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, List, Tuple
from src.ml.features import FeatureEngineer
from src.ml.clustering import KMeansClusterer
from src.ml.train import ModelTrainer


class Predictor:
    """Inference engine for climate vulnerability predictions."""

    def __init__(self, model_type: str = 'kmeans'):
        """
        Initialize the Predictor.

        Args:
            model_type: 'kmeans' or 'random_forest'.
        """
        self.model_type = model_type
        self.scaler = None
        self.model = None
        self.feature_names = []
        logger.debug(f"Initialized Predictor for model type: {model_type}")

    def load_artifacts(self, scaler_path: str, model_path: str, feature_cols: List[str]):
        """
        Load the scaler and the machine learning model.
        """
        logger.info("Loading ML artifacts...")
        self.feature_names = feature_cols
        
        # Load scaler
        fe = FeatureEngineer(feature_cols=feature_cols)
        fe.load_scaler(scaler_path)
        self.scaler = fe
        
        # Load model
        if self.model_type == 'kmeans':
            self.model = KMeansClusterer()
            self.model.load_model(model_path)
        elif self.model_type == 'random_forest':
            self.model = ModelTrainer()
            self.model.load_model(model_path)
            # Ensure feature names match
            if self.model.feature_names != self.feature_names:
                logger.warning("Feature names in loaded model differ from requested feature_cols.")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        logger.info("Artifacts loaded successfully.")

    def predict(self, df: pd.DataFrame) -> List[Dict]:
        """
        Run inference on a new DataFrame of regional features.

        Args:
            df: DataFrame containing the regional features.

        Returns:
            List of dictionaries containing predictions for each region.
        """
        if self.scaler is None or self.model is None:
            raise ValueError("Artifacts not loaded. Call load_artifacts() first.")
            
        logger.info(f"Running inference for {len(df)} regions.")
        
        # Scale features
        X_scaled, _, ids = self.scaler.prepare_data(df, fit_scaler=False)
        
        results = []
        
        if self.model_type == 'kmeans':
            categories = self.model.predict(X_scaled)
            scores = self.model.get_vulnerability_scores(X_scaled)
            
            for idx, reg_id in enumerate(ids):
                results.append({
                    'region_id': reg_id,
                    'model_type': 'KMeans',
                    'vulnerability_score': float(scores[idx]),
                    'vulnerability_category': categories[idx],
                    'confidence_score': None,  # Not applicable for K-Means
                    'feature_importance': None
                })
                
        elif self.model_type == 'random_forest':
            # Assuming labels are categories like 'Low', 'Medium', 'High'
            categories = self.model.model.predict(X_scaled)
            
            # Use predict_proba for score and confidence
            try:
                probas = self.model.model.predict_proba(X_scaled)
                # Find the index for the 'High' class
                classes = list(self.model.model.classes_)
                if 'High' in classes:
                    high_idx = classes.index('High')
                    scores = probas[:, high_idx]
                else:
                    scores = np.max(probas, axis=1) # Fallback to max proba
                    
                confidence = np.max(probas, axis=1)
            except AttributeError:
                scores = np.zeros(len(X_scaled))
                confidence = np.zeros(len(X_scaled))
                
            fi = self.model.get_feature_importance()
            
            for idx, reg_id in enumerate(ids):
                results.append({
                    'region_id': reg_id,
                    'model_type': 'RandomForest',
                    'vulnerability_score': float(scores[idx]),
                    'vulnerability_category': categories[idx],
                    'confidence_score': float(confidence[idx]),
                    'feature_importance': fi
                })
                
        return results
