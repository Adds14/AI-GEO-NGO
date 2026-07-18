"""
Machine Learning Feature Engineering Module.

Handles scaling, imputation, and final preparation of the integrated 
dataset before it enters the ML models.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
from loguru import logger
import joblib
import os


class FeatureEngineer:
    """Prepares and scales features for machine learning."""

    def __init__(self, 
                 feature_cols: List[str], 
                 target_col: Optional[str] = None,
                 id_col: str = 'geographic_id'):
        """
        Initialize the FeatureEngineer.

        Args:
            feature_cols: List of column names to be used as model features.
            target_col: Column name for the target variable (labels).
            id_col: Column name representing the region identifier.
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.id_col = id_col
        self.scaler = StandardScaler()
        self.is_fitted = False
        logger.debug(f"Initialized FeatureEngineer with {len(feature_cols)} features.")

    def prepare_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], pd.Series]:
        """
        Extract features, scale them, and extract target labels if available.

        Args:
            df: The integrated pandas DataFrame.
            fit_scaler: Whether to fit the scaler or transform using an existing fit.

        Returns:
            Tuple containing:
                - X_scaled: Numpy array of scaled features.
                - y: Numpy array of targets (or None if no target_col).
                - ids: Pandas Series of the region IDs.
        """
        logger.info("Preparing data for machine learning.")
        
        # Verify required columns exist
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required feature columns: {missing_cols}")
            raise KeyError(f"Missing features: {missing_cols}")

        X_raw = df[self.feature_cols].copy()
        
        # Final sanity check for NaNs
        if X_raw.isnull().sum().sum() > 0:
            logger.warning("NaNs detected in features. Filling with feature medians.")
            X_raw = X_raw.fillna(X_raw.median())

        # Scaling
        if fit_scaler:
            logger.info("Fitting StandardScaler on feature matrix.")
            X_scaled = self.scaler.fit_transform(X_raw)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler is not fitted yet. Set fit_scaler=True or load a pre-trained scaler.")
            logger.info("Transforming feature matrix using pre-fitted scaler.")
            X_scaled = self.scaler.transform(X_raw)

        # Extract Targets
        y = None
        if self.target_col and self.target_col in df.columns:
            y = df[self.target_col].values
            
        # Extract IDs
        ids = df[self.id_col] if self.id_col in df.columns else pd.Series(df.index)

        return X_scaled, y, ids

    def save_scaler(self, filepath: str):
        """Save the fitted scaler using joblib."""
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted. Cannot save.")
            
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath: str):
        """Load a fitted scaler from a joblib file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
            
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {filepath}")
