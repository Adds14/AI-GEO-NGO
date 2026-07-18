"""
Supervised Machine Learning Module.

Implements Random Forest model training for vulnerability prediction
when labeled ground-truth data is available.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from loguru import logger
from typing import Dict, List, Tuple
import joblib
import os


class ModelTrainer:
    """Supervised learning trainer using Random Forest."""

    def __init__(self, random_state: int = 42):
        """Initialize the ModelTrainer."""
        self.random_state = random_state
        self.model = None
        self.feature_names = []
        self.is_fitted = False
        logger.debug("Initialized ModelTrainer.")

    def train_random_forest(self, 
                            X: np.ndarray, 
                            y: np.ndarray, 
                            feature_names: List[str],
                            tune_hyperparameters: bool = False) -> Dict:
        """
        Train a RandomForestClassifier.

        Args:
            X: Scaled feature matrix.
            y: Target labels.
            feature_names: List of feature names for tracking importance.
            tune_hyperparameters: Whether to run GridSearchCV.

        Returns:
            Dict containing evaluation metrics on a validation split.
        """
        logger.info("Starting Random Forest training.")
        self.feature_names = feature_names

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        if tune_hyperparameters:
            logger.info("Running Hyperparameter Tuning (GridSearchCV).")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            rf = RandomForestClassifier(random_state=self.random_state)
            grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            logger.info(f"Best parameters found: {grid.best_params_}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

        self.is_fitted = True

        # Evaluate
        preds = self.model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        logger.info(f"Validation Accuracy: {acc:.4f}")
        
        report = classification_report(y_val, preds, output_dict=True)
        return report

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importances from the trained model.

        Returns:
            Dictionary mapping feature names to their importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained.")
            
        importances = self.model.feature_importances_
        fi_dict = {name: float(imp) for name, imp in zip(self.feature_names, importances)}
        
        # Sort by importance descending
        fi_dict = dict(sorted(fi_dict.items(), key=lambda item: item[1], reverse=True))
        return fi_dict

    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, filepath)
        logger.info(f"Random Forest model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_fitted = True
        logger.info(f"Random Forest model loaded from {filepath}")
