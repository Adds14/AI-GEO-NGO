"""
Supervised Machine Learning Module.

Implements Random Forest and XGBoost model training for vulnerability prediction
when labeled ground-truth data is available. Evaluates models rigorously.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from loguru import logger
from typing import Dict, List, Tuple
import joblib
import os

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed. XGBoost fallback disabled.")


class ModelTrainer:
    """Supervised learning trainer using Random Forest or XGBoost."""

    def __init__(self, random_state: int = 42, model_type: str = 'rf'):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state: Seed for reproducibility.
            model_type: 'rf' for Random Forest, 'xgboost' for XGBoost.
        """
        self.random_state = random_state
        self.model_type = model_type.lower()
        self.model = None
        self.feature_names = []
        self.is_fitted = False
        self.classes_ = None
        
        if self.model_type == 'xgboost' and not HAS_XGB:
            logger.error("XGBoost requested but not installed. Falling back to Random Forest.")
            self.model_type = 'rf'
            
        logger.debug(f"Initialized ModelTrainer with model_type: {self.model_type}")

    def train_model(self, 
                    X: np.ndarray, 
                    y: np.ndarray, 
                    feature_names: List[str],
                    tune_hyperparameters: bool = False) -> Dict:
        """
        Train the selected classifier and evaluate.

        Args:
            X: Scaled feature matrix.
            y: Target labels.
            feature_names: List of feature names.
            tune_hyperparameters: Whether to run GridSearchCV.

        Returns:
            Dict containing comprehensive evaluation metrics.
        """
        logger.info(f"Starting {self.model_type.upper()} training.")
        self.feature_names = feature_names
        
        # Save classes
        self.classes_ = np.unique(y)

        # Handle string labels for XGBoost by label encoding internally
        y_train_input = y
        label_map = None
        if self.model_type == 'xgboost' and y.dtype.kind in {'U', 'S', 'O'}:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_input = le.fit_transform(y)
            self.classes_ = le.classes_
            label_map = le

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_train_input, test_size=0.2, random_state=self.random_state, stratify=y_train_input
        )

        if self.model_type == 'rf':
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        else:
            base_model = XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='logloss')
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }

        if tune_hyperparameters:
            logger.info("Running Hyperparameter Tuning (GridSearchCV).")
            grid = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            logger.info(f"Best parameters found: {grid.best_params_}")
        else:
            self.model = base_model
            self.model.fit(X_train, y_train)

        self.is_fitted = True

        # Evaluate
        preds = self.model.predict(X_val)
        
        # Comprehensive Metrics
        is_multiclass = len(self.classes_) > 2
        avg_method = 'weighted' if is_multiclass else 'binary'
        
        # If XGBoost label encoded, we need to map back to original for classification report readability,
        # but the scikit-learn metrics need numeric inputs if binary/multiclass.
        acc = accuracy_score(y_val, preds)
        
        # We wrap in try-except in case labels are strings and average='binary' is implicitly called
        try:
            prec = precision_score(y_val, preds, average=avg_method)
            rec = recall_score(y_val, preds, average=avg_method)
            f1 = f1_score(y_val, preds, average=avg_method)
        except ValueError:
            # Handle string targets natively
            prec = precision_score(y_val, preds, average='weighted')
            rec = recall_score(y_val, preds, average='weighted')
            f1 = f1_score(y_val, preds, average='weighted')

        # ROC-AUC
        try:
            probas = self.model.predict_proba(X_val)
            if is_multiclass:
                roc_auc = roc_auc_score(y_val, probas, multi_class='ovr', average='weighted')
            else:
                roc_auc = roc_auc_score(y_val, probas[:, 1])
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            roc_auc = None

        logger.info(f"Validation Metrics -> Acc: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc}")
        
        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'report': classification_report(y_val, preds, output_dict=True)
        }
        
        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model is not trained.")
        importances = self.model.feature_importances_
        fi_dict = {name: float(imp) for name, imp in zip(self.feature_names, importances)}
        return dict(sorted(fi_dict.items(), key=lambda item: item[1], reverse=True))

    def save_model(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'classes': self.classes_
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.model_type = data.get('model_type', 'rf')
        self.classes_ = data.get('classes', None)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
