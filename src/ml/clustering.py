"""
Unsupervised Machine Learning Module.

Implements K-Means clustering and a Weighted Environmental Risk Model
as fallback strategies when labeled ground-truth data is unavailable.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from loguru import logger
from typing import Dict, Tuple, List
import joblib
import os


class KMeansClusterer:
    """K-Means model to cluster regions by environmental vulnerability."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self.cluster_mapping = {}  
        self.is_fitted = False
        logger.debug(f"Initialized KMeansClusterer with {n_clusters} clusters.")

    def fit_predict(self, X_scaled: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
        logger.info("Fitting K-Means model.")
        labels = self.model.fit_predict(X_scaled)
        self.is_fitted = True
        self.cluster_mapping = self._map_clusters_to_vulnerability(feature_names)
        return labels, self.cluster_mapping

    def _map_clusters_to_vulnerability(self, feature_names: List[str]) -> Dict[int, str]:
        centroids = self.model.cluster_centers_
        scores = []
        for i in range(self.n_clusters):
            centroid = centroids[i]
            score = 0
            for j, feature in enumerate(feature_names):
                val = centroid[j]
                feature_lower = feature.lower()
                if any(x in feature_lower for x in ['lst', 'heat', 'urban', 'ndbi', 'risk']):
                    score += val
                elif any(x in feature_lower for x in ['ndvi', 'ndwi', 'rain', 'water']):
                    score -= val
            scores.append((i, score))
            
        scores.sort(key=lambda x: x[1])
        
        mapping = {}
        if self.n_clusters == 3:
            mapping[scores[0][0]] = 'Low'
            mapping[scores[1][0]] = 'Medium'
            mapping[scores[2][0]] = 'High'
        else:
            for idx, (c_id, score) in enumerate(scores):
                mapping[c_id] = f"Level_{idx+1}"
                
        return mapping

    def predict(self, X_scaled: np.ndarray) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        raw_labels = self.model.predict(X_scaled)
        return [self.cluster_mapping.get(label, 'Unknown') for label in raw_labels]

    def get_vulnerability_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        if not self.is_fitted or 'High' not in self.cluster_mapping.values():
            return np.zeros(X_scaled.shape[0])
            
        high_vuln_cluster_id = [k for k, v in self.cluster_mapping.items() if v == 'High'][0]
        distances = self.model.transform(X_scaled)
        dist_to_worst = distances[:, high_vuln_cluster_id]
        
        similarity = 1.0 / (1.0 + dist_to_worst)
        sim_min, sim_max = np.min(similarity), np.max(similarity)
        
        if sim_max == sim_min:
            return np.ones(X_scaled.shape[0]) * 0.5
        return (similarity - sim_min) / (sim_max - sim_min)

    def save_model(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({'model': self.model, 'mapping': self.cluster_mapping}, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.cluster_mapping = data['mapping']
        self.is_fitted = True


class WeightedRiskModel:
    """Computes a vulnerability score using an expert-defined weighted combination of features."""

    def __init__(self, weights: Dict[str, float]):
        """
        Initialize with expert weights.
        
        Args:
            weights: Dictionary mapping feature names to their weights.
                     Positive weights increase vulnerability, negative weights decrease it.
        """
        self.weights = weights
        logger.debug(f"Initialized WeightedRiskModel with weights: {weights}")

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Compute vulnerability scores and categories based on weights.
        
        Args:
            df: DataFrame containing the unscaled or scaled features.
            
        Returns:
            Tuple of (scores array, categories list).
        """
        logger.info("Computing weighted risk scores.")
        scores = np.zeros(len(df))
        
        # Calculate raw score
        for feature, weight in self.weights.items():
            if feature in df.columns:
                # Assuming data is pre-scaled or normalized 0-1 elsewhere for better comparability
                scores += df[feature].values * weight
            else:
                logger.warning(f"Feature '{feature}' missing. Skipping in weighted score.")

        # Min-max scale scores to 0-1
        s_min, s_max = np.min(scores), np.max(scores)
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.ones(len(scores)) * 0.5
            
        # Classify into categories based on percentiles or fixed thresholds
        categories = []
        for score in scores:
            if score < 0.4:
                categories.append('Low')
            elif score < 0.7:
                categories.append('Medium')
            else:
                categories.append('High')
                
        return scores, categories
