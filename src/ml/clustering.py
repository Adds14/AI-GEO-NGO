"""
Unsupervised Machine Learning Module.

Implements K-Means clustering as the primary fallback strategy 
when labeled ground-truth data is unavailable.
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
        """
        Initialize the KMeans clusterer.

        Args:
            n_clusters: Number of clusters (default 3: Low, Medium, High).
            random_state: Seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self.cluster_mapping = {}  # Maps cluster ID (0,1,2) to Vulnerability Category
        self.is_fitted = False
        logger.debug(f"Initialized KMeansClusterer with {n_clusters} clusters.")

    def fit_predict(self, X_scaled: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Fit the K-Means model and predict clusters for the training data.
        Automatically maps clusters to vulnerability levels based on centroid values.

        Args:
            X_scaled: Scaled feature matrix.
            feature_names: List of feature names corresponding to X_scaled columns.

        Returns:
            Tuple containing:
                - Cluster labels (np.ndarray)
                - Mapping dictionary of {cluster_id: 'Category'}
        """
        logger.info("Fitting K-Means model.")
        labels = self.model.fit_predict(X_scaled)
        self.is_fitted = True
        
        self.cluster_mapping = self._map_clusters_to_vulnerability(feature_names)
        logger.info(f"Cluster mapping derived: {self.cluster_mapping}")
        
        return labels, self.cluster_mapping

    def _map_clusters_to_vulnerability(self, feature_names: List[str]) -> Dict[int, str]:
        """
        Heuristic method to map cluster centroids to 'Low', 'Medium', 'High' vulnerability.
        
        Assumes higher values in negative indicators (LST, Heat Stress, NDBI) 
        and lower values in positive indicators (NDVI, NDWI) equal higher vulnerability.
        """
        centroids = self.model.cluster_centers_
        
        # Calculate a simple "vulnerability proxy" for each centroid
        # This requires domain knowledge to know which features indicate vulnerability
        scores = []
        for i in range(self.n_clusters):
            centroid = centroids[i]
            score = 0
            
            for j, feature in enumerate(feature_names):
                val = centroid[j]
                feature_lower = feature.lower()
                # Features where HIGH value = BAD
                if any(x in feature_lower for x in ['lst', 'heat', 'urban', 'ndbi']):
                    score += val
                # Features where LOW value = BAD
                elif any(x in feature_lower for x in ['ndvi', 'ndwi', 'rain', 'water']):
                    score -= val
            scores.append((i, score))
            
        # Sort clusters by proxy score
        scores.sort(key=lambda x: x[1])
        
        mapping = {}
        if self.n_clusters == 3:
            mapping[scores[0][0]] = 'Low'
            mapping[scores[1][0]] = 'Medium'
            mapping[scores[2][0]] = 'High'
        else:
            # Fallback for dynamic cluster sizes
            for idx, (c_id, score) in enumerate(scores):
                mapping[c_id] = f"Level_{idx+1}"
                
        return mapping

    def predict(self, X_scaled: np.ndarray) -> List[str]:
        """
        Predict vulnerability categories for new data.

        Args:
            X_scaled: Scaled feature matrix.

        Returns:
            List of string vulnerability categories.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
            
        logger.info("Predicting clusters for new data.")
        raw_labels = self.model.predict(X_scaled)
        
        mapped_labels = [self.cluster_mapping.get(label, 'Unknown') for label in raw_labels]
        return mapped_labels

    def get_vulnerability_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Generates continuous vulnerability scores (0.0 - 1.0) based on distance 
        to the 'High' vulnerability cluster center.
        """
        if not self.is_fitted or 'High' not in self.cluster_mapping.values():
            # Fallback if categories are non-standard
            return np.zeros(X_scaled.shape[0])
            
        # Find the ID of the 'High' vulnerability cluster
        high_vuln_cluster_id = [k for k, v in self.cluster_mapping.items() if v == 'High'][0]
        
        distances = self.model.transform(X_scaled)
        
        # Distance to the 'High' cluster (index in distances array corresponds to cluster_id)
        dist_to_worst = distances[:, high_vuln_cluster_id]
        
        # Invert and normalize distances so closer to 'High' centroid = higher score
        # max_dist = np.max(dist_to_worst) if np.max(dist_to_worst) > 0 else 1.0
        # normalized_score = 1.0 - (dist_to_worst / max_dist)
        
        # Alternative scaling based on all cluster distances
        # Transform distance into similarity
        similarity = 1.0 / (1.0 + dist_to_worst)
        
        # Min-max scale the similarities to 0-1
        sim_min = np.min(similarity)
        sim_max = np.max(similarity)
        
        if sim_max == sim_min:
            scores = np.ones(X_scaled.shape[0]) * 0.5
        else:
            scores = (similarity - sim_min) / (sim_max - sim_min)
            
        return scores

    def save_model(self, filepath: str):
        """Save the K-Means model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'mapping': self.cluster_mapping
        }, filepath)
        logger.info(f"K-Means model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the K-Means model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        self.model = data['model']
        self.cluster_mapping = data['mapping']
        self.is_fitted = True
        logger.info(f"K-Means model loaded from {filepath}")
