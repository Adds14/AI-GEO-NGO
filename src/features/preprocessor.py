"""
Data Preprocessing Pipeline Module.

Handles missing value imputation, outlier detection, feature scaling, 
feature engineering, and dataset validation for machine learning readiness.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
from typing import Dict, Tuple, Optional, List
import os


class DataPreprocessor:
    """Comprehensive preprocessing pipeline for tabular feature datasets."""

    def __init__(self, id_col: str = 'geographic_id', target_col: Optional[str] = None):
        """
        Initialize the DataPreprocessor.

        Args:
            id_col: Column containing unique region identifiers.
            target_col: Optional column containing ground-truth labels.
        """
        self.id_col = id_col
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.is_fitted = False
        logger.debug("Initialized DataPreprocessor.")

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info(f"Handling missing values using strategy: {strategy}")
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if strategy == 'median':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=numeric_cols)
            
        return df_clean

    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect and cap outliers in numeric columns using the Interquartile Range (IQR) method.
        """
        logger.info(f"Detecting and capping outliers using {method.upper()} method.")
        df_clean = df.copy()
        
        if columns is None:
            columns = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in [self.id_col, self.target_col]]
            
        if method == 'iqr':
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of dropping to preserve region data
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                
        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer derived features such as changes and composite risk scores.
        Assumes presence of current and baseline/historical columns.
        """
        logger.info("Engineering new features (Changes and Risk Scores).")
        df_eng = df.copy()
        
        # Safely compute changes if columns exist, otherwise mock or ignore
        def safe_compute_change(col_curr, col_base, out_col):
            if col_curr in df_eng.columns and col_base in df_eng.columns:
                df_eng[out_col] = df_eng[col_curr] - df_eng[col_base]
                logger.debug(f"Computed {out_col}")

        def safe_compute_rate(col_curr, col_base, out_col):
            if col_curr in df_eng.columns and col_base in df_eng.columns:
                # Avoid division by zero
                base = df_eng[col_base].replace(0, np.nan)
                df_eng[out_col] = ((df_eng[col_curr] - df_eng[col_base]) / base) * 100
                df_eng[out_col] = df_eng[out_col].fillna(0) # Handle where base was 0
                logger.debug(f"Computed {out_col}")
                
        # Engineer standard expected features
        safe_compute_change('ndvi_current', 'ndvi_baseline', 'ndvi_change')
        safe_compute_change('lst_current', 'lst_baseline', 'lst_change')
        safe_compute_change('ndwi_current', 'ndwi_baseline', 'ndwi_change')
        safe_compute_rate('urban_current', 'urban_baseline', 'urban_growth_rate')
        
        # Environmental Risk Score computation
        # A simple composite score based on available adverse indicators
        risk_components = []
        if 'lst_change' in df_eng.columns:
            # Positive LST change increases risk
            risk_components.append(df_eng['lst_change'].fillna(0))
        if 'ndvi_change' in df_eng.columns:
            # Negative NDVI change increases risk, so we subtract
            risk_components.append(-df_eng['ndvi_change'].fillna(0))
        if 'ndwi_change' in df_eng.columns:
            # Negative NDWI change (water loss) increases risk
            risk_components.append(-df_eng['ndwi_change'].fillna(0))
            
        if risk_components:
            # Normalize and sum
            normalized_components = [(c - c.min()) / (c.max() - c.min() + 1e-9) for c in risk_components]
            df_eng['environmental_risk_score'] = sum(normalized_components) / len(normalized_components)
            logger.info("Computed environmental_risk_score.")
        else:
            logger.warning("Could not compute environmental_risk_score due to missing underlying change variables.")
            
        return df_eng

    def scale_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        logger.info("Scaling features.")
        df_scaled = df.copy()
        
        if columns is None:
            columns = [c for c in df_scaled.select_dtypes(include=[np.number]).columns if c not in [self.id_col, self.target_col]]
            
        if not self.is_fitted:
            df_scaled[columns] = self.scaler.fit_transform(df_scaled[columns])
            self.is_fitted = True
        else:
            df_scaled[columns] = self.scaler.transform(df_scaled[columns])
            
        return df_scaled

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate the dataset is ready for ML."""
        logger.info("Validating preprocessed dataset.")
        
        if df.isnull().sum().sum() > 0:
            logger.error("Validation failed: NaN values remain.")
            return False
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if np.isinf(df[numeric_cols]).values.sum() > 0:
            logger.error("Validation failed: Infinite values found.")
            return False
            
        if self.id_col not in df.columns:
            logger.error(f"Validation failed: Join key '{self.id_col}' is missing.")
            return False
            
        logger.success("Dataset validation passed.")
        return True

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test sets."""
        logger.info(f"Splitting data (test_size={test_size}).")
        
        if self.target_col and self.target_col in df.columns:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[self.target_col])
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df

    def run_pipeline(self, 
                     input_filepath: str, 
                     output_dir: str, 
                     test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full preprocessing pipeline.
        
        Args:
            input_filepath: Path to the raw integrated CSV.
            output_dir: Directory to save the processed splits.
            test_size: Proportion of dataset to include in test split.
            
        Returns:
            Tuple of (train_df, test_df).
        """
        logger.info("Starting Data Preprocessing Pipeline.")
        
        # 1. Load Data
        df = pd.read_csv(input_filepath)
        
        # 2. Impute missing
        df = self.handle_missing_values(df)
        
        # 3. Engineer features
        df = self.engineer_features(df)
        
        # 4. Handle outliers
        df = self.detect_outliers(df)
        
        # 5. Scale features
        df = self.scale_features(df)
        
        # 6. Validate
        if not self.validate_dataset(df):
            raise ValueError("Dataset failed validation after preprocessing.")
            
        # 7. Split
        train_df, test_df = self.split_data(df, test_size=test_size)
        
        # 8. Export
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train_features.csv')
        test_path = os.path.join(output_dir, 'test_features.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.success(f"Processed datasets exported to {output_dir}")
        
        return train_df, test_df
