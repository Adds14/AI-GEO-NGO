"""
External Data Integration Module.

Merges satellite-derived features with external datasets such as:
- Population Density
- Rainfall
- Elevation (DEM)
- Flood Susceptibility
- Existing WASH Infrastructure

Handles missing values and validates the final integrated dataset.
"""
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Union
import os


class DataIntegrator:
    """Integrates satellite features with external datasets."""

    def __init__(self, id_column: str = 'geographic_id'):
        """
        Initialize the DataIntegrator.

        Args:
            id_column (str): The common column to join datasets on.
        """
        self.id_column = id_column
        logger.debug(f"Initialized DataIntegrator with join key: {self.id_column}")

    def load_dataset(self, filepath_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Helper to load a dataset from a filepath or return the DataFrame.

        Args:
            filepath_or_df: File path to CSV or a pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        if isinstance(filepath_or_df, pd.DataFrame):
            return filepath_or_df.copy()
        elif isinstance(filepath_or_df, str):
            logger.info(f"Loading dataset from {filepath_or_df}")
            if not os.path.exists(filepath_or_df):
                logger.error(f"File not found: {filepath_or_df}")
                raise FileNotFoundError(f"File not found: {filepath_or_df}")
            return pd.read_csv(filepath_or_df)
        else:
            raise ValueError("Input must be a filepath string or a pandas DataFrame.")

    def merge_datasets(self, 
                       satellite_features: Union[str, pd.DataFrame], 
                       external_datasets: Dict[str, Union[str, pd.DataFrame]],
                       how: str = 'left') -> pd.DataFrame:
        """
        Merge satellite features with external datasets.

        Args:
            satellite_features: The base dataset of satellite features.
            external_datasets: Dictionary mapping dataset names to paths or DataFrames.
            how: Type of merge to be performed. Default is 'left' to keep all satellite regions.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        logger.info(f"Merging satellite features with {len(external_datasets)} external datasets.")
        
        # Load base dataset
        df_merged = self.load_dataset(satellite_features)
        
        if self.id_column not in df_merged.columns:
            logger.error(f"Base dataset is missing the join key '{self.id_column}'.")
            raise KeyError(f"Base dataset missing '{self.id_column}'")

        # Merge external datasets
        for name, data in external_datasets.items():
            logger.info(f"Merging external dataset: {name}")
            df_ext = self.load_dataset(data)
            
            if self.id_column not in df_ext.columns:
                logger.warning(f"External dataset '{name}' is missing the join key '{self.id_column}'. Skipping.")
                continue
                
            # Perform merge
            df_merged = pd.merge(df_merged, df_ext, on=self.id_column, how=how, suffixes=('', f'_{name}'))
            
        logger.info(f"Merged dataset shape: {df_merged.shape}")
        return df_merged

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the integrated dataset.

        Args:
            df (pd.DataFrame): The integrated dataset.
            strategy (str): Strategy to fill numeric NaNs ('median', 'mean', 'zero', 'drop').

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        logger.info(f"Handling missing values using strategy: {strategy}")
        df_clean = df.copy()
        
        # Log missing values before cleaning
        missing_counts = df_clean.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if not missing_cols.empty:
            logger.debug(f"Missing values before cleaning:\n{missing_cols}")
        else:
            logger.debug("No missing values found.")
            return df_clean

        if strategy == 'drop':
            df_clean = df_clean.dropna()
        else:
            # Separate numeric and categorical columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric columns
            if strategy == 'median':
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            elif strategy == 'mean':
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
            elif strategy == 'zero':
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
            else:
                raise ValueError(f"Unknown numeric imputation strategy: {strategy}")
                
            # Fill categorical columns with mode or 'Unknown'
            for col in categorical_cols:
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    
        # Verify no missing values remain
        remaining_missing = df_clean.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"{remaining_missing} missing values still remain after cleaning!")
        else:
            logger.info("Successfully handled all missing values.")
            
        return df_clean

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the integrated dataset for ML readiness.
        Checks for infinite values, missing values, and empty dataframes.

        Args:
            df (pd.DataFrame): The integrated dataset.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        logger.info("Validating integrated dataset.")
        
        if df.empty:
            logger.error("Validation failed: DataFrame is empty.")
            return False
            
        if df.isnull().sum().sum() > 0:
            logger.error("Validation failed: DataFrame contains missing values (NaN).")
            return False
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if np.isinf(df[numeric_cols]).values.sum() > 0:
            logger.error("Validation failed: DataFrame contains infinite values.")
            return False
            
        if self.id_column not in df.columns:
            logger.error(f"Validation failed: Join key '{self.id_column}' is missing.")
            return False
            
        logger.success("Dataset validation passed. Ready for Machine Learning.")
        return True

    def integrate(self, 
                  satellite_features: Union[str, pd.DataFrame], 
                  external_datasets: Dict[str, Union[str, pd.DataFrame]],
                  output_csv_path: Optional[str] = None,
                  imputation_strategy: str = 'median') -> pd.DataFrame:
        """
        End-to-end integration pipeline: merge, clean, validate, and optionally save.

        Args:
            satellite_features: Satellite feature dataset.
            external_datasets: Dictionary of external datasets.
            output_csv_path: Path to save the final integrated dataset.
            imputation_strategy: Strategy for handling missing values.

        Returns:
            pd.DataFrame: Cleaned, validated, and integrated dataset.
        """
        logger.info("Starting Data Integration Pipeline.")
        
        # 1. Merge
        merged_df = self.merge_datasets(satellite_features, external_datasets)
        
        # 2. Clean (Handle Missing Values)
        clean_df = self.handle_missing_values(merged_df, strategy=imputation_strategy)
        
        # 3. Validate
        is_valid = self.validate_data(clean_df)
        if not is_valid:
            logger.warning("Data validation flagged issues. Please inspect the dataset.")
            
        # 4. Export
        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            clean_df.to_csv(output_csv_path, index=False)
            logger.success(f"Integrated dataset saved to {output_csv_path}")
            
        return clean_df
