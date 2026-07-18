"""
File Input/Output utilities for tabular and geospatial data.
"""
import os
from pathlib import Path
from typing import Union, Tuple
import pandas as pd
import geopandas as gpd
import rasterio

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_dataframe(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save dataframe to csv, excel, or parquet based on extension."""
    path = Path(path)
    ensure_dir(path.parent)
    
    ext = path.suffix.lower()
    if ext == '.csv':
        df.to_csv(path, index=False)
    elif ext == '.xlsx':
        df.to_excel(path, index=False)
    elif ext == '.parquet':
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Load dataframe from supported formats."""
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext == '.csv':
        return pd.read_csv(path)
    elif ext == '.xlsx':
        return pd.read_excel(path)
    elif ext == '.parquet':
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def save_geotiff(data, path: Union[str, Path], profile: dict) -> None:
    """Save array as GeoTIFF."""
    path = Path(path)
    ensure_dir(path.parent)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)

def load_geotiff(path: Union[str, Path]) -> Tuple[object, dict]:
    """Load GeoTIFF and return data and profile."""
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile
    return data, profile

def save_geojson(gdf: gpd.GeoDataFrame, path: Union[str, Path]) -> None:
    """Save GeoDataFrame to GeoJSON."""
    path = Path(path)
    ensure_dir(path.parent)
    gdf.to_file(path, driver="GeoJSON")

def load_geojson(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """Load GeoDataFrame from GeoJSON."""
    return gpd.read_file(Path(path))
