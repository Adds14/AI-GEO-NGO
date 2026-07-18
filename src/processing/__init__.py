"""
Geospatial Processing Module.

Provides processors for computing environmental indicators
and derived change metrics from satellite imagery.

Core Indicators:
    - NDVI (Normalized Difference Vegetation Index)
    - NDWI (Normalized Difference Water Index)
    - NDBI (Normalized Difference Built-up Index)
    - LST  (Land Surface Temperature)

Derived Indicators:
    - Vegetation Change
    - Water Body Change
    - Urban Expansion
    - Heat Stress Index
    - Rainfall Anomaly
"""
from src.processing.ndvi import NDVIProcessor
from src.processing.ndwi import NDWIProcessor
from src.processing.ndbi import NDBIProcessor
from src.processing.lst import LSTProcessor
from src.processing.vegetation_change import VegetationChangeProcessor
from src.processing.water_change import WaterChangeProcessor
from src.processing.urban_expansion import UrbanExpansionProcessor
from src.processing.heat_stress import HeatStressProcessor
from src.processing.rainfall_anomaly import RainfallAnomalyProcessor

__all__ = [
    'NDVIProcessor',
    'NDWIProcessor',
    'NDBIProcessor',
    'LSTProcessor',
    'VegetationChangeProcessor',
    'WaterChangeProcessor',
    'UrbanExpansionProcessor',
    'HeatStressProcessor',
    'RainfallAnomalyProcessor',
]
