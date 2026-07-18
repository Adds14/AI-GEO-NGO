"""
GEE data ingestion module.
Provides tools to authenticate with Google Earth Engine and load various environmental datasets.
"""

from .gee_client import GEEClient, get_gee_client
from .sentinel2 import Sentinel2Loader
from .landsat import LandsatLoader
from .srtm import SRTMLoader
from .chirps import CHIRPSLoader
from .boundaries import BoundaryLoader

__all__ = [
    'GEEClient',
    'get_gee_client',
    'Sentinel2Loader',
    'LandsatLoader',
    'SRTMLoader',
    'CHIRPSLoader',
    'BoundaryLoader'
]
