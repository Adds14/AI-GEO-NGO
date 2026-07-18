"""
Google Earth Engine specific configuration.
Contains collection IDs, band mappings, and processing thresholds.
"""

# GEE Collection IDs
GEE_COLLECTIONS = {
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "landsat8": "LANDSAT/LC08/C02/T1_L2",
    "landsat9": "LANDSAT/LC09/C02/T1_L2",
    "srtm": "USGS/SRTMGL1_003",
    "chirps": "UCSB-CHG/CHIRPS/DAILY"
}

# Band mappings per sensor
BAND_MAPPINGS = {
    "sentinel2": {
        "blue": "B2",
        "green": "B3",
        "red": "B4",
        "nir": "B8",
        "swir1": "B11",
        "swir2": "B12"
    },
    "landsat8": {
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "swir2": "SR_B7",
        "thermal": "ST_B10"
    }
}

# Default processing parameters
GEE_DEFAULTS = {
    "cloud_masking": {
        "sentinel2_cloud_probability_threshold": 20,
        "landsat_cloud_cover_max": 20
    },
    "scale": {
        "sentinel2": 10,  # meters
        "landsat": 30,    # meters
        "srtm": 30,       # meters
        "chirps": 5566    # meters (~0.05 degrees)
    },
    "date_ranges": {
        "default_years_back": 1
    }
}
