"""
Shared constants for the AI-GEO-NGO project.
"""

# Environmental Indicator Names
INDICATOR_NAMES = [
    "ndvi", "lst", "ndwi", "ndbi",
    "vegetation_change", "urban_growth", "water_body_change",
    "heat_stress_index", "rainfall_anomaly"
]

# Core indicator names (computed from satellite imagery)
CORE_INDICATORS = ["ndvi", "lst", "ndwi", "ndbi"]

# Derived indicator names (computed from core indicators over time)
DERIVED_INDICATORS = [
    "vegetation_change", "urban_growth", "water_body_change",
    "heat_stress_index", "rainfall_anomaly"
]

# ML Feature columns
FEATURE_COLUMNS = [
    "mean_ndvi", "mean_lst", "mean_ndwi", "mean_ndbi",
    "vegetation_change", "urban_growth_rate", "water_body_change",
    "elevation", "rainfall", "population_density",
    "distance_to_water", "heat_stress_index", "rainfall_anomaly"
]

# Climate Vulnerability Risk Levels
RISK_LEVELS = {
    "low": {"label": "Low Risk", "color": "#2ecc71", "range": (0.0, 0.33)},
    "medium": {"label": "Medium Risk", "color": "#f39c12", "range": (0.34, 0.66)},
    "high": {"label": "High Risk", "color": "#e74c3c", "range": (0.67, 1.0)},
}

# WASH Intervention Priority Levels (0-100 scale)
WASH_PRIORITY_LEVELS = {
    "low": {"label": "Low Priority", "color": "#27ae60", "range": (0, 40)},
    "medium": {"label": "Medium Priority", "color": "#e67e22", "range": (41, 70)},
    "high": {"label": "High Priority", "color": "#c0392b", "range": (71, 100)},
}

# CRS Constants
CRS_WGS84 = "EPSG:4326"
CRS_WEB_MERCATOR = "EPSG:3857"
CRS_UTM_DEFAULT = "EPSG:32643"  # UTM Zone 43N (adjust per study area)

# Visualization Color Maps
COLOR_MAPS = {
    "ndvi": "RdYlGn",
    "lst": "YlOrRd",
    "ndwi": "Blues",
    "ndbi": "Oranges",
    "vulnerability": "RdYlGn_r",
    "wash_priority": "RdYlGn_r",
    "vegetation_change": "RdYlGn",
    "water_body_change": "PuBu",
    "urban_growth": "YlOrBr",
    "heat_stress_index": "hot_r",
    "rainfall_anomaly": "BrBG",
}

# Band name mappings
SENTINEL2_BANDS = {
    "blue": "B2", "green": "B3", "red": "B4",
    "nir": "B8", "swir1": "B11", "swir2": "B12",
    "scl": "SCL",
}

LANDSAT_BANDS = {
    "blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
    "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7",
    "thermal": "ST_B10", "qa": "QA_PIXEL",
}

# Default ML parameters
ML_DEFAULTS = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "n_clusters": 3,  # For K-Means when labels unavailable
}
