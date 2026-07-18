"""
Shared constants across the application.
"""

# Risk Levels and their visualization colors
RISK_LEVELS = {
    "Low": "#00FF00",       # Green
    "Moderate": "#FFFF00",  # Yellow
    "High": "#FFA500",      # Orange
    "Very High": "#FF0000"  # Red
}

# Key indicators
INDICATOR_NAMES = [
    "NDVI",
    "NDWI",
    "NDBI",
    "LST",
    "Precipitation",
    "Elevation",
    "Slope"
]

# Coordinate Reference Systems
CRS_WGS84 = "EPSG:4326"
CRS_WEB_MERCATOR = "EPSG:3857"

# Feature columns for Machine Learning
FEATURE_COLUMNS = [
    "ndvi_mean",
    "ndwi_mean",
    "ndbi_mean",
    "lst_mean",
    "precipitation_sum",
    "elevation_mean",
    "slope_mean"
]

# Color maps for visualization
COLOR_MAPS = {
    "ndvi": ["#FFFFFF", "#CE7E45", "#DF923D", "#F1B555", "#FCD163", "#99B718", "#74A901", "#66A000", "#529400", "#3E8601", "#207401", "#056201", "#004C00", "#023B01", "#012E01", "#011D01", "#011301"],
    "lst": ["#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"],
    "precipitation": ["#FFFFFF", "#00FFFF", "#0000FF"]
}

# Band name mappings
BAND_NAMES = {
    "NDVI": "NDVI",
    "LST": "LST_Day_1km",
    "Precipitation": "precipitation"
}
