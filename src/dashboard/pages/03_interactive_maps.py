"""
Interactive Maps Page.
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium
import os
import sys
import numpy as np
from shapely.geometry import Polygon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.dashboard.components.map_utils import create_base_map, add_choropleth

st.set_page_config(page_title="Interactive Maps", layout="wide")
st.title("🗺️ Interactive Maps")
st.markdown("Explore individual environmental indicators spatially.")

@st.cache_data
def load_mock_geodata():
    # Grid generation
    minx, miny, maxx, maxy = -17, -34, 51, 37
    x = np.linspace(minx, maxx, 6)
    y = np.linspace(miny, maxy, 5)
    polygons = []
    names = []
    idx = 1
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            polygons.append(Polygon([(x[i], y[j]), (x[i+1], y[j]), (x[i+1], y[j+1]), (x[i], y[j+1])]))
            names.append(f"Grid-{idx}")
            idx += 1
            
    np.random.seed(42)
    n = len(polygons)
    data = {
        'geographic_id': names,
        'ndvi': np.random.uniform(0.1, 0.8, n),
        'lst': np.random.uniform(25.0, 45.0, n),
        'ndwi': np.random.uniform(-0.3, 0.5, n),
    }
    return gpd.GeoDataFrame(data, geometry=polygons, crs="EPSG:4326")

gdf = load_mock_geodata()

layer_choice = st.selectbox("Select Layer to Display", ["NDVI (Vegetation)", "LST (Temperature)", "NDWI (Water)"])

with st.spinner("Rendering Map..."):
    bounds = gdf.total_bounds
    m = create_base_map((bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2, 4)
    
    if layer_choice == "NDVI (Vegetation)":
        add_choropleth(m, gdf, 'ndvi', 'NDVI', 'YlGn', 'NDVI', show=True)
    elif layer_choice == "LST (Temperature)":
        add_choropleth(m, gdf, 'lst', 'LST', 'OrRd', 'LST (°C)', show=True)
    elif layer_choice == "NDWI (Water)":
        add_choropleth(m, gdf, 'ndwi', 'NDWI', 'Blues', 'NDWI', show=True)
        
    st_folium(m, width=1200, height=600)
