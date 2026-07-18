"""
Environmental Indicators Data View.
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import os
import numpy as np

st.set_page_config(page_title="Environmental Indicators", layout="wide")
st.title("📊 Environmental Indicators")
st.markdown("View and filter the raw environmental metrics extracted from satellite imagery.")

@st.cache_data
def load_mock_data():
    np.random.seed(42)
    n = 20
    region_names = [f"Grid-{i}" for i in range(1, n+1)]
    
    data = {
        'Region ID': region_names,
        'NDVI (Vegetation)': np.random.uniform(0.1, 0.8, n),
        'NDWI (Water)': np.random.uniform(-0.3, 0.5, n),
        'LST (°C)': np.random.uniform(25.0, 45.0, n),
        'NDBI (Urban)': np.random.uniform(-0.2, 0.4, n),
        'Rainfall Anomaly': np.random.uniform(-50, 50, n),
    }
    return pd.DataFrame(data)

df = load_mock_data()

# Filters
col1, col2 = st.columns(2)
with col1:
    search = st.text_input("Search Region")
with col2:
    sort_by = st.selectbox("Sort By", options=df.columns[1:])

if search:
    df = df[df['Region ID'].str.contains(search, case=False)]
    
df = df.sort_values(by=sort_by, ascending=False)

st.dataframe(
    df,
    use_container_width=True,
    height=600
)
