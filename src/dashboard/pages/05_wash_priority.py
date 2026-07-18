"""
WASH Priority Page.
Replaces the old 09_wash_priority.py.
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

st.set_page_config(page_title="WASH Priority", layout="wide")
st.title("💧 WASH Intervention Priority")
st.markdown("Discover exactly where WASH interventions are needed most based on climate drivers.")

@st.cache_data
def load_mock_priority():
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
            
    np.random.seed(200)
    n = len(polygons)
    scores = np.random.uniform(10, 95, n)
    categories = ['High Priority' if s > 70 else 'Medium Priority' if s > 40 else 'Low Priority' for s in scores]
    
    data = {
        'geographic_id': names,
        'priority_score': scores,
        'priority_class': categories,
        'explanation': [f"Simulated priority factors for Region {i+1}." for i in range(n)]
    }
    return gpd.GeoDataFrame(data, geometry=polygons, crs="EPSG:4326")

gdf = load_mock_priority()

col1, col2 = st.columns([2, 1])

with col1:
    with st.spinner("Rendering Map..."):
        bounds = gdf.total_bounds
        m = create_base_map((bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2, 4)
        add_choropleth(m, gdf, 'priority_score', 'WASH Priority', 'YlOrRd', 'Priority Score (0-100)', show=True)
        st_data = st_folium(m, width=800, height=500, returned_objects=["last_clicked"])

with col2:
    st.subheader("Intervention Details")
    if st_data and st_data.get("last_clicked"):
        lat, lon = st_data["last_clicked"]["lat"], st_data["last_clicked"]["lng"]
        from shapely.geometry import Point
        click_point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        clicked_region = gpd.sjoin(gdf, click_point, how='inner', predicate='intersects')
        
        if not clicked_region.empty:
            row = clicked_region.iloc[0]
            st.metric("Region ID", row.get('geographic_id', 'Unknown'))
            st.metric("Priority Score", f"{row.get('priority_score', 0):.1f}/100")
            st.metric("Priority Class", row.get('priority_class', 'Unknown'))
            st.markdown(f"**Explanation:**\n{row.get('explanation', '')}")
            
            p_class = row.get('priority_class', '')
            if 'High' in p_class:
                st.error("🚨 Immediate Action Required")
            elif 'Medium' in p_class:
                st.warning("⚠️ Proactive Planning Needed")
            else:
                st.success("✅ Routine Monitoring")
        else:
            st.info("Click inside a colored region on the map.")
    else:
        st.info("Click on a region on the map to see details.")
