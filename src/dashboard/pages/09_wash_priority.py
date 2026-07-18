"""
Streamlit Page: Interactive WASH Priority Map.

Displays the primary GIS visualization for the project.
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium
import os
import sys

# Ensure src is in path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.dashboard.components.map_utils import build_integrated_map

st.set_page_config(page_title="WASH Priority Map", layout="wide")

st.title("🌍 WASH Intervention Priority Map")
st.markdown("""
This interactive GIS module displays the **Climate Vulnerability** and **WASH Intervention Priority** across regions.
Use the layer control in the top-right of the map to toggle environmental indicators (NDVI, NDWI, LST, etc.).
Click on any region to view detailed metrics and recommended interventions.
""")

# --- Sidebar Controls ---
st.sidebar.header("Map Controls")

# Search feature
search_query = st.sidebar.text_input("Search Region ID (e.g., REG-101)")

# Time slider (simulated filtering if historical data existed)
time_period = st.sidebar.slider(
    "Select Time Period",
    min_value=2015,
    max_value=2026,
    value=2026,
    step=1,
    help="Filters the dataset to show predictions for a specific year."
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Hover over a region for quick stats. Click for full details.")

# --- Load Data ---
@st.cache_data
def load_data(year):
    """
    Load the prediction output GeoJSON.
    In a real scenario, we might query the FastAPI backend or database here,
    filtering by the selected year.
    """
    # For demonstration, we attempt to load the GeoJSON created by the Pipeline Engine.
    # If not found, we create a mock GeoDataFrame for visualization testing.
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/predictions/predictions_and_priorities.geojson'))
    
    try:
        gdf = gpd.read_file(filepath)
        return gdf
    except Exception as e:
        st.warning(f"Could not load real data from {filepath}. Generating mock data for visualization.")
        
        # Create a mock polygon (square) near Kenya/Turkana
        from shapely.geometry import Polygon
        poly1 = Polygon([(35, 3), (36, 3), (36, 4), (35, 4)])
        poly2 = Polygon([(36, 3), (37, 3), (37, 4), (36, 4)])
        
        data = {
            'geographic_id': ['REG-101', 'REG-102'],
            'priority_score': [85.5, 32.1],
            'priority_class': ['High Priority', 'Low Priority'],
            'vulnerability_score': [0.92, 0.31],
            'vulnerability_category': ['High', 'Low'],
            'explanation': ['Severe water loss and high heat stress.', 'Stable environmental conditions.'],
            'ndvi': [0.15, 0.65],
            'ndwi': [-0.2, 0.4],
            'lst': [38.5, 25.2],
            'urban_growth_rate': [5.2, 1.1],
            'recommendation': ['Immediate Action Required', 'Routine Monitoring']
        }
        
        gdf = gpd.GeoDataFrame(data, geometry=[poly1, poly2], crs="EPSG:4326")
        return gdf


# Load the data based on slider
gdf = load_data(time_period)

# Search Filtering
if search_query:
    filtered_gdf = gdf[gdf['geographic_id'].str.contains(search_query, case=False, na=False)]
    if not filtered_gdf.empty:
        gdf = filtered_gdf
    else:
        st.sidebar.error(f"No region found matching '{search_query}'")

# --- Render Map ---
if not gdf.empty:
    with st.spinner("Rendering map layers..."):
        m = build_integrated_map(gdf)
        
        # Display the map using streamlit-folium
        st_data = st_folium(
            m,
            width=1200,
            height=700,
            returned_objects=["last_active_drawing", "last_clicked"]
        )
        
    # --- Interactivity: Click handling ---
    if st_data and st_data.get("last_clicked"):
        st.subheader("Selected Region Details")
        # Find the row in the GeoDataFrame that matches the clicked coordinates (roughly)
        # Note: In a production app with precise polygons, it's better to use returned feature IDs
        # if using folium GeoJson, but click coords work as a fallback.
        lat, lon = st_data["last_clicked"]["lat"], st_data["last_clicked"]["lng"]
        
        # Spatial join to find which polygon was clicked
        from shapely.geometry import Point
        click_point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        clicked_region = gpd.sjoin(gdf, click_point, how='inner', predicate='intersects')
        
        if not clicked_region.empty:
            row = clicked_region.iloc[0]
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Region ID", row.get('geographic_id', 'Unknown'))
                st.metric("Priority Score", f"{row.get('priority_score', 0):.1f}/100")
                
            with col2:
                st.metric("Priority Class", row.get('priority_class', 'Unknown'))
                vuln = row.get('vulnerability_score', 0)
                st.metric("Vulnerability", f"{vuln:.2f} ({row.get('vulnerability_category', 'Unknown')})")
                
            with col3:
                # Mock a recommended intervention level based on priority class
                p_class = row.get('priority_class', '')
                if 'High' in p_class:
                    rec = "🚨 Immediate Intervention"
                elif 'Medium' in p_class:
                    rec = "⚠️ Proactive Planning"
                else:
                    rec = "✅ Routine Monitoring"
                st.metric("Recommendation", rec)
                
            st.markdown(f"**Explanation:** {row.get('explanation', 'No explanation available.')}")
            
            # Display Raw Indicators
            st.markdown("#### Environmental Indicators")
            i_col1, i_col2, i_col3, i_col4 = st.columns(4)
            i_col1.metric("NDVI", round(row.get('ndvi', 0), 3))
            i_col2.metric("NDWI", round(row.get('ndwi', 0), 3))
            i_col3.metric("LST (°C)", round(row.get('lst', 0), 1))
            i_col4.metric("Urban Growth %", round(row.get('urban_growth_rate', 0), 2))
            
else:
    st.error("No data available to display on the map.")
