"""
Main Streamlit Application Entry Point.
"""
import streamlit as st
import os

st.set_page_config(
    page_title="AI-GEO-NGO Platform",
    page_icon="🌍",
    layout="wide"
)

st.title("AI-Enabled Geospatial Decision Support System")
st.subheader("For Climate-Resilient WASH Planning")

st.markdown("""
Welcome to the AI-GEO-NGO platform. 

This system processes satellite imagery to identify environmental drivers of climate vulnerability and outputs a **WASH Intervention Priority Index** to help prioritize resource allocation.

### Navigation
Please use the sidebar on the left to navigate to the different modules.
- **WASH Priority Map**: The primary GIS Visualization module displaying predictions, regional metrics, and map overlays.

---
*Built with Streamlit, Folium, and Google Earth Engine.*
""")

# Note: Streamlit automatically routes to files in the `pages/` directory.
