"""
Main Streamlit Application Entry Point (Router).
"""
import streamlit as st

st.set_page_config(
    page_title="AI-GEO-NGO Platform",
    page_icon="🌍",
    layout="wide"
)

# Define pages for navigation
pages = {
    "Home": [
        st.Page(
            title="Overview",
            page=lambda: home_page(),
            icon="🏠",
            default=True
        ),
        st.Page(
            "pages/09_about.py",
            title="About the Project",
            icon="ℹ️"
        )
    ],
    "Data Exploration": [
        st.Page(
            "pages/02_indicators.py",
            title="Environmental Indicators",
            icon="📊"
        ),
        st.Page(
            "pages/03_interactive_maps.py",
            title="Interactive Maps",
            icon="🗺️"
        )
    ],
    "Analysis & Priority": [
        st.Page(
            "pages/04_vulnerability.py",
            title="Climate Vulnerability",
            icon="🌡️"
        ),
        st.Page(
            "pages/05_wash_priority.py",
            title="WASH Priority",
            icon="💧"
        ),
        st.Page(
            "pages/06_statistics.py",
            title="Statistics & Trends",
            icon="📈"
        )
    ],
    "Export & Reports": [
        st.Page(
            "pages/07_reports.py",
            title="Generate Reports",
            icon="📄"
        ),
        st.Page(
            "pages/08_downloads.py",
            title="Data Downloads",
            icon="⬇️"
        )
    ]
}

def home_page():
    st.title("🌍 AI-Enabled Geospatial Decision Support System")
    st.subheader("For Climate-Resilient WASH Planning")

    st.markdown("""
    Welcome to the AI-GEO-NGO platform. 

    This system processes satellite imagery (Sentinel-2, Landsat 8) to identify environmental drivers of climate vulnerability and outputs a **WASH Intervention Priority Index** to help prioritize resource allocation.

    ### Quick Start
    Use the sidebar to navigate through the modules:
    
    1. **Data Exploration**: View raw environmental data (NDVI, NDWI, LST) and interactive base maps.
    2. **Analysis & Priority**: Explore the AI-driven Climate Vulnerability and WASH Priority predictions.
    3. **Export & Reports**: Download the final processed datasets as CSV or GeoJSON for external GIS software.
    
    ---
    *Built with Streamlit, Folium, and Google Earth Engine.*
    """)

# Configure navigation
pg = st.navigation(pages)
pg.run()
