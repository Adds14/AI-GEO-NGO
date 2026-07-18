"""
About Page.
"""
import streamlit as st

st.set_page_config(page_title="About the Project", layout="wide")
st.title("ℹ️ About the Project")

st.markdown("""
### AI-Enabled Geospatial Decision Support System

This platform was developed as an NGO initiative to provide data-driven insights into climate vulnerability and prioritize WASH (Water, Sanitation, and Hygiene) interventions globally.

#### Methodology
1. **Data Ingestion**: Satellite imagery is queried from Google Earth Engine (Sentinel-2, Landsat 8/9).
2. **Feature Extraction**: Zonal statistics are computed for key environmental indicators (NDVI, LST, NDWI, NDBI).
3. **Machine Learning**: Random Forest and XGBoost models predict Climate Vulnerability based on historical data.
4. **WASH Priority**: A weighted scoring engine combines climate vulnerability with socio-environmental risk factors to produce a 0-100 Intervention Priority Index.

#### Technologies Used
- **Frontend**: Streamlit, Folium, Plotly
- **Backend**: FastAPI, GeoPandas, Google Earth Engine API
- **Machine Learning**: Scikit-learn, XGBoost

*Version 1.0.0*
""")
