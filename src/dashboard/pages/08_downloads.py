"""
Data Downloads Page.
"""
import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Data Downloads", layout="wide")
st.title("⬇️ Data Downloads")
st.markdown("Export processed datasets, model predictions, and spatial bounds for external use.")

st.subheader("1. Feature Datasets (CSV)")
st.markdown("The complete engineered tabular datasets used for model training and inference.")
df_mock = pd.DataFrame({'geographic_id': ['REG-1'], 'ndvi': [0.5], 'lst': [30.2], 'priority_score': [45.1]})
csv_data = df_mock.to_csv(index=False)
st.download_button(label="Download Features (CSV)", data=csv_data, file_name="features.csv", mime="text/csv")

st.subheader("2. Geospatial Predictions (GeoJSON)")
st.markdown("The final polygons with all associated scores, ready to be loaded into QGIS or ArcGIS.")
geojson_mock = json.dumps({"type": "FeatureCollection", "features": []})
st.download_button(label="Download Predictions (GeoJSON)", data=geojson_mock, file_name="predictions.geojson", mime="application/json")
