"""
Statistics & Trends Page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Statistics & Trends", layout="wide")
st.title("📈 Statistics & Trends")
st.markdown("Analyze distributions and correlations in the environmental data and model predictions.")

@st.cache_data
def load_stats_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'Region': [f'Region_{i}' for i in range(n)],
        'NDVI': np.random.uniform(0.1, 0.8, n),
        'LST': np.random.uniform(25, 45, n),
        'Vulnerability Score': np.random.uniform(0.1, 0.9, n),
        'Priority Class': np.random.choice(['High Priority', 'Medium Priority', 'Low Priority'], n, p=[0.2, 0.5, 0.3])
    })

df = load_stats_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Priority Distribution")
    fig1 = px.pie(df, names='Priority Class', title='Regions by Priority Class', 
                  color='Priority Class', 
                  color_discrete_map={'High Priority':'#ef4444', 'Medium Priority':'#f59e0b', 'Low Priority':'#10b981'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("LST vs NDVI Correlation")
    fig2 = px.scatter(df, x='NDVI', y='LST', color='Priority Class', 
                      title='Vegetation vs Temperature',
                      color_discrete_map={'High Priority':'#ef4444', 'Medium Priority':'#f59e0b', 'Low Priority':'#10b981'})
    st.plotly_chart(fig2, use_container_width=True)
