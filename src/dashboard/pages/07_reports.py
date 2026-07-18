"""
Reports Generator Page.
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Generate Reports", layout="wide")
st.title("📄 Generate Reports")
st.markdown("Generate executive summary reports for specific regions.")

region = st.selectbox("Select Region", [f"Grid-{i}" for i in range(1, 21)])

if st.button("Generate Report"):
    with st.spinner("Compiling report..."):
        report_text = f"""
        # WASH Intervention Report: {region}
        
        ## Executive Summary
        Based on satellite-derived indicators, this region has been flagged by the AI-GEO-NGO platform.
        
        ## Key Metrics
        - **Climate Vulnerability Score**: 0.82 (High)
        - **WASH Priority Score**: 78.5/100 (High Priority)
        
        ## Environmental Drivers
        The primary drivers for this classification are:
        1. **High Land Surface Temperature** (LST) exacerbating evaporation.
        2. **Low Vegetation** (NDVI) indicating drought conditions.
        
        ## Recommended Actions
        - Deploy emergency water trucking.
        - Drill deep boreholes in central sectors.
        - Monitor disease outbreak risks due to heat stress.
        """
        
        st.markdown("### Preview")
        st.info(report_text)
        
        st.download_button(
            label="Download Report as Markdown",
            data=report_text,
            file_name=f"report_{region}.md",
            mime="text/markdown"
        )
