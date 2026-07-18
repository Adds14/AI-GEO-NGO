"""
Streamlit main application entry point.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.settings import settings

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title=settings.APP_NAME,
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title(f"🌍 {settings.APP_NAME}")
    st.markdown("### AI-Enabled Geospatial Decision Support System")

    with st.sidebar:
        st.header("Navigation")
        st.info("Select a page from the sidebar (pages will be added here).")

    st.write("Welcome to the dashboard. Please select a module to begin.")

if __name__ == "__main__":
    main()
