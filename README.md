# 🌍 AI-GEO-NGO — AI-Enabled Geospatial Decision Support System

## Climate-Resilient WASH (Water, Sanitation & Hygiene) Planning

An AI-powered geospatial decision support system that helps NGOs and policymakers identify climate-vulnerable regions and prioritize WASH interventions using satellite imagery, geospatial analysis, and machine learning.

---

## 🎯 Features

- **Satellite Data Ingestion** — Fetch Sentinel-2, Landsat 8/9, SRTM DEM, and CHIRPS rainfall data via Google Earth Engine
- **Environmental Indicators** — Compute NDVI, LST, NDWI, NDBI from satellite imagery
- **ML-Based Vulnerability Prediction** — Random Forest & XGBoost models classify regions as Low / Medium / High risk
- **Interactive GIS Dashboard** — Streamlit-based dashboard with Folium maps, Plotly charts, and downloadable reports
- **RESTful API** — FastAPI backend exposing all data, predictions, and map endpoints

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python, FastAPI, Uvicorn |
| **ML** | Scikit-learn, XGBoost, SHAP |
| **GIS** | Google Earth Engine, GeoPandas, Rasterio, Folium |
| **Frontend** | Streamlit, Plotly, Streamlit-Folium |
| **Database** | SQLite (dev) → PostgreSQL + PostGIS (prod) |
| **Deployment** | Render / Railway / Google Cloud |

## 📁 Project Structure

```
AI-GEO-NGO/
├── config/          # App & GEE configuration
├── data/            # Raw, processed, features, predictions
├── src/
│   ├── ingestion/   # GEE data fetching
│   ├── processing/  # NDVI, LST, NDWI, NDBI computation
│   ├── features/    # Feature engineering & aggregation
│   ├── ml/          # Model training, evaluation, prediction
│   ├── api/         # FastAPI backend
│   └── dashboard/   # Streamlit frontend
├── models/          # Saved ML models
├── notebooks/       # Jupyter notebooks for EDA
├── tests/           # Unit & integration tests
├── scripts/         # Automation scripts
└── docs/            # Documentation
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/AI-GEO-NGO.git
cd AI-GEO-NGO

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your GEE credentials

# Run the API server
uvicorn src.api.main:app --reload

# Run the dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

## 📖 Documentation

- [System Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)
- [User Guide](docs/user_guide.md)

## 📄 License

This project is developed as part of an NGO internship program.
