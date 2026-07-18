"""
FastAPI application for AI-GEO-NGO backend.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import time

from config.settings import settings
from src.api.routes import data, predict, gis, reports

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for Geospatial Decision Support System",
    version="1.0.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - {process_time:.3f}s")
    return response

# Register Routers
app.include_router(data.router, prefix="/api/v1/data", tags=["Data Processing"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["Machine Learning"])
app.include_router(gis.router, prefix="/api/v1/gis", tags=["GIS Maps"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "environment": settings.APP_ENV}

@app.get("/", tags=["System"])
async def root():
    """Root endpoint."""
    return {"message": f"Welcome to {settings.APP_NAME} API. Visit /docs for Swagger UI."}
