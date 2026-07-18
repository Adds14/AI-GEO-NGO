"""
FastAPI application for AI-GEO-NGO backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
import logging

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for Geospatial Decision Support System",
    version="0.1.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "environment": settings.APP_ENV}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": f"Welcome to {settings.APP_NAME} API"}
