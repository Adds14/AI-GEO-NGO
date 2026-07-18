"""
Application settings configuration.
Uses pydantic-settings to load configuration from environment variables or .env file.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

class AppSettings(BaseSettings):
    """Application settings class."""
    
    # App
    APP_NAME: str = "AI-GEO-NGO"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # GEE
    GEE_PROJECT_ID: str = ""
    GEE_SERVICE_ACCOUNT_EMAIL: str = ""
    GEE_KEY_FILE: str = ""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/app.db"
    
    # Paths (relative to BASE_DIR)
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    REPORT_DIR: Path = BASE_DIR / "data" / "reports"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # ML Defaults
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
        
    def setup_directories(self):
        """Auto-create necessary directories."""
        dirs_to_create = [
            self.DATA_DIR / "raw" / "sentinel2",
            self.DATA_DIR / "raw" / "landsat",
            self.DATA_DIR / "raw" / "srtm",
            self.DATA_DIR / "raw" / "chirps",
            self.DATA_DIR / "raw" / "shapefiles",
            self.DATA_DIR / "processed" / "ndvi",
            self.DATA_DIR / "processed" / "lst",
            self.DATA_DIR / "processed" / "ndwi",
            self.DATA_DIR / "processed" / "ndbi",
            self.DATA_DIR / "features",
            self.DATA_DIR / "predictions",
            self.REPORT_DIR,
            self.MODEL_DIR,
            self.LOG_DIR,
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

# Singleton instance
settings = AppSettings()
# Automatically setup directories upon import
settings.setup_directories()
