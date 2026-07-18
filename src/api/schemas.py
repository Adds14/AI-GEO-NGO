"""
Pydantic schemas for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class RegionBase(BaseModel):
    """Base schema for a geographical region."""
    name: str
    geometry: dict  # GeoJSON dict
    crs: str = "EPSG:4326"

class IndicatorValue(BaseModel):
    """Schema for a specific environmental indicator value."""
    indicator: str
    value: float
    unit: str
    timestamp: datetime

class FeatureSet(BaseModel):
    """Schema for a set of features used in prediction."""
    region_id: str
    features: Dict[str, float]
    
class PredictionResult(BaseModel):
    """Schema for model prediction output."""
    region_id: str
    risk_level: str
    risk_score: float
    factors: Dict[str, float]

class RiskSummary(BaseModel):
    """Summary of risks across regions."""
    timestamp: datetime
    high_risk_regions: List[str]
    average_score: float

class TaskStatus(BaseModel):
    """Status of an async background task."""
    task_id: str
    status: str = Field(..., description="e.g., PENDING, RUNNING, COMPLETED, FAILED")
    progress: float = 0.0
    result: Optional[dict] = None
