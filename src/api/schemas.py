"""
Pydantic models for API request and response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# --- Region ---
class RegionBase(BaseModel):
    """Base schema for a region."""
    id: int
    name: str
    admin_level: Optional[str] = None
    area_km2: Optional[float] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None


# --- Indicators ---
class IndicatorValue(BaseModel):
    """Schema for an environmental indicator value."""
    indicator_name: str
    mean_value: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    std_value: Optional[float] = None
    observation_date: Optional[str] = None
    source_dataset: Optional[str] = None


# --- Features ---
class FeatureSet(BaseModel):
    """Schema for ML feature set per region."""
    region_id: int
    mean_ndvi: Optional[float] = None
    mean_lst: Optional[float] = None
    mean_ndwi: Optional[float] = None
    mean_ndbi: Optional[float] = None
    vegetation_change: Optional[float] = None
    urban_growth_rate: Optional[float] = None
    water_body_change: Optional[float] = None
    elevation: Optional[float] = None
    rainfall: Optional[float] = None
    population_density: Optional[float] = None
    distance_to_water: Optional[float] = None
    heat_stress_index: Optional[float] = None
    rainfall_anomaly: Optional[float] = None
    computed_date: Optional[str] = None


# --- Vulnerability Prediction ---
class PredictionResult(BaseModel):
    """Schema for a climate vulnerability prediction."""
    region_id: int
    region_name: Optional[str] = None
    model_name: str
    vulnerability_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str  # Low, Medium, High
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    feature_importance: Optional[dict] = None
    predicted_at: Optional[datetime] = None


# --- WASH Intervention Priority ---
class WASHPriority(BaseModel):
    """Schema for WASH Intervention Priority result."""
    region_id: int
    region_name: Optional[str] = None
    priority_score: float = Field(..., ge=0.0, le=100.0)
    priority_class: str  # Low Priority, Medium Priority, High Priority
    explanation: str  # Human-readable explanation of why this priority
    vulnerability_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_contributing_factors: Optional[List[dict]] = None
    computed_at: Optional[datetime] = None


class WASHPrioritySummary(BaseModel):
    """Aggregated WASH priority statistics."""
    total_regions: int
    priority_distribution: dict  # {"low": 45, "medium": 62, "high": 43}
    avg_priority_score: float
    highest_priority_regions: Optional[List[str]] = None
    last_updated: Optional[datetime] = None


# --- Risk Summary ---
class RiskSummary(BaseModel):
    """Aggregated risk statistics."""
    total_regions: int
    risk_distribution: dict
    avg_confidence: Optional[float] = None
    model_used: Optional[str] = None
    last_updated: Optional[datetime] = None


# --- Task Status ---
class TaskStatus(BaseModel):
    """Status of an async task."""
    task_id: str
    status: str  # pending, running, completed, failed
    message: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    result: Optional[dict] = None
