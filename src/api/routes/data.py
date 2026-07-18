"""
Data routes for handling feature uploads and indicator retrieval.
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from typing import List, Dict, Any
import pandas as pd
from io import StringIO
from src.api.dependencies import verify_api_key
from src.api.schemas import TaskStatus, IndicatorValue

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.post("/upload", response_model=TaskStatus)
async def upload_feature_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV file containing engineered features.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
        
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode("utf-8")))
        
        # Validation
        required_cols = ['geographic_id', 'ndvi', 'lst']
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(400, f"Missing required column: {col}")
                
        # In a real app, save to db or storage here
        return TaskStatus(
            task_id="mock-upload-task",
            status="completed",
            message=f"Successfully ingested {len(df)} rows from {file.filename}."
        )
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")

@router.get("/indicators", response_model=List[IndicatorValue])
async def get_indicators():
    """
    Retrieve aggregated environmental indicators.
    """
    # Mock data return
    return [
        IndicatorValue(
            indicator_name="NDVI",
            mean_value=0.45,
            source_dataset="Sentinel-2 L2A"
        ),
        IndicatorValue(
            indicator_name="LST",
            mean_value=32.1,
            source_dataset="Landsat 8"
        )
    ]
