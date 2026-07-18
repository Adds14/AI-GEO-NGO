"""
Machine Learning routes for climate vulnerability and WASH priority.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from src.api.dependencies import verify_api_key
from src.api.schemas import PredictionResult, WASHPriority, TaskStatus
import uuid

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.post("/vulnerability", response_model=TaskStatus)
async def trigger_vulnerability_prediction():
    """
    Trigger the XGBoost/RandomForest model to score all active regions.
    """
    # Mock task trigger
    task_id = str(uuid.uuid4())
    return TaskStatus(
        task_id=task_id,
        status="pending",
        message="Vulnerability prediction batch job queued."
    )

@router.post("/priority", response_model=List[WASHPriority])
async def calculate_wash_priority():
    """
    Run the WASHPriorityEngine on current predictions to output 0-100 scores.
    """
    # Mock return
    return [
        WASHPriority(
            region_id=1,
            region_name="Turkana",
            priority_score=85.5,
            priority_class="High Priority",
            explanation="Severe water stress coupled with high vulnerability.",
            vulnerability_score=0.92
        ),
        WASHPriority(
            region_id=2,
            region_name="Nairobi",
            priority_score=35.0,
            priority_class="Low Priority",
            explanation="Stable indicators and low climate risk.",
            vulnerability_score=0.21
        )
    ]
