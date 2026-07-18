"""
GIS layers endpoints for GeoJSON rendering in dashboards and external GIS software.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict
from src.api.dependencies import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.get("/layers")
async def get_gis_layers(layer_type: str = "vulnerability") -> Dict[str, Any]:
    """
    Retrieve GeoJSON FeatureCollection for mapping.
    layer_type: 'vulnerability', 'ndvi', 'priority'
    """
    if layer_type not in ["vulnerability", "ndvi", "priority"]:
        raise HTTPException(400, "Invalid layer type requested.")
        
    # Mock GeoJSON structure
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[35.0, 3.0], [36.0, 3.0], [36.0, 4.0], [35.0, 4.0], [35.0, 3.0]]
                    ]
                },
                "properties": {
                    "region": "Grid-1",
                    "score": 0.85,
                    "layer": layer_type
                }
            }
        ]
    }
