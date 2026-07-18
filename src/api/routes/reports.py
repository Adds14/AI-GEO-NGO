"""
Report generation endpoints.
"""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import io
from src.api.dependencies import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.post("/generate")
async def generate_report(region_id: str):
    """
    Generate an executive summary report for a specific region.
    """
    # Mock successful generation
    return {
        "status": "success",
        "message": f"Report generated for {region_id}",
        "file_id": f"rep_{region_id}_123"
    }

@router.get("/download/{file_id}")
async def download_report(file_id: str):
    """
    Download a previously generated report as a Markdown file attachment.
    """
    content = f"# WASH Executive Summary\n\nReport ID: {file_id}\n\nThis region requires immediate attention based on recent GEE satellite metrics."
    
    stream = io.StringIO(content)
    return StreamingResponse(
        stream,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{file_id}.md"}
    )
