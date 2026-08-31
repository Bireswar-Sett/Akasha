from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, List
from services.ai_orchestrator import orchestrate_query
from services.specialist_models import execute_model

router = APIRouter()

@router.post("/validate")
async def validate_input(
    images: List[UploadFile] = File(default=[]),
    query: str = Form(...)
):
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # In a real app, validate image format (GeoTIFF, PNG, JPEG) and size
    image_metas = [{"filename": img.filename} for img in images] if images else None
    
    return {"status": "valid", "query": query, "images": image_metas}

@router.post("/orchestrate")
async def orchestrate(
    query: str = Form(...),
    images: List[UploadFile] = File(default=[])
):
    image_metas = [{"filename": img.filename} for img in images] if images else None
    orchestration_result = orchestrate_query(query, image_metas)
    return orchestration_result

@router.post("/execute")
async def execute(
    model_name: str = Form(...),
    query: str = Form(...),
    images: List[UploadFile] = File(default=[])
):
    image_metas = [{"filename": img.filename} for img in images] if images else None
    result = execute_model(model_name, query, image_metas)
    return result

@router.get("/status")
def get_status():
    return {"status": "idle"}

