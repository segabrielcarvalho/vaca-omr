import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.omr import ENGINE_VERSION, process_image_dynamic

router = APIRouter()


class OmrProcessRequest(BaseModel):
    captureId: str | None = None
    sessionId: str | None = None
    imageBase64: str
    compiledGeometryJson: dict[str, Any] | str
    threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    delta: float = Field(default=0.12, ge=0.0, le=1.0)


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "vaca-omr",
        "engineVersion": ENGINE_VERSION,
    }


@router.post("/omr/process")
async def omr_process(payload: OmrProcessRequest):
    try:
        result = process_image_dynamic(
            capture_id=payload.captureId,
            session_id=payload.sessionId,
            image_base64=payload.imageBase64,
            compiled_geometry_json=payload.compiledGeometryJson,
            threshold=payload.threshold,
            delta=payload.delta,
        )
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            {
                "success": False,
                "engineVersion": ENGINE_VERSION,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                },
                "timestamp": datetime.datetime.now().isoformat(),
            },
            status_code=500,
        )
