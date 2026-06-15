import datetime
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.omr import ENGINE_VERSION, process_image_dynamic

router = APIRouter()
logger = logging.getLogger("vaca_omr.routes_v2")


class OmrProcessRequest(BaseModel):
    captureId: str | None = None
    sessionId: str | None = None
    imageBase64: str
    compiledGeometryJson: dict[str, Any] | str
    masterAnswers: list[int | None] | None = None
    threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    delta: float = Field(default=0.12, ge=0.0, le=1.0)
    includeImages: bool = True


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "vaca-omr",
        "engineVersion": ENGINE_VERSION,
    }


@router.post("/omr/process")
async def omr_process(payload: OmrProcessRequest):
    started = time.perf_counter()
    geometry_summary = _summarize_geometry(payload.compiledGeometryJson)
    _log_info(
        "omr.process.request_started",
        captureId=payload.captureId,
        sessionId=payload.sessionId,
        imageBase64Chars=len(payload.imageBase64 or ""),
        threshold=payload.threshold,
        delta=payload.delta,
        **geometry_summary,
    )
    try:
        result = process_image_dynamic(
            capture_id=payload.captureId,
            session_id=payload.sessionId,
            image_base64=payload.imageBase64,
            compiled_geometry_json=payload.compiledGeometryJson,
            master_answers=payload.masterAnswers,
            threshold=payload.threshold,
            delta=payload.delta,
            include_images=payload.includeImages,
        )
        _log_info(
            "omr.process.request_finished",
            captureId=payload.captureId,
            sessionId=payload.sessionId,
            success=result.get("success"),
            errorCode=(result.get("error") or {}).get("code")
            if isinstance(result.get("error"), dict)
            else None,
            engineVersion=result.get("engineVersion"),
            registrationStatus=(result.get("registration") or {}).get("status")
            if isinstance(result.get("registration"), dict)
            else None,
            answersCount=len(result.get("answers") or result.get("answers_numeric") or []),
            durationMs=_elapsed_ms(started),
            timings=result.get("timings"),
        )
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        _log_exception(
            "omr.process.request_error",
            captureId=payload.captureId,
            sessionId=payload.sessionId,
            errorType=type(exc).__name__,
            errorMessage=str(exc),
            durationMs=_elapsed_ms(started),
        )
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


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _summarize_geometry(compiled_geometry_json: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(compiled_geometry_json, str):
        try:
            geometry = json.loads(compiled_geometry_json)
        except json.JSONDecodeError:
            return {"geometryType": "string", "geometryValidJson": False}
    else:
        geometry = compiled_geometry_json

    registration = geometry.get("registration") if isinstance(geometry, dict) else None
    questions = geometry.get("questions") if isinstance(geometry, dict) else None

    return {
        "geometryType": "object" if isinstance(geometry, dict) else type(geometry).__name__,
        "registrationDigits": registration.get("digits")
        if isinstance(registration, dict)
        else None,
        "questionCount": questions.get("questionCount")
        if isinstance(questions, dict)
        else None,
    }


def _log_info(event: str, **payload: Any) -> None:
    logger.info(
        json.dumps(
            {
                "event": event,
                **payload,
            },
            ensure_ascii=False,
            default=str,
        )
    )


def _log_exception(event: str, **payload: Any) -> None:
    logger.exception(
        json.dumps(
            {
                "event": event,
                **payload,
            },
            ensure_ascii=False,
            default=str,
        )
    )
