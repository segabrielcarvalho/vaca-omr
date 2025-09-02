from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import datetime

from ..services.omr import process_image_simple

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok", "service": "vaca-omr"}

@router.post("/omr")
async def omr_multipart(
    file: UploadFile = File(...), 
    threshold: float = 0.50, 
    delta: float = 0.12
):
    
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        data = await file.read()
        
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Dados não correspondem a uma imagem válida")
        
        result = process_image_simple(img, threshold=threshold, delta=delta)
        
        if result["success"]:
            result["upload_info"] = {
                "original_filename": file.filename,
                "file_size_bytes": len(data),
                "content_type": file.content_type
            }
        
        return JSONResponse(result)
                
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Erro no processamento: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        }, status_code=500)

@router.post("/omr-bytes")
async def omr_bytes(
    request: Request,
    threshold: float = 0.50,
    delta: float = 0.12
):
    
    try:
        buf = bytearray()
        async for chunk in request.stream():
            buf.extend(chunk)
        data = bytes(buf)
        
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Nenhum dado recebido")
        
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Dados não correspondem a uma imagem válida")
        
        result = process_image_simple(img, threshold=threshold, delta=delta)
        
        if result["success"]:
            result["upload_info"] = {
                "file_size_bytes": len(data),
                "content_type": "application/octet-stream"
            }
        
        return JSONResponse(result)
                
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Erro no processamento: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        }, status_code=500)
