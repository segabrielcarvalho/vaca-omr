import cv2
import datetime
from pathlib import Path

from ...utils import (
    detect_sheet, base_centers, detect_choices, detect_qr
)

ABC = "ABCDE"


class OMRService:
    
    def __init__(self, threshold=0.50, delta=0.12):
        self.threshold = threshold
        self.delta = delta
    
    def process_image(self, img_path_or_array):
        
        if isinstance(img_path_or_array, str):
            img = cv2.imread(img_path_or_array)
            filename = Path(img_path_or_array).name
        else:
            img = img_path_or_array
            filename = "uploaded_image"
        
        if img is None:
            return {
                "success": False,
                "error": "Arquivo de imagem inválido",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        disp, warped, th, status, ids = detect_sheet(img)
        
        if status != "ok":
            return {
                "success": False,
                "error": status,
                "aruco_detected": ids.tolist() if ids is not None else [],
                "aruco_count": len(ids) if ids is not None else 0,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        centers = base_centers()
        fills, picks = detect_choices(th, centers, self.threshold, self.delta)
        qr_info = detect_qr(warped if warped is not None else img)
        
        readable_answers = []
        for pick in picks:
            if pick == -1:
                readable_answers.append("não_marcado")
            elif pick == -2:
                readable_answers.append("ambíguo")
            elif 0 <= pick <= 4:
                readable_answers.append(ABC[pick])
            else:
                readable_answers.append("erro")
        
        total_questions = len(picks)
        answered = len([p for p in picks if p >= 0])
        not_answered = len([p for p in picks if p == -1])
        ambiguous = len([p for p in picks if p == -2])
        
        return {
            "success": True,
            "results": {
                "filename": filename,
                "total_questions": total_questions,
                "answered": answered,
                "not_answered": not_answered,
                "ambiguous": ambiguous,
                "answers_numeric": picks,
                "answers_letters": readable_answers,
                "confidence_scores": fills,
                "detection_status": status,
                "aruco_count": len(ids) if ids is not None else 0,
                "input_shape": list(img.shape),
                "qr": qr_info,
                "parameters": {
                    "threshold": self.threshold,
                    "delta": self.delta
                }
            },
            "timestamp": datetime.datetime.now().isoformat()
        }


def process_image_simple(img_path_or_array, threshold=0.50, delta=0.12):
    service = OMRService(threshold=threshold, delta=delta)
    return service.process_image(img_path_or_array)
