import base64
import json
import logging
import os
import time
from itertools import combinations
from typing import Any

import cv2
import numpy as np

from .debug_dump import write_omr_debug_dump

ENGINE_VERSION = os.environ.get("OMR_ENGINE_VERSION", "2.1.0-template-driven-no-qr")
OMR_DEBUG_TRACE = os.environ.get("OMR_DEBUG_TRACE", "false").lower() == "true"
logger = logging.getLogger("vaca_omr.service_v2")
REQUIRED_ARUCO_IDS = (0, 1, 2, 3)
ARUCO_VARIANT_UPSCALE = 2.0
LOW_SIGNAL_THRESHOLD = 0.08
LOW_SIGNAL_DELTA = 0.035
LOW_SIGNAL_MEDIAN_GAP = 0.05
REGISTRATION_NOMINAL_SIGNAL_THRESHOLD = 0.12


def process_image_dynamic(
    capture_id: str | None,
    session_id: str | None,
    image_base64: str,
    compiled_geometry_json: dict[str, Any] | str,
    master_answers: list[int | None] | None = None,
    threshold: float = 0.50,
    delta: float = 0.12,
) -> dict[str, Any]:
    timings: dict[str, Any] = {}
    started = time.perf_counter()
    image: np.ndarray | None = None
    geometry: dict[str, Any] | None = None
    page: dict[str, Any] | None = None
    sheet: dict[str, Any] | None = None
    registration: dict[str, Any] | None = None
    answers: dict[str, Any] | None = None
    registration_overlay: np.ndarray | None = None
    answers_overlay: np.ndarray | None = None
    overlay: np.ndarray | None = None
    result: dict[str, Any] | None = None
    unhandled_error: dict[str, Any] | None = None

    try:
        _trace(
            "omr.service.started",
            captureId=capture_id,
            sessionId=session_id,
            threshold=threshold,
            delta=delta,
            imageBase64Chars=len(image_base64 or ""),
        )
        try:
            t = time.perf_counter()
            image = _decode_base64_image(image_base64)
            timings["decodeMs"] = _elapsed_ms(t)
            _trace(
                "omr.service.image_decoded",
                captureId=capture_id,
                sessionId=session_id,
                decodeMs=timings["decodeMs"],
                imageShape=list(image.shape),
            )
        except ValueError as exc:
            timings["totalMs"] = _elapsed_ms(started)
            result = _error_response("IMAGE_DECODE_FAILED", str(exc), timings)
            _trace(
                "omr.service.failed",
                captureId=capture_id,
                sessionId=session_id,
                errorCode="IMAGE_DECODE_FAILED",
                totalMs=timings["totalMs"],
            )
            return result

        try:
            geometry = _normalize_geometry(compiled_geometry_json)
            page = _normalize_page(geometry)
            _trace(
                "omr.service.geometry_normalized",
                captureId=capture_id,
                sessionId=session_id,
                geometry=_summarize_geometry(geometry),
            )
        except ValueError as exc:
            timings["totalMs"] = _elapsed_ms(started)
            result = _error_response("GEOMETRY_INVALID", str(exc), timings)
            _trace(
                "omr.service.failed",
                captureId=capture_id,
                sessionId=session_id,
                errorCode="GEOMETRY_INVALID",
                totalMs=timings["totalMs"],
            )
            return result

        t = time.perf_counter()
        sheet = _detect_and_rectify_sheet(image, geometry, page)
        timings["rectifyMs"] = _elapsed_ms(t)
        _trace(
            "omr.service.sheet_rectified",
            captureId=capture_id,
            sessionId=session_id,
            ok=sheet["ok"],
            rectifyMs=timings["rectifyMs"],
            arucoDetected=sheet.get("arucoDetected", []),
            arucoVariant=sheet.get("arucoVariant"),
        )

        if not sheet["ok"]:
            timings["totalMs"] = _elapsed_ms(started)
            result = _error_response(
                sheet["errorCode"],
                sheet["errorMessage"],
                timings,
                extra={"arucoDetected": sheet.get("arucoDetected", [])},
            )
            _trace(
                "omr.service.failed",
                captureId=capture_id,
                sessionId=session_id,
                errorCode=sheet["errorCode"],
                totalMs=timings["totalMs"],
            )
            return result

        warped = sheet["warped"]
        gray_warped = sheet["grayWarped"]
        thresholded = sheet["thresholded"]

        t = time.perf_counter()
        try:
            registration = _read_registration(gray_warped, geometry, page, threshold, delta)
        except ValueError as exc:
            timings["registrationMs"] = _elapsed_ms(t)
            timings["totalMs"] = _elapsed_ms(started)
            result = _error_response("GEOMETRY_INVALID", str(exc), timings)
            _trace(
                "omr.service.failed",
                captureId=capture_id,
                sessionId=session_id,
                errorCode="GEOMETRY_INVALID",
                totalMs=timings["totalMs"],
            )
            return result
        timings["registrationMs"] = _elapsed_ms(t)
        _trace(
            "omr.service.registration_read",
            captureId=capture_id,
            sessionId=session_id,
            registrationMs=timings["registrationMs"],
            status=registration["status"],
            value=registration["value"],
        )

        t = time.perf_counter()
        try:
            answers = _read_answers(gray_warped, geometry, page, threshold, delta)
        except ValueError as exc:
            timings["answersMs"] = _elapsed_ms(t)
            timings["totalMs"] = _elapsed_ms(started)
            result = _error_response("GEOMETRY_INVALID", str(exc), timings)
            _trace(
                "omr.service.failed",
                captureId=capture_id,
                sessionId=session_id,
                errorCode="GEOMETRY_INVALID",
                totalMs=timings["totalMs"],
            )
            return result
        timings["answersMs"] = _elapsed_ms(t)
        _trace(
            "omr.service.answers_read",
            captureId=capture_id,
            sessionId=session_id,
            answersMs=timings["answersMs"],
            answersCount=len(answers["answers"]),
        )

        t = time.perf_counter()
        registration_overlay = _draw_overlay(
            warped,
            registration["details"],
            [],
            None,
        )
        answers_overlay = _draw_overlay(
            warped,
            [],
            answers["details"],
            master_answers,
        )
        overlay = _draw_overlay(
            warped,
            registration["details"],
            answers["details"],
            master_answers,
        )
        rectified_base64 = _encode_image_base64(warped)
        overlay_base64 = _encode_image_base64(overlay)
        timings["overlayMs"] = _elapsed_ms(t)

        timings["totalMs"] = _elapsed_ms(started)
        _trace(
            "omr.service.overlay_encoded",
            captureId=capture_id,
            sessionId=session_id,
            overlayMs=timings["overlayMs"],
            totalMs=timings["totalMs"],
        )

        result = {
            "success": True,
            "engineVersion": ENGINE_VERSION,
            "registration": {
                "value": registration["value"],
                "status": registration["status"],
            },
            "answers": answers["answers"],
            "answers_numeric": answers["answers_numeric"],
            "timings": timings,
            "images": {
                "rectifiedBase64": rectified_base64,
                "overlayBase64": overlay_base64,
            },
        }
        _trace(
            "omr.service.finished",
            captureId=capture_id,
            sessionId=session_id,
            success=True,
            totalMs=timings["totalMs"],
        )
        return result
    except Exception as exc:
        unhandled_error = {
            "code": "UNHANDLED_ERROR",
            "message": str(exc),
            "type": type(exc).__name__,
        }
        _trace(
            "omr.service.unhandled_error",
            captureId=capture_id,
            sessionId=session_id,
            errorType=type(exc).__name__,
            errorMessage=str(exc),
            timings=timings,
        )
        raise
    finally:
        _write_debug_dump_safe(
            capture_id=capture_id,
            session_id=session_id,
            threshold=threshold,
            delta=delta,
            compiled_geometry_json=compiled_geometry_json,
            master_answers=master_answers,
            geometry=geometry,
            page=page,
            sheet=sheet,
            registration=registration,
            answers=answers,
            timings=timings,
            result=result,
            error=unhandled_error,
            artifacts={
                "00_input.jpg": image,
                "01_gray.jpg": None if sheet is None else sheet.get("gray"),
                "02_aruco_detected.jpg": None if sheet is None else sheet.get("arucoOverlay"),
                "03_rectified.jpg": None if sheet is None else sheet.get("warped"),
                "04_thresholded.jpg": None if sheet is None else sheet.get("thresholded"),
                "05_registration_overlay.jpg": registration_overlay,
                "06_answers_overlay.jpg": answers_overlay,
                "07_overlay_final.jpg": overlay,
            },
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _trace(event: str, **payload: Any) -> None:
    if not OMR_DEBUG_TRACE:
        return
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


def _error_response(
    code: str,
    message: str,
    timings: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "success": False,
        "engineVersion": ENGINE_VERSION,
        "registration": {"value": None, "status": "invalid"},
        "answers": [],
        "answers_numeric": [],
        "timings": timings,
        "error": {"code": code, "message": message},
    }
    if extra:
        payload.update(extra)
    return payload


def _write_debug_dump_safe(
    *,
    capture_id: str | None,
    session_id: str | None,
    threshold: float,
    delta: float,
    compiled_geometry_json: dict[str, Any] | str,
    master_answers: list[int | None] | None,
    geometry: dict[str, Any] | None,
    page: dict[str, Any] | None,
    sheet: dict[str, Any] | None,
    registration: dict[str, Any] | None,
    answers: dict[str, Any] | None,
    timings: dict[str, Any],
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
    artifacts: dict[str, np.ndarray | None],
) -> None:
    metadata = {
        "engineVersion": ENGINE_VERSION,
        "request": {
            "threshold": threshold,
            "delta": delta,
            "compiledGeometrySummary": _summarize_geometry(
                geometry if geometry is not None else compiled_geometry_json
            ),
            "masterAnswers": master_answers,
        },
        "page": page,
        "arucoDetected": [] if sheet is None else sheet.get("arucoDetected", []),
        "arucoVariant": None if sheet is None else sheet.get("arucoVariant"),
        "sourceMarkerCentersPx": None if sheet is None else sheet.get("sourceMarkerCentersPx"),
        "targetAnchorCentersPx": None if sheet is None else sheet.get("targetAnchorCentersPx"),
        "timings": timings,
        "registration": _summarize_registration(registration),
        "answers": _summarize_answers(answers),
        "success": None if result is None else result.get("success"),
        "error": error if error is not None else None if result is None else result.get("error"),
        "response": _summarize_response(result),
    }
    try:
        dump_paths = write_omr_debug_dump(
            capture_id=capture_id,
            session_id=session_id,
            artifacts=artifacts,
            metadata=metadata,
        )
        if dump_paths:
            print(
                "[vaca-omr] debug dump written:",
                json.dumps(
                    {
                        "captureId": capture_id,
                        "sessionId": session_id,
                        "paths": dump_paths,
                    },
                    ensure_ascii=False,
                ),
            )
    except Exception as exc:
        print(f"[vaca-omr] failed to write debug dump: {exc}")


def _decode_base64_image(image_base64: str) -> np.ndarray:
    if not image_base64 or not isinstance(image_base64, str):
        raise ValueError("imageBase64 ausente ou invalido.")

    raw = image_base64
    if image_base64.startswith("data:") and "," in image_base64:
        raw = image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("imageBase64 nao e um base64 valido.") from exc

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Nao foi possivel decodificar a imagem.")

    return image


def _normalize_geometry(compiled_geometry_json: dict[str, Any] | str) -> dict[str, Any]:
    geometry = compiled_geometry_json

    if isinstance(geometry, str):
        try:
            geometry = json.loads(geometry)
        except json.JSONDecodeError as exc:
            raise ValueError("compiledGeometryJson JSON invalido.") from exc

    if not isinstance(geometry, dict):
        raise ValueError("compiledGeometryJson deve ser um objeto JSON.")

    if "questions" not in geometry:
        raise ValueError("compiledGeometryJson.questions e obrigatorio.")

    return geometry


def _normalize_page(geometry: dict[str, Any]) -> dict[str, Any]:
    page = geometry.get("page", {})
    if not isinstance(page, dict):
        page = {}

    width_mm = _num(page.get("widthMm"), 210.0)
    height_mm = _num(page.get("heightMm"), 297.0)
    px_per_mm = _num(page.get("pxPerMm"), 10.0)

    width_px = max(500, int(round(width_mm * px_per_mm)))
    height_px = max(700, int(round(height_mm * px_per_mm)))

    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "width_px": width_px,
        "height_px": height_px,
        "x_scale": width_px / width_mm,
        "y_scale": height_px / height_mm,
    }


def _detect_and_rectify_sheet(
    image: np.ndarray,
    geometry: dict[str, Any],
    page: dict[str, Any],
) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected = _detect_aruco_markers(gray, image)
    aruco_overlay = detected["overlay"]

    target_map = _target_anchor_centers(geometry, page)

    if not detected["detectedIds"]:
        return {
            "ok": False,
            "errorCode": "ARUCO_NOT_FOUND",
            "errorMessage": "Nenhuma ancora ArUco encontrada.",
            "gray": gray,
            "arucoOverlay": aruco_overlay,
            "arucoDetected": [],
            "arucoVariant": detected["variant"],
            "targetAnchorCentersPx": {
                marker_id: [float(coords[0]), float(coords[1])]
                for marker_id, coords in target_map.items()
            },
        }

    detected_ids = detected["detectedIds"]
    marker_map = detected["markerMap"]
    marker_corners = detected["markerCorners"]

    required_detected_ids = [marker_id for marker_id in REQUIRED_ARUCO_IDS if marker_id in marker_map]

    if len(required_detected_ids) < 3:
        return {
            "ok": False,
            "errorCode": "ARUCO_MISSING_IDS",
            "errorMessage": "Menos de tres ancoras ArUco obrigatorias (0..3) foram encontradas.",
            "gray": gray,
            "arucoOverlay": aruco_overlay,
            "arucoDetected": detected_ids,
            "arucoVariant": detected["variant"],
            "sourceMarkerCentersPx": {
                marker_id: [float(point[0]), float(point[1])]
                for marker_id, point in marker_map.items()
            },
            "targetAnchorCentersPx": {
                marker_id: [float(coords[0]), float(coords[1])]
                for marker_id, coords in target_map.items()
            },
        }

    source = np.vstack([marker_corners[marker_id] for marker_id in required_detected_ids]).astype(np.float32)
    target = np.vstack([_target_anchor_corners(marker_id, geometry, page) for marker_id in required_detected_ids]).astype(
        np.float32
    )

    matrix, _ = cv2.findHomography(source, target, 0)
    if matrix is None:
        return {
            "ok": False,
            "errorCode": "ARUCO_HOMOGRAPHY_FAILED",
            "errorMessage": "Nao foi possivel retificar a folha a partir das ancoras ArUco.",
            "gray": gray,
            "arucoOverlay": aruco_overlay,
            "arucoDetected": detected_ids,
            "arucoVariant": detected["variant"],
            "sourceMarkerCentersPx": {
                marker_id: [float(point[0]), float(point[1])]
                for marker_id, point in marker_map.items()
            },
            "targetAnchorCentersPx": {
                marker_id: [float(coords[0]), float(coords[1])]
                for marker_id, coords in target_map.items()
            },
        }

    warped = cv2.warpPerspective(image, matrix, (page["width_px"], page["height_px"]))
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresholded = _build_threshold_map(gray_warped)

    return {
        "ok": True,
        "gray": gray,
        "arucoOverlay": aruco_overlay,
        "warped": warped,
        "grayWarped": gray_warped,
        "thresholded": thresholded,
        "arucoDetected": detected_ids,
        "arucoVariant": detected["variant"],
        "sourceMarkerCentersPx": {
            marker_id: [float(point[0]), float(point[1])]
            for marker_id, point in marker_map.items()
        },
        "targetAnchorCentersPx": {
            marker_id: [float(coords[0]), float(coords[1])]
            for marker_id, coords in target_map.items()
        },
    }


def _target_anchor_centers(
    geometry: dict[str, Any],
    page: dict[str, Any],
) -> dict[int, tuple[float, float]]:
    default = {
        0: (10.0, 10.0),
        1: (page["width_mm"] - 10.0, 10.0),
        2: (page["width_mm"] - 10.0, page["height_mm"] - 10.0),
        3: (10.0, page["height_mm"] - 10.0),
    }

    anchors = geometry.get("anchors", [])
    if isinstance(anchors, list):
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            marker_id = int(_num(anchor.get("id"), -1))
            if marker_id not in REQUIRED_ARUCO_IDS:
                continue
            x_mm = _num(anchor.get("xMm"), default[marker_id][0])
            y_mm = _num(anchor.get("yMm"), default[marker_id][1])
            size_mm = _num(anchor.get("sizeMm"), 10.0)
            default[marker_id] = (x_mm + (size_mm / 2.0), y_mm + (size_mm / 2.0))

    return {
        marker_id: (
            float(_mm_x_to_px(coords[0], page)),
            float(_mm_y_to_px(coords[1], page)),
        )
        for marker_id, coords in default.items()
    }


def _target_anchor_corners(
    marker_id: int,
    geometry: dict[str, Any],
    page: dict[str, Any],
) -> np.ndarray:
    anchors = geometry.get("anchors", [])
    size_mm = 10.0

    if isinstance(anchors, list):
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            if int(_num(anchor.get("id"), -1)) != marker_id:
                continue
            size_mm = _num(anchor.get("sizeMm"), size_mm)
            x_mm = _num(anchor.get("xMm"), 0.0)
            y_mm = _num(anchor.get("yMm"), 0.0)
            return np.array(
                [
                    [_mm_x_to_px(x_mm, page), _mm_y_to_px(y_mm, page)],
                    [_mm_x_to_px(x_mm + size_mm, page), _mm_y_to_px(y_mm, page)],
                    [_mm_x_to_px(x_mm + size_mm, page), _mm_y_to_px(y_mm + size_mm, page)],
                    [_mm_x_to_px(x_mm, page), _mm_y_to_px(y_mm + size_mm, page)],
                ],
                dtype=np.float32,
            )

    center = _target_anchor_centers(geometry, page).get(marker_id, (0.0, 0.0))
    half_size_px = (size_mm * page["x_scale"]) / 2.0
    return np.array(
        [
            [center[0] - half_size_px, center[1] - half_size_px],
            [center[0] + half_size_px, center[1] - half_size_px],
            [center[0] + half_size_px, center[1] + half_size_px],
            [center[0] - half_size_px, center[1] + half_size_px],
        ],
        dtype=np.float32,
    )


def _detect_aruco_markers(gray: np.ndarray, original: np.ndarray) -> dict[str, Any]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = _build_aruco_detector(dictionary)
    best: dict[str, Any] | None = None

    for variant in _build_aruco_variants(gray):
        corners, ids, _ = detector.detectMarkers(variant["image"])
        corners = corners or []
        detected_ids = [] if ids is None else [int(value) for value in ids.flatten().tolist()]

        scaled_corners = [_rescale_marker_corners(corner[0], variant["scale"]) for corner in corners]
        marker_map: dict[int, np.ndarray] = {}
        marker_corners: dict[int, np.ndarray] = {}

        for index, marker_id in enumerate(detected_ids):
            current_corners = scaled_corners[index].astype(np.float32)
            marker_corners[marker_id] = current_corners
            marker_map[marker_id] = current_corners.mean(axis=0).astype(np.float32)

        current = {
            "variant": variant["name"],
            "detectedIds": detected_ids,
            "markerMap": marker_map,
            "markerCorners": marker_corners,
        }
        score = _score_aruco_detection(current)
        if best is None or score > _score_aruco_detection(best):
            best = current

        if score[0] == len(REQUIRED_ARUCO_IDS):
            break

    if best is None:
        best = {
            "variant": "gray",
            "detectedIds": [],
            "markerMap": {},
            "markerCorners": {},
        }

    overlay = original.copy()
    if best["detectedIds"]:
        draw_corners = [corners.reshape(1, 4, 2) for corners in best["markerCorners"].values()]
        ids = np.array([[marker_id] for marker_id in best["markerCorners"].keys()], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(overlay, draw_corners, ids)

    best["overlay"] = overlay
    return best


def _build_aruco_detector(dictionary: Any) -> Any:
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 101
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.008
    params.maxMarkerPerimeterRate = 1.0
    params.polygonalApproxAccuracyRate = 0.1
    params.minCornerDistanceRate = 0.01
    params.minDistanceToBorder = 2

    if hasattr(params, "cornerRefinementMethod") and hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)

    class _LegacyDetector:
        def __init__(self, legacy_dictionary: Any, legacy_params: Any) -> None:
            self.dictionary = legacy_dictionary
            self.params = legacy_params

        def detectMarkers(self, image: np.ndarray) -> tuple[Any, Any, Any]:
            return cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.params)

    return _LegacyDetector(dictionary, params)


def _build_aruco_variants(gray: np.ndarray) -> list[dict[str, Any]]:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = cv2.equalizeHist(gray)
    clahe_equalized = clahe.apply(gray)
    adaptive_gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    adaptive_soft = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        7,
    )
    adaptive_clahe = cv2.adaptiveThreshold(
        clahe_equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    variants = [
        {"name": "gray", "image": gray, "scale": 1.0},
        {"name": "equalized", "image": equalized, "scale": 1.0},
        {"name": "clahe", "image": clahe_equalized, "scale": 1.0},
        {"name": "adaptive-gray", "image": adaptive_gray, "scale": 1.0},
        {"name": "adaptive-soft", "image": adaptive_soft, "scale": 1.0},
        {"name": "adaptive-clahe", "image": adaptive_clahe, "scale": 1.0},
    ]

    for variant in list(variants):
        if variant["name"].startswith("adaptive"):
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_CUBIC
        variants.append(
            {
                "name": f'{variant["name"]}-2x',
                "image": cv2.resize(
                    variant["image"],
                    None,
                    fx=ARUCO_VARIANT_UPSCALE,
                    fy=ARUCO_VARIANT_UPSCALE,
                    interpolation=interpolation,
                ),
                "scale": ARUCO_VARIANT_UPSCALE,
            }
        )

    return variants


def _rescale_marker_corners(corners: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return corners.astype(np.float32)
    return (corners.astype(np.float32) / float(scale)).astype(np.float32)


def _score_aruco_detection(detected: dict[str, Any]) -> tuple[int, int]:
    required_count = sum(1 for marker_id in REQUIRED_ARUCO_IDS if marker_id in detected["markerMap"])
    return required_count, len(detected["detectedIds"])


def _build_threshold_map(gray_warped: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray_warped, (5, 5), 0)
    otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    adaptive = cv2.adaptiveThreshold(
        cv2.equalizeHist(blurred),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    combined = cv2.bitwise_or(otsu, adaptive)
    return cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))


def _estimate_registration_offset(
    gray_warped: np.ndarray,
    registration: dict[str, Any],
    page: dict[str, Any],
    radius: int,
) -> tuple[int, int, bool]:
    start_x = _mm_x_to_px(_num(registration.get("startXmm"), 20.0), page)
    start_y = _mm_y_to_px(_num(registration.get("startYmm"), 40.0), page)
    columns = int(_num(registration.get("columns"), _num(registration.get("digits"), 7)))
    rows = int(_num(registration.get("rows"), 10))
    col_gap_px = _num(registration.get("colGapMm"), 7.0) * page["x_scale"]
    row_gap_px = _num(registration.get("rowGapMm"), 6.0) * page["y_scale"]

    nominal_score = 0.0
    for col in range(columns):
        for row in range(rows):
            nominal_score += _bubble_presence_score(
                gray_warped,
                start_x + int(round(col * col_gap_px)),
                start_y + int(round(row * row_gap_px)),
                radius,
            )

    bubble_count = max(1, columns * rows)
    nominal_avg_score = nominal_score / bubble_count
    if nominal_avg_score >= REGISTRATION_NOMINAL_SIGNAL_THRESHOLD:
        return 0, 0, False

    best_score = nominal_score
    best_offset = (0, 0)
    for offset_x in range(-80, 81, 4):
        for offset_y in range(-80, 81, 4):
            score = 0.0
            for col in range(columns):
                for row in range(rows):
                    score += _bubble_presence_score(
                        gray_warped,
                        start_x + int(round(col * col_gap_px)) + offset_x,
                        start_y + int(round(row * row_gap_px)) + offset_y,
                        radius,
                    )
            if score > best_score:
                best_score = score
                best_offset = (offset_x, offset_y)

    refined = best_score > nominal_score + 0.5
    return best_offset[0], best_offset[1], refined


def _estimate_answers_grid(
    gray_warped: np.ndarray,
    questions: dict[str, Any],
    page: dict[str, Any],
    radius: int,
) -> dict[str, dict[str, Any]]:
    question_count = int(_num(questions.get("questionCount"), 0))
    columns = int(_num(questions.get("columns"), 1))
    rows_per_column = int(_num(questions.get("rowsPerColumn"), question_count))
    option_gap = _num(questions.get("optionGapMm"), 7.0) * page["x_scale"]
    row_gap = _num(questions.get("rowGapMm"), 6.0) * page["y_scale"]
    block_gap = _num(questions.get("colGapMm"), 84.0)
    start_x = _num(questions.get("startXmm"), 20.0)
    start_y = _num(questions.get("startYmm"), 108.0)
    alternatives = questions.get("alternatives", ["A", "B", "C", "D", "E"])
    alternatives_count = len(alternatives) if isinstance(alternatives, list) and alternatives else 5

    refined_blocks: dict[str, dict[str, Any]] = {}

    block_count = max(1, min(columns, int(np.ceil(question_count / max(rows_per_column, 1)))))
    for block_column in range(block_count):
        block_rows = min(rows_per_column, question_count - (block_column * rows_per_column))
        if block_rows <= 0:
            continue

        base_x_mm = start_x + (block_column * block_gap)
        base_y_mm = start_y
        roi_margin_x = max(radius * 3, 24)
        roi_margin_y = max(radius * 3, 24)
        block_width_px = int(round((max(alternatives_count - 1, 0) * option_gap) + (radius * 3)))
        block_height_px = int(round((max(block_rows - 1, 0) * row_gap) + (radius * 3)))
        x0 = max(0, _mm_x_to_px(base_x_mm, page) - roi_margin_x)
        y0 = max(0, _mm_y_to_px(base_y_mm, page) - roi_margin_y)
        x1 = min(gray_warped.shape[1], _mm_x_to_px(base_x_mm, page) + block_width_px + roi_margin_x)
        y1 = min(gray_warped.shape[0], _mm_y_to_px(base_y_mm, page) + block_height_px + roi_margin_y)

        roi = gray_warped[y0:y1, x0:x1]
        circles = _detect_roi_circles(roi, radius)
        if circles is None or len(circles) == 0:
            continue

        x_candidates = _cluster_axis([circle[0] for circle in circles], tolerance=16.0)
        y_candidates = _cluster_axis([circle[1] for circle in circles], tolerance=16.0)
        expected_x = (_mm_x_to_px(base_x_mm, page) - x0)
        expected_y = (_mm_y_to_px(base_y_mm, page) - y0)

        selected_x = _select_axis_candidates(
            x_candidates,
            alternatives_count,
            expected_x,
            option_gap,
            lower_ratio=0.2,
        )
        selected_y = _select_axis_candidates(
            y_candidates,
            block_rows,
            expected_y,
            row_gap,
            lower_ratio=0.15,
        )
        if len(selected_x) != alternatives_count or len(selected_y) != block_rows:
            continue

        refined_blocks[str(block_column)] = {
            "columns": [float(x0 + value) for value in selected_x],
            "rows": [float(y0 + value) for value in selected_y],
            "radius": float(np.median([circle[2] for circle in circles])),
        }

    return refined_blocks


def _detect_roi_circles(roi: np.ndarray, radius: int) -> np.ndarray | None:
    if roi.size == 0:
        return None

    filtered = cv2.medianBlur(roi, 5)
    circles = cv2.HoughCircles(
        filtered,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(28, int(radius * 1.8)),
        param1=80,
        param2=18,
        minRadius=max(8, int(radius * 0.55)),
        maxRadius=max(12, int(radius * 1.35)),
    )
    if circles is None:
        return None
    return np.round(circles[0]).astype(np.int32)


def _cluster_axis(values: list[int], tolerance: float) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(float(item) for item in values):
        if not clusters or abs(value - float(np.mean(clusters[-1]))) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.mean(cluster)) for cluster in clusters]


def _select_axis_candidates(
    candidates: list[float],
    count: int,
    expected_start: float,
    expected_step: float,
    lower_ratio: float,
) -> list[float]:
    if count <= 0:
        return []

    if len(candidates) < count:
        return []

    upper_bound = expected_start + expected_step * (count - 1 + 0.4)
    lower_bound = expected_start - (expected_step * lower_ratio)
    filtered = [value for value in candidates if lower_bound <= value <= upper_bound]
    if len(filtered) < count:
        filtered = candidates

    best_score = None
    best_values: list[float] | None = None
    for combo in combinations(filtered, count):
        values = np.array(combo, dtype=np.float32)
        axis = np.arange(count, dtype=np.float32)
        slope, intercept = np.polyfit(axis, values, 1)
        prediction = intercept + (slope * axis)
        residual = float(np.mean((values - prediction) ** 2))
        score = residual + (0.4 * float((slope - expected_step) ** 2))
        if best_score is None or score < best_score:
            best_score = score
            best_values = list(float(item) for item in values)

    return [] if best_values is None else best_values


def _bubble_presence_score(gray_warped: np.ndarray, cx: int, cy: int, radius: int) -> float:
    roi, distances = _bubble_roi(gray_warped, cx, cy, radius)
    if roi.size == 0:
        return -1.0

    ring = (distances >= radius * 0.72) & (distances <= radius * 1.18)
    outer = (distances >= radius * 1.28) & (distances <= radius * 1.95)
    if not np.any(ring) or not np.any(outer):
        return -1.0

    ring_mean = float(roi[ring].mean())
    outer_mean = float(roi[outer].mean())
    return (outer_mean - ring_mean) / 255.0


def _fill_score(gray_warped: np.ndarray, cx: int, cy: int, radius: int) -> float:
    roi, distances = _bubble_roi(gray_warped, cx, cy, radius)
    if roi.size == 0:
        return 0.0

    inner = distances <= radius * 0.50
    outer = (distances >= radius * 1.10) & (distances <= radius * 1.80)
    if not np.any(inner) or not np.any(outer):
        return 0.0

    inner_mean = float(roi[inner].mean())
    outer_mean = float(roi[outer].mean())
    return max(0.0, min(1.0, (outer_mean - inner_mean) / max(outer_mean, 1.0)))


def _bubble_roi(gray_warped: np.ndarray, cx: int, cy: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    x0 = max(0, int(cx - (radius * 2)))
    y0 = max(0, int(cy - (radius * 2)))
    x1 = min(gray_warped.shape[1], int(cx + (radius * 2)) + 1)
    y1 = min(gray_warped.shape[0], int(cy + (radius * 2)) + 1)
    roi = gray_warped[y0:y1, x0:x1]
    if roi.size == 0:
        return roi, np.array([])

    yy, xx = np.ogrid[: roi.shape[0], : roi.shape[1]]
    distances = np.sqrt((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2)
    return roi, distances


def _read_registration(
    gray_warped: np.ndarray,
    geometry: dict[str, Any],
    page: dict[str, Any],
    threshold: float,
    delta: float,
) -> dict[str, Any]:
    reg = geometry.get("registration", {})
    if not isinstance(reg, dict):
        reg = {}

    digits = int(_num(reg.get("digits"), 7))
    rows = int(_num(reg.get("rows"), 10))
    columns = int(_num(reg.get("columns"), digits))

    start_x = _num(reg.get("startXmm"), 20.0)
    start_y = _num(reg.get("startYmm"), 40.0)
    col_gap = _num(reg.get("colGapMm"), 7.0)
    row_gap = _num(reg.get("rowGapMm"), 6.0)
    bubble_diameter = _num(reg.get("bubbleDiameterMm"), 4.0)
    strict_one_mark = bool(reg.get("strictOneMarkPerColumn", True))

    radius = max(4, int(round((bubble_diameter / 2.0) * page["x_scale"])))
    offset_x, offset_y, refined = _estimate_registration_offset(gray_warped, reg, page, radius)

    values: list[str] = []
    details: list[dict[str, Any]] = []
    has_missing = False
    has_ambiguous = False

    for col in range(columns):
        bubbles: list[tuple[int, int, int]] = []
        ratios: list[float] = []

        x_mm = start_x + (col * col_gap)
        cx = _mm_x_to_px(x_mm, page) + offset_x

        for row in range(rows):
            y_mm = start_y + (row * row_gap)
            cy = _mm_y_to_px(y_mm, page) + offset_y
            bubbles.append((cx, cy, radius))
            ratios.append(_fill_score(gray_warped, cx, cy, radius))

        decision, best_index, second_index = _choose(ratios, threshold, delta)
        details.append(
            {
                "bubbles": bubbles,
                "decision": decision,
                "bestIndex": best_index,
                "secondIndex": second_index,
                "ratios": ratios,
            }
        )

        if decision == -1:
            has_missing = True
            continue
        if decision == -2:
            has_ambiguous = True
            continue

        values.append(str(decision))

    status = "ok"
    if has_ambiguous:
        status = "ambiguous"
    elif has_missing and strict_one_mark:
        status = "missing"
    elif len(values) != columns:
        status = "invalid"

    value = "".join(values) if status == "ok" and len(values) == columns else None

    return {
        "status": status,
        "value": value,
        "details": details,
        "refinement": {
            "mode": "block_offset" if refined else "nominal",
            "offsetPx": [offset_x, offset_y],
        },
    }


def _read_answers(
    gray_warped: np.ndarray,
    geometry: dict[str, Any],
    page: dict[str, Any],
    threshold: float,
    delta: float,
) -> dict[str, Any]:
    questions = geometry.get("questions", {})
    if not isinstance(questions, dict):
        questions = {}

    question_count = int(_num(questions.get("questionCount"), 0))
    if question_count <= 0:
        raise ValueError("questions.questionCount invalido.")

    alternatives = questions.get("alternatives", ["A", "B", "C", "D", "E"])
    if not isinstance(alternatives, list) or len(alternatives) == 0:
        alternatives = ["A", "B", "C", "D", "E"]

    alternatives_count = len(alternatives)

    columns = int(_num(questions.get("columns"), 1))
    rows_per_column = int(_num(questions.get("rowsPerColumn"), question_count))
    start_x = _num(questions.get("startXmm"), 20.0)
    start_y = _num(questions.get("startYmm"), 108.0)
    col_gap = _num(questions.get("colGapMm"), 84.0)
    row_gap = _num(questions.get("rowGapMm"), 6.0)
    option_gap = _num(questions.get("optionGapMm"), 7.0)
    bubble_diameter = _num(questions.get("bubbleDiameterMm"), 4.0)

    radius = max(4, int(round((bubble_diameter / 2.0) * page["x_scale"])))
    refined_grid = _estimate_answers_grid(gray_warped, questions, page, radius)

    answers: list[dict[str, Any]] = []
    answers_numeric: list[int] = []
    details: list[dict[str, Any]] = []

    for index in range(question_count):
        block_column = min(columns - 1, index // rows_per_column)
        block_row = index % rows_per_column

        base_x = start_x + (block_column * col_gap)
        base_y = start_y + (block_row * row_gap)
        refined_block = refined_grid.get(str(block_column))
        refined_row = (
            None
            if refined_block is None or block_row >= len(refined_block["rows"])
            else refined_block["rows"][block_row]
        )

        bubbles: list[tuple[int, int, int]] = []
        ratios: list[float] = []
        for option in range(alternatives_count):
            nominal_cx = _mm_x_to_px(base_x + (option * option_gap), page)
            nominal_cy = _mm_y_to_px(base_y, page)
            if refined_block is None or option >= len(refined_block["columns"]) or refined_row is None:
                cx = nominal_cx
                cy = nominal_cy
            else:
                cx = int(round(refined_block["columns"][option]))
                cy = int(round(refined_row))
            bubbles.append((cx, cy, radius))
            ratios.append(_fill_score(gray_warped, cx, cy, radius))

        decision, best_index, second_index = _choose(ratios, threshold, delta)
        details.append(
            {
                "bubbles": bubbles,
                "decision": decision,
                "bestIndex": best_index,
                "secondIndex": second_index,
                "ratios": ratios,
            }
        )

        selected = None
        is_ambiguous = False
        numeric = -1

        if decision == -2:
            is_ambiguous = True
            numeric = -2
        elif decision >= 0:
            selected = decision + 1
            numeric = decision

        answers.append(
            {
                "question": index + 1,
                "selected": selected,
                "isAmbiguous": is_ambiguous,
                "confidence": [round(value, 4) for value in ratios],
            }
        )
        answers_numeric.append(numeric)

    return {
        "answers": answers,
        "answers_numeric": answers_numeric,
        "details": details,
        "refinement": {
            "mode": "local_grid",
            "blocks": refined_grid,
        },
    }


def _draw_overlay(
    warped: np.ndarray,
    registration_details: list[dict[str, Any]],
    answers_details: list[dict[str, Any]],
    master_answers: list[int | None] | None = None,
) -> np.ndarray:
    overlay = warped.copy()

    _draw_cells(overlay, registration_details)
    _draw_answer_cells(overlay, answers_details, master_answers)

    return overlay


def _draw_cells(image: np.ndarray, details: list[dict[str, Any]]) -> None:
    for row in details:
        bubbles = row.get("bubbles", [])
        decision = int(row.get("decision", -1))
        best_index = int(row.get("bestIndex", -1))
        second_index = int(row.get("secondIndex", -1))

        for (cx, cy, radius) in bubbles:
            cv2.circle(image, (int(cx), int(cy)), int(radius), (160, 160, 160), 1)

        if decision >= 0 and decision < len(bubbles):
            cx, cy, radius = bubbles[decision]
            cv2.circle(image, (int(cx), int(cy)), int(max(2, radius - 2)), (0, 170, 0), -1)
            cv2.circle(image, (int(cx), int(cy)), int(radius), (0, 200, 0), 2)

        if decision == -2:
            if 0 <= best_index < len(bubbles):
                cx, cy, radius = bubbles[best_index]
                cv2.circle(image, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            if 0 <= second_index < len(bubbles):
                cx, cy, radius = bubbles[second_index]
                cv2.circle(image, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)


def _draw_answer_cells(
    image: np.ndarray,
    details: list[dict[str, Any]],
    master_answers: list[int | None] | None,
) -> None:
    for index, row in enumerate(details):
        bubbles = row.get("bubbles", [])
        decision = int(row.get("decision", -1))
        best_index = int(row.get("bestIndex", -1))
        second_index = int(row.get("secondIndex", -1))
        master_answer = _normalize_master_answer(
            None if master_answers is None or index >= len(master_answers) else master_answers[index],
            len(bubbles),
        )

        for (cx, cy, radius) in bubbles:
            cv2.circle(image, (int(cx), int(cy)), int(radius), (160, 160, 160), 1)

        if master_answer is None:
            if decision >= 0 and decision < len(bubbles):
                _draw_filled_choice(image, bubbles[decision], (0, 170, 0), (0, 200, 0))
        else:
            _draw_choice_ring(image, bubbles[master_answer - 1], (0, 200, 0), 2, 1)

            if decision >= 0 and decision < len(bubbles):
                selected_answer = decision + 1
                if selected_answer == master_answer:
                    _draw_filled_choice(image, bubbles[decision], (0, 170, 0), (0, 200, 0))
                else:
                    _draw_filled_choice(image, bubbles[decision], (0, 140, 255), (0, 165, 255))

        if decision == -2:
            if 0 <= best_index < len(bubbles):
                _draw_choice_ring(image, bubbles[best_index], (0, 0, 255), 2, 0)
            if 0 <= second_index < len(bubbles):
                _draw_choice_ring(image, bubbles[second_index], (0, 0, 255), 2, 0)


def _draw_filled_choice(
    image: np.ndarray,
    bubble: tuple[int, int, int],
    fill_color: tuple[int, int, int],
    stroke_color: tuple[int, int, int],
) -> None:
    cx, cy, radius = bubble
    cv2.circle(image, (int(cx), int(cy)), int(max(2, radius - 2)), fill_color, -1)
    cv2.circle(image, (int(cx), int(cy)), int(radius), stroke_color, 2)


def _draw_choice_ring(
    image: np.ndarray,
    bubble: tuple[int, int, int],
    color: tuple[int, int, int],
    thickness: int,
    radius_offset: int,
) -> None:
    cx, cy, radius = bubble
    cv2.circle(image, (int(cx), int(cy)), int(radius + radius_offset), color, thickness)


def _normalize_master_answer(
    value: int | None,
    bubbles_count: int,
) -> int | None:
    if not isinstance(value, int):
        return None
    if value < 1 or value > bubbles_count:
        return None
    return value


def _summarize_geometry(compiled_geometry_json: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if compiled_geometry_json is None:
        return None

    geometry = compiled_geometry_json
    if isinstance(geometry, str):
        try:
            geometry = json.loads(geometry)
        except json.JSONDecodeError:
            return {"rawType": "string", "parseable": False}

    if not isinstance(geometry, dict):
        return {"rawType": type(compiled_geometry_json).__name__}

    questions = geometry.get("questions")
    registration = geometry.get("registration")
    return {
        "page": geometry.get("page"),
        "anchors": geometry.get("anchors"),
        "registration": registration,
        "questions": {
            "questionCount": None if not isinstance(questions, dict) else questions.get("questionCount"),
            "columns": None if not isinstance(questions, dict) else questions.get("columns"),
            "rowsPerColumn": None if not isinstance(questions, dict) else questions.get("rowsPerColumn"),
            "alternatives": None if not isinstance(questions, dict) else questions.get("alternatives"),
        },
        "registrationColumns": None
        if not isinstance(registration, dict)
        else registration.get("columns"),
    }


def _summarize_registration(registration: dict[str, Any] | None) -> dict[str, Any] | None:
    if registration is None:
        return None
    return {
        "status": registration.get("status"),
        "value": registration.get("value"),
        "details": registration.get("details"),
        "refinement": registration.get("refinement"),
    }


def _summarize_answers(answers: dict[str, Any] | None) -> dict[str, Any] | None:
    if answers is None:
        return None
    return {
        "answers": answers.get("answers"),
        "answers_numeric": answers.get("answers_numeric"),
        "details": answers.get("details"),
        "refinement": answers.get("refinement"),
    }


def _summarize_response(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None

    payload = {
        key: value
        for key, value in result.items()
        if key != "images"
    }
    if "images" in result:
        payload["images"] = {
            "hasRectifiedBase64": bool(result["images"].get("rectifiedBase64")),
            "hasOverlayBase64": bool(result["images"].get("overlayBase64")),
        }
    return payload


def _encode_image_base64(image: np.ndarray) -> str | None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _fill_ratio(thresholded: np.ndarray, cx: int, cy: int, radius: int) -> float:
    inner_radius = int(max(3, radius * 0.55))
    x0 = max(0, cx - inner_radius)
    y0 = max(0, cy - inner_radius)
    x1 = min(thresholded.shape[1], cx + inner_radius)
    y1 = min(thresholded.shape[0], cy + inner_radius)

    roi = thresholded[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[: roi.shape[0], : roi.shape[1]]
    mx, my = roi.shape[1] // 2, roi.shape[0] // 2
    mask = (xx - mx) ** 2 + (yy - my) ** 2 <= inner_radius * inner_radius
    total = int(mask.sum())
    if total == 0:
        return 0.0

    filled = int(np.count_nonzero(roi[mask]))
    return float(filled) / float(total)


def _choose(values: list[float], threshold: float, delta: float) -> tuple[int, int, int]:
    if not values:
        return -1, -1, -1

    arr = np.array(values, dtype=np.float32)
    best_index = int(np.argmax(arr))
    max_value = float(arr[best_index])
    sorted_idx = np.argsort(arr)
    second_index = int(sorted_idx[-2]) if len(arr) > 1 else best_index
    second_value = float(arr[second_index]) if len(arr) > 1 else 0.0

    if max_value < threshold:
        median_value = float(np.median(arr))
        if (
            max_value >= LOW_SIGNAL_THRESHOLD
            and max_value - second_value >= LOW_SIGNAL_DELTA
            and max_value - median_value >= LOW_SIGNAL_MEDIAN_GAP
        ):
            return best_index, best_index, second_index
        return -1, best_index, second_index

    if max_value - second_value < delta:
        return -2, best_index, second_index

    return best_index, best_index, second_index


def _mm_x_to_px(value_mm: float, page: dict[str, Any]) -> int:
    return int(round(value_mm * page["x_scale"]))


def _mm_y_to_px(value_mm: float, page: dict[str, Any]) -> int:
    return int(round(value_mm * page["y_scale"]))


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
