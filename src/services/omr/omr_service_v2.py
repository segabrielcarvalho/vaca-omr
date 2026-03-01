import base64
import json
import os
import time
from typing import Any

import cv2
import numpy as np

ENGINE_VERSION = os.environ.get("OMR_ENGINE_VERSION", "2.0.0-template-driven")
REQUIRED_ARUCO_IDS = (0, 1, 2, 3)


def process_image_dynamic(
    image_base64: str,
    compiled_geometry_json: dict[str, Any] | str,
    threshold: float = 0.50,
    delta: float = 0.12,
) -> dict[str, Any]:
    timings: dict[str, Any] = {}
    started = time.perf_counter()

    try:
        t = time.perf_counter()
        image = _decode_base64_image(image_base64)
        timings["decodeMs"] = _elapsed_ms(t)
    except ValueError as exc:
        timings["totalMs"] = _elapsed_ms(started)
        return _error_response("IMAGE_DECODE_FAILED", str(exc), timings)

    try:
        geometry = _normalize_geometry(compiled_geometry_json)
        page = _normalize_page(geometry)
    except ValueError as exc:
        timings["totalMs"] = _elapsed_ms(started)
        return _error_response("GEOMETRY_INVALID", str(exc), timings)

    t = time.perf_counter()
    sheet = _detect_and_rectify_sheet(image, geometry, page)
    timings["rectifyMs"] = _elapsed_ms(t)

    if not sheet["ok"]:
        timings["totalMs"] = _elapsed_ms(started)
        return _error_response(
            sheet["errorCode"],
            sheet["errorMessage"],
            timings,
            extra={"arucoDetected": sheet.get("arucoDetected", [])},
        )

    warped = sheet["warped"]
    thresholded = sheet["thresholded"]

    t = time.perf_counter()
    qr_data = _read_qr(warped, geometry, page)
    timings["qrMs"] = _elapsed_ms(t)

    t = time.perf_counter()
    try:
        registration = _read_registration(thresholded, geometry, page, threshold, delta)
    except ValueError as exc:
        timings["registrationMs"] = _elapsed_ms(t)
        timings["totalMs"] = _elapsed_ms(started)
        return _error_response("GEOMETRY_INVALID", str(exc), timings)
    timings["registrationMs"] = _elapsed_ms(t)

    t = time.perf_counter()
    try:
        answers = _read_answers(thresholded, geometry, page, threshold, delta)
    except ValueError as exc:
        timings["answersMs"] = _elapsed_ms(t)
        timings["totalMs"] = _elapsed_ms(started)
        return _error_response("GEOMETRY_INVALID", str(exc), timings)
    timings["answersMs"] = _elapsed_ms(t)

    t = time.perf_counter()
    overlay = _draw_overlay(
        warped,
        registration["details"],
        answers["details"],
        _qr_rect(geometry, page),
    )
    rectified_base64 = _encode_image_base64(warped)
    overlay_base64 = _encode_image_base64(overlay)
    timings["overlayMs"] = _elapsed_ms(t)

    timings["totalMs"] = _elapsed_ms(started)

    return {
        "success": True,
        "engineVersion": ENGINE_VERSION,
        "qr": {
            "data": qr_data.get("data"),
            "points": qr_data.get("points"),
        },
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


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _error_response(
    code: str,
    message: str,
    timings: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "success": False,
        "engineVersion": ENGINE_VERSION,
        "qr": {"data": None, "points": None},
        "registration": {"value": None, "status": "invalid"},
        "answers": [],
        "answers_numeric": [],
        "timings": timings,
        "error": {"code": code, "message": message},
    }
    if extra:
        payload.update(extra)
    return payload


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
    corners, ids = _detect_aruco_markers(gray)

    if ids is None or len(ids) == 0:
        return {
            "ok": False,
            "errorCode": "ARUCO_NOT_FOUND",
            "errorMessage": "Nenhuma ancora ArUco encontrada.",
            "arucoDetected": [],
        }

    detected_ids = [int(value) for value in ids.flatten().tolist()]
    marker_map: dict[int, np.ndarray] = {}
    for index, marker_id in enumerate(detected_ids):
        marker_map[marker_id] = corners[index][0].mean(axis=0).astype(np.float32)

    if not all(marker_id in marker_map for marker_id in REQUIRED_ARUCO_IDS):
        return {
            "ok": False,
            "errorCode": "ARUCO_MISSING_IDS",
            "errorMessage": "Nem todas as ancoras ArUco obrigatorias (0..3) foram encontradas.",
            "arucoDetected": detected_ids,
        }

    target_map = _target_anchor_centers(geometry, page)
    source = np.array([marker_map[marker_id] for marker_id in REQUIRED_ARUCO_IDS], np.float32)
    target = np.array([target_map[marker_id] for marker_id in REQUIRED_ARUCO_IDS], np.float32)

    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(image, matrix, (page["width_px"], page["height_px"]))
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.threshold(
        gray_warped,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    return {
        "ok": True,
        "warped": warped,
        "thresholded": thresholded,
        "arucoDetected": detected_ids,
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


def _detect_aruco_markers(gray: np.ndarray) -> tuple[Any, Any]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids


def _read_qr(
    warped: np.ndarray,
    geometry: dict[str, Any],
    page: dict[str, Any],
) -> dict[str, Any]:
    detector = cv2.QRCodeDetector()

    qr_x0, qr_y0, qr_x1, qr_y1 = _qr_rect(geometry, page)
    roi = warped[qr_y0:qr_y1, qr_x0:qr_x1]

    data = ""
    points = None

    if roi.size > 0:
        data, points, _ = detector.detectAndDecode(roi)
        if data and points is not None:
            points[:, :, 0] += qr_x0
            points[:, :, 1] += qr_y0

    if not data:
        data, points, _ = detector.detectAndDecode(warped)

    return {
        "data": data or None,
        "points": points.tolist() if points is not None else None,
    }


def _qr_rect(geometry: dict[str, Any], page: dict[str, Any]) -> tuple[int, int, int, int]:
    qr = geometry.get("qr", {})
    if not isinstance(qr, dict):
        qr = {}

    x_mm = _num(qr.get("xMm"), 170.0)
    y_mm = _num(qr.get("yMm"), 24.0)
    size_mm = _num(qr.get("sizeMm"), 28.0)
    margin_mm = _num(qr.get("marginMm"), 3.0)

    x0 = _clamp(_mm_x_to_px(x_mm - margin_mm, page), 0, page["width_px"] - 1)
    y0 = _clamp(_mm_y_to_px(y_mm - margin_mm, page), 0, page["height_px"] - 1)
    x1 = _clamp(
        _mm_x_to_px(x_mm + size_mm + margin_mm, page),
        x0 + 1,
        page["width_px"],
    )
    y1 = _clamp(
        _mm_y_to_px(y_mm + size_mm + margin_mm, page),
        y0 + 1,
        page["height_px"],
    )

    return x0, y0, x1, y1


def _read_registration(
    thresholded: np.ndarray,
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

    values: list[str] = []
    details: list[dict[str, Any]] = []
    has_missing = False
    has_ambiguous = False

    for col in range(columns):
        bubbles: list[tuple[int, int, int]] = []
        ratios: list[float] = []

        x_mm = start_x + (col * col_gap)
        cx = _mm_x_to_px(x_mm, page)

        for row in range(rows):
            y_mm = start_y + (row * row_gap)
            cy = _mm_y_to_px(y_mm, page)
            bubbles.append((cx, cy, radius))
            ratios.append(_fill_ratio(thresholded, cx, cy, radius))

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
    }


def _read_answers(
    thresholded: np.ndarray,
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

    answers: list[dict[str, Any]] = []
    answers_numeric: list[int] = []
    details: list[dict[str, Any]] = []

    for index in range(question_count):
        block_column = min(columns - 1, index // rows_per_column)
        block_row = index % rows_per_column

        base_x = start_x + (block_column * col_gap)
        base_y = start_y + (block_row * row_gap)

        bubbles: list[tuple[int, int, int]] = []
        ratios: list[float] = []
        for option in range(alternatives_count):
            cx = _mm_x_to_px(base_x + (option * option_gap), page)
            cy = _mm_y_to_px(base_y, page)
            bubbles.append((cx, cy, radius))
            ratios.append(_fill_ratio(thresholded, cx, cy, radius))

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
    }


def _draw_overlay(
    warped: np.ndarray,
    registration_details: list[dict[str, Any]],
    answers_details: list[dict[str, Any]],
    qr_rect: tuple[int, int, int, int],
) -> np.ndarray:
    overlay = warped.copy()

    _draw_cells(overlay, registration_details)
    _draw_cells(overlay, answers_details)

    x0, y0, x1, y1 = qr_rect
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 200, 0), 2)

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
        return -1, best_index, second_index

    if max_value - second_value < delta:
        return -2, best_index, second_index

    return best_index, best_index, second_index


def _mm_x_to_px(value_mm: float, page: dict[str, Any]) -> int:
    return int(round(value_mm * page["x_scale"]))


def _mm_y_to_px(value_mm: float, page: dict[str, Any]) -> int:
    return int(round(value_mm * page["y_scale"]))


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
