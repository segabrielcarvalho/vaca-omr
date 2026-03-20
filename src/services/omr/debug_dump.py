from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEBUG_ROOT_DIR = Path("/tmp/vaca-omr-debug")
HOST_VISIBLE_DEBUG_ROOT_DIR = Path("/app/.tmp/vaca-omr-debug")
DEBUG_ROOT_DIRS = (
    DEBUG_ROOT_DIR,
    HOST_VISIBLE_DEBUG_ROOT_DIR,
)
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def write_omr_debug_dump(
    *,
    capture_id: str | None,
    session_id: str | None,
    artifacts: dict[str, np.ndarray | None],
    metadata: dict[str, Any],
) -> list[str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session_segment = _sanitize_path_segment(session_id, "session-unknown")
    capture_segment = _sanitize_path_segment(capture_id, "capture-unknown")
    payload = {
        "captureId": capture_id,
        "sessionId": session_id,
        "savedAt": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    successful_paths: list[str] = []
    failures: list[str] = []

    unique_roots = list(dict.fromkeys(DEBUG_ROOT_DIRS))
    for root_dir in unique_roots:
        dump_dir = root_dir / session_segment / f"{capture_segment}__{timestamp}"
        try:
            dump_dir.mkdir(parents=True, exist_ok=True)

            for filename, image in artifacts.items():
                if image is None:
                    continue
                _write_image(dump_dir / filename, image)

            (dump_dir / "metadata.json").write_text(
                json.dumps(
                    _to_json_compatible(payload),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            successful_paths.append(str(dump_dir))
        except Exception as exc:
            failures.append(f"{dump_dir}: {exc}")

    if not successful_paths and failures:
        raise RuntimeError("; ".join(failures))

    return successful_paths


def _sanitize_path_segment(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    sanitized = _SAFE_SEGMENT_RE.sub("_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or fallback


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Não foi possível gravar artefato de debug em {path}.")


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value
