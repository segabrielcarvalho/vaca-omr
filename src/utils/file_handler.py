import cv2
import json
import datetime
from pathlib import Path


def save_img(path, img):
    cv2.imwrite(str(path), img)


def save_metadata(outdir, log_data):
    with open(outdir / "metadata.json", "w", encoding="utf-8") as mf:
        json.dump(log_data, mf, ensure_ascii=False, indent=2)


def create_log_data(img_path, status, ids, t, d, img, warped, picks, dt_detect, dt_choice, dt_total):
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "file": Path(img_path).name,
        "status": status,
        "aruco_count": 0 if ids is None else int(len(ids)),
        "threshold_t": t,
        "delta_d": d,
        "input_shape": list(img.shape) if img is not None else None,
        "warp_shape": list(warped.shape) if warped is not None else None,
        "time_ms_detect_sheet": round(dt_detect, 2),
        "time_ms_detect_choices": round(dt_choice, 2),
        "time_ms_total": round(dt_total, 2),
        "picks": picks if picks else []
    }


def save_all_images(outdir, disp, warped, th, base_prev, result_prev, panel, bd):
    save_img(outdir / "disp.png", disp)
    
    if warped is not None:
        save_img(outdir / "warp.png", warped)
    
    if th is not None:
        save_img(outdir / "th.png", th)
    
    if base_prev is not None:
        save_img(outdir / "base.png", base_prev)
    
    if result_prev is not None:
        save_img(outdir / "result.png", result_prev)
    
    save_img(outdir / "panel.png", panel)
    save_img(outdir / "board.png", bd)
