import cv2
import numpy as np
from .sheet_geometry import group_sir

W, H = 1400, 1000


def draw_result(img, centers, picks):
    out = img.copy()
    rows = group_sir(centers)
    
    for ri, row in enumerate(rows):
        c = picks[ri] if ri < len(picks) else -1
        for j, (cx, cy, r) in enumerate(row):
            col = (0, 200, 0)
            if c == j:
                col = (0, 0, 255)
                cv2.circle(out, (cx, cy), r-3, col, -1)
            cv2.circle(out, (cx, cy), r, col, 2)
    
    return out


def centers_preview(img, centers, color=(0, 200, 0)):
    out = img.copy()
    for cx, cy, r in centers:
        cv2.circle(out, (cx, cy), r, color, 2)
    return out


def wrap_lines(text, max_chars):
    words = text.split(' ')
    lines = []
    cur = ""
    
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    
    if cur:
        lines.append(cur)
    
    return lines


def info_panel(picks, h=720, w=300):
    panel = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(panel, (0, 0), (w-1, h-1), (80, 80, 80), 2)
    cv2.putText(panel, "Posicoes", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 40, 40), 2, cv2.LINE_AA)
    
    arr = "[" + ", ".join(str(x) for x in picks) + "]"
    y = 62
    
    for line in wrap_lines(arr, 38):
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        y += 18
    
    y += 6
    
    for i, v in enumerate(picks, start=1):
        cv2.putText(panel, f"{i:02d}: {v}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 20
        if y > h - 10:
            break
    
    return panel


def fit_h(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*s), h), interpolation=cv2.INTER_AREA)


def board(cam, warped, th, base_prev, result_prev, picks, h=720):
    A = fit_h(cam, h)
    B = fit_h(warped if warped is not None else np.zeros((H, W, 3), np.uint8), h)
    
    C = th if th is not None else np.zeros((H, W), np.uint8)
    if len(C.shape) == 2:
        C = cv2.cvtColor(C, cv2.COLOR_GRAY2BGR)
    C = fit_h(C, h)
    
    P = info_panel(picks, h=h, w=300)
    top = cv2.hconcat([A, B, C, P])
    
    h2 = h//2
    D = fit_h(base_prev if base_prev is not None else np.zeros((H, W, 3), np.uint8), h2)
    E = fit_h(result_prev if result_prev is not None else np.zeros((H, W, 3), np.uint8), h2)
    
    right_w = max(0, top.shape[1] - D.shape[1] - E.shape[1])
    bottom = cv2.hconcat([D, E, np.full((h2, right_w, 3), 255, np.uint8)])
    
    out = cv2.vconcat([top, bottom])
    
    cv2.putText(out, "Imagem", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(out, "Warp", (A.shape[1]+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(out, "Binarizado", (A.shape[1]+B.shape[1]+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(out, "Grade base", (20, h+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(out, "Respostas", (A.shape[1]//2+20, h+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
    
    return out
