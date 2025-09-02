import cv2
import numpy as np

W, H = 1400, 1000


def order_ids(corners, ids):
    idx = {int(i): k for k, i in enumerate(ids.flatten())}
    need = [0, 1, 2, 3]
    if not all(i in idx for i in need):
        return None
    C = [corners[idx[i]][0].mean(axis=0) for i in need]
    return np.array(C, np.float32)


def detect_sheet(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    
    disp = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(disp, corners, ids)
    
    if ids is None or len(ids) < 4:
        return disp, None, None, "Aruco não encontrado", ids
    
    src = order_ids(corners, ids)
    if src is None:
        return disp, None, None, "IDs 0..3 ausentes", ids
    
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (W, H))
    
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    return disp, warped, th, "ok", ids
