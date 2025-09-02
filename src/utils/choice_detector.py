import numpy as np
from .sheet_geometry import group_sir


def ratio_fill(th, cx, cy, r):
    rr = int(max(4, r*0.55))
    x0 = max(0, cx-rr)
    y0 = max(0, cy-rr)
    x1 = min(th.shape[1], cx+rr)
    y1 = min(th.shape[0], cy+rr)
    
    roi = th[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    
    y, x = np.ogrid[:roi.shape[0], :roi.shape[1]]
    mx, my = roi.shape[1]//2, roi.shape[0]//2
    mask = (x-mx)**2 + (y-my)**2 <= rr*rr
    
    return float(np.count_nonzero(roi[mask]))/float(mask.sum())


def choose(vals, t=0.50, d=0.12):
    a = int(np.argmax(vals))
    m = float(vals[a])
    b = float(np.partition(vals, -2)[-2]) if len(vals) > 1 else 0.0
    
    if m < t:
        return -1 
    if m - b < d:
        return -2
    
    return a


def detect_choices(th, centers, t=0.50, d=0.12):
    rows = group_sir(centers)
    fills = []
    picks = []
    
    for row in rows:
        vs = [ratio_fill(th, cx, cy, r) for (cx, cy, r) in row]
        fills.append(vs)
        picks.append(choose(vs, t, d))
    
    return fills, picks
