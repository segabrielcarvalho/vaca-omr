import numpy as np

W, H = 1400, 1000
MM = {
    "block_w": 180.0,
    "block_h": 112.0,
    "pad_x": 4.0,
    "pad_y": 4.0,
    "gap": 4.0,
    "num_w": 7.0,
    "d": 6.2,
    "sx": 8.6,
    "sy": 8.4,
    "header_h": 9.0
}


def _scaled_mm():
    bw = MM["block_w"]
    bh = MM["block_h"]
    d = MM["d"]
    sx = MM["sx"]
    sy = MM["sy"]
    num_w = MM["num_w"]
    gap = MM["gap"]
    pad_x = MM["pad_x"]
    pad_y = MM["pad_y"]
    header_h = MM["header_h"]
    
    sec_w = num_w + (d + 4*sx)
    total_w = 2*pad_x + 3*sec_w + 2*gap
    scale = bw / total_w
    
    d *= scale
    sx *= scale
    sy *= scale
    num_w *= scale
    gap *= scale
    pad_x *= scale
    pad_y *= scale
    header_h *= scale
    sec_w *= scale
    
    kx = W / bw
    ky = H / bh
    
    return dict(
        d=d, sx=sx, sy=sy, num_w=num_w, gap=gap,
        pad_x=pad_x, pad_y=pad_y, header_h=header_h,
        sec_w=sec_w, kx=kx, ky=ky
    )


def base_centers():
    P = _scaled_mm()
    bx = P["pad_x"]
    by = P["pad_y"] + P["header_h"]
    xs = [bx, bx + P["sec_w"] + P["gap"], bx + 2*(P["sec_w"] + P["gap"])]
    
    C = []
    r = int(round((P["d"]/2)*P["kx"]))
    
    for s in range(3):
        s_off = xs[s]
        for i in range(10):
            y_mm = by + i*P["sy"]
            for j in range(5):
                cx_mm = s_off + P["num_w"] + j*P["sx"]
                cy_mm = y_mm
                cx = int(round(cx_mm*P["kx"]))
                cy = int(round(cy_mm*P["ky"]))
                C.append((cx, cy, r))
    
    return C


def group_sir(centers):
    out = []
    k = 0
    
    for s in range(3):
        for i in range(10):
            row = []
            for j in range(5):
                row.append(centers[k])
                k += 1
            out.append(row)
    
    return out
