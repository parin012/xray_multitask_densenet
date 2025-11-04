#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json, cv2, numpy as np, torch

def rasterize_polygon_mask(json_path, H=None, W=None):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        meta = json.load(f)
    if H is None or W is None:
        H = meta.get("imageHeight", 512)
        W = meta.get("imageWidth",  512)
    mask = np.zeros((H, W), dtype=np.uint8)
    for shp in meta.get("shapes", []):
        pts = np.array(shp.get("points", []), dtype=np.int32)
        if pts.ndim == 2 and len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask

def preprocess_xray_1ch_array(gray_u8, out_hw=(448, 448)):
    g = cv2.resize(gray_u8, out_hw[::-1], interpolation=cv2.INTER_AREA)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    # 1~99 퍼센타일 스트레치
    p1, p99 = np.percentile(g, [1, 99])
    g = np.clip((g - p1) / max(p99 - p1, 1e-6), 0, 1)
    g = (g * 255).astype(np.uint8)
    # 표준화 텐서
    x = torch.from_numpy(g.astype(np.float32)/255.0).unsqueeze(0)
    x = (x - 0.5) / 0.25
    return x

