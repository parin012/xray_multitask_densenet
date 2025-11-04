#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, glob, cv2, torch
from torch.utils.data import Dataset
from .preprocess import rasterize_polygon_mask, preprocess_xray_1ch_array

def _list_subdirs(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def make_items_by_order(raw_base, label_base, num_classes=5):
    """
    (img_path, json_path, class_id) 리스트와 클래스 폴더명 목록 반환
    - raw_base/클래스명/*.png
    - label_base/클래스명/*.json
    """
    raw_sub = _list_subdirs(raw_base)
    lbl_sub = _list_subdirs(label_base)
    assert len(raw_sub) == len(lbl_sub) == num_classes, \
        f"class 수 불일치: raw={len(raw_sub)}, lbl={len(lbl_sub)}, expect={num_classes}"
    items = []
    for i, (rf, lf) in enumerate(zip(raw_sub, lbl_sub), start=1):
        img_dir = os.path.join(raw_base, rf)
        lbl_dir = os.path.join(label_base, lf)
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(lbl_dir, stem + ".json")
            if os.path.exists(json_path):
                items.append((img_path, json_path, i))
    return items, raw_sub

class XraySegClsDataset(Dataset):
    def __init__(self, items, img_size=448):
        self.items = items
        self.img_size = img_size
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img_path, json_path, cls_id = self.items[i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        mask = rasterize_polygon_mask(json_path, H, W)
        x = preprocess_xray_1ch_array(img, (self.img_size, self.img_size))
        m = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        m = torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0)
        y = torch.tensor(cls_id - 1, dtype=torch.long)
        return x, m, y

